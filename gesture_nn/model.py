"""
Lightweight gesture classifier: 1D-CNN + Self-Attention for real-time AR inference.

Architecture (matching thesis Direction 1 design):
  Input:  [T=32, J=26, C=3]  → normalize to wrist-relative
  Spatial: 1×1 Conv projects (J, C) → d_model
  Temporal: 3-layer dilated 1D-CNN (k=3, d=1,2,4)
  Global:   1-layer lightweight Self-Attention (4 heads, over T dim)
  Pool:     Global Average Pooling → d_model
  Head:     Linear(d_model, 7) → Softmax

Total params: ~45K, target inference <2ms on Snapdragon XR2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config import D_MODEL, N_HEADS, N_CNN_LAYERS, CNN_KERNEL, DILATIONS, DROPOUT, N_GESTURE_CLASSES


class SpatialEmbedding(nn.Module):
    """Project per-frame joint coords (J, C) → d_model using 1×1 Conv."""

    def __init__(self, in_channels: int = 3, d_model: int = 64):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, J, C]
        Returns:
            [B, T, J, d_model]
        """
        B, T, J, C = x.shape
        x = x.view(B * T, J, C).permute(0, 2, 1)  # [B*T, C, J]
        x = self.proj(x)                             # [B*T, d_model, J]
        x = x.permute(0, 2, 1)                       # [B*T, J, d_model]
        x = self.norm(x)
        x = x.view(B, T, J, -1)
        return x


class JointWisePool(nn.Module):
    """Pool across joints: [B, T, J, D] → [B, T, D]."""

    def __init__(self, pool_type: str = "mean"):
        super().__init__()
        self.pool_type = pool_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "mean":
            return x.mean(dim=2)
        elif self.pool_type == "max":
            return x.max(dim=2).values
        elif self.pool_type == "meanmax":
            return torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=-1)
        return x.mean(dim=2)


class DilatedTemporalCNN(nn.Module):
    """Multi-scale dilated 1D-CNN for temporal feature extraction."""

    def __init__(self, d_model: int = 64, n_layers: int = 3,
                 kernel_size: int = 3, dilations: list = [1, 2, 4],
                 dropout: float = 0.1):
        super().__init__()
        assert n_layers == len(dilations), "n_layers must match len(dilations)"

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i, dilation in enumerate(dilations):
            padding = (kernel_size - 1) * dilation // 2
            self.convs.append(
                nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                          dilation=dilation, padding=padding, groups=1)
            )
            self.norms.append(nn.BatchNorm1d(d_model))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        residual = x
        x = x.permute(0, 2, 1)  # [B, D, T]

        for conv, norm in zip(self.convs, self.norms):
            out = conv(x)
            out = norm(out)
            out = self.activation(out)
            out = self.dropout(out)
            x = out + x  # residual connection

        x = x.permute(0, 2, 1)  # [B, T, D]
        return x + residual  # global residual


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal dimension."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class LightweightSelfAttention(nn.Module):
    """Single-layer multi-head self-attention over temporal dimension."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        B, T, D = x.shape
        residual = x

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # [B, heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.proj(out)
        out = self.dropout(out)

        return self.norm(out + residual)


class GestureClassifier(nn.Module):
    """
    Lightweight 1D-CNN + Transformer gesture classifier for AR headsets.

    Input:  skeleton sequence [B, T=32, J=26, C=3]
    Output: gesture logits   [B, 7]
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_cnn_layers: int = N_CNN_LAYERS, cnn_kernel: int = CNN_KERNEL,
                 dilations: list = DILATIONS, dropout: float = DROPOUT,
                 n_classes: int = N_GESTURE_CLASSES, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.d_model = d_model

        # Stage 1: Spatial embedding (J,C) → d_model
        self.spatial_embed = SpatialEmbedding(in_channels=3, d_model=d_model)

        # Stage 2: Joint pooling → [B, T, D]
        self.joint_pool = JointWisePool(pool_type="mean")

        # Stage 3: Positional encoding
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=64)

        # Stage 4: Dilated temporal CNN
        self.temporal_cnn = DilatedTemporalCNN(
            d_model=d_model, n_layers=n_cnn_layers,
            kernel_size=cnn_kernel, dilations=dilations, dropout=dropout
        )

        # Stage 5: Self-attention (optional, for ablation)
        if use_attention:
            self.attention = LightweightSelfAttention(
                d_model=d_model, n_heads=n_heads, dropout=dropout
            )
        else:
            self.attention = None

        # Stage 6: Global pooling + classification head
        self.norm_final = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()
        self._count_params()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GestureClassifier: {total:,} total params ({trainable:,} trainable)")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T, J, C] skeleton sequence
        Returns:
            logits: [B, n_classes]
            attn_weights: [B, heads, T, T] or None
        """
        # Normalize to wrist-relative coordinates
        wrist = x[:, :, 0:1, :]  # [B, T, 1, C] — joint 0 is wrist
        x = x - wrist

        # Stage 1: Spatial embedding
        x = self.spatial_embed(x)  # [B, T, J, D]

        # Stage 2: Joint pooling
        x = self.joint_pool(x)     # [B, T, D]

        # Stage 3: Positional encoding
        x = self.pos_enc(x)

        # Stage 4: Temporal CNN
        x = self.temporal_cnn(x)

        # Stage 5: Self-attention
        attn_weights = None
        if self.attention is not None:
            x = self.attention(x)
            # (attn_weights not stored for efficiency; can add hook if needed)

        # Stage 6: Global pooling + classify
        x = x.mean(dim=1)          # [B, D] — average over time
        x = self.norm_final(x)
        logits = self.classifier(x)

        return logits, attn_weights

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return class predictions and confidence scores."""
        logits, _ = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        conf, preds = probs.max(dim=-1)
        return preds, conf


def build_model(use_attention: bool = True, d_model: int = D_MODEL) -> GestureClassifier:
    """Factory function with default config."""
    return GestureClassifier(
        d_model=d_model,
        n_heads=N_HEADS,
        n_cnn_layers=N_CNN_LAYERS,
        cnn_kernel=CNN_KERNEL,
        dilations=DILATIONS,
        dropout=DROPOUT,
        use_attention=use_attention,
    )


def build_ablation_cnn_only(d_model: int = D_MODEL) -> GestureClassifier:
    """CNN-only variant for ablation study."""
    return GestureClassifier(
        d_model=d_model,
        n_heads=N_HEADS,
        n_cnn_layers=N_CNN_LAYERS,
        cnn_kernel=CNN_KERNEL,
        dilations=DILATIONS,
        dropout=DROPOUT,
        use_attention=False,
    )


if __name__ == "__main__":
    # Quick smoke test
    model = build_model()
    x = torch.randn(2, 32, 26, 3)
    logits, _ = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {logits.shape}")
    print(f"Prediction: {model.predict(x)}")

    model_cnn = build_ablation_cnn_only()
    logits_cnn, _ = model_cnn(x)
    print(f"\nCNN-only params: {sum(p.numel() for p in model_cnn.parameters()):,}")
