"""
ST-GCN gesture classifier for 26-joint hand skeleton sequences.
Thesis: Direction 1 — replaces 1D-CNN with spatial-temporal graph convolution.

Architecture:
  Input [B, T=32, J=26, C=3]
  → Input Projection (1x1 conv, 3→64)
  → 6 ST-GCN Blocks (channel progression: 64→64→128→128→256→256)
  → Global Average Pooling
  → Classifier Head (256→128→7)

Key features:
  - Spatial GCN with fixed hand-skeleton adjacency + learnable refinement
  - Temporal depthwise-separable conv for parameter efficiency
  - Residual connections throughout
  - ~75K parameters, target <3ms inference on Snapdragon XR2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from graph_utils import build_hand_adjacency_matrix


class SpatialGraphConv(nn.Module):
    """
    Spatial graph convolution with fixed + learnable adjacency.
    CTR-GCN style channel-wise topology refinement (simplified).
    """
    def __init__(self, in_channels: int, out_channels: int, num_joints: int = 26):
        super().__init__()
        self.num_joints = num_joints

        A = build_hand_adjacency_matrix("spatial_full")
        self.register_buffer("A_fixed", A)

        # Learnable adjacency residual (CTR-GCN idea: let model adapt topology)
        self.A_learn = nn.Parameter(torch.zeros(num_joints, num_joints))
        nn.init.normal_(self.A_learn, 0, 0.01)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, J]
        Returns:
            [B, C_out, T, J]
        """
        residual = self.residual(x)
        B, C, T, J = x.shape

        A = self.A_fixed + F.relu(self.A_learn)
        x = x.permute(0, 2, 1, 3).contiguous().view(B * T, C, J)
        x = torch.einsum("ncj,jk->nck", x, A)
        x = x.view(B, T, C, J).permute(0, 2, 1, 3).contiguous()

        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x + residual)


class TemporalConv(nn.Module):
    """
    Depthwise-separable temporal convolution.
    More parameter-efficient than standard conv for edge deployment.
    """
    def __init__(self, channels: int, kernel_size: int = 9):
        super().__init__()
        padding = kernel_size // 2

        self.depthwise = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=channels, bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T, J] → [B, C, T, J]"""
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x + residual)


class STGCNBlock(nn.Module):
    """One ST-GCN unit: Spatial GCN → Temporal Conv."""
    def __init__(self, in_channels: int, out_channels: int,
                 temporal_kernel: int = 9):
        super().__init__()
        self.sgc = SpatialGraphConv(in_channels, out_channels)
        self.tc = TemporalConv(out_channels, kernel_size=temporal_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tc(self.sgc(x))


class STGCNGestureClassifier(nn.Module):
    """
    ST-GCN gesture classifier for 26-joint hand skeleton.

    Input:  [B, T, J, C] skeleton sequence (T=32, J=26, C=3)
    Output: [B, 7] gesture class logits

    Target: ~75K params, <3ms inference on Snapdragon XR2.
    """

    def __init__(self, num_classes: int = 7, base_channels: int = 64,
                 num_joints: int = 26):
        super().__init__()
        self.num_joints = num_joints
        self.base_channels = base_channels

        # Channel progression after input_proj (3 → base_channels):
        # Smaller channels for ~75K total params (standard) / ~45K (small)
        block_channels = [base_channels, base_channels,
                         base_channels * 2, base_channels * 2]

        # Input projection: (J, C) → base_channels per frame
        self.input_proj = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # ST-GCN blocks
        self.blocks = nn.ModuleList()
        prev_ch = base_channels
        for ch in block_channels:
            self.blocks.append(STGCNBlock(prev_ch, ch))
            prev_ch = ch

        self.output_channels = block_channels[-1]

        # Output head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(self.output_channels, self.output_channels // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_channels // 2, num_classes),
        )

        self._init_weights()
        self._count_params()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"STGCNGestureClassifier: {total:,} total params ({trainable:,} trainable)")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x: [B, T, J, C]
        Returns:
            (logits [B, 7], None) — None for attn_weights API compatibility
        """
        # Normalize to wrist-relative coordinates
        wrist = x[:, :, 0:1, :]
        x = x - wrist

        # [B, T, J, C] → [B, C, T, J]
        x = x.permute(0, 3, 1, 2).contiguous()

        # Input projection
        x = self.input_proj(x)

        # ST-GCN blocks
        for block in self.blocks:
            x = block(x)

        # Global pooling + classify
        x = self.global_pool(x).flatten(1)
        logits = self.classifier(x)

        return logits, None

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, _ = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        conf, preds = probs.max(dim=-1)
        return preds, conf


class STGCNSmall(nn.Module):
    """
    Smaller ST-GCN variant (~45K params) for tighter edge budget.
    Channel progression: 3 → 48 → 48 → 96 → 96 → 192 → 192
    """
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self._full = STGCNGestureClassifier(
            num_classes=num_classes, base_channels=48,
        )

    def forward(self, x):
        return self._full(x)

    def predict(self, x):
        return self._full.predict(x)


def build_stgcn(num_classes: int = 7, base_channels: int = 64) -> STGCNGestureClassifier:
    """Factory: standard ST-GCN."""
    return STGCNGestureClassifier(num_classes=num_classes, base_channels=base_channels)


def build_stgcn_small(num_classes: int = 7) -> STGCNSmall:
    """Factory: smaller variant."""
    return STGCNSmall(num_classes=num_classes)


if __name__ == "__main__":
    model = build_stgcn()
    x = torch.randn(2, 32, 26, 3)
    logits, _ = model(x)
    preds, conf = model.predict(x)
    print(f"Input:  {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Preds:  {preds}, Conf: {conf}")

    model_small = build_stgcn_small()
    logits_s, _ = model_small(x)
    print(f"Small variant: {logits_s.shape}")
