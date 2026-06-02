"""
Training pipeline for DBEW-NN gesture classifier.

Includes:
  - Warmup phase (CNN only, attention frozen)
  - Full training with Focal Loss or CrossEntropy + class weights
  - Temporal smoothness regularization
  - Early stopping and checkpointing
  - ONNX export for Unity integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import time
import sys

from config import *
from model import GestureClassifier, build_model, build_ablation_cnn_only
from dataset import create_dataloaders, get_class_names


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class TemporalSmoothLoss(nn.Module):
    """Penalize large prediction changes between adjacent windows during training."""
    def forward(self, logits_t: torch.Tensor, logits_tp1: torch.Tensor) -> torch.Tensor:
        probs_t = F.softmax(logits_t, dim=-1)
        probs_tp1 = F.softmax(logits_tp1, dim=-1)
        return F.kl_div(probs_t.log(), probs_tp1, reduction='batchmean')


class MetricsTracker:
    """Track and aggregate metrics across epochs."""
    def __init__(self, class_names: list):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0
        self.running_loss = 0.0
        self.n_batches = 0
        self.per_class_correct = torch.zeros(N_GESTURE_CLASSES)
        self.per_class_total = torch.zeros(N_GESTURE_CLASSES)
        self.all_preds = []
        self.all_targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float):
        preds = logits.argmax(dim=-1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.size(0)
        self.running_loss += loss
        self.n_batches += 1

        for c in range(N_GESTURE_CLASSES):
            mask = targets == c
            self.per_class_correct[c] += (preds[mask] == c).sum().item()
            self.per_class_total[c] += mask.sum().item()

        self.all_preds.extend(preds.cpu().tolist())
        self.all_targets.extend(targets.cpu().tolist())

    def compute(self) -> Dict:
        accuracy = self.correct / max(self.total, 1)

        # Per-class metrics
        per_class = {}
        for c in range(N_GESTURE_CLASSES):
            tp = self.per_class_correct[c].item()
            total = self.per_class_total[c].item()
            recall = tp / max(total, 1)

            # Precision: of all preds for class c, how many were correct
            preds_arr = np.array(self.all_preds)
            targets_arr = np.array(self.all_targets)
            pred_count = (preds_arr == c).sum()
            precision = tp / max(pred_count, 1)

            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            per_class[self.class_names[c]] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'support': int(total),
            }

        # Macro-average F1
        macro_f1 = np.mean([per_class[c]['f1'] for c in self.class_names
                            if per_class[c]['support'] > 0])

        return {
            'accuracy': round(accuracy, 4),
            'macro_f1': round(macro_f1, 4),
            'loss': round(self.running_loss / max(self.n_batches, 1), 4),
            'per_class': per_class,
        }


def train_epoch(model: nn.Module, loader, criterion, optimizer,
                scaler: Optional[GradScaler], device: str,
                smooth_loss_fn: Optional[TemporalSmoothLoss] = None,
                track_metrics: bool = True) -> Dict:
    """Single training epoch."""
    model.train()
    metrics = MetricsTracker(get_class_names())
    prev_logits = None

    for batch_idx, (skeletons, targets) in enumerate(loader):
        skeletons = skeletons.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                logits, _ = model(skeletons)
                loss = criterion(logits, targets)
                if smooth_loss_fn is not None and prev_logits is not None:
                    # Temporal smoothness between consecutive batches (approximate)
                    min_batch = min(logits.size(0), prev_logits.size(0))
                    loss += TEMPORAL_SMOOTH_LAMBDA * smooth_loss_fn(
                        logits[:min_batch], prev_logits[:min_batch].detach()
                    )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(skeletons)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        if track_metrics:
            metrics.update(logits.detach(), targets, loss.item())

        prev_logits = logits.detach()

    return metrics.compute() if track_metrics else {}


@torch.no_grad()
def validate(model: nn.Module, loader, criterion, device: str) -> Dict:
    """Validation epoch."""
    model.eval()
    metrics = MetricsTracker(get_class_names())

    for skeletons, targets in loader:
        skeletons = skeletons.to(device)
        targets = targets.to(device)

        logits, _ = model(skeletons)
        loss = criterion(logits, targets)
        metrics.update(logits, targets, loss.item())

    return metrics.compute()


@torch.no_grad()
def measure_latency(model: nn.Module, device: str, n_warmup: int = 50,
                    n_measure: int = 500) -> Dict:
    """Measure inference latency."""
    model.eval()
    x = torch.randn(1, WINDOW_SIZE, N_JOINTS, JOINT_DIMS, device=device)

    # Warmup
    for _ in range(n_warmup):
        _ = model(x)

    # Measure
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_measure):
        _ = model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / n_measure) * 1000
    return {
        'latency_ms_mean': round(latency_ms, 3),
        'throughput_fps': round(1000 / latency_ms, 1),
        'n_params': sum(p.numel() for p in model.parameters()),
    }


def train_model(model_type: str = "full", device: str = "cuda",
                data_dir: str = "data", output_dir: str = "checkpoints") -> Dict:
    """
    Full training pipeline.

    Args:
        model_type: "full" (CNN+Attention) or "cnn_only" (ablation)
        device: "cuda" or "cpu"
        data_dir: path to generated data
        output_dir: checkpoint save directory
    """
    output_path = Path(output_dir) / model_type
    output_path.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(data_dir=data_dir)

    # Model
    if model_type == "full":
        model = build_model(use_attention=True)
    else:
        model = build_ablation_cnn_only()
    model = model.to(device)

    # Loss
    class_weights = train_loader.dataset.get_class_weights().to(device)
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    smooth_loss_fn = TemporalSmoothLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    # Mixed precision
    scaler = GradScaler() if device == 'cuda' else None

    # Freeze attention during warmup
    if model.use_attention and WARMUP_EPOCHS > 0:
        print(f"\nWarmup: freezing self-attention for {WARMUP_EPOCHS} epochs")
        for param in model.attention.parameters():
            param.requires_grad = False

    # Training loop
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': []}

    print(f"\n{'='*60}")
    print(f"Training {model_type} model")
    print(f"Device: {device}, Batch size: {BATCH_SIZE}, Epochs: {N_EPOCHS}")
    print(f"{'='*60}")

    for epoch in range(N_EPOCHS):
        # Unfreeze attention after warmup
        if model.use_attention and epoch == WARMUP_EPOCHS:
            print(f"\nEpoch {epoch}: unfreezing self-attention")
            for param in model.attention.parameters():
                param.requires_grad = True

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, smooth_loss_fn
        )
        history['train'].append(train_metrics)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        history['val'].append(val_metrics)

        scheduler.step()

        # Print progress
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
              f"Macro F1: {val_metrics['macro_f1']:.4f}")

        # Checkpointing
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            patience_counter = 0

            safe_config = {}
            for k, v in list(globals().items()):
                if k.isupper() and not k.startswith('_'):
                    if isinstance(v, (int, float, str, bool, list, tuple, dict, type(None))):
                        safe_config[k] = v
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': safe_config,
            }, output_path / "best_model.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    checkpoint = torch.load(output_path / "best_model.pt", map_location=device,
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation on test set
    print(f"\n{'='*60}")
    print(f"Test set evaluation (best epoch: {best_epoch})")
    print(f"{'='*60}")

    test_metrics = validate(model, test_loader, criterion, device)

    print(f"\nTest Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"\nPer-class metrics:")
    print(f"{'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print("-" * 60)
    for cls_name, cls_metrics in test_metrics['per_class'].items():
        if cls_metrics['support'] > 0:
            print(f"{cls_name:<20s} {cls_metrics['precision']:>10.4f} "
                  f"{cls_metrics['recall']:>10.4f} {cls_metrics['f1']:>10.4f} "
                  f"{cls_metrics['support']:>10d}")

    # Latency measurement
    latency = measure_latency(model, device)
    print(f"\nInference latency: {latency['latency_ms_mean']:.3f} ms "
          f"({latency['throughput_fps']:.0f} FPS)")
    print(f"Model parameters: {latency['n_params']:,}")

    # Save results
    results = {
        'model_type': model_type,
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
        'latency': latency,
        'history': history,
    }
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Export to ONNX
    export_onnx(model, output_path, device)

    return results


def export_onnx(model: nn.Module, output_path: Path, device: str):
    """Export model to ONNX format for Unity Barracuda / ONNX Runtime."""
    model.eval()
    dummy_input = torch.randn(1, WINDOW_SIZE, N_JOINTS, JOINT_DIMS, device=device)

    onnx_path = output_path / "gesture_classifier.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=[EXPORT_INPUT_NAME],
        output_names=[EXPORT_OUTPUT_NAME],
        opset_version=ONNX_OPSET,
        dynamic_axes={
            EXPORT_INPUT_NAME: {0: 'batch'},
            EXPORT_OUTPUT_NAME: {0: 'batch'},
        },
        export_params=True,
        do_constant_folding=True,
        dynamo=False,  # use legacy TorchScript exporter for Unity Barracuda compat
    )
    print(f"\nONNX model exported to: {onnx_path}")

    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: OK")
    except ImportError:
        print("ONNX verification skipped (onnx not installed)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="full",
                        choices=["full", "cnn_only"])
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    results = train_model(
        model_type=args.model,
        device=args.device,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
