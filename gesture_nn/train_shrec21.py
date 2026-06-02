"""
Train DBEW-NN on SHREC 2021 — continuous stream detection with NONE class.
18 classes: 0=NONE + 17 SHREC gesture classes.
Mimics our DBEW--Gesture pipeline: sliding window over continuous input.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import pickle
import json

from config import WINDOW_SIZE, N_JOINTS, JOINT_DIMS, BATCH_SIZE
from model import build_model, build_ablation_cnn_only

# ── Configuration ──
N_ALL_CLASSES = 18  # 0=NONE, 1-17=gesture
CLASS_NAMES = ['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'OK', 'MENU',
               'LEFT', 'RIGHT', 'CIRCLE', 'V', 'CROSS',
               'GRAB', 'PINCH', 'TAP', 'DENY', 'KNOB', 'EXPAND']
N_EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 15
WARMUP_EPOCHS = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ShrecDataset(Dataset):
    def __init__(self, windows, labels, augment=False):
        self.windows = windows
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].copy().astype(np.float32)
        y = self.labels[idx]
        if self.augment:
            if np.random.random() < 0.5:
                x = x + np.random.normal(0, 0.001, x.shape).astype(np.float32)
            if np.random.random() < 0.3:
                shift = np.random.randint(1, 4)
                x = np.roll(x, shift, axis=0)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    def get_class_weights(self):
        labels = np.array(self.labels)
        counts = np.bincount(labels, minlength=N_ALL_CLASSES).astype(np.float32)
        counts = np.maximum(counts, 1)
        weights = 1.0 / counts
        weights = weights / weights.sum() * N_ALL_CLASSES
        return torch.from_numpy(weights)


def load_shrec_data(data_dir='shrec21/processed_continuous'):
    """Load continuous-stream SHREC data (with NONE class)."""
    with open(f'{data_dir}/shrec21_training_continuous.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(f'{data_dir}/shrec21_test_continuous.pkl', 'rb') as f:
        test_data = pickle.load(f)

    print(f'Train windows: {len(train_data["windows"])}')
    print(f'Test windows:  {len(test_data["windows"])}')

    # Split test into val/test (50/50)
    n_test = len(test_data['windows'])
    indices = np.random.RandomState(42).permutation(n_test)
    n_val = n_test // 2

    train_ds = ShrecDataset(train_data['windows'], train_data['labels'], augment=True)
    test_ds = ShrecDataset(
        [test_data['windows'][i] for i in indices[n_val:]],
        [test_data['labels'][i] for i in indices[n_val:]],
        augment=False)
    val_ds = ShrecDataset(
        [test_data['windows'][i] for i in indices[:n_val]],
        [test_data['labels'][i] for i in indices[:n_val]],
        augment=False)

    # Balanced batch sampling: each batch has ~50% NONE, ~50% gestures
    # NONE frames are easy but essential for learning "no gesture" state
    none_indices = [i for i, l in enumerate(train_ds.labels) if l == 0]
    gesture_indices = [i for i, l in enumerate(train_ds.labels) if l != 0]
    n_half = BATCH_SIZE // 2

    class GestureBalancedSampler(torch.utils.data.Sampler):
        def __init__(self, none_idx, gesture_idx, batch_size, n_batches):
            self.none_idx = none_idx
            self.gesture_idx = gesture_idx
            self.batch_size = batch_size
            self.n_batches = n_batches
        def __iter__(self):
            for _ in range(self.n_batches):
                n_none = self.batch_size // 2
                n_gest = self.batch_size - n_none
                batch = (list(np.random.choice(self.none_idx, n_none, replace=True)) +
                         list(np.random.choice(self.gesture_idx, n_gest, replace=True)))
                np.random.shuffle(batch)
                yield from batch
        def __len__(self):
            return self.n_batches * self.batch_size

    sampler = GestureBalancedSampler(none_indices, gesture_indices, BATCH_SIZE,
                                     n_batches=len(gesture_indices) // (BATCH_SIZE // 2))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
    return train_loader, val_loader, test_loader, train_ds


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(-1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    per_class = {}
    for c in range(N_ALL_CLASSES):
        tp = ((preds == c) & (targets == c)).sum()
        gt = (targets == c).sum()
        pc = (preds == c).sum()
        rec = tp / max(gt, 1)
        pre = tp / max(pc, 1)
        f1 = 2 * pre * rec / max(pre + rec, 1e-8)
        per_class[CLASS_NAMES[c]] = {
            'precision': round(pre, 4), 'recall': round(rec, 4),
            'f1': round(f1, 4), 'support': int(gt)}

    # Macro F1: exclude NONE (focus on gesture classes)
    gesture_f1s = [per_class[c]['f1'] for c in CLASS_NAMES[1:]
                   if per_class[c]['support'] > 0]
    macro_f1 = np.mean(gesture_f1s) if gesture_f1s else 0.0

    # Detection metrics: treat non-NONE as "gesture detected"
    true_gesture = targets != 0
    pred_gesture = preds != 0
    det_tp = (true_gesture & pred_gesture).sum()
    det_precision = det_tp / max(pred_gesture.sum(), 1)
    det_recall = det_tp / max(true_gesture.sum(), 1)
    det_f1 = 2 * det_precision * det_recall / max(det_precision + det_recall, 1e-8)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'macro_f1': round(macro_f1, 4),
        'per_class': per_class,
        'detection_precision': round(det_precision, 4),
        'detection_recall': round(det_recall, 4),
        'detection_f1': round(det_f1, 4),
    }


def train_shrec(model_type='full', output_dir='checkpoints/shrec21_continuous'):
    output_path = Path(output_dir) / model_type
    output_path.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, train_ds = load_shrec_data()

    model = build_model(use_attention=(model_type == 'full'))
    model.classifier[3] = nn.Linear(32, N_ALL_CLASSES)  # 18 classes
    model = model.to(DEVICE)
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')

    # Don't use per-class alpha weights — balanced batch sampler already
    # handles class distribution. Alpha=1 for all classes lets FocalLoss
    # focus on hard examples naturally (NONE is easy, gestures are hard).
    criterion = FocalLoss(gamma=2.0, alpha=None)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    if model.use_attention and WARMUP_EPOCHS > 0:
        print(f'Warmup: freezing attention for {WARMUP_EPOCHS} epochs')
        for param in model.attention.parameters():
            param.requires_grad = False

    best_val_f1 = 0
    best_epoch = 0
    patience = 0
    history = {'train': [], 'val': []}

    print(f'\n{"="*60}')
    print(f'Training {model_type} on SHREC 2021 (18 classes: NONE + 17 gestures)')
    print(f'Device: {DEVICE}')
    print(f'{"="*60}')

    for epoch in range(N_EPOCHS):
        if model.use_attention and epoch == WARMUP_EPOCHS:
            print(f'Epoch {epoch}: unfreezing attention')
            for param in model.attention.parameters():
                param.requires_grad = True

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        history['train'].append({'loss': round(train_loss, 4), 'accuracy': round(train_acc, 4)})

        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        history['val'].append(val_metrics)

        scheduler.step()

        print(f'Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Acc: {val_metrics["accuracy"]:.4f} F1: {val_metrics["macro_f1"]:.4f} '
              f'Det: {val_metrics["detection_f1"]:.4f}')

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            best_epoch = epoch
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'class_names': CLASS_NAMES,
            }, output_path / 'best_model.pt')
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break

    # Test evaluation
    ckpt = torch.load(output_path / 'best_model.pt', map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)

    print(f'\n{"="*60}')
    print(f'TEST RESULTS (best epoch: {best_epoch})')
    print(f'{"="*60}')
    print(f'Accuracy:        {test_metrics["accuracy"]:.4f}')
    print(f'Macro F1 (gest): {test_metrics["macro_f1"]:.4f}')
    print(f'Detection P:     {test_metrics["detection_precision"]:.4f}')
    print(f'Detection R:     {test_metrics["detection_recall"]:.4f}')
    print(f'Detection F1:    {test_metrics["detection_f1"]:.4f}')
    print(f'\nPer-class:')
    print(f'{"Class":<12s} {"Prec":>8s} {"Rec":>8s} {"F1":>8s} {"Sup":>6s}')
    print('-' * 44)
    for c in CLASS_NAMES:
        m = test_metrics['per_class'][c]
        if m['support'] > 0:
            print(f'{c:<12s} {m["precision"]:>8.4f} {m["recall"]:>8.4f} {m["f1"]:>8.4f} {m["support"]:>6d}')

    results = {
        'model_type': model_type,
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
        'history': history,
    }
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='full', choices=['full', 'cnn_only'])
    args = parser.parse_args()
    train_shrec(model_type=args.model)
