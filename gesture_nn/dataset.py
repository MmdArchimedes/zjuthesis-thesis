"""
PyTorch Dataset and DataLoader for hand skeleton gesture sequences.

Handles sliding window extraction, normalization, and augmentation.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, List
from config import *


class GestureDataset(Dataset):
    """Sliding-window dataset over gesture skeleton sequences."""

    def __init__(self, sequences: List[np.ndarray], labels: List[np.ndarray],
                 metadata: List[dict], window_size: int = WINDOW_SIZE,
                 window_stride: int = WINDOW_STRIDE, augment: bool = False,
                 participant_ids: Optional[set] = None):
        """
        Args:
            sequences: list of [T_i, 26, 3] skeleton arrays
            labels: list of [T_i] label arrays
            metadata: list of per-sequence metadata dicts
            window_size: frames per input window
            window_stride: stride between windows
            augment: enable training augmentation
            participant_ids: if set, only include these participants
        """
        self.window_size = window_size
        self.window_stride = window_stride
        self.augment = augment

        # Filter by participant if specified
        self.windows = []
        self.window_labels = []
        self.window_metadata = []

        for seq, lab, meta in zip(sequences, labels, metadata):
            if participant_ids is not None and meta['person_id'] not in participant_ids:
                continue

            T = len(seq)
            if T < window_size:
                # Pad short sequences
                pad_len = window_size - T
                seq = np.concatenate([seq, np.tile(seq[-1:], (pad_len, 1, 1))], axis=0)
                lab = np.concatenate([lab, np.full(pad_len, lab[-1])])
                T = len(seq)

            # Slide window
            for start in range(0, T - window_size + 1, window_stride):
                end = start + window_size
                window = seq[start:end].copy()

                # Label: majority vote in window (excluding transition frames)
                window_lab = lab[start:end]
                non_transition = window_lab[window_lab != 0]
                if len(non_transition) > 0:
                    final_label = int(np.bincount(non_transition).argmax())
                else:
                    final_label = 0  # NONE

                self.windows.append(window)
                self.window_labels.append(final_label)
                self.window_metadata.append(meta)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        skeleton = self.windows[idx].copy().astype(np.float32)
        label = self.window_labels[idx]

        if self.augment:
            skeleton = self._augment(skeleton)

        # Normalize: wrist-relative + unit scale
        wrist = skeleton[:, 0:1, :]  # [T, 1, 3]
        skeleton = skeleton - wrist

        # Global scale normalization (to ~unit variance)
        scale = np.std(skeleton)
        if scale > 1e-6:
            skeleton = skeleton / (scale * 10)

        return torch.from_numpy(skeleton), torch.tensor(label, dtype=torch.long)

    def _augment(self, skeleton: np.ndarray) -> np.ndarray:
        """Apply training augmentations."""
        T = skeleton.shape[0]
        # Time crop jitter: randomly shift window slightly
        if T > self.window_size + 2 * AUG_TIME_CROP_JITTER:
            shift = np.random.randint(-AUG_TIME_CROP_JITTER, AUG_TIME_CROP_JITTER + 1)
            start = max(0, shift)
            end = min(T, T + shift - (self.window_size - T))
            skeleton = skeleton[start:end]
        if len(skeleton) > self.window_size:
            start = np.random.randint(0, len(skeleton) - self.window_size)
            skeleton = skeleton[start:start + self.window_size]

        # Coordinate noise (simulate sensor jitter)
        noise = np.random.normal(0, AUG_COORD_NOISE_SIGMA, skeleton.shape).astype(np.float32)
        skeleton = skeleton + noise

        # Random mirror (left ↔ right hand, swap x coordinate sign)
        if np.random.random() < AUG_MIRROR_PROB:
            skeleton = skeleton.copy()
            skeleton[:, :, 0] = -skeleton[:, :, 0]

        return skeleton

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balancing."""
        labels = np.array(self.window_labels)
        counts = np.bincount(labels, minlength=N_GESTURE_CLASSES).astype(np.float32)
        counts = np.maximum(counts, 1)  # avoid division by zero
        weights = 1.0 / counts
        weights = weights / weights.sum() * N_GESTURE_CLASSES
        return torch.from_numpy(weights)

    def get_sample_weights(self) -> np.ndarray:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        labels = np.array(self.window_labels)
        return class_weights[labels]


def load_data(data_dir: str = "data") -> Tuple[List, List, List]:
    """Load pre-generated dataset."""
    data_path = Path(data_dir)
    with open(data_path / "sequences.pkl", "rb") as f:
        sequences = pickle.load(f)
    with open(data_path / "labels.pkl", "rb") as f:
        labels = pickle.load(f)
    with open(data_path / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return sequences, labels, metadata


def create_dataloaders(data_dir: str = "data", batch_size: int = BATCH_SIZE,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with participant-level split."""
    sequences, labels, metadata = load_data(data_dir)

    # Get all participant IDs
    all_pids = sorted(set(m['person_id'] for m in metadata))
    n_participants = len(all_pids)
    np.random.seed(42)
    pids_shuffled = np.random.permutation(all_pids)

    # Split by participant
    n_train = int(n_participants * TRAIN_SPLIT)
    n_val = int(n_participants * VAL_SPLIT)

    train_pids = set(pids_shuffled[:n_train])
    val_pids = set(pids_shuffled[n_train:n_train + n_val])
    test_pids = set(pids_shuffled[n_train + n_val:])

    print(f"Train participants: {sorted(train_pids)}")
    print(f"Val participants:   {sorted(val_pids)}")
    print(f"Test participants:  {sorted(test_pids)}")

    train_ds = GestureDataset(sequences, labels, metadata,
                              augment=True, participant_ids=train_pids)
    val_ds = GestureDataset(sequences, labels, metadata,
                            augment=False, participant_ids=val_pids)
    test_ds = GestureDataset(sequences, labels, metadata,
                             augment=False, participant_ids=test_pids)

    print(f"\nTrain windows: {len(train_ds)}")
    print(f"Val windows:   {len(val_ds)}")
    print(f"Test windows:  {len(test_ds)}")

    # Weighted sampling for class balance in training
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def get_class_names() -> List[str]:
    return [GESTURE_MAP[i] for i in range(N_GESTURE_CLASSES)]
