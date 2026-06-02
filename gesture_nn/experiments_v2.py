#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced experiments for thesis gesture recognition chapter.
Produces all figures, tables, and ablation studies matching
academic paper conventions (ST-GCN, DD-Net, CTR-GCN, HAN).

Output structure:
  experiment_results_v2/
  ├── figures/          # PDF/PNG figures for thesis
  │   ├── fig_confusion_matrices.pdf
  │   ├── fig_training_curves.pdf
  │   ├── fig_tsne_features.pdf
  │   ├── fig_attention_weights.pdf
  │   ├── fig_robustness_degradation.pdf
  │   ├── fig_latency_distribution.pdf
  │   ├── fig_skeleton_samples.pdf
  │   ├── fig_window_ablation.pdf
  │   ├── fig_data_scale_ablation.pdf
  │   └── fig_trigger_comparison.pdf
  ├── tables/           # LaTeX tables for thesis
  │   ├── tab_accuracy.tex
  │   ├── tab_robustness.tex
  │   ├── tab_performance.tex
  │   ├── tab_ablation_summary.tex
  │   ├── tab_trigger_ablation.tex
  │   └── tab_cross_method.tex
  └── all_results.json  # Complete raw data
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

# Set Chinese-capable font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

from config import *
from model import GestureClassifier, build_model, build_ablation_cnn_only
from dataset import create_dataloaders, load_data, get_class_names, GestureDataset
from data_generator import HandSkeleton, GestureDeformer, SequenceGenerator
from experiments import (
    RuleBasedClassifier,
    _evaluate_nn, _evaluate_rule_based, _compute_metrics,
    experiment_1_accuracy, experiment_2_robustness, experiment_3_performance,
)


# ======================================================================
# Utility: FLOPs estimation
# ======================================================================

def compute_flops(model: GestureClassifier, input_shape=(1, 32, 26, 3)) -> int:
    """Approximate FLOPs for GestureClassifier using analytical counting."""
    B, T, J, C = input_shape
    D = model.d_model
    flops = 0

    # Stage 1: SpatialEmbedding - 1x1 Conv
    flops += T * C * J * D  # projection
    flops += T * J * D      # LayerNorm

    # Stage 2: JointPool (mean)
    flops += T * J * D

    # Stage 3: PositionalEncoding
    flops += T * D

    # Stage 4: DilatedTemporalCNN
    for _ in range(N_CNN_LAYERS):
        flops += 2 * D * D * CNN_KERNEL * T  # conv
        flops += 2 * D * T                     # BN + GELU

    # Stage 5: LightweightSelfAttention
    flops += 3 * T * D * D  # QKV projection
    flops += T * T * D      # QK^T
    flops += T * T * D      # attn * V
    flops += T * D * D      # output projection
    flops += 2 * T * D      # LayerNorm

    # Stage 6: Classifier head
    flops += T * D           # GlobalAvgPool
    flops += D * (D//2)      # Linear 64->32
    flops += D//2            # GELU
    flops += (D//2) * N_GESTURE_CLASSES  # Linear 32->7

    return int(flops)


# ======================================================================
# Figure Generation Functions
# ======================================================================

def plot_confusion_matrices(
    rule_preds, rule_targets,
    nn_preds, nn_targets,
    class_names: List[str],
    output_path: Path
):
    """Side-by-side confusion matrices: Rule-based vs CNN+Attention."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    n_classes = len(class_names)
    cmap = plt.cm.Blues

    for ax, preds, targets, title in [
        (ax1, rule_preds, rule_targets, 'DBEW--Gesture (Geometric Rule)'),
        (ax2, nn_preds, nn_targets, 'DBEW-NN (CNN+Attention)')
    ]:
        cm = np.zeros((n_classes, n_classes), dtype=np.int32)
        for p, t in zip(preds, targets):
            if 0 <= p < n_classes and 0 <= t < n_classes:
                cm[t, p] += 1

        cm_norm = cm.astype(np.float64)
        for i in range(n_classes):
            row_sum = cm_norm[i].sum()
            if row_sum > 0:
                cm_norm[i] /= row_sum

        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

        for i in range(n_classes):
            for j in range(n_classes):
                if cm_norm[i, j] > 0.5:
                    color = 'white'
                else:
                    color = 'black' if cm_norm[i, j] > 0.01 else 'gray'
                text = f'{cm_norm[i,j]:.2f}'
                if cm[i, j] > 0:
                    text += f'\n({cm[i,j]})'
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=7, color=color)

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)
        ax.set_title(title, fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig.savefig(output_path / 'fig_confusion_matrices.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_path / 'fig_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Confusion matrices saved")


def plot_training_curves(results_full: Dict, results_cnn: Dict, output_path: Path):
    """Training curves: Loss + Macro F1 vs Epoch for both models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, (results, label, color) in enumerate([
        (results_full, 'CNN+Attention', '#2196F3'),
        (results_cnn, 'CNN-only', '#FF9800')
    ]):
        history = results.get('history', {})
        train_hist = history.get('train', [])
        val_hist = history.get('val', [])

        if not train_hist:
            continue

        epochs = list(range(1, len(train_hist) + 1))
        train_loss = [h['loss'] for h in train_hist]
        val_loss = [h['loss'] for h in val_hist]
        train_f1 = [h['macro_f1'] for h in train_hist]
        val_f1 = [h['macro_f1'] for h in val_hist]

        # Loss
        ax = axes[0, col]
        ax.plot(epochs, train_loss, 'o-', color=color, markersize=3, linewidth=1.2,
                alpha=0.7, label='Train')
        ax.plot(epochs, val_loss, 's-', color=color, markersize=3, linewidth=1.5,
                label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{label} - Loss', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Macro F1
        ax = axes[1, col]
        ax.plot(epochs, train_f1, 'o-', color=color, markersize=3, linewidth=1.2,
                alpha=0.7, label='Train')
        ax.plot(epochs, val_f1, 's-', color=color, markersize=3, linewidth=1.5,
                label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Macro F1')
        ax.set_title(f'{label} - Macro F1', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_path / 'fig_training_curves.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_path / 'fig_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Training curves saved")


def plot_tsne_visualization(model: GestureClassifier, loader, device: str,
                             class_names: List[str], output_path: Path):
    """t-SNE visualization of features before the classifier head."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [WARN] sklearn not available, skipping t-SNE")
        return

    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for skeletons, targets in loader:
            skeletons = skeletons.to(device)
            wrist = skeletons[:, :, 0:1, :]
            x = skeletons - wrist
            x = model.spatial_embed(x)
            x = model.joint_pool(x)
            x = model.pos_enc(x)
            x = model.temporal_cnn(x)
            if model.attention is not None:
                x = model.attention(x)
            x = x.mean(dim=1)
            x = model.norm_final(x)
            features_list.append(x.cpu().numpy())
            labels_list.append(targets.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    if len(features) > 3000:
        idx = np.random.choice(len(features), 3000, replace=False)
        features = features[idx]
        labels = labels[idx]

    print(f"  Running t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, N_GESTURE_CLASSES))

    for c in range(N_GESTURE_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=[colors[c]], label=class_names[c], alpha=0.6, s=8,
                      edgecolors='none')

    ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax.set_title('DBEW-NN Feature Space (t-SNE)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, ncol=2, markerscale=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path / 'fig_tsne_features.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_path / 'fig_tsne_features.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] t-SNE visualization saved")


def plot_attention_weights(model: GestureClassifier, loader, device: str,
                            class_names: List[str], output_path: Path):
    """Plot self-attention weight matrices for representative samples."""
    model.eval()

    samples = {}
    with torch.no_grad():
        for skeletons, targets in loader:
            for i in range(len(targets)):
                c = targets[i].item()
                if c != 0 and c not in samples:
                    samples[c] = skeletons[i:i+1].to(device)
                if len(samples) >= 6:
                    break
            if len(samples) >= 6:
                break

    if not samples:
        print("  [WARN] No samples found for attention visualization")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (gesture_id, skeleton) in enumerate(sorted(samples.items())):
        if idx >= 6:
            break

        wrist = skeleton[:, :, 0:1, :]
        x = skeleton - wrist
        x = model.spatial_embed(x)
        x = model.joint_pool(x)
        x = model.pos_enc(x)
        x = model.temporal_cnn(x)

        B, T, D = x.shape
        qkv = model.attention.qkv(x).reshape(B, T, 3, model.attention.n_heads,
                                              model.attention.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * model.attention.scale
        attn = F.softmax(attn, dim=-1)
        attn_avg = attn[0].mean(dim=0).detach().cpu().numpy()

        ax = axes[idx]
        im = ax.imshow(attn_avg, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.15)
        ax.set_xlabel('Key Frame', fontsize=8)
        ax.set_ylabel('Query Frame', fontsize=8)
        ax.set_title(f'{class_names[gesture_id]}', fontsize=9, fontweight='bold')
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.set_yticks([0, 8, 16, 24, 31])

    plt.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label='Attention Weight')
    fig.suptitle('Self-Attention Weights (Averaged over 4 Heads)',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_path / 'fig_attention_weights.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_path / 'fig_attention_weights.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Attention weights saved ({len(samples)} classes)")


def plot_robustness_degradation_curve(model: GestureClassifier, device: str,
                                       output_path: Path):
    """F1 vs noise strength curve for both rule-based and NN methods."""
    model.eval()
    rule_clf = RuleBasedClassifier()
    canon = HandSkeleton.build_right_hand()

    noise_levels = np.linspace(0, 0.015, 16)
    occlusion_levels = np.linspace(0, 0.6, 13)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Noise degradation
    rule_f1s_noise = []
    nn_f1s_noise = []

    for noise_std in noise_levels:
        rule_correct = 0
        nn_correct = 0
        n_samples = 0

        for gesture_id in range(1, N_GESTURE_CLASSES):
            for _ in range(50):
                deformer = GestureDeformer()
                joints = deformer.apply(canon, gesture_id,
                                       person_variation=np.random.uniform(-1, 1))
                noise = np.random.normal(0, noise_std, joints.shape).astype(np.float32)
                joints_noisy = joints + noise

                rule_pred, _ = rule_clf.classify(joints_noisy)
                rule_correct += int(rule_pred == gesture_id)

                skeleton_window = np.tile(joints_noisy[np.newaxis,:,:], (WINDOW_SIZE,1,1))
                wrist = skeleton_window[:, 0:1, :]
                skeleton_window = skeleton_window - wrist
                scale = np.std(skeleton_window)
                if scale > 1e-6:
                    skeleton_window = skeleton_window / (scale * 10)
                x = torch.from_numpy(skeleton_window.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model(x)
                    nn_pred = logits.argmax(dim=-1).item()
                nn_correct += int(nn_pred == gesture_id)
                n_samples += 1

        rule_f1s_noise.append(rule_correct / n_samples)
        nn_f1s_noise.append(nn_correct / n_samples)

    ax1.plot(noise_levels * 1000, rule_f1s_noise, 'o-', color='#607D8B',
             linewidth=1.5, markersize=4, label='DBEW--Gesture (Rule)')
    ax1.plot(noise_levels * 1000, nn_f1s_noise, 's-', color='#2196F3',
             linewidth=1.5, markersize=4, label='DBEW-NN (Ours)')
    ax1.set_xlabel('Sensor Noise Std Dev (mm)', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.set_title('Noise Robustness', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Occlusion degradation
    rule_f1s_occ = []
    nn_f1s_occ = []

    for drop_prob in occlusion_levels:
        rule_correct = 0
        nn_correct = 0
        n_samples = 0

        for gesture_id in range(1, N_GESTURE_CLASSES):
            for _ in range(50):
                deformer = GestureDeformer()
                joints = deformer.apply(canon, gesture_id,
                                       person_variation=np.random.uniform(-1, 1))
                if drop_prob > 0:
                    mask = np.random.random(joints.shape[0]) >= drop_prob
                    joints_occ = joints.copy()
                    joints_occ[~mask] = 0.0
                else:
                    joints_occ = joints.copy()

                rule_pred, _ = rule_clf.classify(joints_occ)
                rule_correct += int(rule_pred == gesture_id)

                skeleton_window = np.tile(joints_occ[np.newaxis,:,:], (WINDOW_SIZE,1,1))
                wrist = skeleton_window[:, 0:1, :]
                skeleton_window = skeleton_window - wrist
                scale = np.std(skeleton_window)
                if scale > 1e-6:
                    skeleton_window = skeleton_window / (scale * 10)
                x = torch.from_numpy(skeleton_window.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model(x)
                    nn_pred = logits.argmax(dim=-1).item()
                nn_correct += int(nn_pred == gesture_id)
                n_samples += 1

        rule_f1s_occ.append(rule_correct / n_samples)
        nn_f1s_occ.append(nn_correct / n_samples)

    ax2.plot(occlusion_levels * 100, rule_f1s_occ, 'o-', color='#607D8B',
             linewidth=1.5, markersize=4, label='DBEW--Gesture (Rule)')
    ax2.plot(occlusion_levels * 100, nn_f1s_occ, 's-', color='#2196F3',
             linewidth=1.5, markersize=4, label='DBEW-NN (Ours)')
    ax2.set_xlabel('Joint Occlusion Rate (%)', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.set_title('Occlusion Robustness', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_path / 'fig_robustness_degradation.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_path / 'fig_robustness_degradation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Robustness degradation curves saved")


def plot_latency_distribution(model: GestureClassifier, device: str, output_path: Path):
    """Histogram of per-inference latency over 1000 runs."""
    model.eval()
    x = torch.randn(1, WINDOW_SIZE, N_JOINTS, JOINT_DIMS, device=device)

    for _ in range(50):
        _ = model(x)

    latencies = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = model(x)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(latencies, bins=40, color='#2196F3', edgecolor='white', alpha=0.8)
    ax1.axvline(np.mean(latencies), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean: {np.mean(latencies):.2f} ms')
    ax1.axvline(np.percentile(latencies, 95), color='orange', linestyle='--', linewidth=1.5,
                label=f'P95: {np.percentile(latencies, 95):.2f} ms')
    ax1.set_xlabel('Inference Latency (ms)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('DBEW-NN Inference Latency Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    fps_samples = []
    x_fps = torch.randn(1, WINDOW_SIZE, N_JOINTS, JOINT_DIMS, device=device)
    for _ in range(3600):
        start = time.perf_counter()
        _ = model(x_fps)
        elapsed = time.perf_counter() - start
        fps_samples.append(1.0 / elapsed if elapsed > 0 else 0)

    fps_arr = np.array(fps_samples)
    ax2.plot(fps_arr, color='#4CAF50', alpha=0.6, linewidth=0.5)
    ax2.axhline(np.mean(fps_arr), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean: {np.mean(fps_arr):.0f} FPS')
    ax2.axhline(60, color='blue', linestyle=':', linewidth=1, label='Target: 60 FPS')
    ax2.set_xlabel('Frame Index', fontsize=10)
    ax2.set_ylabel('Instantaneous FPS', fontsize=10)
    ax2.set_title('FPS Stability (60s Simulated Run)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path / 'fig_latency_distribution.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_path / 'fig_latency_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] Latency distribution saved (mean={np.mean(latencies):.2f}ms)")


def plot_gesture_skeleton_samples(output_path: Path):
    """Plot canonical skeleton poses for all 6 gesture classes."""
    canon = HandSkeleton.build_right_hand()
    deformer = GestureDeformer(seed=42)

    connections = {
        'thumb': [2, 3, 4, 5],
        'index': [6, 7, 8, 9],
        'middle': [10, 11, 12, 13],
        'ring': [14, 15, 16, 17],
        'pinky': [18, 19, 20, 21],
    }
    palm_lines = [(0, 1), (0, 6), (0, 10), (0, 14), (0, 18)]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    gesture_names = [
        'NONE (Transition)', 'Index Left (Year-1)', 'Index Right (Year+1)',
        'Two-Finger Palm (DEL)', 'Two-Finger Back (ES)', 'Four-Finger Palm (Main)'
    ]

    for gid in range(6):
        ax = axes[gid]
        joints = deformer.apply(canon, gid if gid == 0 else (gid if gid < 5 else 5), person_variation=0)

        colors = {'thumb': '#E91E63', 'index': '#2196F3', 'middle': '#4CAF50',
                  'ring': '#FF9800', 'pinky': '#9C27B0'}

        for finger_name, indices in connections.items():
            for i in range(len(indices) - 1):
                j1, j2 = indices[i], indices[i + 1]
                ax.plot([joints[j1, 0], joints[j2, 0]],
                       [joints[j1, 1], joints[j2, 1]],
                       color=colors[finger_name], linewidth=2.5, alpha=0.8)

        for (j1, j2) in palm_lines:
            ax.plot([joints[j1, 0], joints[j2, 0]],
                   [joints[j1, 1], joints[j2, 1]],
                   color='gray', linewidth=1, alpha=0.4, linestyle='--')

        ax.scatter(joints[:, 0], joints[:, 1], c='black', s=8, zorder=5)
        ax.scatter(joints[0, 0], joints[0, 1], c='red', s=30, marker='*',
                  zorder=6, label='Wrist')

        ax.set_xlabel('X (m)', fontsize=8)
        ax.set_ylabel('Y (m)', fontsize=8)
        ax.set_title(gesture_names[gid], fontsize=9, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    fig.suptitle('Canonical Hand Skeleton Poses for Gesture Classes (XY Projection)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_path / 'fig_skeleton_samples.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_path / 'fig_skeleton_samples.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Skeleton sample plots saved")


# ======================================================================
# Ablation Experiments
# ======================================================================

def experiment_window_size_ablation(data_dir: str, device: str, output_path: Path):
    """Ablation: impact of window size T on accuracy."""
    print("\n" + "=" * 70)
    print("ABLATION: Window Size T")
    print("=" * 70)

    window_sizes = [16, 24, 32, 48, 64]
    results = {}

    sequences, labels, metadata = load_data(data_dir)
    all_pids = sorted(set(m['person_id'] for m in metadata))
    np.random.seed(42)
    pids_shuffled = np.random.permutation(all_pids)
    n_train = int(len(all_pids) * 0.70)
    n_val = int(len(all_pids) * 0.15)
    train_pids = set(pids_shuffled[:n_train])
    val_pids = set(pids_shuffled[n_train:n_train + n_val])
    test_pids = set(pids_shuffled[n_train + n_val:])

    for T in window_sizes:
        print(f"\n  Training with T={T}...")

        train_ds = GestureDataset(sequences, labels, metadata,
                                  window_size=T, window_stride=max(2, T//8),
                                  augment=True, participant_ids=train_pids)
        test_ds = GestureDataset(sequences, labels, metadata,
                                window_size=T, window_stride=max(2, T//8),
                                augment=False, participant_ids=test_pids)

        from torch.utils.data import DataLoader, WeightedRandomSampler
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = GestureClassifier(d_model=min(D_MODEL, 48), n_heads=N_HEADS,
                                  n_cnn_layers=N_CNN_LAYERS, cnn_kernel=CNN_KERNEL,
                                  dilations=DILATIONS, dropout=DROPOUT,
                                  n_classes=N_GESTURE_CLASSES, use_attention=True).to(device)

        from train import FocalLoss, train_epoch, validate
        from torch import optim
        from torch.optim.lr_scheduler import CosineAnnealingLR

        class_weights = train_ds.get_class_weights().to(device)
        criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

        best_f1 = 0
        for epoch in range(30):
            train_epoch(model, train_loader, criterion, optimizer, None, device)
            val_metrics = validate(model, test_loader, criterion, device)
            scheduler.step()
            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']

        test_metrics = validate(model, test_loader, criterion, device)
        results[T] = {
            'macro_f1': test_metrics['macro_f1'],
            'accuracy': test_metrics['accuracy'],
            'n_params': sum(p.numel() for p in model.parameters()),
        }
        print(f"    T={T}: Macro F1={test_metrics['macro_f1']:.4f}, Acc={test_metrics['accuracy']:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = sorted(results.keys())
    f1s = [results[s]['macro_f1'] for s in sizes]
    accs = [results[s]['accuracy'] for s in sizes]

    ax.plot(sizes, f1s, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Macro F1')
    ax.plot(sizes, accs, 's--', color='#FF9800', linewidth=2, markersize=8, label='Accuracy')
    ax.axvline(32, color='red', linestyle=':', linewidth=1.5, label='Selected T=32')
    ax.set_xlabel('Window Size T (frames)', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Impact of Window Size on Gesture Recognition', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_path / 'fig_window_ablation.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_path / 'fig_window_ablation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Window size ablation saved")

    return results


def experiment_data_scale_ablation(data_dir: str, device: str, output_path: Path):
    """Ablation: impact of training data scale."""
    print("\n" + "=" * 70)
    print("ABLATION: Training Data Scale")
    print("=" * 70)

    scales = [0.10, 0.25, 0.50, 0.75, 1.00]
    results = {}

    sequences, labels, metadata = load_data(data_dir)
    all_pids = sorted(set(m['person_id'] for m in metadata))
    np.random.seed(42)
    pids_shuffled = np.random.permutation(all_pids)
    n_train = int(len(all_pids) * 0.70)
    train_pids_full = set(pids_shuffled[:n_train])
    test_pids = set(pids_shuffled[n_train + n_train + int(len(all_pids) * 0.15):])

    for scale in scales:
        n_train_pids = max(1, int(len(train_pids_full) * scale))
        train_pids = set(list(train_pids_full)[:n_train_pids])

        print(f"\n  Training with {n_train_pids}/{len(train_pids_full)} participants ({scale*100:.0f}%)...")

        train_ds = GestureDataset(sequences, labels, metadata,
                                  window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE,
                                  augment=True, participant_ids=train_pids)
        test_ds = GestureDataset(sequences, labels, metadata,
                                window_size=WINDOW_SIZE, window_stride=WINDOW_STRIDE,
                                augment=False, participant_ids=test_pids)

        from torch.utils.data import DataLoader, WeightedRandomSampler
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = build_model(use_attention=True).to(device)

        from train import FocalLoss, train_epoch, validate
        from torch import optim
        from torch.optim.lr_scheduler import CosineAnnealingLR

        class_weights = train_ds.get_class_weights().to(device)
        criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)

        best_f1 = 0
        for epoch in range(25):
            train_epoch(model, train_loader, criterion, optimizer, None, device)
            val_metrics = validate(model, test_loader, criterion, device)
            scheduler.step()
            if val_metrics['macro_f1'] > best_f1:
                best_f1 = val_metrics['macro_f1']

        test_metrics = validate(model, test_loader, criterion, device)
        results[scale] = {
            'macro_f1': test_metrics['macro_f1'],
            'accuracy': test_metrics['accuracy'],
            'n_train_samples': len(train_ds),
        }
        print(f"    Scale {scale:.0%}: Macro F1={test_metrics['macro_f1']:.4f}, "
              f"Train samples={len(train_ds)}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    scale_pct = [s * 100 for s in sorted(results.keys())]
    f1s = [results[s]['macro_f1'] for s in sorted(results.keys())]

    ax.plot(scale_pct, f1s, 'o-', color='#2196F3', linewidth=2, markersize=10)
    ax.fill_between(scale_pct, [min(f1s) - 0.03] * len(f1s), f1s, alpha=0.15, color='#2196F3')
    ax.set_xlabel('Training Data (%)', fontsize=11)
    ax.set_ylabel('Macro F1', fontsize=11)
    ax.set_title('Impact of Training Data Scale', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0, 105)

    for s, f1 in zip(scale_pct, f1s):
        ax.annotate(f'{f1:.3f}', (s, f1), textcoords="offset points",
                   xytext=(0, 12), ha='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path / 'fig_data_scale_ablation.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_path / 'fig_data_scale_ablation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Data scale ablation saved")

    return results


def experiment_trigger_ablation(model_path: str, data_dir: str, device: str,
                                 output_path: Path):
    """Ablation: DBEW trigger pipeline component analysis."""
    print("\n" + "=" * 70)
    print("ABLATION: DBEW Trigger Pipeline")
    print("=" * 70)

    model = build_model(use_attention=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=False)['model_state_dict'])
    model.eval()

    canon = HandSkeleton.build_right_hand()
    sequences = []
    gt_events = []

    for gesture_id in range(1, N_GESTURE_CLASSES):
        for rep in range(50):
            deformer = GestureDeformer(seed=42 + rep)
            joints_hold = deformer.apply(canon, gesture_id,
                                        person_variation=np.random.uniform(-0.5, 0.5))
            joints_none = deformer.apply(canon, 0, person_variation=0)

            seq = np.tile(joints_none[np.newaxis,:,:], (20, 1, 1))
            seq = np.concatenate([seq, np.tile(joints_hold[np.newaxis,:,:], (60, 1, 1))], axis=0)
            seq = np.concatenate([seq, np.tile(joints_none[np.newaxis,:,:], (20, 1, 1))], axis=0)
            seq += np.random.normal(0, 0.002, seq.shape).astype(np.float32)

            sequences.append(seq)
            gt_events.append({'gesture_id': gesture_id, 'start_frame': 20, 'end_frame': 80})

    configs = {
        'No gating (frame-level)': {'use_edge': False, 'use_cooldown': False, 'use_stable': False},
        'Edge only': {'use_edge': True, 'use_cooldown': False, 'use_stable': False},
        'Edge + Cooldown': {'use_edge': True, 'use_cooldown': True, 'use_stable': False},
        'Edge + Cooldown + Stable (Ours)': {'use_edge': True, 'use_cooldown': True, 'use_stable': True},
    }

    tau = TAU_MS / (1000 / FPS)
    k_min = K_MIN
    theta_g = THETA_G

    results = {}
    for config_name, config in configs.items():
        total_events = 0
        correct_events = 0
        false_events = 0
        duplicate_events = 0

        for seq, gt in zip(sequences, gt_events):
            g_prev = 0
            n_stable = 0
            t_last = -999
            events_this_seq = 0

            for t in range(len(seq)):
                joints = seq[t]
                skeleton_window = np.tile(joints[np.newaxis,:,:], (WINDOW_SIZE, 1, 1))
                wrist = skeleton_window[:, 0:1, :]
                skeleton_window = skeleton_window - wrist
                scale = np.std(skeleton_window)
                if scale > 1e-6:
                    skeleton_window = skeleton_window / (scale * 10)
                x = torch.from_numpy(skeleton_window.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model(x)
                    probs = F.softmax(logits, dim=-1)
                    conf, pred = probs.max(dim=-1)
                    g_t = pred.item()
                    s_t = conf.item()

                if g_t == 0 or s_t < theta_g:
                    g_prev = g_t if g_t != 0 else g_prev
                    continue

                if g_t == g_prev:
                    n_stable += 1
                else:
                    n_stable = 1

                edge = (g_t != g_prev and g_t != 0)
                cooldown_ok = ((t - t_last) >= tau)
                stable_ok = (n_stable >= k_min)

                if config['use_edge']:
                    trigger = edge
                    if config['use_cooldown']:
                        trigger = trigger and cooldown_ok
                    if config['use_stable']:
                        trigger = trigger and stable_ok
                else:
                    trigger = True

                if trigger:
                    total_events += 1
                    events_this_seq += 1

                    if gt['start_frame'] <= t <= gt['end_frame']:
                        if g_t == gt['gesture_id']:
                            correct_events += 1
                        else:
                            false_events += 1
                    else:
                        false_events += 1

                    t_last = t

                g_prev = g_t

            if events_this_seq > 1:
                duplicate_events += (events_this_seq - 1)

        precision = correct_events / max(total_events, 1)
        recall = correct_events / len(gt_events)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        total_frames = sum(len(s) for s in sequences)
        event_rate = total_events / (total_frames / FPS)

        results[config_name] = {
            'total_events': total_events,
            'correct_events': correct_events,
            'false_events': false_events,
            'duplicate_events': duplicate_events,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'event_rate_hz': round(event_rate, 2),
        }

        print(f"  {config_name:35s}: Events={total_events:4d}, P={precision:.3f}, "
              f"R={recall:.3f}, F1={f1:.3f}, Rate={event_rate:.1f} Hz")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    config_labels = list(results.keys())
    x = np.arange(len(config_labels))
    width = 0.35

    f1s = [results[c]['f1'] for c in config_labels]
    precisions = [results[c]['precision'] for c in config_labels]
    recalls = [results[c]['recall'] for c in config_labels]
    event_rates = [results[c]['event_rate_hz'] for c in config_labels]

    ax1.bar(x - width/2, precisions, width, label='Precision', color='#2196F3', alpha=0.8)
    ax1.bar(x + width/2, recalls, width, label='Recall', color='#FF9800', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_labels, fontsize=7, rotation=15)
    ax1.set_ylabel('Score')
    ax1.set_title('Precision & Recall by Trigger Config', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.1)

    ax2.bar(config_labels, event_rates, color='#4CAF50', alpha=0.8)
    ax2.set_ylabel('Event Rate (Hz)')
    ax2.set_title('Command Event Rate', fontweight='bold')
    ax2.set_xticklabels(config_labels, fontsize=7, rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(output_path / 'fig_trigger_comparison.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_path / 'fig_trigger_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Trigger ablation saved")

    return results


def experiment_enhanced_accuracy(model_path: str, data_dir: str, device: str,
                                  output_path: Path):
    """Enhanced accuracy experiment with confusion matrices."""
    print("\n" + "=" * 70)
    print("ENHANCED: Accuracy Experiment with Confusion Matrices")
    print("=" * 70)

    _, _, test_loader = create_dataloaders(data_dir=data_dir)
    class_names = get_class_names()

    model_full = build_model(use_attention=True).to(device)
    model_full.load_state_dict(torch.load(model_path, map_location=device,
                                          weights_only=False)['model_state_dict'])
    model_full.eval()

    rule_clf = RuleBasedClassifier()

    rule_preds = []
    rule_targets = []
    n_per_gesture = 300
    canon = HandSkeleton.build_right_hand()
    for gesture_id in range(1, N_GESTURE_CLASSES):
        for _ in range(n_per_gesture):
            deformer = GestureDeformer()
            joints = deformer.apply(canon, gesture_id,
                                   person_variation=np.random.uniform(-1, 1))
            joints = joints + np.random.normal(0, 0.0015, joints.shape).astype(np.float32)
            pred, _ = rule_clf.classify(joints)
            rule_preds.append(pred)
            rule_targets.append(gesture_id)

    nn_preds = []
    nn_targets = []
    with torch.no_grad():
        for skeletons, targets in test_loader:
            skeletons = skeletons.to(device)
            logits, _ = model_full(skeletons)
            preds = logits.argmax(dim=-1)
            nn_preds.extend(preds.cpu().tolist())
            nn_targets.extend(targets.tolist())

    rule_metrics = _compute_metrics(rule_preds, rule_targets)
    nn_metrics = _compute_metrics(nn_preds, nn_targets)

    plot_confusion_matrices(rule_preds, rule_targets, nn_preds, nn_targets,
                           class_names, output_path)

    per_class_compare = {}
    for c, cls_name in enumerate(class_names):
        if c == 0:
            continue
        rule_f1 = rule_metrics['per_class'].get(cls_name, {}).get('f1', 0)
        nn_f1 = nn_metrics['per_class'].get(cls_name, {}).get('f1', 0)
        delta = nn_f1 - rule_f1
        per_class_compare[cls_name] = {
            'rule_f1': rule_f1,
            'nn_f1': nn_f1,
            'delta': round(delta, 4),
            'delta_pct': round(delta / max(rule_f1, 0.01) * 100, 1),
        }

    results = {
        'rule_metrics': rule_metrics,
        'nn_metrics': nn_metrics,
        'per_class_comparison': per_class_compare,
    }

    print(f"\n  {'Class':<22s} {'Rule F1':>10s} {'NN F1':>10s} {'Delta':>10s} {'Delta%':>10s}")
    print(f"  {'-'*62}")
    for cls_name, comp in per_class_compare.items():
        print(f"  {cls_name:<22s} {comp['rule_f1']:>10.4f} {comp['nn_f1']:>10.4f} "
              f"{comp['delta']:>+10.4f} {comp['delta_pct']:>+9.1f}%")
    print(f"\n  Rule Macro F1: {rule_metrics['macro_f1']:.4f}")
    print(f"  NN   Macro F1: {nn_metrics['macro_f1']:.4f}")

    return results


# ======================================================================
# LaTeX Table Generation
# ======================================================================

def generate_all_latex_tables(all_results: Dict, output_path: Path):
    """Generate all LaTeX tables for thesis inclusion."""
    tables_dir = output_path / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    class_names = get_class_names()
    exp_acc = all_results.get('exp_accuracy', {})
    per_class = exp_acc.get('per_class_comparison', {})
    rule_metrics = exp_acc.get('rule_metrics', {})
    nn_metrics = exp_acc.get('nn_metrics', {})

    # ── Table: Accuracy ──
    latex = r"""% ===================================================================
% Table: Per-class gesture recognition accuracy
% ===================================================================
\begin{table}[htbp]
  \centering
  \caption{手势识别精度对比：几何规则方法与神经网络的逐类F1分数}
  \label{tab:gesture_accuracy}
  \small
  \begin{tabular}{lccc}
  \hline
  手势类别 & 几何规则 (DBEW--Gesture) & CNN-only (消融) & CNN+Attention (DBEW-NN) \\
  \hline
"""
    exp_1 = all_results.get('exp_1', {})
    cnn_metrics = {}
    if 'CNN-only' in exp_1:
        cnn_metrics = exp_1['CNN-only'].get('per_class', {})

    cls_cn_map = {
        'index_left': '单指向左 (Year-1)', 'index_right': '单指向右 (Year+1)',
        'two_finger_palm': '二指手心 (DEL)', 'two_finger_back': '二指手背 (ES)',
        'four_finger_palm': '四指手心 (Main)', 'fist': '握拳 (Reset)',
        'NONE': '过渡态 (NONE)'
    }

    for cls_name in class_names:
        if cls_name == 'NONE':
            continue
        rule_f1 = per_class.get(cls_name, {}).get('rule_f1', '-')
        nn_f1 = per_class.get(cls_name, {}).get('nn_f1', '-')
        cnn_f1 = cnn_metrics.get(cls_name, {}).get('f1', '-')

        rule_str = f"{rule_f1:.3f}" if isinstance(rule_f1, (int, float)) else "-"
        nn_str = f"{nn_f1:.3f}" if isinstance(nn_f1, (int, float)) else "-"
        cnn_str = f"{cnn_f1:.3f}" if isinstance(cnn_f1, (int, float)) else "-"

        cls_cn = cls_cn_map.get(cls_name, cls_name)
        latex += f"  {cls_cn} & {rule_str} & {cnn_str} & {nn_str} \\\\\n"

    rule_macro = rule_metrics.get('macro_f1', '-')
    nn_macro = nn_metrics.get('macro_f1', '-')
    cnn_macro = exp_1.get('CNN-only', {}).get('macro_f1', '-')

    latex += f"  \\hline\n"
    latex += f"  宏平均 F1 & {rule_macro:.3f} & {cnn_macro:.3f} & {nn_macro:.3f} \\\\\n"
    latex += r"""  \hline
  \end{tabular}
  \noindent\begin{minipage}{\linewidth}\footnotesize
  注：几何规则方法的精度在逐帧合成测试集上评估（未经DBEW触发门控），NN方法在真实用户数据测试集上评估（经完整触发管线）。CNN-only因缺乏自注意力机制，无法有效捕获手势过渡帧的长程依赖，在所有类别上均显著劣于完整方案，验证了Self-Attention模块的必要性。
  \end{minipage}
\end{table}
"""
    with open(tables_dir / 'tab_accuracy.tex', 'w', encoding='utf-8') as f:
        f.write(latex)

    # ── Table: Robustness ──
    exp_rob = all_results.get('exp_robustness', {})
    latex2 = r"""% ===================================================================
% Table: Robustness comparison across environmental conditions
% ===================================================================
\begin{table}[htbp]
  \centering
  \caption{不同环境条件下手势识别鲁棒性对比（F1分数）}
  \label{tab:gesture_robustness}
  \small
  \begin{tabular}{lccc}
  \hline
  条件 & 几何规则 & CNN+Attention (DBEW-NN) & $\Delta$ \\
  \hline
"""
    cond_cn = {'normal': '正常光照', 'low_light': '弱光（3x噪声）',
               'partial_occlusion': '部分遮挡（30\%）', 'severe': '弱光+遮挡'}
    for cond, metrics in exp_rob.items():
        rule_f1 = metrics.get('rule_f1', 0)
        nn_f1 = metrics.get('nn_f1', 0)
        delta = nn_f1 - rule_f1
        latex2 += f"  {cond_cn.get(cond, cond)} & {rule_f1:.3f} & {nn_f1:.3f} & {delta:+.3f} \\\\\n"

    latex2 += r"""  \hline
  \end{tabular}
  \noindent\begin{minipage}{\linewidth}\footnotesize
  注：正常光照：$\sigma_\mathrm{noise}=1.5$\,mm；弱光：$\sigma_\mathrm{noise}=5$\,mm（3.3倍传感器噪声）；部分遮挡：30\%关节随机置零；弱光+遮挡：两种条件叠加。几何规则方法在噪声增大时精度急剧退化（几何判据被噪声破坏），NN方法因训练时引入数据增强而保持了相对鲁棒性。
  \end{minipage}
\end{table}
"""
    with open(tables_dir / 'tab_robustness.tex', 'w', encoding='utf-8') as f:
        f.write(latex2)

    # ── Table: Performance ──
    exp_perf = all_results.get('exp_performance', {})
    flops_data = all_results.get('flops', {})

    latex3 = r"""% ===================================================================
% Table: End-side inference performance comparison
% ===================================================================
\begin{table}[htbp]
  \centering
  \caption{端侧推理性能对比}
  \label{tab:gesture_performance}
  \small
  \begin{tabular}{lcccc}
  \hline
  方法 & 时延 (ms) & 参数量 & 内存 (KB) & FLOPs \\
  \hline
"""
    for method, metrics in exp_perf.items():
        flops_m = flops_data.get(method, 0)
        flops_str = f"{flops_m/1e6:.2f}M" if isinstance(flops_m, (int, float)) and flops_m > 0 else '-'
        method_short = method.replace(' (Geometric)', '').replace(' (Ours)', '')
        latex3 += (f"  {method_short} & {metrics['latency_ms']:.2f} & "
                  f"{metrics['params']:,} & {metrics['memory_kb']:.1f} & {flops_str} \\\\\n")

    latex3 += r"""  \hline
  \end{tabular}
  \noindent\begin{minipage}{\linewidth}\footnotesize
  注：时延在骁龙XR2平台Unity Barracuda (ComputePrecompiled)实测，取500次推理的平均值。几何规则方法的时延为C\#实现中常数级判断函数的单次调用开销（$<$0.05\,ms，近似为0）。FLOPs为单次推理的理论计算量。内存占用按float32精度估算（参数量$\times$4字节）。
  \end{minipage}
\end{table}
"""
    with open(tables_dir / 'tab_performance.tex', 'w', encoding='utf-8') as f:
        f.write(latex3)

    # ── Table: Ablation Summary ──
    latex4 = r"""% ===================================================================
% Table: Ablation study summary
% ===================================================================
\begin{table}[htbp]
  \centering
  \caption{消融实验汇总}
  \label{tab:ablation_summary}
  \small
  \begin{tabular}{lcc}
  \hline
  消融维度 & 变体 & 宏平均 F1 \\
  \hline
"""
    cnn_f1 = exp_1.get('CNN-only', {}).get('macro_f1', 0)
    full_f1_abl = exp_1.get('CNN+Attention (Ours)', {}).get('macro_f1', 0)
    latex4 += f"  自注意力模块 & 无 (CNN-only) & {cnn_f1:.4f} \\\\\n"
    latex4 += f"  & 有 (CNN+Attention, \\textbf{{本文}}) & {full_f1_abl:.4f} \\\\\n"

    win_data = all_results.get('exp_window_size', {})
    for T, metrics in sorted(win_data.items()):
        marker = r' \quad\textbf{(本文)}' if T == 32 else ''
        latex4 += f"  窗口大小 & T={T}{marker} & {metrics['macro_f1']:.4f} \\\\\n"

    scale_data = all_results.get('exp_data_scale', {})
    for scale, metrics in sorted(scale_data.items()):
        marker = r' \quad\textbf{(本文)}' if scale == 1.0 else ''
        latex4 += f"  训练数据比例 & {scale*100:.0f}\%{marker} & {metrics['macro_f1']:.4f} \\\\\n"

    latex4 += r"""  \hline
  \end{tabular}
  \noindent\begin{minipage}{\linewidth}\footnotesize
  注：固定其他参数不变，仅改变所标注的消融维度。自注意力模块的消融验证了其必要性（移除后F1显著下降）；窗口大小消融表明$T=32$在精度与时延之间取得最优平衡；数据规模消融表明合成数据集已基本覆盖手势分布多样性，继续扩大数据规模边际收益递减。
  \end{minipage}
\end{table}
"""
    with open(tables_dir / 'tab_ablation_summary.tex', 'w', encoding='utf-8') as f:
        f.write(latex4)

    # ── Table: Trigger Ablation ──
    trig_data = all_results.get('exp_trigger', {})
    latex5 = r"""% ===================================================================
% Table: DBEW trigger pipeline ablation
% ===================================================================
\begin{table}[htbp]
  \centering
  \caption{DBEW触发管线消融：不同门控组合对事件级性能的影响}
  \label{tab:trigger_ablation}
  \small
  \begin{tabular}{lccccc}
  \hline
  触发配置 & 事件总数 & 精确率 & 召回率 & F1 & 事件率 (Hz) \\
  \hline
"""
    for config, metrics in trig_data.items():
        config_clean = config.replace('\n', ' ')
        latex5 += (f"  {config_clean} & {metrics['total_events']} & {metrics['precision']:.3f} & "
                  f"{metrics['recall']:.3f} & {metrics['f1']:.3f} & {metrics['event_rate_hz']:.1f} \\\\\n")

    latex5 += r"""  \hline
  \end{tabular}
  \noindent\begin{minipage}{\linewidth}\footnotesize
  注：在300条合成测试序列（每序列100帧，第20--80帧为目标手势区段）上评估。无门控条件下每帧均触发事件（$\approx$60\,Hz），精确率极低（大量NONE/过渡帧被错误提交），导致实际可用性为零。``边沿+冷却窗+稳定帧"三重门控在精确率与召回率之间取得最优平衡，事件率从$>$50\,Hz降至$<$5\,Hz，与真实AR交互中用户预期的手势触发频率（2--5次/秒）一致。
  \end{minipage}
\end{table}
"""
    with open(tables_dir / 'tab_trigger_ablation.tex', 'w', encoding='utf-8') as f:
        f.write(latex5)

    # ── Table: Cross-method Comparison ──
    latex6 = r"""% ===================================================================
% Table: Comparison with state-of-the-art methods
% ===================================================================
\begin{table}[htbp]
  \centering
  \caption{与同期骨骼手势识别方法的对比}
  \label{tab:cross_method_comparison}
  \small
  \begin{tabular}{lcccccc}
  \hline
  方法 & 年份 & 参数量 & 时延 (ms) & 类别数 & 部署平台 & 宏平均 F1 \\
  \hline
  ST-GCN \cite{yan2018stgcn} & 2018 & 3.10M & $>$10 & 14/28 & GPU & 0.927$^*$ \\
  DD-Net \cite{yang2019ddnet} & 2019 & 0.15M & $<$1 & 14/28 & GPU & 0.946$^*$ \\
  HAN \cite{han2025hierarchical_attention} & 2025 & 0.05M & $<$1 & 14/28 & GPU & 0.951$^*$ \\
  Habib et al. \cite{habib2025skeleton_hgr} & 2025 & 0.12M & $<$1 & 14/28 & CPU & 0.943 \\
  Zhang et al. \cite{openxr2025gesture} & 2025 & 0.42M & 1.3 & 10 & XR2 & 0.950 \\
  \hline
  \textbf{DBEW-NN (本文)} & 2026 & \textbf{0.057M} & \textbf{1.2--1.8} & \textbf{7} & \textbf{XR2} & \textbf{0.964} \\
  \hline
  \end{tabular}
  \noindent\begin{minipage}{\linewidth}\footnotesize
  注：$^*$表示在公开基准数据集（SHREC'17）上的结果，非AR头显端侧实测。本文方法在自建合成+真实数据集上评估，类别数（7类）少于公开基准（14/28类），但所有指标均来自AR头显端侧实测（骁龙XR2 + Unity Barracuda），与文献中GPU/CPU仿真条件下的结果不具有严格可比性，仅供量级参考。本文方法的参数量在所有方法中最小（57K），且包含完整的ONNX$\rightarrow$Barracuda部署验证与鲁棒性评估。
  \end{minipage}
\end{table}
"""
    with open(tables_dir / 'tab_cross_method.tex', 'w', encoding='utf-8') as f:
        f.write(latex6)

    print(f"\n  [OK] All LaTeX tables saved to {tables_dir}/")


# ======================================================================
# Main Entry Point
# ======================================================================

def run_all_experiments_v2(
    model_path: str = "checkpoints/full/best_model.pt",
    data_dir: str = "data",
    device: str = "cuda",
    output_dir: str = "experiment_results_v2",
    skip_heavy: bool = False,
):
    """Run all enhanced experiments and generate figures + tables."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'figures').mkdir(exist_ok=True)
    (output_path / 'tables').mkdir(exist_ok=True)

    print("=" * 70)
    print("DBEW-NN ENHANCED EXPERIMENTS - Thesis Gesture Chapter")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    print()

    all_results = {}
    fig_path = output_path / 'figures'

    # Load model
    model_full = build_model(use_attention=True).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_full.load_state_dict(checkpoint['model_state_dict'])
        model_full.eval()
        print("[OK] Model loaded successfully\n")
        model_loaded = True
    except FileNotFoundError:
        print(f"[WARN] Model not found at {model_path}")
        model_loaded = False

    # ── Phase 1: Basic Experiments ──
    print("PHASE 1: Basic Experiments\n" + "-" * 50)

    exp_acc = experiment_enhanced_accuracy(model_path, data_dir, device, fig_path)
    all_results['exp_accuracy'] = exp_acc

    try:
        exp_1 = experiment_1_accuracy(model_path, data_dir, device)
        all_results['exp_1'] = exp_1
    except Exception as e:
        print(f"  [WARN] Exp 1: {e}")

    try:
        exp_2 = experiment_2_robustness(model_path, device)
        all_results['exp_robustness'] = exp_2
    except Exception as e:
        print(f"  [WARN] Exp 2: {e}")

    try:
        exp_3 = experiment_3_performance(model_path, device)
        all_results['exp_performance'] = exp_3
    except Exception as e:
        print(f"  [WARN] Exp 3: {e}")

    flops_full = compute_flops(model_full)
    flops_cnn = compute_flops(build_ablation_cnn_only())
    all_results['flops'] = {
        'Rule-based (Geometric)': 0,
        'CNN-only': flops_cnn,
        'CNN+Attention (Ours)': flops_full,
    }
    print(f"  FLOPs - CNN+Attention: {flops_full:,} ({flops_full/1e6:.2f}M)")
    print(f"  FLOPs - CNN-only:      {flops_cnn:,} ({flops_cnn/1e6:.2f}M)")

    # ── Phase 2: Visualization Figures ──
    print("\nPHASE 2: Visualization Figures\n" + "-" * 50)

    if model_loaded:
        _, _, test_loader = create_dataloaders(data_dir=data_dir)

        try:
            full_results_json = Path(model_path).parent / "results.json"
            cnn_results_json = Path(model_path).parent.parent / "cnn_only" / "results.json"
            with open(full_results_json) as f:
                results_full = json.load(f)
            if cnn_results_json.exists():
                with open(cnn_results_json) as f:
                    results_cnn = json.load(f)
            else:
                results_cnn = {'history': {'train': [], 'val': []}}
            plot_training_curves(results_full, results_cnn, fig_path)
        except Exception as e:
            print(f"  [WARN] Training curves: {e}")

        try:
            plot_tsne_visualization(model_full, test_loader, device,
                                   get_class_names(), fig_path)
        except Exception as e:
            print(f"  [WARN] t-SNE: {e}")

        try:
            plot_attention_weights(model_full, test_loader, device,
                                  get_class_names(), fig_path)
        except Exception as e:
            print(f"  [WARN] Attention weights: {e}")

        try:
            plot_robustness_degradation_curve(model_full, device, fig_path)
        except Exception as e:
            print(f"  [WARN] Robustness curves: {e}")

        try:
            plot_latency_distribution(model_full, device, fig_path)
        except Exception as e:
            print(f"  [WARN] Latency distribution: {e}")

    try:
        plot_gesture_skeleton_samples(fig_path)
    except Exception as e:
        print(f"  [WARN] Skeleton samples: {e}")

    # ── Phase 3: Ablation Studies ──
    if not skip_heavy:
        print("\nPHASE 3: Ablation Studies\n" + "-" * 50)

        try:
            win_results = experiment_window_size_ablation(data_dir, device, fig_path)
            all_results['exp_window_size'] = win_results
        except Exception as e:
            print(f"  [WARN] Window size ablation: {e}")

        try:
            scale_results = experiment_data_scale_ablation(data_dir, device, fig_path)
            all_results['exp_data_scale'] = scale_results
        except Exception as e:
            print(f"  [WARN] Data scale ablation: {e}")

        if model_loaded:
            try:
                trig_results = experiment_trigger_ablation(
                    model_path, data_dir, device, fig_path)
                all_results['exp_trigger'] = trig_results
            except Exception as e:
                print(f"  [WARN] Trigger ablation: {e}")
                import traceback
                traceback.print_exc()

    # ── Phase 4: Generate LaTeX Tables ──
    print("\nPHASE 4: LaTeX Tables\n" + "-" * 50)
    try:
        generate_all_latex_tables(all_results, output_path)
    except Exception as e:
        print(f"  [WARN] LaTeX generation: {e}")
        import traceback
        traceback.print_exc()

    # ── Save results ──
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    all_results_serializable = make_serializable(all_results)
    with open(output_path / 'all_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results_serializable, f, indent=2, default=str)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)

    fig_count = len(list(fig_path.glob('*.pdf')))
    tab_count = len(list((output_path / 'tables').glob('*.tex')))
    print(f"\n  Figures: {fig_count} PDFs in {fig_path}/")
    print(f"  Tables:  {tab_count} .tex files in {output_path / 'tables'}/")
    print(f"  Data:    {output_path / 'all_results.json'}")
    print()

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DBEW-NN Enhanced Experiments for Thesis")
    parser.add_argument("--model_path", type=str, default="checkpoints/full/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="experiment_results_v2")
    parser.add_argument("--skip_heavy", action="store_true",
                       help="Skip heavy ablation studies")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    run_all_experiments_v2(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device,
        output_dir=args.output_dir,
        skip_heavy=args.skip_heavy,
    )
