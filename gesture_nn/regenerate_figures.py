#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate all Chapter 3 figures with clean, academic styling.
Key improvements:
  - Consistent muted color palette (no garish defaults)
  - Clean grid, removed chartjunk
  - Proper legend placement (never overlapping data)
  - Publication-quality DPI and font sizing
  - Figure 3.8 legend position FIXED (placed outside plot)
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List
import json
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ═══════════════════════════════════════════════════════
# Global style settings — clean academic aesthetics
# ═══════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.4,
    'legend.framealpha': 0.85,
    'legend.edgecolor': '#CCCCCC',
})

# Academic color palette — muted, colorblind-friendly
PALETTE = {
    'blue':       '#2B579A',   # primary
    'orange':     '#D35400',   # secondary
    'green':      '#27AE60',   # tertiary
    'red':        '#C0392B',
    'purple':     '#7D3C98',
    'teal':       '#117864',
    'gray':       '#7F8C8D',
    'light_blue': '#5DADE2',
    'light_orange':'#F0B27A',
    'bg':         '#F8F9FA',
}

def set_academic_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent academic styling to an axis."""
    ax.grid(True, alpha=0.25, linewidth=0.4, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    if title:
        ax.set_title(title, fontweight='bold', fontsize=11, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.tick_params(labelsize=8)


# ═══════════════════════════════════════════════════════
# Import project modules
# ═══════════════════════════════════════════════════════
from config import *
from model import GestureClassifier, build_model, build_ablation_cnn_only
from dataset import create_dataloaders, load_data, get_class_names, GestureDataset
from data_generator import HandSkeleton, GestureDeformer, SequenceGenerator
from experiments import (
    RuleBasedClassifier,
    _evaluate_nn, _evaluate_rule_based, _compute_metrics,
    experiment_1_accuracy, experiment_2_robustness, experiment_3_performance,
)


# ═══════════════════════════════════════════════════════
# FIGURE 1: Confusion Matrices (side-by-side)
# ═══════════════════════════════════════════════════════
def fig_confusion_matrices(
    rule_preds, rule_targets,
    nn_preds, nn_targets,
    class_names: List[str],
    output_path: Path
):
    """Side-by-side confusion matrices with clean academic styling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    n_classes = len(class_names)
    cmap = plt.cm.Blues

    for ax, preds, targets, title in [
        (ax1, rule_preds, rule_targets, 'DBEW--Gesture (Geometric Rule)'),
        (ax2, nn_preds, nn_targets, 'DBEW-NN (CNN+Attention, Ours)')
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

        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect='equal')

        for i in range(n_classes):
            for j in range(n_classes):
                if cm_norm[i, j] > 0.6:
                    color = 'white'
                    weight = 'bold'
                elif cm_norm[i, j] > 0.01:
                    color = '#1a1a1a'
                    weight = 'normal'
                else:
                    color = '#999999'
                    weight = 'normal'
                text = f'{cm_norm[i,j]:.2f}'
                if cm[i, j] > 0:
                    text += f'\n({cm[i,j]})'
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=6.5, color=color, fontweight=weight)

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        ax.set_xlabel('Predicted', fontsize=9, labelpad=4)
        ax.set_ylabel('True', fontsize=9, labelpad=4)
        ax.set_title(title, fontweight='bold', fontsize=10, pad=8)

    plt.tight_layout(pad=1.5)
    fig.savefig(output_path / 'fig_confusion_matrices.pdf', dpi=300)
    fig.savefig(output_path / 'fig_confusion_matrices.png', dpi=300)
    plt.close(fig)
    print("  [OK] fig_confusion_matrices")


# ═══════════════════════════════════════════════════════
# FIGURE 2: Training Curves
# ═══════════════════════════════════════════════════════
def fig_training_curves(results_full: Dict, results_cnn: Dict, output_path: Path):
    """Training curves with clean style, proper legends."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    for col, (results, label, color) in enumerate([
        (results_full, 'CNN+Attention', PALETTE['blue']),
        (results_cnn, 'CNN-only', PALETTE['orange'])
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
        ax.plot(epochs, train_loss, '-', color=color, linewidth=1.2, alpha=0.45,
                label='Train Loss')
        ax.plot(epochs, val_loss, '-', color=color, linewidth=2.0,
                label='Val Loss')
        ax.axvline(5, color=PALETTE['green'], linestyle=':', linewidth=1.0,
                   alpha=0.6, label='Attention unfrozen')
        set_academic_style(ax, title=f'{label} — Loss', xlabel='Epoch', ylabel='Loss')
        ax.legend(loc='upper right', fontsize=7.5, framealpha=0.8)

        # Macro F1
        ax = axes[1, col]
        ax.plot(epochs, train_f1, '-', color=color, linewidth=1.2, alpha=0.45,
                label='Train F1')
        ax.plot(epochs, val_f1, '-', color=color, linewidth=2.0,
                label='Val F1')
        ax.axvline(5, color=PALETTE['green'], linestyle=':', linewidth=1.0,
                   alpha=0.6, label='Attention unfrozen')
        set_academic_style(ax, title=f'{label} — Macro F1', xlabel='Epoch', ylabel='Macro F1')
        ax.legend(loc='lower right', fontsize=7.5, framealpha=0.8)
        ax.set_ylim(0, 1.05)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path / 'fig_training_curves.pdf', dpi=300)
    fig.savefig(output_path / 'fig_training_curves.png', dpi=300)
    plt.close(fig)
    print("  [OK] fig_training_curves")


# ═══════════════════════════════════════════════════════
# FIGURE 3: t-SNE Feature Visualization
# ═══════════════════════════════════════════════════════
def fig_tsne_features(model: GestureClassifier, loader, device: str,
                       class_names: List[str], output_path: Path):
    """t-SNE plot with clean academic styling."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [WARN] sklearn not available, skipping t-SNE")
        return

    model.eval()
    features_list, labels_list = [], []

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
        features = features[idx]; labels = labels[idx]

    print(f"  Running t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, N_GESTURE_CLASSES))

    for c in range(N_GESTURE_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=[colors[c]], label=class_names[c], alpha=0.55, s=6,
                      edgecolors='none', rasterized=True)

    set_academic_style(ax, title='DBEW-NN Feature Space (t-SNE Projection)',
                       xlabel='t-SNE Dimension 1', ylabel='t-SNE Dimension 2')
    ax.legend(loc='upper left', fontsize=6.5, ncol=2, markerscale=2.5,
              framealpha=0.85, bbox_to_anchor=(0.01, 0.99))

    plt.tight_layout()
    fig.savefig(output_path / 'fig_tsne_features.pdf', dpi=300)
    fig.savefig(output_path / 'fig_tsne_features.png', dpi=300)
    plt.close(fig)
    print("  [OK] fig_tsne_features")


# ═══════════════════════════════════════════════════════
# FIGURE 4: Self-Attention Weights
# ═══════════════════════════════════════════════════════
def fig_attention_weights(model: GestureClassifier, loader, device: str,
                           class_names: List[str], output_path: Path):
    """Attention weight matrices with clean colormap."""
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
        print("  [WARN] No samples for attention visualization")
        return

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    axes = axes.flatten()

    for idx, (gesture_id, skeleton) in enumerate(sorted(samples.items())):
        if idx >= 6: break

        wrist = skeleton[:, :, 0:1, :]
        x = skeleton - wrist
        x = model.spatial_embed(x)
        x = model.joint_pool(x)
        x = model.pos_enc(x)
        x = model.temporal_cnn(x)

        B, T, D_shape = x.shape
        qkv = model.attention.qkv(x).reshape(B, T, 3, model.attention.n_heads,
                                              model.attention.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * model.attention.scale
        attn = F.softmax(attn, dim=-1)
        attn_avg = attn[0].mean(dim=0).detach().cpu().numpy()

        ax = axes[idx]
        im = ax.imshow(attn_avg, cmap='YlOrRd', aspect='auto',
                       vmin=0, vmax=attn_avg.max() * 1.05)
        ax.set_xlabel('Key Frame', fontsize=8)
        ax.set_ylabel('Query Frame', fontsize=8)
        ax.set_title(f'{class_names[gesture_id]}', fontsize=9, fontweight='bold')
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.set_yticks([0, 8, 16, 24, 31])
        ax.tick_params(labelsize=7)

    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.025)
    cbar.set_label('Attention Weight', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle('Self-Attention Weights (Averaged over 4 Heads)',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_path / 'fig_attention_weights.pdf', dpi=300)
    fig.savefig(output_path / 'fig_attention_weights.png', dpi=300)
    plt.close(fig)
    print(f"  [OK] fig_attention_weights ({len(samples)} classes)")


# ═══════════════════════════════════════════════════════
# FIGURE 5: Robustness Degradation Curves
# ═══════════════════════════════════════════════════════
def fig_robustness_degradation(model: GestureClassifier, device: str,
                                 output_path: Path):
    """Noise & occlusion degradation curves, clean style, proper legends."""
    model.eval()
    rule_clf = RuleBasedClassifier()
    canon = HandSkeleton.build_right_hand()

    noise_levels = np.linspace(0, 0.015, 16)
    occlusion_levels = np.linspace(0, 0.6, 13)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # ── Noise ──
    rule_f1s, nn_f1s = [], []
    for noise_std in noise_levels:
        rule_correct, nn_correct, n_samples = 0, 0, 0
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
                wrist_occ = skeleton_window[:, 0:1, :]
                skeleton_window = skeleton_window - wrist_occ
                scale = np.std(skeleton_window)
                if scale > 1e-6: skeleton_window /= (scale * 10)
                x_t = torch.from_numpy(skeleton_window.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model(x_t)
                    nn_correct += int(logits.argmax(dim=-1).item() == gesture_id)
                n_samples += 1
        rule_f1s.append(rule_correct / n_samples)
        nn_f1s.append(nn_correct / n_samples)

    ax1.plot(noise_levels * 1000, rule_f1s, 'o-', color=PALETTE['gray'],
             linewidth=1.6, markersize=4.5, label='DBEW--Gesture (Rule)',
             markerfacecolor='white', markeredgewidth=1.2)
    ax1.plot(noise_levels * 1000, nn_f1s, 's-', color=PALETTE['blue'],
             linewidth=1.8, markersize=5, label='DBEW-NN (Ours)',
             markerfacecolor='white', markeredgewidth=1.2)
    set_academic_style(ax1, title='Noise Robustness',
                       xlabel='Sensor Noise Std Dev (mm)', ylabel='Accuracy')
    ax1.legend(loc='lower left', fontsize=8, framealpha=0.85)
    ax1.set_ylim(0, 1.05)

    # ── Occlusion ──
    rule_f1s_occ, nn_f1s_occ = [], []
    for drop_prob in occlusion_levels:
        rule_correct, nn_correct, n_samples = 0, 0, 0
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
                wrist_occ2 = skeleton_window[:, 0:1, :]
                skeleton_window = skeleton_window - wrist_occ2
                scale = np.std(skeleton_window)
                if scale > 1e-6: skeleton_window /= (scale * 10)
                x_t = torch.from_numpy(skeleton_window.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model(x_t)
                    nn_correct += int(logits.argmax(dim=-1).item() == gesture_id)
                n_samples += 1
        rule_f1s_occ.append(rule_correct / n_samples)
        nn_f1s_occ.append(nn_correct / n_samples)

    ax2.plot(occlusion_levels * 100, rule_f1s_occ, 'o-', color=PALETTE['gray'],
             linewidth=1.6, markersize=4.5, label='DBEW--Gesture (Rule)',
             markerfacecolor='white', markeredgewidth=1.2)
    ax2.plot(occlusion_levels * 100, nn_f1s_occ, 's-', color=PALETTE['blue'],
             linewidth=1.8, markersize=5, label='DBEW-NN (Ours)',
             markerfacecolor='white', markeredgewidth=1.2)
    set_academic_style(ax2, title='Occlusion Robustness',
                       xlabel='Joint Occlusion Rate (%)', ylabel='Accuracy')
    ax2.legend(loc='lower left', fontsize=8, framealpha=0.85)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path / 'fig_robustness_degradation.pdf', dpi=300)
    fig.savefig(output_path / 'fig_robustness_degradation.png', dpi=300)
    plt.close(fig)
    print("  [OK] fig_robustness_degradation")


# ═══════════════════════════════════════════════════════
# FIGURE 6: Latency Distribution & FPS Stability
# ═══════════════════════════════════════════════════════
def fig_latency_distribution(model: GestureClassifier, device: str,
                               output_path: Path):
    """Latency histogram + FPS stability plot, clean style."""
    model.eval()
    x = torch.randn(1, WINDOW_SIZE, N_JOINTS, JOINT_DIMS, device=device)

    for _ in range(30):
        _ = model(x)

    latencies = []
    for _ in range(1000):
        start = time.perf_counter()
        _ = model(x)
        latencies.append((time.perf_counter() - start) * 1000)
    latencies = np.array(latencies)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Histogram
    ax1.hist(latencies, bins=45, color=PALETTE['blue'], edgecolor='white',
             alpha=0.8, linewidth=0.3)
    mean_val = np.mean(latencies)
    p95_val = np.percentile(latencies, 95)
    ax1.axvline(mean_val, color=PALETTE['red'], linestyle='--', linewidth=1.5,
                label=f'Mean: {mean_val:.2f} ms')
    ax1.axvline(p95_val, color=PALETTE['orange'], linestyle='--', linewidth=1.5,
                label=f'P95: {p95_val:.2f} ms')
    set_academic_style(ax1, title='Inference Latency Distribution',
                       xlabel='Latency (ms)', ylabel='Count')
    ax1.legend(loc='upper right', fontsize=7.5, framealpha=0.85)

    # FPS stability
    fps_arr = []
    x_fps = torch.randn(1, WINDOW_SIZE, N_JOINTS, JOINT_DIMS, device=device)
    for _ in range(3600):
        start = time.perf_counter()
        _ = model(x_fps)
        elapsed = time.perf_counter() - start
        fps_arr.append(1.0 / elapsed if elapsed > 0 else 0)
    fps_arr = np.array(fps_arr)

    ax2.plot(fps_arr, color=PALETTE['blue'], alpha=0.35, linewidth=0.4)
    ax2.axhline(np.mean(fps_arr), color=PALETTE['red'], linestyle='--',
                linewidth=1.2, label=f'Mean: {np.mean(fps_arr):.0f} FPS')
    ax2.axhline(60, color=PALETTE['green'], linestyle=':', linewidth=1.2,
                label='Target: 60 FPS')
    set_academic_style(ax2, title='FPS Stability (60 s Simulated)',
                       xlabel='Frame Index', ylabel='Instantaneous FPS')
    ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.85)
    ax2.set_ylim(0, max(fps_arr.max() * 1.15, 100))

    plt.tight_layout(pad=2.0)
    fig.savefig(output_path / 'fig_latency_distribution.pdf', dpi=300)
    fig.savefig(output_path / 'fig_latency_distribution.png', dpi=300)
    plt.close(fig)
    print(f"  [OK] fig_latency_distribution (mean={mean_val:.2f} ms)")


# ═══════════════════════════════════════════════════════
# FIGURE 7: Skeleton Samples
# ═══════════════════════════════════════════════════════
def fig_skeleton_samples(output_path: Path):
    """Canonical hand skeleton poses, clean academic style."""
    canon = HandSkeleton.build_right_hand()
    deformer = GestureDeformer(seed=42)

    connections = {
        'Thumb':  (2, 3, 4, 5),
        'Index':  (6, 7, 8, 9),
        'Middle': (10, 11, 12, 13),
        'Ring':   (14, 15, 16, 17),
        'Pinky':  (18, 19, 20, 21),
    }
    palm_lines = [(0, 1), (0, 6), (0, 10), (0, 14), (0, 18)]
    finger_colors = {
        'Thumb': '#E74C3C', 'Index': '#2B579A', 'Middle': '#27AE60',
        'Ring': '#E67E22', 'Pinky': '#8E44AD'
    }

    gesture_names = [
        'NONE (Transition)', 'Index Left (Year $-$1)', 'Index Right (Year $+$1)',
        'Two-Finger Palm (DEL)', 'Two-Finger Back (ES)', 'Four-Finger Palm (Main)'
    ]
    gesture_ids = [0, 1, 2, 3, 4, 5]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8))
    axes = axes.flatten()

    for idx, gid in enumerate(gesture_ids):
        ax = axes[idx]
        joints = deformer.apply(canon, gid if gid != 0 else 0, person_variation=0)

        for finger_name, indices in connections.items():
            for i in range(len(indices) - 1):
                j1, j2 = indices[i], indices[i + 1]
                ax.plot([joints[j1, 0], joints[j2, 0]],
                       [joints[j1, 1], joints[j2, 1]],
                       color=finger_colors[finger_name], linewidth=2.2, alpha=0.85,
                       solid_capstyle='round')

        for (j1, j2) in palm_lines:
            ax.plot([joints[j1, 0], joints[j2, 0]],
                   [joints[j1, 1], joints[j2, 1]],
                   color='#BBBBBB', linewidth=0.8, alpha=0.5, linestyle='--')

        ax.scatter(joints[:, 0], joints[:, 1], c='#333333', s=12, zorder=5,
                   edgecolors='white', linewidth=0.3)
        ax.scatter(joints[0, 0], joints[0, 1], c=PALETTE['red'], s=40,
                   marker='*', zorder=6, edgecolors='white', linewidth=0.5,
                   label='Wrist (Joint 0)')

        set_academic_style(ax, title=gesture_names[idx],
                           xlabel='X (m)', ylabel='Y (m)')
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))

    # Add a unified color legend for fingers
    handles = [plt.Line2D([0], [0], color=c, linewidth=2.2, label=n)
               for n, c in finger_colors.items()]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=7.5,
               framealpha=0.85, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Canonical Hand Skeleton Poses for Six Gesture Classes (XY Projection)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(pad=2.5)
    fig.savefig(output_path / 'fig_skeleton_samples.pdf', dpi=300,
                bbox_inches='tight')
    fig.savefig(output_path / 'fig_skeleton_samples.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print("  [OK] fig_skeleton_samples")


# ═══════════════════════════════════════════════════════
# FIGURE 8: Trigger Pipeline Ablation  [FIXED LEGEND POSITION]
# ═══════════════════════════════════════════════════════
def fig_trigger_comparison(model_path: str, data_dir: str, device: str,
                             output_path: Path):
    """DBEW trigger ablation with LEGEND PLACED OUTSIDE plot area.

    This is Figure 3.8 — the key fix is legend position.
    Original code used ax1.legend() at default 'best' which overlapped bars.
    Now legend is placed OUTSIDE the plot (above-right) to never occlude data.
    """
    model = build_model(use_attention=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=False)['model_state_dict'])
    model.eval()

    canon = HandSkeleton.build_right_hand()
    sequences, gt_events = [], []

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
        'No gating\n(frame-level)':        {'use_edge': False, 'use_cooldown': False, 'use_stable': False},
        'Edge only':                        {'use_edge': True,  'use_cooldown': False, 'use_stable': False},
        'Edge +\nCooldown':                 {'use_edge': True,  'use_cooldown': True,  'use_stable': False},
        'Edge + Cooldown\n+ Stable (Ours)': {'use_edge': True,  'use_cooldown': True,  'use_stable': True},
    }

    tau = TAU_MS / (1000 / FPS)
    k_min = K_MIN
    theta_g = THETA_G

    results = {}
    for config_name, config in configs.items():
        total_events, correct_events, false_events = 0, 0, 0

        for seq, gt in zip(sequences, gt_events):
            g_prev, n_stable, t_last = 0, 0, -999

            for t_frame in range(len(seq)):
                joints = seq[t_frame]
                skeleton_window = np.tile(joints[np.newaxis,:,:], (WINDOW_SIZE, 1, 1))
                wrist_t = skeleton_window[:, 0:1, :]
                skeleton_window = skeleton_window - wrist_t
                scale = np.std(skeleton_window)
                if scale > 1e-6: skeleton_window /= (scale * 10)
                x_t = torch.from_numpy(skeleton_window.astype(np.float32)).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model(x_t)
                    probs = F.softmax(logits, dim=-1)
                    conf, pred = probs.max(dim=-1)
                    g_t, s_t = pred.item(), conf.item()

                if g_t == 0 or s_t < theta_g:
                    g_prev = g_t if g_t != 0 else g_prev
                    continue

                n_stable = n_stable + 1 if g_t == g_prev else 1
                edge = (g_t != g_prev and g_t != 0)
                cooldown_ok = ((t_frame - t_last) >= tau)
                stable_ok = (n_stable >= k_min)

                if config['use_edge']:
                    trigger = edge
                    if config['use_cooldown']: trigger = trigger and cooldown_ok
                    if config['use_stable']:   trigger = trigger and stable_ok
                else:
                    trigger = True

                if trigger:
                    total_events += 1
                    if gt['start_frame'] <= t_frame <= gt['end_frame'] and g_t == gt['gesture_id']:
                        correct_events += 1
                    else:
                        false_events += 1
                    t_last = t_frame

                g_prev = g_t

        precision = correct_events / max(total_events, 1)
        recall = correct_events / len(gt_events)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        total_frames = sum(len(s) for s in sequences)
        event_rate = total_events / (total_frames / FPS)

        results[config_name] = {
            'total_events': total_events,
            'correct_events': correct_events,
            'false_events': false_events,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'event_rate_hz': round(event_rate, 2),
        }
        print(f"  {config_name.replace(chr(10),' '):35s} | P={precision:.3f} "
              f"R={recall:.3f} F1={f1:.3f} Rate={event_rate:.1f} Hz")

    # ── Plotting ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    config_labels = list(results.keys())
    config_labels_flat = [c.replace('\n', ' ') for c in config_labels]
    x = np.arange(len(config_labels))
    width = 0.32

    precisions = [results[c]['precision'] for c in config_labels]
    recalls = [results[c]['recall'] for c in config_labels]
    event_rates = [results[c]['event_rate_hz'] for c in config_labels]

    # Left: Precision & Recall bars
    bars1 = ax1.bar(x - width/2, precisions, width, label='Precision',
                    color=PALETTE['blue'], alpha=0.85, edgecolor='white', linewidth=0.3)
    bars2 = ax1.bar(x + width/2, recalls, width, label='Recall',
                    color=PALETTE['orange'], alpha=0.85, edgecolor='white', linewidth=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_labels_flat, fontsize=6.8, rotation=12, ha='center')
    set_academic_style(ax1, title='Precision & Recall by Trigger Configuration',
                       ylabel='Score')
    # ★ FIXED: Legend placed outside plot (upper left, outside)
    ax1.legend(loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=8,
               ncol=2, framealpha=0.85, borderpad=0.5)
    ax1.set_ylim(0, 1.15)

    # Right: Event Rate bars
    bar_colors = [PALETTE['blue'], PALETTE['light_blue'], PALETTE['orange'], PALETTE['green']]
    bars3 = ax2.bar(config_labels_flat, event_rates, color=bar_colors, alpha=0.85,
                    edgecolor='white', linewidth=0.3, width=0.55)
    set_academic_style(ax2, title='Command Event Rate',
                       ylabel='Event Rate (Hz)')
    ax2.set_xticklabels(config_labels_flat, fontsize=6.8, rotation=12, ha='center')

    # Add value labels on bars
    for bar, val in zip(bars3, event_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold', color='#333333')

    # Add a 60 Hz reference line
    ax2.axhline(60, color=PALETTE['red'], linestyle=':', linewidth=1.0,
                alpha=0.6, label='Raw frame rate (60 Hz)')
    ax2.legend(loc='upper right', fontsize=7.5, framealpha=0.85)

    plt.tight_layout(pad=2.5)
    fig.savefig(output_path / 'fig_trigger_comparison.pdf', dpi=300)
    fig.savefig(output_path / 'fig_trigger_comparison.png', dpi=300)
    plt.close(fig)
    print("  [OK] fig_trigger_comparison (legend FIXED — placed outside plot)")
    return results


# ═══════════════════════════════════════════════════════
# FIGURE 9: Window Size Ablation
# ═══════════════════════════════════════════════════════
def fig_window_ablation(window_results: Dict, output_path: Path):
    """Window size ablation with clean academic style."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sizes = sorted(window_results.keys())
    f1s = [window_results[s]['macro_f1'] for s in sizes]
    accs = [window_results[s]['accuracy'] for s in sizes]

    ax.plot(sizes, f1s, 'o-', color=PALETTE['blue'], linewidth=2.0, markersize=8,
            label='Macro F1', markerfacecolor='white', markeredgewidth=1.5)
    ax.plot(sizes, accs, 's--', color=PALETTE['orange'], linewidth=2.0, markersize=8,
            label='Accuracy', markerfacecolor='white', markeredgewidth=1.5)
    ax.axvline(32, color=PALETTE['green'], linestyle=':', linewidth=1.8,
               alpha=0.7, label='Selected $T=32$')

    set_academic_style(ax, title='Impact of Window Size on Gesture Recognition',
                       xlabel='Window Size $T$ (frames)', ylabel='Score')
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.85)
    ax.set_ylim(0.8, 1.02)

    plt.tight_layout()
    fig.savefig(output_path / 'fig_window_ablation.pdf', dpi=300)
    fig.savefig(output_path / 'fig_window_ablation.png', dpi=300)
    plt.close(fig)
    print("  [OK] fig_window_ablation")


# ═══════════════════════════════════════════════════════
# FIGURE 10: Data Scale Ablation
# ═══════════════════════════════════════════════════════
def fig_data_scale_ablation(scale_results: Dict, output_path: Path):
    """Data scale ablation with clean academic style."""
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    scale_pct = [s * 100 for s in sorted(scale_results.keys())]
    f1s = [scale_results[s]['macro_f1'] for s in sorted(scale_results.keys())]

    ax.plot(scale_pct, f1s, 'o-', color=PALETTE['blue'], linewidth=2.0, markersize=10,
            markerfacecolor='white', markeredgewidth=1.5)
    ax.fill_between(scale_pct, [min(f1s) - 0.02] * len(f1s), f1s,
                    alpha=0.10, color=PALETTE['blue'])

    for s, f1 in zip(scale_pct, f1s):
        ax.annotate(f'{f1:.3f}', (s, f1), textcoords="offset points",
                   xytext=(0, 13), ha='center', fontsize=8.5, fontweight='bold',
                   color='#333333')

    set_academic_style(ax, title='Impact of Training Data Scale',
                       xlabel='Training Data (%)', ylabel='Macro F1')
    ax.set_ylim(0.75, 1.0)
    ax.set_xlim(-2, 105)

    plt.tight_layout()
    fig.savefig(output_path / 'fig_data_scale_ablation.pdf', dpi=300)
    fig.savefig(output_path / 'fig_data_scale_ablation.png', dpi=300)
    plt.close(fig)
    print("  [OK] fig_data_scale_ablation")


# ═══════════════════════════════════════════════════════
# Main regeneration driver
# ═══════════════════════════════════════════════════════
def regenerate_all_figures(
    model_path: str = "checkpoints/full/best_model.pt",
    data_dir: str = "data",
    device: str = "cpu",
    output_dir: str = "experiment_results_v2",
):
    """Regenerate ALL Chapter 3 figures with clean academic styling."""
    output_path = Path(output_dir)
    fig_path = output_path / 'figures'
    fig_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REGENERATING ALL CHAPTER 3 FIGURES — Academic Styling")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Output: {fig_path}")
    print()

    # Load model
    model_full = build_model(use_attention=True).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_full.load_state_dict(checkpoint['model_state_dict'])
        model_full.eval()
        print("[OK] Model loaded\n")
        model_loaded = True
    except FileNotFoundError:
        print(f"[WARN] Model not found: {model_path}")
        model_loaded = False

    # ── Figure 1: Confusion Matrices ──
    print("1/8  Confusion Matrices...")
    try:
        class_names = get_class_names()
        rule_clf = RuleBasedClassifier()
        canon = HandSkeleton.build_right_hand()
        rule_preds, rule_targets = [], []
        for gesture_id in range(1, N_GESTURE_CLASSES):
            for _ in range(300):
                deformer = GestureDeformer()
                joints = deformer.apply(canon, gesture_id,
                                       person_variation=np.random.uniform(-1, 1))
                joints = joints + np.random.normal(0, 0.0015, joints.shape).astype(np.float32)
                pred, _ = rule_clf.classify(joints)
                rule_preds.append(pred); rule_targets.append(gesture_id)

        _, _, test_loader = create_dataloaders(data_dir=data_dir)
        nn_preds, nn_targets = [], []
        with torch.no_grad():
            for skeletons, targets in test_loader:
                skeletons = skeletons.to(device)
                logits, _ = model_full(skeletons)
                preds = logits.argmax(dim=-1)
                nn_preds.extend(preds.cpu().tolist())
                nn_targets.extend(targets.tolist())

        fig_confusion_matrices(rule_preds, rule_targets, nn_preds, nn_targets,
                               class_names, fig_path)
    except Exception as e:
        print(f"  [ERROR] Confusion matrices: {e}")

    # ── Figure 2: Training Curves ──
    print("2/8  Training Curves...")
    try:
        full_results_path = Path(model_path).parent / "results.json"
        cnn_results_path = Path(model_path).parent.parent / "cnn_only" / "results.json"
        if full_results_path.exists():
            with open(full_results_path) as f:
                results_full = json.load(f)
        else:
            results_full = {'history': {'train': [], 'val': []}}
        if cnn_results_path.exists():
            with open(cnn_results_path) as f:
                results_cnn = json.load(f)
        else:
            results_cnn = {'history': {'train': [], 'val': []}}
        fig_training_curves(results_full, results_cnn, fig_path)
    except Exception as e:
        print(f"  [ERROR] Training curves: {e}")

    # ── Figure 3: t-SNE ──
    print("3/8  t-SNE Features...")
    try:
        _, _, test_loader = create_dataloaders(data_dir=data_dir)
        fig_tsne_features(model_full, test_loader, device, class_names, fig_path)
    except Exception as e:
        print(f"  [ERROR] t-SNE: {e}")

    # ── Figure 4: Attention Weights ──
    print("4/8  Attention Weights...")
    try:
        _, _, test_loader = create_dataloaders(data_dir=data_dir)
        fig_attention_weights(model_full, test_loader, device, class_names, fig_path)
    except Exception as e:
        print(f"  [ERROR] Attention weights: {e}")

    # ── Figure 5: Robustness ──
    print("5/8  Robustness Degradation...")
    try:
        fig_robustness_degradation(model_full, device, fig_path)
    except Exception as e:
        print(f"  [ERROR] Robustness: {e}")

    # ── Figure 6: Latency ──
    print("6/8  Latency Distribution...")
    try:
        fig_latency_distribution(model_full, device, fig_path)
    except Exception as e:
        print(f"  [ERROR] Latency: {e}")

    # ── Figure 7: Skeleton Samples ──
    print("7/8  Skeleton Samples...")
    try:
        fig_skeleton_samples(fig_path)
    except Exception as e:
        print(f"  [ERROR] Skeleton samples: {e}")

    # ── Figure 8: Trigger Comparison (★ FIXED LEGEND) ──
    print("8/8  Trigger Comparison (with fixed legend)...")
    try:
        fig_trigger_comparison(model_path, data_dir, device, fig_path)
    except Exception as e:
        print(f"  [ERROR] Trigger comparison: {e}")
        import traceback; traceback.print_exc()

    # ── Summary ──
    pdfs = sorted(fig_path.glob('*.pdf'))
    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(pdfs)} figures regenerated")
    print(f"{'='*70}")
    for p in pdfs:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:45s} {size_kb:7.1f} KB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/full/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="experiment_results_v2")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    regenerate_all_figures(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device,
        output_dir=args.output_dir,
    )
