"""
Interactive demo: generate test gesture samples and classify with trained model.
Visualizes 3D hand skeleton and model predictions with confidence scores.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

from config import *
from model import build_model
from data_generator import HandSkeleton, GestureDeformer
from experiments import RuleBasedClassifier

# ── setup ──
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GESTURE_NAMES = {
    0: 'NONE (无手势)',
    1: 'index_left (食指向左)',
    2: 'index_right (食指向右)',
    3: 'two_finger_palm (双指掌心)',
    4: 'two_finger_back (双指手背)',
    5: 'four_finger_palm (四指掌心)',
    6: 'fist (握拳)',
}
# Joint connectivity for skeleton drawing
BONES = [
    # thumb
    (0, 2), (2, 3), (3, 4), (4, 5),
    # index
    (0, 6), (6, 7), (7, 8), (8, 9),
    # middle
    (0, 10), (10, 11), (11, 12), (12, 13),
    # ring
    (0, 14), (14, 15), (15, 16), (16, 17),
    # pinky
    (0, 18), (18, 19), (19, 20), (20, 21),
    # palm cross connections
    (1, 6), (1, 10), (1, 14), (1, 18), (0, 1),
]


def load_model():
    ckpt_path = 'checkpoints/full/best_model.pt'
    model = build_model(use_attention=True).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded model (epoch {ckpt["epoch"]}, val_f1={ckpt["val_metrics"]["macro_f1"]:.4f})')
    return model


def generate_sample(gesture_id, noise_std=0.0015, person_var=0.5):
    """Generate one clean gesture sample (single frame with variation)."""
    canon = HandSkeleton.build_right_hand()
    deformer = GestureDeformer()
    joints = deformer.apply(canon, gesture_id, person_variation=person_var)
    joints = joints + np.random.normal(0, noise_std, joints.shape).astype(np.float32)
    return joints


def joints_to_window(joints):
    """Convert single frame [26,3] to model input [1, 32, 26, 3] with normalization."""
    window = np.tile(joints[np.newaxis, :, :], (WINDOW_SIZE, 1, 1)).astype(np.float32)
    # normalize: wrist-relative
    wrist = window[:, 0:1, :]
    window = window - wrist
    scale = np.std(window)
    if scale > 1e-6:
        window = window / (scale * 10)
    return torch.from_numpy(window).unsqueeze(0).to(DEVICE)


def classify_sample(model, joints):
    """Classify a single frame with the NN model. Returns (pred_id, confidence, all_probs)."""
    x = joints_to_window(joints)
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred = int(probs.argmax())
    conf = float(probs[pred])
    return pred, conf, probs


def draw_skeleton_2d(ax, joints, title='', pred_label='', true_label='',
                     color='#2563EB', correct=True):
    """Draw 2D projection of 3D hand skeleton (top-down: x-z plane)."""
    # Project to x-z plane (top-down view of hand)
    x = joints[:, 0]
    z = joints[:, 2]  # depth → vertical in 2D

    # Draw bones
    for (i, j) in BONES:
        ax.plot([x[i], x[j]], [z[i], z[j]], '-', color=color, linewidth=1.5, alpha=0.8)

    # Draw joints
    ax.scatter(x, z, c=color, s=20, zorder=5, alpha=0.9)
    ax.scatter(x[0], z[0], c='red', s=50, zorder=6, marker='s', label='Wrist')  # wrist

    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(-0.05, 0.15)
    ax.set_aspect('equal')
    ax.axis('off')

    # Color-coded border based on correctness
    border_color = '#16A34A' if correct else '#DC2626'
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)

    # Title
    status = '[OK]' if correct else '[FAIL]'
    ax.set_title(f'{status} {title}', fontsize=9, fontweight='bold',
                 color=border_color, pad=3)

    # Subtitle with prediction info
    ax.text(0.5, -0.08, pred_label, transform=ax.transAxes, fontsize=7,
            ha='center', va='top', color='#6B7280')


# ── main demo ──
def main():
    print('=' * 60)
    print('DBEW-NN Gesture Classifier — Interactive Demo')
    print('=' * 60)

    model = load_model()
    rule_clf = RuleBasedClassifier()

    # Generate samples: 3 per gesture class (skip NONE=0, create clean + noisy variants)
    np.random.seed(42)

    all_cases = []
    for gid in range(1, N_GESTURE_CLASSES):
        # clean sample
        joints_clean = generate_sample(gid, noise_std=0.0015, person_var=0.3)
        all_cases.append((gid, 'clean', joints_clean))
        # medium noise
        joints_med = generate_sample(gid, noise_std=0.004, person_var=0.8)
        all_cases.append((gid, 'noisy', joints_med))
        # challenging: high noise + large person variation
        joints_hard = generate_sample(gid, noise_std=0.006, person_var=1.0)
        all_cases.append((gid, 'hard', joints_hard))

    # ── classify all ──
    results = []
    for gid, variant, joints in all_cases:
        nn_pred, nn_conf, nn_probs = classify_sample(model, joints)
        rule_pred, rule_conf = rule_clf.classify(joints)
        results.append({
            'gid': gid, 'variant': variant, 'joints': joints,
            'nn_pred': nn_pred, 'nn_conf': nn_conf, 'nn_probs': nn_probs,
            'rule_pred': rule_pred, 'rule_conf': rule_conf,
            'nn_correct': nn_pred == gid,
            'rule_correct': rule_pred == gid,
        })

    # ── summary stats ──
    nn_acc = sum(r['nn_correct'] for r in results) / len(results)
    rule_acc = sum(r['rule_correct'] for r in results) / len(results)
    print(f'\nSummary: NN accuracy={nn_acc:.2%}, Rule accuracy={rule_acc:.2%}')
    print(f'Total test samples: {len(results)} ({len(results)//3} gestures × 3 variants)\n')

    # Print per-sample results
    for r in results:
        nn_mark = '[OK]' if r['nn_correct'] else '[FAIL]'
        rule_mark = '[OK]' if r['rule_correct'] else '[FAIL]'
        print(f'{GESTURE_NAMES[r["gid"]][:20]:20s} | {r["variant"]:5s} | '
              f'NN→ {GESTURE_NAMES[r["nn_pred"]][:20]:20s} conf={r["nn_conf"]:.3f} {nn_mark} | '
              f'Rule→ {GESTURE_NAMES[r["rule_pred"]][:20]:20s} {rule_mark}')

    # ── plot: one big figure with all samples ──
    n_gestures = N_GESTURE_CLASSES - 1  # 6
    n_variants = 3
    fig, axes = plt.subplots(n_gestures, n_variants + 1,
                             figsize=(4 * (n_variants + 1), 3.5 * n_gestures),
                             gridspec_kw={'width_ratios': [1, 1, 1, 0.35]})

    variant_titles = {'clean': 'Clean\n(σ=1.5mm)', 'noisy': 'Noisy\n(σ=4mm)',
                      'hard': 'Hard\n(σ=6mm+var)'}

    for row, gid in enumerate(range(1, N_GESTURE_CLASSES)):
        row_results = [r for r in results if r['gid'] == gid]

        for col, variant in enumerate(['clean', 'noisy', 'hard']):
            ax = axes[row, col]
            r = row_results[col]
            correct = r['nn_correct']
            conf = r['nn_conf']

            # Build prediction label text
            pred_name = GESTURE_NAMES[r['nn_pred']].split('(')[0].strip()
            true_name = GESTURE_NAMES[gid].split('(')[0].strip()
            pred_text = f'Pred: {pred_name} ({conf:.1%})\nTrue: {true_name}'

            draw_skeleton_2d(ax, r['joints'],
                           title=variant_titles[variant] if row == 0 else '',
                           pred_label=pred_text,
                           correct=correct,
                           color='#2563EB' if correct else '#DC2626')

        # Bar chart column: top-3 probabilities
        ax = axes[row, 3]
        r = row_results[0]  # clean sample probs
        probs = r['nn_probs']
        top3_idx = np.argsort(probs)[::-1][:3]
        top3_names = [GESTURE_NAMES[i].split('(')[0].strip() for i in top3_idx]
        top3_vals = probs[top3_idx]
        colors = ['#16A34A' if idx == gid else '#E5E7EB' if idx != gid else '#DC2626'
                  for idx in top3_idx]
        # highlight correct class
        colors = ['#16A34A' if idx == gid else '#9CA3AF' for idx in top3_idx]

        ax.barh(range(3), top3_vals[::-1], color=colors[::-1], height=0.6)
        ax.set_yticks(range(3))
        ax.set_yticklabels(top3_names[::-1], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Confidence', fontsize=7)
        if row == 0:
            ax.set_title('Top-3\nProbs', fontsize=9, fontweight='bold')

        # Mark true class
        true_rank = list(np.argsort(probs)[::-1]).index(gid) + 1
        ax.text(0.95, 0.05, f'True rank: #{true_rank}', transform=ax.transAxes,
                fontsize=7, ha='right', color='#6B7280')

    # Row labels
    for row, gid in enumerate(range(1, N_GESTURE_CLASSES)):
        name = GESTURE_NAMES[gid]
        axes[row, 0].set_ylabel(name, fontsize=10, fontweight='bold', rotation=90,
                                labelpad=10, ha='center', va='center')

    fig.suptitle('DBEW-NN Gesture Classifier — Test Demo\n'
                 f'(NN Accuracy: {nn_acc:.1%} | Rule Accuracy: {rule_acc:.1%} | '
                 f'{len(results)} samples total)',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    out_path = 'demo_test_results.png'
    plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'\nSaved: {out_path}')

    # ── Also generate a confusion-matrix-style summary ──
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    cm = np.zeros((N_GESTURE_CLASSES - 1, N_GESTURE_CLASSES))
    for r in results:
        cm[r['gid'] - 1, r['nn_pred']] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    im = ax2.imshow(cm_norm, cmap='YlGn', vmin=0, vmax=1)
    ax2.set_xticks(range(N_GESTURE_CLASSES))
    short_names = [GESTURE_NAMES[i].split('(')[1].rstrip(')') if '(' in GESTURE_NAMES[i]
                   else GESTURE_NAMES[i] for i in range(N_GESTURE_CLASSES)]
    ax2.set_xticklabels(short_names, fontsize=7, rotation=45, ha='right')
    ax2.set_yticks(range(N_GESTURE_CLASSES - 1))
    ax2.set_yticklabels([GESTURE_NAMES[i].split('(')[0].strip() for i in range(1, N_GESTURE_CLASSES)],
                         fontsize=8)
    ax2.set_ylabel('True Gesture', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Predicted Gesture', fontsize=11, fontweight='bold')
    ax2.set_title('Confusion Matrix (Demo Test Set)', fontsize=12, fontweight='bold')

    for i in range(N_GESTURE_CLASSES - 1):
        for j in range(N_GESTURE_CLASSES):
            count = int(cm[i, j])
            if count > 0:
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax2.text(j, i, f'{count}', ha='center', va='center', fontsize=9,
                        color=color, fontweight='bold')

    plt.colorbar(im, ax=ax2, label='Fraction')
    plt.tight_layout()
    plt.savefig('demo_confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
    print('Saved: demo_confusion_matrix.png')


if __name__ == '__main__':
    main()
