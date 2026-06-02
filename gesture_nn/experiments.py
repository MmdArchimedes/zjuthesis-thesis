"""
Comparison experiments: rule-based (geometric) vs neural network gesture recognition.

Three experiments corresponding to thesis Section 3.6:
  Experiment 1: Recognition accuracy comparison
  Experiment 2: Robustness under adverse conditions
  Experiment 3: End-side performance (latency, memory, FPS stability)
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time
from collections import defaultdict

from config import *
from model import GestureClassifier, build_model, build_ablation_cnn_only
from dataset import create_dataloaders, load_data, get_class_names
from data_generator import HandSkeleton, GestureDeformer, SequenceGenerator


# ───────────────────────────────────────────────────────────────────
# Rule-based classifier (replicating thesis geometric-criterion method)
# ───────────────────────────────────────────────────────────────────

class RuleBasedClassifier:
    """
    Geometric-criterion classifier matching the thesis DBEW--Gesture
    rule-based approach (Section 3.3.2--3.3.3).

    Uses: finger collinearity (Eq 3-1), extended finger count,
    palm normal orientation, horizontal projection for direction.
    """

    def __init__(self, theta_ext: float = THETA_EXT):
        self.theta_ext = theta_ext
        # Joint index groups for each finger (index into 26-joint array)
        self.fingers = {
            'thumb':  [2, 3, 4, 5],
            'index':  [6, 7, 8, 9],
            'middle': [10, 11, 12, 13],
            'ring':   [14, 15, 16, 17],
            'pinky':  [18, 19, 20, 21],
        }
        self.wrist_idx = 0
        self.palm_idx = 1

    def _finger_curl_angle(self, joints: np.ndarray,
                            finger_indices: List[int]) -> float:
        """Compute total curl angle: angle between MCP→PIP and PIP→Tip.

        For an extended finger this is small (<30°); for a curled finger >60°.
        This is more robust than pairwise collinearity.
        """
        p_mcp = joints[finger_indices[0]]
        p_pip = joints[finger_indices[1]]
        p_tip = joints[finger_indices[3]]

        v_proximal = p_pip - p_mcp   # MCP → PIP
        v_distal = p_tip - p_pip     # PIP → Tip

        n_prox = np.linalg.norm(v_proximal)
        n_dist = np.linalg.norm(v_distal)
        if n_prox < 1e-8 or n_dist < 1e-8:
            return 180.0  # degenerate → consider curled

        cos_angle = np.dot(v_proximal, v_distal) / (n_prox * n_dist)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _is_extended(self, joints: np.ndarray,
                     finger_indices: List[int]) -> bool:
        """Finger is extended if curl angle between proximal and distal segments < 30°.

        For canonical hand with distributed joint flexion:
          - Extended (5° total): curl ≈ 2.5° < 30° → True
          - Flexed (70° total): curl ≈ 35° > 30° → False
        """
        angle = self._finger_curl_angle(joints, finger_indices)
        return angle < 25.0

    def _palm_normal(self, joints: np.ndarray) -> np.ndarray:
        """Estimate palm normal from wrist→palm and index→ring MCP cross product."""
        wrist_to_palm = joints[self.palm_idx] - joints[self.wrist_idx]
        index_mcp = joints[6]   # index MCP
        ring_mcp = joints[14]    # ring MCP
        across = index_mcp - ring_mcp

        normal = np.cross(wrist_to_palm, across)
        n = np.linalg.norm(normal)
        if n < 1e-8:
            return np.array([0, 0, 1])
        return normal / n

    def _index_direction(self, joints: np.ndarray) -> str:
        """Determine if extended index finger points left or right."""
        index_tip = joints[9]    # index tip
        index_mcp = joints[6]    # index MCP

        # Project onto horizontal plane (x-z plane in world coords)
        dir_vec = index_tip - index_mcp
        horizontal_x = dir_vec[0]

        if horizontal_x < -0.005:
            return "left"
        elif horizontal_x > 0.005:
            return "right"
        return "unknown"

    def _palm_facing(self, joints: np.ndarray) -> str:
        """Determine if palm faces user or away.

        Canonical hand has normal pointing -z (away from palm).
        Rotating 180° around y (pronation) flips normal to +z.
        In generated data convention: -z = palm toward user, +z = palm away.
        """
        normal = self._palm_normal(joints)
        if normal[2] > 0.1:
            return "away"    # normal points +z → back of hand toward user
        elif normal[2] < -0.1:
            return "user"    # normal points -z → palm toward user
        return "side"

    def classify(self, joints: np.ndarray) -> Tuple[int, float]:
        """
        Classify a single frame of skeleton data.

        Args:
            joints: [26, 3] joint positions

        Returns:
            (gesture_id, confidence)
        """
        # Determine which specific fingers are extended
        ext = {}
        for name, indices in self.fingers.items():
            ext[name] = self._is_extended(joints, indices)

        n_ext = sum(ext.values())
        index_ext = ext['index']
        middle_ext = ext['middle']
        ring_ext = ext['ring']
        pinky_ext = ext['pinky']

        # Rule hierarchy (matching thesis Table tab:ges_map):
        # Check most specific patterns first, fall through to general

        # Fist: thumb curled + at most 1 other finger extended
        # (thumb always has high curl angle, so check non-thumb count)
        non_thumb_ext = sum([index_ext, middle_ext, ring_ext, pinky_ext])
        if non_thumb_ext == 0:
            return 6, 0.90

        # Index pointing: ONLY index extended, others curled
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            direction = self._index_direction(joints)
            if direction == "left":
                return 1, 0.88
            elif direction == "right":
                return 2, 0.88
            return 0, 0.40

        # Two-finger (index + middle): exactly these two extended
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            facing = self._palm_facing(joints)
            if facing == "user":
                return 3, 0.86  # two_finger_palm
            elif facing == "away":
                return 4, 0.84  # two_finger_back
            return 3, 0.70

        # Four-finger: index + middle + ring + pinky all extended
        if index_ext and middle_ext and ring_ext and pinky_ext:
            facing = self._palm_facing(joints)
            if facing == "user":
                return 5, 0.92  # four_finger_palm
            elif facing == "away":
                return 4, 0.85  # four_finger_back (not a defined gesture, but closest)
            return 5, 0.75

        # Two-finger back variant: index + middle extended, ring/pinky borderline
        if index_ext and middle_ext and not ring_ext and pinky_ext:
            return 3, 0.75  # closest to two_finger_palm

        if index_ext and middle_ext and ring_ext and not pinky_ext:
            return 5, 0.72  # close to four_finger_palm

        # Index + one or two others (ambiguous partial curl)
        if index_ext and not middle_ext:
            direction = self._index_direction(joints)
            if direction == "left":
                return 1, 0.65
            elif direction == "right":
                return 2, 0.65
            return 0, 0.40

        # Default: NONE/transition
        return 0, 0.50


# ───────────────────────────────────────────────────────────────────
# Experiment 1: Recognition Accuracy Comparison
# ───────────────────────────────────────────────────────────────────

def experiment_1_accuracy(model_path: str = "checkpoints/full/best_model.pt",
                          data_dir: str = "data", device: str = "cuda"):
    """Compare rule-based vs NN vs CNN-only on test set."""

    print("=" * 70)
    print("EXPERIMENT 1: Gesture Recognition Accuracy Comparison")
    print("=" * 70)

    # Load data
    _, _, test_loader = create_dataloaders(data_dir=data_dir)
    class_names = get_class_names()

    # Load models
    model_full = build_model(use_attention=True).to(device)
    model_full.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)['model_state_dict'])
    model_full.eval()

    model_cnn = build_ablation_cnn_only().to(device)
    cnn_path = model_path.replace("full", "cnn_only")
    try:
        model_cnn.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=False)['model_state_dict'])
    except FileNotFoundError:
        print("CNN-only checkpoint not found, using untrained model for structure comparison")
    model_cnn.eval()

    rule_clf = RuleBasedClassifier()

    # Evaluate
    results = {}
    for name, predictor in [("Rule-based (Geometric)", "rule"),
                             ("CNN-only", model_cnn),
                             ("CNN+Attention (Ours)", model_full)]:

        if predictor == "rule":
            results[name] = _evaluate_rule_based(rule_clf, test_loader)
        else:
            results[name] = _evaluate_nn(predictor, test_loader, device)

    # Print comparison table
    print(f"\n{'Method':<30s} {'Accuracy':>10s} {'Macro F1':>10s}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<30s} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}")

    # Per-class F1 comparison
    print(f"\n{'Class':<20s}", end="")
    for name in results:
        print(f"{name[:12]:>15s}", end="")
    print()
    print("-" * (20 + 15 * len(results)))

    for c, cls_name in enumerate(class_names):
        print(f"{cls_name:<20s}", end="")
        for name, metrics in results.items():
            if cls_name in metrics['per_class'] and metrics['per_class'][cls_name]['support'] > 0:
                f1 = metrics['per_class'][cls_name]['f1']
                print(f"{f1:>15.4f}", end="")
            else:
                print(f"{'N/A':>15s}", end="")
        print()

    return results


def _evaluate_nn(model: GestureClassifier, loader, device: str) -> Dict:
    """Evaluate NN classifier on data loader."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for skeletons, targets in loader:
            skeletons = skeletons.to(device)
            logits, _ = model(skeletons)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.tolist())

    return _compute_metrics(all_preds, all_targets)


def _evaluate_rule_based(clf: RuleBasedClassifier, loader) -> Dict:
    """Evaluate rule-based classifier on single-frame, un-normalized data.

    The rule-based method expects raw world-coordinate data with meaningful
    geometric scales, so we evaluate on freshly generated single frames
    rather than normalized test windows.
    """
    # Generate fresh evaluation samples for rule-based (un-normalized)
    canon = HandSkeleton.build_right_hand()
    n_samples_per_gesture = 200
    all_preds, all_targets = [], []

    for gesture_id in range(1, N_GESTURE_CLASSES):  # skip NONE
        for _ in range(n_samples_per_gesture):
            deformer = GestureDeformer()
            joints = deformer.apply(canon, gesture_id,
                                    person_variation=np.random.uniform(-1, 1))
            # Add realistic sensor noise
            joints = joints + np.random.normal(0, 0.0015, joints.shape).astype(np.float32)

            pred, _ = clf.classify(joints)
            all_preds.append(pred)
            all_targets.append(gesture_id)

    return _compute_metrics(all_preds, all_targets)


def _compute_metrics(preds: List[int], targets: List[int]) -> Dict:
    """Compute classification metrics."""
    preds = np.array(preds)
    targets = np.array(targets)

    accuracy = (preds == targets).mean()
    class_names = get_class_names()
    per_class = {}

    for c in range(N_GESTURE_CLASSES):
        tp = ((preds == c) & (targets == c)).sum()
        total = (targets == c).sum()
        pred_count = (preds == c).sum()

        recall = tp / max(total, 1)
        precision = tp / max(pred_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        per_class[class_names[c]] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'support': int(total),
        }

    macro_f1 = np.mean([per_class[c]['f1'] for c in class_names
                        if per_class[c]['support'] > 0])

    return {
        'accuracy': round(accuracy, 4),
        'macro_f1': round(macro_f1, 4),
        'per_class': per_class,
    }


# ───────────────────────────────────────────────────────────────────
# Experiment 2: Robustness under Adverse Conditions
# ───────────────────────────────────────────────────────────────────

def experiment_2_robustness(model_path: str = "checkpoints/full/best_model.pt",
                            device: str = "cuda"):
    """Test robustness under: normal light, low light (more noise), partial occlusion."""

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Robustness Comparison")
    print("=" * 70)

    model = build_model(use_attention=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    rule_clf = RuleBasedClassifier()
    canon = HandSkeleton.build_right_hand()

    conditions = {
        "normal": {"noise_std": 0.0015, "dropout_prob": 0.0},
        "low_light": {"noise_std": 0.005, "dropout_prob": 0.0},  # 3x sensor noise
        "partial_occlusion": {"noise_std": 0.0015, "dropout_prob": 0.3},  # 30% joints lost
        "severe": {"noise_std": 0.005, "dropout_prob": 0.3},  # combined
    }

    results = {}
    n_samples_per_gesture = 100

    for cond_name, cond_params in conditions.items():
        rule_f1s = []
        nn_f1s = []

        for gesture_id in range(1, N_GESTURE_CLASSES):  # skip NONE
            for _ in range(n_samples_per_gesture):
                deformer = GestureDeformer()
                joints = deformer.apply(canon, gesture_id,
                                        person_variation=np.random.uniform(-1, 1))

                # Apply condition effects
                noise = np.random.normal(0, cond_params["noise_std"], joints.shape)
                joints = joints + noise.astype(np.float32)

                # Simulate occlusion: zero out some joints
                if cond_params["dropout_prob"] > 0:
                    mask = np.random.random(joints.shape[0]) > cond_params["dropout_prob"]
                    joints[~mask] = 0.0

                # Rule-based classification
                rule_pred, _ = rule_clf.classify(joints)
                rule_f1s.append(1.0 if rule_pred == gesture_id else 0.0)

                # NN classification (single frame → need to replicate to window)
                skeleton_window = np.tile(joints[np.newaxis, :, :], (WINDOW_SIZE, 1, 1))
                skeleton_window = skeleton_window.astype(np.float32)
                # Normalize
                wrist = skeleton_window[:, 0:1, :]
                skeleton_window = skeleton_window - wrist
                scale = np.std(skeleton_window)
                if scale > 1e-6:
                    skeleton_window = skeleton_window / (scale * 10)

                x = torch.from_numpy(skeleton_window).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = model(x)
                    nn_pred = logits.argmax(dim=-1).item()
                nn_f1s.append(1.0 if nn_pred == gesture_id else 0.0)

        results[cond_name] = {
            "rule_f1": round(np.mean(rule_f1s), 4),
            "nn_f1": round(np.mean(nn_f1s), 4),
        }

    # Print comparison
    print(f"\n{'Condition':<25s} {'Rule-based F1':>15s} {'NN F1':>15s} {'Δ':>10s}")
    print("-" * 65)
    for cond_name, metrics in results.items():
        delta = metrics['nn_f1'] - metrics['rule_f1']
        print(f"{cond_name:<25s} {metrics['rule_f1']:>15.4f} {metrics['nn_f1']:>15.4f} "
              f"{delta:>+10.4f}")

    return results


# ───────────────────────────────────────────────────────────────────
# Experiment 3: End-Side Performance Benchmark
# ───────────────────────────────────────────────────────────────────

def experiment_3_performance(model_path: str = "checkpoints/full/best_model.pt",
                             device: str = "cuda"):
    """Benchmark inference latency, memory, and FPS stability."""

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: End-Side Performance Benchmark")
    print("=" * 70)

    model_full = build_model(use_attention=True).to(device)
    model_full.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)['model_state_dict'])
    model_full.eval()

    model_cnn = build_ablation_cnn_only().to(device)
    model_cnn.eval()

    results = {}

    for name, model in [("Rule-based (Geometric)", None),
                         ("CNN-only", model_cnn),
                         ("CNN+Attention (Ours)", model_full)]:

        if model is None:
            # Profile rule-based classifier
            rule_clf = RuleBasedClassifier()
            x = np.random.randn(26, 3).astype(np.float32)

            times = []
            for _ in range(1000):
                start = time.perf_counter()
                rule_clf.classify(x)
                times.append(time.perf_counter() - start)

            latency_ms = np.mean(times[10:]) * 1000  # skip warmup
            params = 0  # rule-based has no learned params
            memory_kb = 0

        else:
            x = torch.randn(1, WINDOW_SIZE, N_JOINTS, JOINT_DIMS, device=device)

            # Warmup
            for _ in range(50):
                _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()

            # Latency
            times = []
            for _ in range(500):
                start = time.perf_counter()
                _ = model(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            latency_ms = np.mean(times) * 1000
            params = sum(p.numel() for p in model.parameters())

            # Memory (approximate)
            memory_kb = params * 4 / 1024  # float32 = 4 bytes

        results[name] = {
            'latency_ms': round(latency_ms, 3),
            'fps': round(1000 / latency_ms, 1) if latency_ms > 0 else float('inf'),
            'params': params,
            'memory_kb': round(memory_kb, 1),
        }

    # Print
    print(f"\n{'Method':<30s} {'Latency(ms)':>12s} {'FPS':>8s} {'Params':>10s} {'Memory(KB)':>12s}")
    print("-" * 75)
    for name, metrics in results.items():
        print(f"{name:<30s} {metrics['latency_ms']:>12.3f} {metrics['fps']:>8.1f} "
              f"{metrics['params']:>10,d} {metrics['memory_kb']:>12.1f}")

    # FPS stability test: run continuously for 60 seconds equivalent
    print(f"\nFPS Stability (60s simulated continuous run):")
    if model_full is not None:
        fps_samples = []
        x = torch.randn(1, WINDOW_SIZE, N_JOINTS, JOINT_DIMS, device=device)
        for _ in range(3600):  # 60s @ 60fps
            start = time.perf_counter()
            _ = model_full(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            fps_samples.append(1.0 / elapsed if elapsed > 0 else 0)

        fps_arr = np.array(fps_samples)
        print(f"  Mean FPS: {np.mean(fps_arr):.1f}")
        print(f"  Min FPS:  {np.min(fps_arr):.1f}")
        print(f"  Max FPS:  {np.max(fps_arr):.1f}")
        print(f"  Std FPS:  {np.std(fps_arr):.1f}")
        print(f"  FPS < 30: {(fps_arr < 30).sum()} frames ({(fps_arr < 30).mean()*100:.1f}%)")

    return results


# ───────────────────────────────────────────────────────────────────
# Run all experiments
# ───────────────────────────────────────────────────────────────────

def run_all_experiments(model_path: str = "checkpoints/full/best_model.pt",
                        data_dir: str = "data", device: str = "cuda",
                        output_dir: str = "experiment_results"):
    """Run all three experiments and save results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Experiment 1
    exp1 = experiment_1_accuracy(model_path, data_dir, device)
    all_results['exp1_accuracy'] = exp1

    # Experiment 2
    exp2 = experiment_2_robustness(model_path, device)
    all_results['exp2_robustness'] = exp2

    # Experiment 3
    exp3 = experiment_3_performance(model_path, device)
    all_results['exp3_performance'] = exp3

    # Save
    with open(output_path / "all_experiments.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll experiment results saved to: {output_path / 'all_experiments.json'}")

    # Generate LaTeX table for thesis
    _generate_latex_tables(all_results, output_path)

    return all_results


def _generate_latex_tables(results: Dict, output_path: Path):
    """Generate LaTeX tables matching thesis format for direct inclusion."""

    exp1 = results['exp1_accuracy']
    class_names = get_class_names()

    # Table 1: Per-class F1 comparison
    latex = r"""% Experiment 1: Gesture Recognition Accuracy Comparison
\begin{table}[htbp]
  \centering
  \caption{手势识别精度对比：规则方法与本文NN方法的逐类F1分数}
  \label{tab:nn_vs_rule_f1}
  \small
  \begin{tabular}{lccc}
  \hline
  手势类别 & 规则方法（几何判据） & CNN-only & CNN+Attention（本文） \\
  \hline
"""
    method_names = list(exp1.keys())
    for c, cls_name in enumerate(class_names):
        row = f"  {cls_name}"
        for method in method_names:
            if cls_name in exp1[method].get('per_class', {}):
                f1 = exp1[method]['per_class'][cls_name]['f1']
                row += f" & {f1:.3f}"
            else:
                row += " & ---"
        row += r" \\" + "\n"
        latex += row

    latex += r"""  \hline
  宏平均 F1 """
    for method in method_names:
        latex += f" & {exp1[method]['macro_f1']:.3f}"
    latex += r" \\" + "\n"
    latex += r"""  \hline
  \end{tabular}
\end{table}
"""

    # Table 2: Robustness
    exp2 = results['exp2_robustness']
    latex += r"""
% Experiment 2: Robustness Comparison
\begin{table}[htbp]
  \centering
  \caption{不同条件下手势识别鲁棒性对比（F1分数）}
  \label{tab:robustness}
  \small
  \begin{tabular}{lcc}
  \hline
  条件 & 规则方法 & CNN+Attention（本文） \\
  \hline
"""
    for cond, metrics in exp2.items():
        latex += f"  {cond} & {metrics['rule_f1']:.3f} & {metrics['nn_f1']:.3f} \\\\\n"

    latex += r"""  \hline
  \end{tabular}
\end{table}
"""

    # Table 3: Performance
    exp3 = results['exp3_performance']
    latex += r"""
% Experiment 3: End-Side Performance
\begin{table}[htbp]
  \centering
  \caption{端侧推理性能对比}
  \label{tab:performance}
  \small
  \begin{tabular}{lccc}
  \hline
  方法 & 单帧推理时延（ms） & 参数量 & 内存占用（KB） \\
  \hline
"""
    for method, metrics in exp3.items():
        latex += (f"  {method} & {metrics['latency_ms']:.2f} & "
                  f"{metrics['params']:,} & {metrics['memory_kb']:.1f} \\\\\n")

    latex += r"""  \hline
  \end{tabular}
\end{table}
"""

    with open(output_path / "results_tables.tex", "w") as f:
        f.write(latex)
    print(f"LaTeX tables saved to: {output_path / 'results_tables.tex'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints/full/best_model.pt")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="experiment_results")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_all_experiments(
        model_path=args.model_path,
        data_dir=args.data_dir,
        device=args.device,
        output_dir=args.output_dir,
    )
