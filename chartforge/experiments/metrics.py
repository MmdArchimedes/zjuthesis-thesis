"""
Evaluation metrics for ChartForge experiments.

Paper metrics:
  CA  (Chart Accuracy): chart type + data mapping + visual encoding all correct
  SVAS: semantic-visual alignment score (see svas/)
  US  (User Satisfaction): 1-5 Likert scale
  Latency: end-to-end generation time (seconds)
"""

import time
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

from ..pcg.sampler import ChartSpec
from ..cif.schema import CIFTriple
from ..svas.scorer import SVASScorer, svas, phi_sem, phi_vis, phi_int


@dataclass
class ChartAccuracy:
    """Chart accuracy breakdown."""
    type_correct: bool = False       # chart type matches
    data_correct: bool = False       # data fields correctly mapped
    encoding_correct: bool = False   # visual encodings correct
    overall_correct: bool = False    # all three correct

    @property
    def accuracy(self) -> float:
        return float(self.overall_correct)


def compute_chart_accuracy(
    generated: ChartSpec,
    ground_truth: ChartSpec,
) -> ChartAccuracy:
    """Compute chart accuracy between generated and ground truth specs.

    Chart Accuracy = type_match AND data_match AND encoding_match.
    """
    # Type match
    type_correct = generated.chart_type == ground_truth.chart_type

    # Data match: all required fields present
    gt_fields = set(ground_truth.data_bindings.keys())
    gen_fields = set(generated.data_bindings.keys())
    data_correct = gt_fields.issubset(gen_fields)

    # Encoding match: at least x and y are correctly assigned
    encoding_correct = True
    for channel in ["x", "y"]:
        gt_field = None
        gen_field = None
        for f, ch in ground_truth.data_bindings.items():
            if ch == channel:
                gt_field = f
        for f, ch in generated.data_bindings.items():
            if ch == channel:
                gen_field = f
        if gt_field and gen_field and gt_field != gen_field:
            encoding_correct = False
            break

    overall = type_correct and data_correct and encoding_correct

    return ChartAccuracy(
        type_correct=type_correct,
        data_correct=data_correct,
        encoding_correct=encoding_correct,
        overall_correct=overall,
    )


def compute_metrics(
    generated_specs: List[ChartSpec],
    ground_truth_specs: List[ChartSpec],
    cifs: List[CIFTriple] = None,
) -> Dict[str, Any]:
    """Compute all evaluation metrics for a batch of generated charts.

    Returns:
        Dict with CA, SVAS, per-chart-type breakdown, and latency stats.
    """
    n = len(generated_specs)

    # Chart Accuracy
    accuracies = [
        compute_chart_accuracy(gen, gt)
        for gen, gt in zip(generated_specs, ground_truth_specs)
    ]
    ca = sum(1 for a in accuracies if a.overall_correct) / max(n, 1)
    type_acc = sum(1 for a in accuracies if a.type_correct) / max(n, 1)
    data_acc = sum(1 for a in accuracies if a.data_correct) / max(n, 1)
    encoding_acc = sum(1 for a in accuracies if a.encoding_correct) / max(n, 1)

    # SVAS
    if cifs:
        scorer = SVASScorer()
        svas_scores = [scorer.score(gen, cif) for gen, cif in zip(generated_specs, cifs)]
        avg_svas = np.mean([s["svas"] for s in svas_scores])
    else:
        svas_scores = []
        avg_svas = 0.0

    # Per-chart-type breakdown
    per_type = {}
    for gen, gt in zip(generated_specs, ground_truth_specs):
        ct = gt.chart_type
        if ct not in per_type:
            per_type[ct] = {"total": 0, "correct": 0}
        per_type[ct]["total"] += 1
        acc = compute_chart_accuracy(gen, gt)
        if acc.overall_correct:
            per_type[ct]["correct"] += 1

    for ct in per_type:
        per_type[ct]["accuracy"] = per_type[ct]["correct"] / per_type[ct]["total"]

    return {
        "chart_accuracy": ca,
        "type_accuracy": type_acc,
        "data_accuracy": data_acc,
        "encoding_accuracy": encoding_acc,
        "avg_svas": avg_svas,
        "per_chart_type": per_type,
        "total_samples": n,
        "individual_accuracies": [a.overall_correct for a in accuracies],
        "svas_scores": svas_scores,
    }


def compute_svas_batch(
    generated_specs: List[ChartSpec],
    cifs: List[CIFTriple],
) -> List[Dict[str, float]]:
    """Compute full SVAS breakdown for batch evaluation."""
    scorer = SVASScorer()
    return scorer.batch_score(generated_specs, cifs)


class ExperimentTimer:
    """Measure generation latency with statistics."""

    def __init__(self):
        self.times: List[float] = []

    def measure(self, fn, *args, **kwargs) -> Tuple[Any, float]:
        """Measure execution time of fn."""
        t_start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t_start
        self.times.append(elapsed)
        return result, elapsed

    @property
    def stats(self) -> Dict[str, float]:
        """Compute latency statistics."""
        if not self.times:
            return {"mean": 0, "median": 0, "p95": 0, "min": 0, "max": 0}
        arr = np.array(self.times)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }
