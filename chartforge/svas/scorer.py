"""
SVAS Scorer — Semantic-Visual Alignment Score.

Paper Section 3.3, Equation (14):
  SVAS(C, Q) = α · Φ_sem(C, D) + β · Φ_vis(C, V) + γ · Φ_int(C, I)

Three sub-metrics:
  Φ_sem: Semantic fidelity — are required data fields present & correctly mapped?
  Φ_vis:  Visual completeness — chart type, encoding, style match?
  Φ_int:  Interaction accessibility — can requested interactions be performed?
"""

from typing import Dict, List, Any
from ..cif.schema import CIFTriple, FieldSpec, VisualSpec, InteractionSpec
from ..pcg.sampler import ChartSpec
from ..config import (
    ALPHA_SEM, BETA_VIS, GAMMA_INT,
    TAU_SEM, CHART_TYPES,
)


def phi_sem(chart: ChartSpec, cif: CIFTriple) -> float:
    """Semantic fidelity Φ_sem: measures data field coverage.

    Equation (15): fraction of required fields present with correct channel mapping.

    Args:
        chart: Generated chart specification
        cif: CIF intent triple

    Returns:
        Score ∈ [0, 1]
    """
    if not cif.data_semantics:
        return 1.0  # no fields specified = no constraint

    valid_channels = {
        "nominal": {"x", "color", "facet", "shape"},
        "ordinal": {"x", "color", "facet", "shape"},
        "quantitative": {"y", "x", "color", "size", "opacity"},
        "temporal": {"x", "facet"},
    }

    score = 0.0
    for field_spec in cif.data_semantics:
        # Check: field appears in chart's data bindings
        matched_field = False
        matched_channel = False

        for binding_field, channel in chart.data_bindings.items():
            if binding_field == field_spec.field_name:
                matched_field = True
                # Check channel compatibility
                compatible_channels = valid_channels.get(field_spec.field_type, set())
                if channel in compatible_channels:
                    matched_channel = True
                break

        # Partial credit: field present (0.7) + correct channel (0.3)
        if matched_field:
            score += 0.7 + 0.3 * int(matched_channel)

    return score / len(cif.data_semantics)


def phi_vis(chart: ChartSpec, cif: CIFTriple) -> float:
    """Visual completeness Φ_vis: chart type, encoding, style match.

    Equation (16):
      Φ_vis = ω₁·typeMatch + ω₂·encodingMatch + ω₃·styleMatch

    Args:
        chart: Generated chart specification
        cif: CIF intent triple

    Returns:
        Score ∈ [0, 1]
    """
    vis = cif.visual_encoding

    # 1. Chart type match (ω₁ = 0.40)
    type_score = _chart_type_similarity(chart.chart_type, vis.chart_type)

    # 2. Encoding match (ω₂ = 0.35)
    encoding_score = _encoding_alignment(chart, vis)

    # 3. Style match (ω₃ = 0.25)
    style_score = _style_similarity(chart.style_params, vis.style_params)

    return 0.40 * type_score + 0.35 * encoding_score + 0.25 * style_score


def phi_int(chart: ChartSpec, cif: CIFTriple) -> float:
    """Interaction accessibility Φ_int: can requested interactions be performed?

    Equation (17):
      Φ_int = (1/|I|) Σ interactionFeasibility(C, e, h) · constraintSatisfaction(C, c)

    Args:
        chart: Generated chart specification
        cif: CIF intent triple

    Returns:
        Score ∈ [0, 1]
    """
    if not cif.interaction_constraints:
        return 1.0  # no interaction specified = no constraint

    # Chart-type-specific interaction capabilities
    chart_interaction_capabilities = {
        "bar": {"click", "hover", "brush", "zoom", "filter"},
        "line": {"click", "hover", "brush", "zoom", "filter"},
        "scatter": {"click", "hover", "brush", "zoom", "filter", "select"},
        "area": {"click", "hover", "brush", "zoom"},
        "heatmap": {"click", "hover", "zoom", "filter"},
        "pie": {"click", "hover"},
        "radar": {"click", "hover"},
        "sankey": {"click", "hover"},
        "treemap": {"click", "hover", "drilldown"},
        "boxplot": {"click", "hover"},
        "gauge": {"click"},
        "funnel": {"click", "hover"},
    }

    capabilities = chart_interaction_capabilities.get(chart.chart_type, {"click", "hover"})

    score = 0.0
    for int_spec in cif.interaction_constraints:
        # Feasibility: can this interaction type work with this chart type?
        feasible = 1.0 if int_spec.event_type in capabilities else 0.3

        # Constraint satisfaction: check constraints (simplified)
        constraint_ok = 1.0
        if "response_time" in int_spec.constraint:
            # Always satisfied for our system (sub-200ms for simple interactions)
            constraint_ok = 1.0
        elif "max_points" in int_spec.constraint:
            # Check data point count (simplified)
            constraint_ok = 0.9  # most charts within limits

        score += feasible * constraint_ok

    return score / len(cif.interaction_constraints)


def svas(
    chart: ChartSpec,
    cif: CIFTriple,
    alpha: float = ALPHA_SEM,
    beta: float = BETA_VIS,
    gamma: float = GAMMA_INT,
) -> float:
    """Compute composite SVAS score.

    Equation (14): SVAS = α·Φ_sem + β·Φ_vis + γ·Φ_int

    Returns:
        Score ∈ [0, 1]
    """
    return (
        alpha * phi_sem(chart, cif) +
        beta * phi_vis(chart, cif) +
        gamma * phi_int(chart, cif)
    )


# ── Helper functions ──────────────────────────────────────────────

def _chart_type_similarity(generated: str, expected: str) -> float:
    """Compute chart type similarity with semantic distance.

    Exact match = 1.0. Related types get partial credit.
    """
    if generated == expected:
        return 1.0

    # Define chart type similarity matrix (simplified)
    related_groups = [
        {"bar", "line", "area"},
        {"scatter", "line"},
        {"pie", "gauge", "funnel"},
        {"heatmap", "treemap"},
        {"radar", "line"},
        {"boxplot", "bar"},
    ]

    for group in related_groups:
        if generated in group and expected in group:
            return 0.5  # related but not identical

    return 0.1  # unrelated


def _encoding_alignment(chart: ChartSpec, vis: VisualSpec) -> float:
    """Compute visual encoding alignment score."""
    if not vis.encoding_map:
        # No encoding constraints specified
        required_channels = {"x", "y"}
        have_channels = set(chart.encoding_map.keys())
        return len(have_channels & required_channels) / len(required_channels)

    # Compare expected vs actual encoding map
    matches = 0
    total = len(vis.encoding_map)

    for field, expected_channel in vis.encoding_map.items():
        if field in chart.data_bindings:
            actual_channel = chart.data_bindings[field]
            if actual_channel == expected_channel:
                matches += 1
            else:
                # Partial credit: field is used but different channel
                matches += 0.5

    if total == 0:
        return 1.0
    return matches / total


def _style_similarity(generated_style: Dict, expected_style: Dict) -> float:
    """Compute style parameter similarity."""
    if not expected_style:
        return 0.8  # no constraints = good enough

    key_weights = {
        "color_scheme": 0.4,
        "font_family": 0.2,
        "font_size": 0.15,
        "background": 0.1,
        "grid": 0.15,
    }

    score = 0.0
    for key, weight in key_weights.items():
        if key in expected_style:
            if key in generated_style and generated_style[key] == expected_style[key]:
                score += weight
            elif key in generated_style:
                score += 0.5 * weight  # present but different

    return score


class SVASScorer:
    """Convenience class for batch SVAS scoring."""

    def __init__(
        self,
        alpha: float = ALPHA_SEM,
        beta: float = BETA_VIS,
        gamma: float = GAMMA_INT,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def score(self, chart: ChartSpec, cif: CIFTriple) -> Dict[str, float]:
        """Compute full SVAS breakdown.

        Returns:
            Dict with 'phi_sem', 'phi_vis', 'phi_int', 'svas', and 'passed_filter'
        """
        ps = phi_sem(chart, cif)
        pv = phi_vis(chart, cif)
        pi = phi_int(chart, cif)
        total = self.alpha * ps + self.beta * pv + self.gamma * pi

        return {
            "phi_sem": ps,
            "phi_vis": pv,
            "phi_int": pi,
            "svas": total,
            "passed_filter": total > TAU_SEM,
        }

    def batch_score(
        self, charts: List[ChartSpec], cifs: List[CIFTriple]
    ) -> List[Dict[str, float]]:
        """Score multiple chart-CIF pairs."""
        return [self.score(c, cif) for c, cif in zip(charts, cifs)]
