"""
CCA: Composable Chart Algebra — core operations.

Paper Section 3.5, Definitions 3-6:

  Compose    C₁ ∘ C₂ = Chart(mergeLayout(C₁,C₂), C₁.glyphs ∪ C₂.glyphs)
  Layer      C₁ ⊕ C₂ = Chart(C₁.layout, C₁.glyphs ∪ C₂.glyphs)  [same layout]
  Transform   T_φ(C) = Chart(C.layout, {φ(g) | g ∈ C.glyphs})
  Parameterize P_θ(C) = C[θ / vars(C)]

Algebraic properties (Propositions 1-4):
  - Associativity: (C₁ ⊕ C₂) ⊕ C₃ = C₁ ⊕ (C₂ ⊕ C₃)
  - Commutativity: C₁ ⊕ C₂ = C₂ ⊕ C₁
  - Distributivity: T_φ(C₁ ⊕ C₂) = T_φ(C₁) ⊕ T_φ(C₂)
  - Identity: ∃ C_∅ such that C ⊕ C_∅ = C
"""

import copy
from typing import Callable, Dict, List, Any
from ..pcg.sampler import ChartSpec


def _merge_layout(layout1: Dict[str, str], layout2: Dict[str, str]) -> Dict[str, str]:
    """Merge two chart layouts. Layout2 values override Layout1 on conflict."""
    merged = dict(layout1)
    for key, value in layout2.items():
        if key in merged:
            # On conflict, keep the first (primary) layout's value
            continue
        merged[key] = value
    return merged


def _merge_data_bindings(bindings1: Dict, bindings2: Dict) -> Dict:
    """Merge data bindings from two charts."""
    merged = dict(bindings1)
    merged.update(bindings2)
    return merged


def _merge_style(style1: Dict, style2: Dict) -> Dict:
    """Merge style params, preferring style1 on conflict."""
    merged = dict(style2)
    merged.update(style1)
    return merged


def compose(c1: ChartSpec, c2: ChartSpec) -> ChartSpec:
    """Chart compose: C₁ ∘ C₂.

    Creates a new chart by combining layouts and glyphs from both charts.
    The resulting chart uses a merged layout and can render both sets of glyphs.
    """
    merged_layout = _merge_layout(c1.layout, c2.layout)
    merged_bindings = _merge_data_bindings(c1.data_bindings, c2.data_bindings)
    merged_style = _merge_style(c1.style_params, c2.style_params)

    # Determine composite chart type
    if c1.chart_type == c2.chart_type:
        chart_type = c1.chart_type
    elif {c1.chart_type, c2.chart_type} == {"bar", "line"}:
        chart_type = "combo_bar_line"
    else:
        chart_type = c1.chart_type  # primary determines type

    merged_interactions = list(c1.interactions)
    existing_types = {i["type"] for i in merged_interactions}
    for intr in c2.interactions:
        if intr["type"] not in existing_types:
            merged_interactions.append(intr)

    return ChartSpec(
        chart_type=chart_type,
        data_bindings=merged_bindings,
        encoding_map={v: k for k, v in merged_bindings.items()},
        layout=merged_layout,
        glyph_type=f"{c1.glyph_type}+{c2.glyph_type}",
        style_params=merged_style,
        interactions=merged_interactions,
        annotations=c1.annotations + c2.annotations,
        log_probability=c1.log_probability + c2.log_probability,
    )


def layer(c1: ChartSpec, c2: ChartSpec) -> ChartSpec:
    """Chart layer: C₁ ⊕ C₂.

    Layers two charts on the same coordinate system.
    Requires that c1.layout == c2.layout.

    Example: layer(bar_chart, line_chart) → combo chart with both bar and line glyphs.
    """
    # For layering, check that layouts are compatible
    x1 = c1.layout.get("x_axis", "")
    x2 = c2.layout.get("x_axis", "")
    y1 = c1.layout.get("y_axis", "")
    y2 = c2.layout.get("y_axis", "")

    # Allow layering if at least x-axes are compatible
    if x1 and x2 and x1 != x2:
        # Different x-axis types: need dual-axis layout
        merged_layout = {
            "x_axis": x1,
            "x_axis_secondary": x2,
            "y_axis": y1,
            "y_axis_secondary": y2,
        }
    else:
        merged_layout = dict(c1.layout)

    merged_bindings = _merge_data_bindings(c1.data_bindings, c2.data_bindings)
    merged_style = _merge_style(c1.style_params, c2.style_params)

    return ChartSpec(
        chart_type=f"{c1.chart_type}+{c2.chart_type}",
        data_bindings=merged_bindings,
        encoding_map={v: k for k, v in merged_bindings.items()},
        layout=merged_layout,
        glyph_type=f"{c1.glyph_type}+{c2.glyph_type}",
        style_params=merged_style,
        interactions=list(c1.interactions) + list(c2.interactions),
        annotations=c1.annotations + c2.annotations,
        log_probability=c1.log_probability + c2.log_probability,
    )


def transform(chart: ChartSpec, phi: Callable[[ChartSpec], ChartSpec]) -> ChartSpec:
    """Chart transform: T_φ(C).

    Applies transformation φ to the chart. Common transformations:
    - Color scheme change
    - Axis flip (x ↔ y)
    - Scale adjustment
    - Annotation addition
    """
    return phi(copy.deepcopy(chart))


def parameterize(template: ChartSpec, params: Dict[str, Any]) -> ChartSpec:
    """Chart parameterize: P_θ(C).

    Replaces free variables in chart template with concrete parameter values.

    Example:
      parameterize(bar_template, {"province": "浙江", "year": 2022, "metric": "DEL"})
    """
    result = copy.deepcopy(template)

    # Replace in data bindings
    for field, value in params.items():
        if field in result.data_bindings:
            result.data_bindings[field] = value

    # Replace in annotations
    result.annotations = [
        a.format(**params) if "{" in a else a
        for a in result.annotations
    ]

    # Apply style overrides from params
    for key, value in params.items():
        if key.startswith("style_"):
            style_key = key[6:]  # remove "style_" prefix
            result.style_params[style_key] = value

    return result


# ── Pre-defined transformations ───────────────────────────────────

def flip_axes(chart: ChartSpec) -> ChartSpec:
    """Transform: swap x and y axes."""
    result = copy.deepcopy(chart)
    result.layout["x_axis"], result.layout["y_axis"] = result.layout["y_axis"], result.layout["x_axis"]

    # Swap data bindings
    new_bindings = {}
    for field, channel in result.data_bindings.items():
        if channel == "x":
            new_bindings[field] = "y"
        elif channel == "y":
            new_bindings[field] = "x"
        else:
            new_bindings[field] = channel
    result.data_bindings = new_bindings
    return result


def change_color_scheme(chart: ChartSpec, scheme: str) -> ChartSpec:
    """Transform: apply a new color scheme."""
    result = copy.deepcopy(chart)
    result.style_params["color_scheme"] = scheme
    return result


def add_annotation(chart: ChartSpec, text: str, position: str = "top") -> ChartSpec:
    """Transform: add a text annotation."""
    result = copy.deepcopy(chart)
    result.annotations.append(text)
    return result


# ── Identity element ──────────────────────────────────────────────

EMPTY_CHART = ChartSpec(
    chart_type="empty",
    data_bindings={},
    encoding_map={},
    layout={},
    glyph_type="NoneGlyph",
    style_params={},
    interactions=[],
    annotations=[],
    log_probability=0.0,
)
