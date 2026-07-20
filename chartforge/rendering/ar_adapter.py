"""
Chart Spec → Unity AR chart format adapter.

Converts ChartSpec to JSON format consumable by Unity ChartRenderer.
This bridges the Python AI pipeline with the AR visualization frontend.
"""

from typing import Dict, Any, List
from ..pcg.sampler import ChartSpec


def to_ar_format(spec: ChartSpec, position: List[float] = None) -> Dict[str, Any]:
    """Convert ChartSpec to Unity AR chart JSON format.

    Args:
        spec: Generated chart specification
        position: [x, y, z] world-space position in AR (meters).
                  Default places chart 1.5m in front of user at eye level.

    Returns:
        JSON object for Unity ChartRenderer consumption
    """
    if position is None:
        position = [0.0, 0.0, 1.5]  # default: 1.5m forward

    ar_spec = {
        "chartType": spec.chart_type,
        "chartId": f"cf_{spec.chart_type}_{hash(str(spec.data_bindings)) % 100000}",
        "layout": {
            "position": {"x": position[0], "y": position[1], "z": position[2]},
            "size": {"width": 0.8, "height": 0.6},  # meters in AR space
            "rotation": {"x": 0, "y": 0, "z": 0},
        },
        "data": {
            "bindings": spec.data_bindings,
            "source": "data_engine",  # points to Unity DataManager
        },
        "encodings": {
            "x": _get_encoding(spec, "x"),
            "y": _get_encoding(spec, "y"),
            "color": _get_encoding(spec, "color"),
            "size": _get_encoding(spec, "size"),
        },
        "style": {
            "colorScheme": spec.style_params.get("color_scheme", "tableau10"),
            "fontSize": spec.style_params.get("font_size", 12),
            "titleFontSize": spec.style_params.get("title_font_size", 16),
            "backgroundColor": spec.style_params.get("background", "#FFFFFF"),
            "showGrid": spec.style_params.get("grid", True),
            "gridAlpha": spec.style_params.get("grid_alpha", 0.3),
        },
        "interactions": [
            _map_interaction(interaction)
            for interaction in spec.interactions
        ],
        "annotations": [
            {"text": a, "position": "top"}
            for a in spec.annotations
        ],
    }

    return ar_spec


def _get_encoding(spec: ChartSpec, channel: str) -> Dict:
    """Get encoding info for a specific visual channel."""
    for field, ch in spec.data_bindings.items():
        if ch == channel:
            return {
                "field": field,
                "enabled": True,
            }
    return {"field": None, "enabled": False}


def _map_interaction(interaction: Dict) -> Dict:
    """Map a ChartSpec interaction to Unity interaction format."""
    interaction_map = {
        "hover": "OnHover",
        "click": "OnClick",
        "brush": "OnBrush",
        "zoom": "OnZoom",
        "filter": "OnFilter",
        "drilldown": "OnDrilldown",
    }

    unity_event = interaction_map.get(interaction["type"], "OnClick")
    return {
        "event": unity_event,
        "handler": interaction.get("handler", "default"),
        "enabled": interaction.get("enabled", True),
    }


def batch_to_ar_format(specs: List[ChartSpec]) -> List[Dict[str, Any]]:
    """Convert multiple ChartSpecs to AR format with auto-layout.

    Places charts in a horizontal row in AR space.
    """
    results = []
    spacing = 1.0  # meters between charts
    start_x = -(len(specs) - 1) * spacing / 2

    for i, spec in enumerate(specs):
        position = [start_x + i * spacing, 0.0, 1.5]
        results.append(to_ar_format(spec, position=position))

    return results
