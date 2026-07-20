"""
Chart Spec → ECharts option converter.

Used for baseline comparison and web-based chart rendering.
"""

from typing import Dict, Any
from ..pcg.sampler import ChartSpec


def to_echarts(spec: ChartSpec) -> Dict[str, Any]:
    """Convert a ChartSpec to ECharts option object.

    ECharts is the most widely-used charting library in China,
    making it suitable for thesis demos and baseline comparisons.
    """
    # Map chart type to ECharts series type
    series_type_map = {
        "bar": "bar", "line": "line", "scatter": "scatter",
        "area": "line", "pie": "pie", "heatmap": "heatmap",
        "radar": "radar", "sankey": "sankey", "treemap": "treemap",
        "boxplot": "boxplot", "gauge": "gauge", "funnel": "funnel",
    }
    series_type = series_type_map.get(spec.chart_type, "bar")

    # Build ECharts option
    option = {
        "title": {"text": "", "left": "center"},
        "tooltip": {"trigger": "item" if spec.chart_type in ("pie", "treemap") else "axis"},
        "series": [{"type": series_type, "data": []}],
    }

    # Area chart: set areaStyle
    if spec.chart_type == "area":
        option["series"][0]["areaStyle"] = {}
        option["series"][0]["smooth"] = True

    # Color scheme
    style = spec.style_params or {}
    if "color_scheme" in style:
        option["color"] = _get_color_palette(style["color_scheme"])

    # Build axes for Cartesian charts
    cartesian_types = {"bar", "line", "scatter", "area", "boxplot"}
    if spec.chart_type in cartesian_types:
        x_fields = [f for f, ch in spec.data_bindings.items() if ch == "x"]
        y_fields = [f for f, ch in spec.data_bindings.items() if ch == "y"]

        option["xAxis"] = {
            "type": "category",
            "name": x_fields[0] if x_fields else "",
        }
        option["yAxis"] = {
            "type": "value",
            "name": y_fields[0] if y_fields else "",
        }

    # Enable interactions from spec
    interaction_types = {i["type"] for i in spec.interactions}
    if "zoom" in interaction_types:
        option["dataZoom"] = [{"type": "slider"}, {"type": "inside"}]
    if "brush" in interaction_types:
        option["brush"] = {"toolbox": ["rect", "polygon", "clear"]}

    return option


def _get_color_palette(scheme: str) -> list:
    """Get color palette for a named scheme."""
    palettes = {
        "tableau10": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
                       "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"],
        "viridis": ["#440154", "#482878", "#3E4A89", "#31688E", "#26828E",
                    "#1F9E89", "#35B779", "#6DCD59", "#B4DE2C", "#FDE725"],
        "blues": ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1", "#6BAED6",
                  "#4292C6", "#2171B5", "#08519C", "#08306B"],
        "reds": ["#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A",
                 "#EF3B2D", "#CB181D", "#A50F15", "#67000D"],
    }
    return palettes.get(scheme, palettes["tableau10"])
