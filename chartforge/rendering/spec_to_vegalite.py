"""
Chart Spec → Vega-Lite JSON converter.

Used for: web display, baseline comparison (LLM-Direct generates Vega-Lite),
and as intermediate format for AR rendering.
"""

from typing import Dict, Any
from ..pcg.sampler import ChartSpec


def to_vegalite(spec: ChartSpec) -> Dict[str, Any]:
    """Convert a ChartSpec to Vega-Lite JSON specification.

    This is the primary rendering target for paper experiments.
    """
    # Map chart type to Vega-Lite mark
    mark_map = {
        "bar": "bar", "line": "line", "scatter": "point",
        "area": "area", "pie": "arc", "heatmap": "rect",
        "radar": "line", "boxplot": "boxplot", "gauge": "arc",
        "treemap": "rect", "funnel": "bar",
    }
    mark = mark_map.get(spec.chart_type, "bar")

    vega = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": {"type": mark},
        "encoding": {},
    }

    # Title
    title_prefix = {
        "bar": "Bar Chart of ", "line": "Line Chart of ", "scatter": "Scatter Plot of ",
        "area": "Area Chart of ", "pie": "Pie Chart of ", "heatmap": "Heatmap of ",
        "radar": "Radar Chart of ", "sankey": "Sankey Diagram of ",
        "treemap": "Treemap of ", "boxplot": "Box Plot of ", "gauge": "Gauge of ",
        "funnel": "Funnel Chart of ",
    }.get(spec.chart_type, "Chart of ")

    y_fields = [f for f, ch in spec.data_bindings.items() if ch == "y"]
    vega["title"] = title_prefix + (y_fields[0] if y_fields else "Data")

    # Build encodings from data_bindings
    type_hints = {
        "province": "nominal", "region": "nominal",
        "year": "temporal",
        "DEL": "quantitative", "ES": "quantitative", "PGDP": "quantitative",
        "URBAN": "quantitative", "INDS": "quantitative",
    }

    for field, channel in spec.data_bindings.items():
        dtype = type_hints.get(field, "quantitative")

        if channel == "x":
            vega["encoding"]["x"] = {"field": field, "type": dtype, "title": field}
        elif channel == "y":
            vega["encoding"]["y"] = {"field": field, "type": "quantitative", "title": field}
        elif channel == "color":
            vega["encoding"]["color"] = {"field": field, "type": dtype, "title": field}
        elif channel == "size":
            vega["encoding"]["size"] = {"field": field, "type": "quantitative"}

    # Apply style
    style = spec.style_params or {}
    if "color_scheme" in style:
        vega["config"] = {
            "range": {"category": {"scheme": style["color_scheme"]}},
            "axis": {"labelFontSize": style.get("font_size", 12)},
            "title": {"fontSize": style.get("title_font_size", 16)},
        }

    return vega
