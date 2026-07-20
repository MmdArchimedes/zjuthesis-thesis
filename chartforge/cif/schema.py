"""
CIF Data Model: Chart Intent Formalization triple ⟨D, V, I⟩.
Matching paper Section 3.1, equations (2)-(5).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class FieldSpec:
    """D layer: Data semantics for a single field.

    Equation (2): D = {(f_i, t_i, s_i, a_i)}
    """
    field_name: str
    field_type: str          # nominal | ordinal | quantitative | temporal
    suggested_channel: str   # x | y | color | size | shape | facet | text
    aggregation: str = "none"  # sum | mean | count | none | min | max | median

    def __post_init__(self):
        valid_types = {"nominal", "ordinal", "quantitative", "temporal"}
        valid_channels = {"x", "y", "color", "size", "shape", "facet", "text", "opacity"}
        valid_aggs = {"sum", "mean", "count", "none", "min", "max", "median"}

        if self.field_type not in valid_types:
            raise ValueError(f"Invalid field type: {self.field_type}")
        if self.suggested_channel not in valid_channels:
            raise ValueError(f"Invalid channel: {self.suggested_channel}")
        if self.aggregation not in valid_aggs:
            raise ValueError(f"Invalid aggregation: {self.aggregation}")


@dataclass
class VisualSpec:
    """V layer: Visual encoding specification.

    Equation (3): V = ⟨G, E, Θ⟩
    """
    chart_type: str          # bar | line | scatter | area | heatmap | pie | ...
    encoding_map: Dict[str, str] = field(default_factory=dict)  # field → visual channel
    style_params: Dict[str, Any] = field(default_factory=dict)  # color scheme, font, etc.

    def __post_init__(self):
        from ..config import CHART_TYPES
        if self.chart_type not in CHART_TYPES:
            raise ValueError(f"Unknown chart type: {self.chart_type}. "
                           f"Must be one of {CHART_TYPES}")


@dataclass
class InteractionSpec:
    """I layer: Interaction constraint.

    Equation (4): I = {(e_j, h_j, c_j)}
    """
    event_type: str          # click | hover | brush | zoom | filter | drilldown
    handler_spec: str        # handler specification
    constraint: str = ""     # e.g. "response_time < 200ms"

    def __post_init__(self):
        valid_events = {"click", "hover", "brush", "zoom", "filter", "drilldown", "select"}
        if self.event_type not in valid_events:
            raise ValueError(f"Invalid event type: {self.event_type}")


@dataclass
class CIFTriple:
    """Complete CIF representation: ⟨D, V, I⟩.

    Equation (1): CIF(Q) = ⟨D, V, I⟩
    """
    data_semantics: List[FieldSpec]          # D
    visual_encoding: VisualSpec               # V
    interaction_constraints: List[InteractionSpec]  # I
    raw_query: str = ""                       # original NL query
    confidence: float = 1.0                   # parsing confidence ∈ [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize CIF to JSON-compatible dict."""
        return {
            "data_semantics": [
                {
                    "field_name": f.field_name,
                    "field_type": f.field_type,
                    "suggested_channel": f.suggested_channel,
                    "aggregation": f.aggregation,
                }
                for f in self.data_semantics
            ],
            "visual_encoding": {
                "chart_type": self.visual_encoding.chart_type,
                "encoding_map": self.visual_encoding.encoding_map,
                "style_params": self.visual_encoding.style_params,
            },
            "interaction_constraints": [
                {
                    "event_type": i.event_type,
                    "handler_spec": i.handler_spec,
                    "constraint": i.constraint,
                }
                for i in self.interaction_constraints
            ],
            "raw_query": self.raw_query,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CIFTriple":
        """Deserialize CIF from dict."""
        return cls(
            data_semantics=[
                FieldSpec(**f) for f in d["data_semantics"]
            ],
            visual_encoding=VisualSpec(**d["visual_encoding"]),
            interaction_constraints=[
                InteractionSpec(**i) for i in d.get("interaction_constraints", [])
            ],
            raw_query=d.get("raw_query", ""),
            confidence=d.get("confidence", 1.0),
        )

    def __repr__(self) -> str:
        return (f"CIFTriple(chart={self.visual_encoding.chart_type}, "
                f"fields={len(self.data_semantics)}, "
                f"interactions={len(self.interaction_constraints)}, "
                f"conf={self.confidence:.2f})")
