"""
PCG Beam Search Sampler — generates chart spec candidates from CIF intent.

Paper Algorithm 1: PCG-BeamSearch
  Complexity: O(k * |R| * d), k=5, |R|≈200, d≤10 → ~10K evaluations per chart

Paper Equation (13): T* = argmax P(T | PCG) · P(CIF(Q) | T)
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field

from .grammar import ChartGrammar, SyntaxNode, SyntaxTree, Symbol, NodeType
from ..cif.schema import CIFTriple
from ..config import BEAM_SIZE, MAX_DEPTH, NON_TERMINALS


@dataclass
class BeamItem:
    """A single item in the beam search."""
    tree: SyntaxNode
    score: float          # log probability
    depth: int = 0

    def __lt__(self, other):
        return self.score > other.score  # higher score = better


@dataclass
class ChartSpec:
    """Output chart specification from PCG sampling."""
    chart_type: str
    data_bindings: Dict[str, str]      # field → visual channel
    encoding_map: Dict[str, str]       # channel → field (inverted)
    layout: Dict[str, str]            # axis specifications
    glyph_type: str                    # BarGlyph | LinePath | ...
    style_params: Dict[str, object] = field(default_factory=dict)
    interactions: List[Dict] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    log_probability: float = 0.0

    def to_dict(self) -> dict:
        return {
            "chart_type": self.chart_type,
            "data_bindings": self.data_bindings,
            "encoding_map": self.encoding_map,
            "layout": self.layout,
            "glyph_type": self.glyph_type,
            "style_params": self.style_params,
            "interactions": self.interactions,
            "annotations": self.annotations,
            "log_probability": self.log_probability,
        }


class PCGBeamSampler:
    """Beam search sampler over PCG grammar."""

    def __init__(self, grammar: ChartGrammar, beam_size: int = BEAM_SIZE):
        self.grammar = grammar
        self.beam_size = beam_size
        self.max_depth = MAX_DEPTH

    def sample(
        self,
        cif: CIFTriple,
        beam_size: int = None,
    ) -> List[ChartSpec]:
        """Generate top-k chart spec candidates from CIF.

        Args:
            cif: Parsed CIF triple ⟨D, V, I⟩
            beam_size: Beam size (default from config)

        Returns:
            List of ChartSpec candidates, sorted by score descending
        """
        k = beam_size or self.beam_size

        # Initialize beam with start symbol
        start_node = SyntaxNode(symbol=Symbol(self.grammar.start_symbol, NodeType.NON_TERMINAL))
        beams: List[BeamItem] = [BeamItem(tree=start_node, score=0.0, depth=0)]

        # Iteratively expand non-terminals
        while any(self._has_non_terminal(b.tree) for b in beams):
            candidates: List[BeamItem] = []

            for beam in beams:
                if beam.depth >= self.max_depth:
                    candidates.append(beam)
                    continue

                nt = self._leftmost_non_terminal(beam.tree)
                if nt is None:
                    candidates.append(beam)
                    continue

                # Expand with each applicable production rule
                rules = self.grammar.get_rules_for(nt)
                for rule in rules:
                    new_tree = self._expand(beam.tree, nt, rule)
                    new_score = beam.score + math.log(max(rule.probability, 1e-10))
                    new_score += self._cif_match_score(new_tree, cif)

                    candidates.append(BeamItem(
                        tree=new_tree,
                        score=new_score,
                        depth=beam.depth + 1,
                    ))

            # Keep top-k
            candidates.sort(key=lambda x: -x.score)
            beams = candidates[:k]

            # Safety: if no progress, break
            if len(beams) == 0:
                break

        # Finalize: convert trees to ChartSpec objects
        results = []
        for beam in beams:
            try:
                spec = self._tree_to_spec(beam.tree, beam.score)
                results.append(spec)
            except Exception:
                continue

        # Sort by score descending
        results.sort(key=lambda s: -s.log_probability)
        return results[:k]

    def _has_non_terminal(self, node: SyntaxNode) -> bool:
        """Check if tree has any non-terminal to expand."""
        if node.symbol.node_type == NodeType.NON_TERMINAL and not node.children:
            return True
        for child in node.children:
            if self._has_non_terminal(child):
                return True
        return False

    def _leftmost_non_terminal(self, node: SyntaxNode) -> Optional[str]:
        """Find the leftmost unexpanded non-terminal."""
        if node.symbol.node_type == NodeType.NON_TERMINAL and not node.children:
            return node.symbol.name
        for child in node.children:
            result = self._leftmost_non_terminal(child)
            if result is not None:
                return result
        return None

    def _expand(
        self,
        node: SyntaxNode,
        target_nt: str,
        rule,
    ) -> SyntaxNode:
        """Expand the leftmost occurrence of target_nt using rule."""
        import copy
        new_node = copy.deepcopy(node)

        def _expand_recursive(n: SyntaxNode) -> bool:
            if (n.symbol.node_type == NodeType.NON_TERMINAL and
                n.symbol.name == target_nt and not n.children):
                # Expand this node
                for sym in rule.rhs:
                    child_nt = NodeType.NON_TERMINAL if sym.name in self.grammar.non_terminals else NodeType.TERMINAL
                    n.children.append(SyntaxNode(symbol=Symbol(sym.name, child_nt)))
                n.rule = rule
                return True
            for child in n.children:
                if _expand_recursive(child):
                    return True
            return False

        _expand_recursive(new_node)
        return new_node

    def _cif_match_score(self, tree: SyntaxNode, cif: CIFTriple) -> float:
        """Compute log P(CIF | tree) — how well the partial tree matches intent.

        This is the "intent matching" term in Equation (13).
        """
        score = 0.0

        # Extract terminals so far
        terminals = self._extract_terminals(tree)

        # Match glyph type to CIF chart type
        glyph_to_chart = {
            "BarGlyph": "bar", "LinePath": "line", "PointGlyph": "scatter",
            "AreaPath": "area", "ArcGlyph": "pie", "RectCell": "heatmap",
        }
        for t in terminals:
            if t in glyph_to_chart:
                if glyph_to_chart[t] == cif.visual_encoding.chart_type:
                    score += 2.0  # strong positive match
                else:
                    score -= 1.0  # mismatch penalty

        # Match axis types to data semantics
        axis_types = {"OrdinalAxis": "nominal", "QuantitativeAxis": "quantitative",
                      "TemporalAxis": "temporal"}
        for t in terminals:
            if t in axis_types:
                expected_type = axis_types[t]
                for field in cif.data_semantics:
                    if field.field_type == expected_type:
                        score += 0.5  # axis type matches field type

        return score

    def _extract_terminals(self, node: SyntaxNode) -> List[str]:
        """Extract terminal symbols from a syntax tree."""
        terminals = []
        if node.symbol.node_type == NodeType.TERMINAL and not node.children:
            terminals.append(node.symbol.name)
        for child in node.children:
            terminals.extend(self._extract_terminals(child))
        return terminals

    def _tree_to_spec(self, tree: SyntaxNode, log_prob: float) -> ChartSpec:
        """Convert a complete syntax tree to a ChartSpec."""
        terminals = self._extract_terminals(tree)

        # Determine chart type from glyph
        glyph_to_chart = {
            "BarGlyph": "bar", "LinePath": "line", "PointGlyph": "scatter",
            "AreaPath": "area", "ArcGlyph": "pie", "RectCell": "heatmap",
        }
        chart_type = "bar"  # default
        glyph_type = "BarGlyph"
        for t in terminals:
            if t in glyph_to_chart:
                chart_type = glyph_to_chart[t]
                glyph_type = t
                break

        # Determine layout
        x_axis = "QuantitativeAxis"
        y_axis = "OrdinalAxis"
        for t in terminals:
            if "Axis" in t:
                if x_axis == "QuantitativeAxis":
                    x_axis = t
                else:
                    y_axis = t

        # Build data bindings (will be filled by CIF alignment in MS-GRP)
        data_bindings = {"x": "province", "y": "DEL"}  # placeholder

        return ChartSpec(
            chart_type=chart_type,
            data_bindings=data_bindings,
            encoding_map={v: k for k, v in data_bindings.items()},
            layout={"x_axis": x_axis, "y_axis": y_axis},
            glyph_type=glyph_type,
            log_probability=log_prob,
        )
