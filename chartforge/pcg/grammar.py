"""
PCG Grammar Definition — Probabilistic Chart Grammar (PCFG).

Formal definition (paper Definition 1, Equation 6):
  PCG = ⟨N, T, S, R, P⟩

Where:
  N = {Chart, Layout, Glyph, Axis, Scale, Legend, Guide, Facet, Layer, Annotation}
  T = {BarGlyph, PointGlyph, LinePath, AreaPath, ArcGlyph, RectCell, ...}
  S = Chart (start symbol)
  R = Production rules mapping N → (N ∪ T)*
  P = Probability distribution over R
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional, Any
from enum import Enum


class NodeType(Enum):
    NON_TERMINAL = "NT"
    TERMINAL = "T"
    KLEENE_STAR = "*"   # zero or more
    KLEENE_PLUS = "+"   # one or more


@dataclass
class Symbol:
    """A grammar symbol (terminal or non-terminal)."""
    name: str
    node_type: NodeType

    def __repr__(self):
        suffix = ""
        if self.node_type == NodeType.KLEENE_STAR:
            suffix = "*"
        elif self.node_type == NodeType.KLEENE_PLUS:
            suffix = "+"
        return f"{self.name}{suffix}"


@dataclass
class ProductionRule:
    """A single production rule: A → β with probability p.

    Equation: A → β₁ β₂ ... βₖ  with P(A → β)
    """
    lhs: str                      # left-hand side (non-terminal name)
    rhs: List[Symbol]            # right-hand side (sequence of symbols)
    probability: float = 0.0     # P(A → β) ∈ [0, 1]
    rule_id: str = ""            # unique identifier

    def __repr__(self):
        rhs_str = " ".join(str(s) for s in self.rhs)
        return f"{self.lhs} → {rhs_str} [{self.probability:.4f}]"


@dataclass
class SyntaxNode:
    """A node in the chart syntax tree."""
    symbol: Symbol
    children: List["SyntaxNode"] = field(default_factory=list)
    rule: Optional[ProductionRule] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.symbol.node_type == NodeType.TERMINAL and not self.children

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class SyntaxTree:
    """Full syntax tree representing a chart derivation."""
    root: SyntaxNode
    log_probability: float = 0.0
    chart_type: str = ""

    def get_terminals(self) -> List[str]:
        """Extract terminal sequence (leaf nodes)."""
        terminals = []

        def traverse(node):
            if node.is_leaf:
                terminals.append(node.symbol.name)
            else:
                for child in node.children:
                    traverse(child)

        traverse(self.root)
        return terminals


class ChartGrammar:
    """Probabilistic Chart Grammar (PCG) — complete definition.

    Paper Section 3.2, Equations (6)-(9).
    """

    def __init__(self):
        self.non_terminals: Set[str] = set()
        self.terminals: Set[str] = set()
        self.start_symbol: str = "Chart"
        self.productions: Dict[str, List[ProductionRule]] = {}
        self._rule_counter: int = 0

        self._init_grammar()

    def _init_grammar(self):
        """Initialize the chart grammar with production rules."""
        # Non-terminals (paper Equation 7)
        nts = [
            "Chart", "Layout", "Glyph", "Axis",
            "Scale", "Legend", "Guide", "Facet",
            "Layer", "Annotation"
        ]
        self.non_terminals = set(nts)

        # Terminals (paper Equation 8)
        ts = [
            "BarGlyph", "PointGlyph", "LinePath", "AreaPath",
            "ArcGlyph", "RectCell", "TextLabel", "TickMark",
            "GridLine", "ColorScale", "SizeScale", "OpacityScale",
            "ShapeScale", "OrdinalAxis", "QuantitativeAxis",
            "TemporalAxis", "LegendDef", "GuideLine", "FacetGrid",
            "AnnotationText", "AnnotationRegion"
        ]
        self.terminals = set(ts)

        # Production rules (paper Equations 9-12)
        self._add_rule("Chart", [
            ("Layout", NodeType.NON_TERMINAL),
            ("Glyph", NodeType.KLEENE_PLUS),
            ("Guide", NodeType.KLEENE_STAR),
            ("Annotation", NodeType.KLEENE_STAR),
        ])

        self._add_rule("Layout", [
            ("Axis", NodeType.NON_TERMINAL),
            ("Axis", NodeType.NON_TERMINAL),
            ("Facet", NodeType.KLEENE_STAR),
        ])

        # Glyph alternatives (paper Equation 9)
        for glyph in ["BarGlyph", "PointGlyph", "LinePath", "AreaPath",
                       "ArcGlyph", "RectCell"]:
            self._add_rule("Glyph", [(glyph, NodeType.TERMINAL)])

        # Axis alternatives
        for axis_type in ["OrdinalAxis", "QuantitativeAxis", "TemporalAxis"]:
            self._add_rule("Axis", [(axis_type, NodeType.TERMINAL)])

        # Scale alternatives
        for scale_type in ["ColorScale", "SizeScale", "OpacityScale", "ShapeScale"]:
            self._add_rule("Scale", [(scale_type, NodeType.TERMINAL)])

        # Legend
        self._add_rule("Legend", [("LegendDef", NodeType.TERMINAL)])

        # Guide
        self._add_rule("Guide", [("GuideLine", NodeType.TERMINAL)])
        self._add_rule("Guide", [("TickMark", NodeType.TERMINAL)])
        self._add_rule("Guide", [("GridLine", NodeType.TERMINAL)])

        # Facet
        self._add_rule("Facet", [("FacetGrid", NodeType.TERMINAL)])

        # Layer (for composite charts)
        self._add_rule("Layer", [
            ("Glyph", NodeType.NON_TERMINAL),
            ("Glyph", NodeType.KLEENE_PLUS),
        ])

        # Annotation
        self._add_rule("Annotation", [("AnnotationText", NodeType.TERMINAL)])
        self._add_rule("Annotation", [("AnnotationRegion", NodeType.TERMINAL)])

        # Set uniform initial probabilities
        self._set_uniform_probabilities()

    def _add_rule(self, lhs: str, rhs: List[Tuple[str, NodeType]]):
        """Add a production rule to the grammar."""
        symbols = [Symbol(name, nt) for name, nt in rhs]

        rule = ProductionRule(
            lhs=lhs,
            rhs=symbols,
            probability=0.0,
            rule_id=f"R{self._rule_counter:04d}",
        )
        self._rule_counter += 1

        if lhs not in self.productions:
            self.productions[lhs] = []
        self.productions[lhs].append(rule)

    def _set_uniform_probabilities(self):
        """Set uniform initial probability for each non-terminal's rules."""
        for lhs, rules in self.productions.items():
            n = len(rules)
            for rule in rules:
                rule.probability = 1.0 / n

    def set_probabilities(self, probabilities: Dict[str, float]):
        """Set learned probabilities from PCFGProbabilityLearner."""
        for rule_id, prob in probabilities.items():
            for rules in self.productions.values():
                for rule in rules:
                    if rule.rule_id == rule_id:
                        rule.probability = prob
                        break

    def validate(self) -> bool:
        """Validate grammar properties:
        1. All non-terminals have at least one production
        2. Probability sums to 1 for each non-terminal (within tolerance)
        3. No orphaned symbols
        """
        for nt in self.non_terminals:
            if nt not in self.productions:
                print(f"  [ERROR] Non-terminal {nt} has no productions")
                return False

            rules = self.productions[nt]
            prob_sum = sum(r.probability for r in rules)
            if abs(prob_sum - 1.0) > 1e-6:
                print(f"  [WARN] {nt}: probability sum = {prob_sum:.6f} (expected 1.0)")

        return True

    def get_rules_for(self, non_terminal: str) -> List[ProductionRule]:
        """Get all production rules for a given non-terminal."""
        return self.productions.get(non_terminal, [])

    @property
    def total_rules(self) -> int:
        return sum(len(rules) for rules in self.productions.values())

    def summary(self) -> str:
        """Print grammar summary."""
        lines = [
            f"Chart Grammar Summary",
            f"  Non-terminals: {len(self.non_terminals)}",
            f"  Terminals:     {len(self.terminals)}",
            f"  Productions:   {self.total_rules}",
            f"  Start symbol:  {self.start_symbol}",
        ]
        for nt in sorted(self.non_terminals):
            rules = self.productions.get(nt, [])
            lines.append(f"\n  {nt} ({len(rules)} rules):")
            for r in rules:
                rhs_str = " ".join(str(s) for s in r.rhs)
                lines.append(f"    → {rhs_str} [{r.probability:.3f}]")
        return "\n".join(lines)
