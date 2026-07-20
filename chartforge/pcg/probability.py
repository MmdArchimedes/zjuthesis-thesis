"""
PCFG Probability Learning — Maximum Likelihood Estimation from chart corpus.

Paper Section 3.2.2, Equation (10):
  L_PCG = Σ log P(r | parent(r)) + λ · R(P)

Learns production rule probabilities from ChartIntent-10K training set
using count-based MLE with smoothing.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .grammar import ChartGrammar, ProductionRule
from ..config import CHECKPOINT_DIR


class PCFGProbabilityLearner:
    """Learn PCFG production probabilities from chart corpus."""

    def __init__(self, grammar: ChartGrammar, smoothing: str = "laplace"):
        """
        Args:
            grammar: ChartGrammar instance with rule definitions
            smoothing: Smoothing method — "laplace", "add_k", or "none"
        """
        self.grammar = grammar
        self.smoothing = smoothing
        self.rule_counts: Dict[str, int] = defaultdict(int)  # rule_id → count
        self.nt_counts: Dict[str, int] = defaultdict(int)     # non-terminal → total count

    def fit(self, chart_samples: List[dict], verbose: bool = True) -> Dict[str, float]:
        """Learn probabilities from chart corpus.

        Args:
            chart_samples: List of chart spec dicts from ChartIntent-10K
            verbose: Print learning progress

        Returns:
            Dict mapping rule_id → learned probability
        """
        if verbose:
            print(f"Learning PCFG probabilities from {len(chart_samples)} charts...")

        # Step 1: Parse each chart spec into a syntax tree, count rule usage
        for i, chart in enumerate(chart_samples):
            try:
                tree = self._parse_chart_to_tree(chart)
                self._count_rules(tree)
            except Exception as e:
                if verbose and i < 5:
                    print(f"  [SKIP] Chart {i}: {e}")
                continue

        if verbose:
            n_rules_used = len([c for c in self.rule_counts.values() if c > 0])
            print(f"  Parsed {sum(1 for c in self.rule_counts.values() if c > 0)}/{len(self.rule_counts)} rules used")

        # Step 2: Compute MLE probabilities with smoothing
        probabilities = {}
        for nt in self.grammar.non_terminals:
            rules = self.grammar.get_rules_for(nt)
            counts = [self.rule_counts.get(r.rule_id, 0) for r in rules]
            total = sum(counts)

            if self.smoothing == "laplace":
                # Laplace (add-1) smoothing
                k = len(rules)
                probs = [(c + 1) / (total + k) for c in counts]
            elif self.smoothing == "add_k":
                # Add-k smoothing with k=0.1
                k_val = 0.1
                probs = [(c + k_val) / (total + k_val * len(rules)) for c in counts]
            else:
                # No smoothing — fall back to uniform for unseen rules
                if total == 0:
                    probs = [1.0 / len(rules)] * len(rules)
                else:
                    probs = [max(c / total, 1e-6) for c in counts]

            for rule, prob in zip(rules, probs):
                probabilities[rule.rule_id] = prob

        # Step 3: Apply learned probabilities to grammar
        self.grammar.set_probabilities(probabilities)

        if verbose:
            print(f"  Learned probabilities for {len(probabilities)} rules")
            self._print_top_rules(probabilities, n=10)

        return probabilities

    def _parse_chart_to_tree(self, chart: dict) -> "SyntaxNode":
        """Parse a chart specification dict into a syntax tree using grammar rules.

        This performs a simplified parse — in production, use a proper CYK parser.
        Here we use a heuristic mapping from Vega-Lite spec features to grammar rules.
        """
        from .grammar import SyntaxNode, Symbol, NodeType

        chart_type = chart.get("chart_type", "bar")
        mark_type = chart.get("mark", {}).get("type", "bar") if isinstance(chart, dict) else "bar"
        encoding = chart.get("encoding", {}) if isinstance(chart, dict) else {}

        # Map Vega-Lite mark to grammar glyph terminal
        mark_to_glyph = {
            "bar": "BarGlyph", "line": "LinePath", "point": "PointGlyph",
            "area": "AreaPath", "arc": "ArcGlyph", "rect": "RectCell",
        }
        glyph_name = mark_to_glyph.get(mark_type, "BarGlyph")

        # Map encoding channels to axis types
        has_x = "x" in encoding
        has_y = "y" in encoding
        has_color = "color" in encoding
        has_facet = "facet" in encoding or "column" in encoding or "row" in encoding

        x_type = encoding.get("x", {}).get("type", "nominal")
        y_type = encoding.get("y", {}).get("type", "quantitative")

        type_to_axis = {
            "nominal": "OrdinalAxis", "ordinal": "OrdinalAxis",
            "quantitative": "QuantitativeAxis",
            "temporal": "TemporalAxis",
        }
        axis_x = type_to_axis.get(x_type, "OrdinalAxis")
        axis_y = type_to_axis.get(y_type, "QuantitativeAxis")

        # Build simplified syntax tree
        # Chart → Layout Glyph+ Guide* Annotation*
        chart_node = SyntaxNode(symbol=Symbol("Chart", NodeType.NON_TERMINAL))

        # Layout → Axis Axis Facet*
        layout_node = SyntaxNode(symbol=Symbol("Layout", NodeType.NON_TERMINAL))
        if has_x:
            layout_node.children.append(SyntaxNode(symbol=Symbol(axis_x, NodeType.TERMINAL)))
        if has_y:
            layout_node.children.append(SyntaxNode(symbol=Symbol(axis_y, NodeType.TERMINAL)))
        if has_facet:
            layout_node.children.append(SyntaxNode(symbol=Symbol("FacetGrid", NodeType.TERMINAL)))

        # Glyph
        glyph_node = SyntaxNode(symbol=Symbol("Glyph", NodeType.NON_TERMINAL))
        glyph_node.children.append(SyntaxNode(symbol=Symbol(glyph_name, NodeType.TERMINAL)))

        # Guide (optional)
        guide_node = SyntaxNode(symbol=Symbol("Guide", NodeType.NON_TERMINAL))
        if has_color:
            guide_node.children.append(SyntaxNode(symbol=Symbol("ColorScale", NodeType.TERMINAL)))

        # Annotation (optional)
        annot_node = SyntaxNode(symbol=Symbol("Annotation", NodeType.NON_TERMINAL))
        text_labels = encoding.get("text", None) or chart.get("title", None)
        if text_labels:
            annot_node.children.append(SyntaxNode(symbol=Symbol("AnnotationText", NodeType.TERMINAL)))

        chart_node.children = [layout_node, glyph_node]
        if guide_node.children:
            chart_node.children.append(guide_node)
        if annot_node.children:
            chart_node.children.append(annot_node)

        return chart_node

    def _count_rules(self, node: "SyntaxNode"):
        """Count rule usage in a syntax tree."""
        if node.rule:
            self.rule_counts[node.rule.rule_id] += 1
            self.nt_counts[node.rule.lhs] += 1

        for child in node.children:
            self._count_rules(child)

    def _print_top_rules(self, probabilities: Dict[str, float], n: int = 10):
        """Print the top-N highest probability rules."""
        sorted_rules = sorted(probabilities.items(), key=lambda x: -x[1])
        print(f"\n  Top-{n} highest probability rules:")
        for rule_id, prob in sorted_rules[:n]:
            # Find the actual rule
            rule = None
            for rules in self.grammar.productions.values():
                for r in rules:
                    if r.rule_id == rule_id:
                        rule = r
                        break
                if rule:
                    break
            if rule:
                print(f"    {rule}")

    def save(self, path: str = None):
        """Save learned probabilities to JSON."""
        path = Path(path) if path else CHECKPOINT_DIR / "pcg" / "probabilities.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "smoothing": self.smoothing,
            "rule_probabilities": {
                rule_id: float(prob)
                for rule_id, prob in self._get_all_probabilities().items()
            },
            "rule_counts": dict(self.rule_counts),
            "total_charts_processed": sum(self.nt_counts.values()),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved PCG probabilities to {path}")

    def load(self, path: str = None):
        """Load learned probabilities from JSON."""
        path = Path(path) if path else CHECKPOINT_DIR / "pcg" / "probabilities.json"
        with open(path) as f:
            data = json.load(f)

        probabilities = data["rule_probabilities"]
        self.grammar.set_probabilities(probabilities)
        self.smoothing = data.get("smoothing", self.smoothing)
        print(f"Loaded PCG probabilities from {path}")

    def _get_all_probabilities(self) -> Dict[str, float]:
        """Get current probability for all rules."""
        result = {}
        for rules in self.grammar.productions.values():
            for rule in rules:
                result[rule.rule_id] = rule.probability
        return result
