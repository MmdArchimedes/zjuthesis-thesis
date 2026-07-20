"""
MS-GRP Pipeline — Multi-Stage Generative Refinement Pipeline.

Four stages:
  1. Coarse Generation: PCG beam search from CIF intent
  2. Semantic Verification: SVAS filtering (threshold τ=0.7)
  3. Visual Refinement: REINFORCE-style parameter optimization
  4. Interaction Injection: Add interaction handlers to chart spec

Paper Section 3.4, Equations (14)-(20).
"""

import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..cif.schema import CIFTriple
from ..cif.parser import CIFParser
from ..pcg.grammar import ChartGrammar
from ..pcg.sampler import PCGBeamSampler, ChartSpec
from ..pcg.probability import PCFGProbabilityLearner
from ..svas.scorer import SVASScorer, svas
from ..config import BEAM_SIZE, TAU_SEM


@dataclass
class GenerationResult:
    """Complete MS-GRP generation result with metadata."""
    chart_spec: ChartSpec
    cif: CIFTriple
    svas_score: float
    svas_breakdown: Dict[str, float]
    stage_times: Dict[str, float]  # timing per stage
    candidates_count: int
    final_chart_json: Optional[Dict] = None  # rendered chart (Vega-Lite/Plotly/ECharts)


class MSGRPPipeline:
    """Complete MS-GRP generation pipeline.

    Usage:
        pipeline = MSGRPPipeline(grammar, cif_parser, openai_client)
        result = pipeline.generate("Show me a bar chart of DEL by province in 2022")
    """

    def __init__(
        self,
        grammar: ChartGrammar = None,
        cif_parser: CIFParser = None,
        scorer: SVASScorer = None,
        openai_client=None,
    ):
        self.grammar = grammar or ChartGrammar()
        self.cif_parser = cif_parser
        self.scorer = scorer or SVASScorer()
        self.client = openai_client

        # Initialize sub-components
        self.sampler = PCGBeamSampler(self.grammar, beam_size=BEAM_SIZE)

    def generate(
        self,
        nl_query: str,
        available_fields: List[str] = None,
        context: str = "",
        return_all_stages: bool = False,
    ) -> GenerationResult:
        """Full MS-GRP pipeline: NL → refined chart spec.

        Args:
            nl_query: Natural language chart request
            available_fields: List of available data field names
            context: Domain context (e.g., "provincial economic data")
            return_all_stages: If True, return intermediate stage outputs

        Returns:
            GenerationResult with final chart spec and metadata
        """
        stage_times = {}
        t_start = time.time()

        # ── Stage 0: Parse intent (NL → CIF) ──
        t0 = time.time()
        if self.cif_parser:
            cif = self.cif_parser.parse(
                nl_query,
                available_fields=available_fields,
                context=context,
            )
        else:
            # Minimal CIF without LLM parser (for testing)
            cif = self._fallback_cif(nl_query)
        stage_times["cif_parsing"] = time.time() - t0

        # ── Stage 1: Coarse Generation ──
        t1 = time.time()
        candidates = self._stage1_coarse(cif)
        stage_times["coarse_generation"] = time.time() - t1

        # ── Stage 2: Semantic Verification ──
        t2 = time.time()
        verified = self._stage2_verify(candidates, cif)
        stage_times["semantic_verification"] = time.time() - t2

        # ── Stage 3: Visual Refinement ──
        t3 = time.time()
        refined = self._stage3_refine(verified, cif)
        stage_times["visual_refinement"] = time.time() - t3

        # ── Stage 4: Interaction Injection ──
        t4 = time.time()
        final = self._stage4_inject(refined, cif)
        stage_times["interaction_injection"] = time.time() - t4

        # ── Final evaluation ──
        svas_score = svas(final, cif)
        svas_breakdown = self.scorer.score(final, cif)

        stage_times["total"] = time.time() - t_start

        return GenerationResult(
            chart_spec=final,
            cif=cif,
            svas_score=svas_score,
            svas_breakdown=svas_breakdown,
            stage_times=stage_times,
            candidates_count=len(candidates),
        )

    def _stage1_coarse(self, cif: CIFTriple) -> List[ChartSpec]:
        """Stage 1: Coarse generation via PCG beam search.

        Paper Equation (14):
          C_coarse = argmax P(T | PCG) · P(CIF(Q) | T)
        """
        return self.sampler.sample(cif, beam_size=BEAM_SIZE)

    def _stage2_verify(
        self, candidates: List[ChartSpec], cif: CIFTriple
    ) -> ChartSpec:
        """Stage 2: Semantic verification via SVAS filtering.

        Paper Equation (15):
          C_filtered = {C ∈ C_coarse | SVAS(C, Q) > τ_sem}
        """
        # Score all candidates
        scored = [(c, svas(c, cif)) for c in candidates]
        scored.sort(key=lambda x: -x[1])

        # Filter by threshold
        filtered = [(c, s) for c, s in scored if s > TAU_SEM]

        if not filtered:
            # Fallback: use top candidate with lowered threshold
            filtered = [scored[0]]

        # Return best candidate
        return filtered[0][0]

    def _stage3_refine(self, chart: ChartSpec, cif: CIFTriple) -> ChartSpec:
        """Stage 3: Visual refinement.

        Paper Equations (16)-(19):
          Θ* = argmin L_visual(C[Θ], Q)

        Uses REINFORCE-style gradient estimation for non-differentiable visual loss.
        """
        import copy
        import numpy as np
        from ..config import (
            N_VISUAL_REFINE_STEPS, REINFORCE_N_SAMPLES,
            REINFORCE_SIGMA_INIT, REINFORCE_SIGMA_DECAY, REINFORCE_LR,
        )

        refined = copy.deepcopy(chart)

        # Collect optimizable style parameters
        style_params = refined.style_params
        if not style_params:
            style_params = {
                "color_scheme": "tableau10",
                "font_size": 12,
                "grid_alpha": 0.3,
                "title_font_size": 16,
            }

        # Parameter vector (continuous parameters only)
        param_keys = [k for k in style_params if isinstance(style_params[k], (int, float))]
        if not param_keys:
            return refined

        theta = np.array([float(style_params[k]) for k in param_keys])
        sigma = float(REINFORCE_SIGMA_INIT)

        for step in range(N_VISUAL_REFINE_STEPS):
            # Sample N perturbations
            epsilons = np.random.normal(0, sigma, (REINFORCE_N_SAMPLES, len(theta)))
            losses = []

            for eps in epsilons:
                # Apply perturbation
                candidate_style = dict(style_params)
                for i, key in enumerate(param_keys):
                    candidate_style[key] = theta[i] + eps[i]

                # Clip to reasonable ranges
                if "font_size" in candidate_style:
                    candidate_style["font_size"] = max(6, min(24, candidate_style["font_size"]))
                if "grid_alpha" in candidate_style:
                    candidate_style["grid_alpha"] = max(0, min(1, candidate_style["grid_alpha"]))

                candidate = copy.deepcopy(refined)
                candidate.style_params = candidate_style

                # Compute visual loss
                loss = self._visual_loss(candidate, cif)
                losses.append(loss)

            losses = np.array(losses)

            # REINFORCE gradient estimate
            grad = np.mean([
                L * eps / (sigma ** 2)
                for L, eps in zip(losses, epsilons)
            ], axis=0)

            # Update parameters
            theta -= REINFORCE_LR * grad
            sigma *= REINFORCE_SIGMA_DECAY

        # Apply optimized parameters
        for i, key in enumerate(param_keys):
            refined.style_params[key] = float(theta[i])

        return refined

    def _visual_loss(self, chart: ChartSpec, cif: CIFTriple) -> float:
        """Compute visual loss L_visual.

        Equation (19):
          L_visual = λ₁·||palette - preferred|| + λ₂·clutterPenalty + λ₃·accessibilityScore
        """
        from ..config import LAMBDA_PALETTE, LAMBDA_CLUTTER, LAMBDA_ACCESSIBILITY

        # Palette match loss
        palette_loss = 0.0
        expected_scheme = cif.visual_encoding.style_params.get("color_scheme", "tableau10")
        actual_scheme = chart.style_params.get("color_scheme", "tableau10")
        if expected_scheme != actual_scheme:
            palette_loss = 0.5

        # Clutter penalty (simplified: penalty for having too many glyphs)
        # More glyphs = more visual clutter
        num_glyphs = len(chart.data_bindings)
        clutter_penalty = min(1.0, max(0.0, (num_glyphs - 3) / 10))

        # Accessibility score (simplified: font size adequacy)
        font_size = chart.style_params.get("font_size", 12)
        if font_size >= 14:
            accessibility = 0.0  # no penalty
        elif font_size >= 10:
            accessibility = 0.3
        else:
            accessibility = 0.7

        return (
            LAMBDA_PALETTE * palette_loss +
            LAMBDA_CLUTTER * clutter_penalty +
            LAMBDA_ACCESSIBILITY * accessibility
        )

    def _stage4_inject(self, chart: ChartSpec, cif: CIFTriple) -> ChartSpec:
        """Stage 4: Interaction injection.

        Paper Equation (20):
          C_final = injectInteractions(C_refined, I)

        Adds interaction handlers to the chart spec based on CIF constraints.
        """
        import copy
        final = copy.deepcopy(chart)

        injected = []
        for int_spec in cif.interaction_constraints:
            interaction = {
                "type": int_spec.event_type,
                "handler": int_spec.handler_spec,
                "constraint": int_spec.constraint if int_spec.constraint else "default",
                "enabled": True,
            }
            injected.append(interaction)

        # Add default interactions based on chart type
        default_interactions = {
            "bar": [{"type": "hover", "handler": "show_tooltip", "constraint": "default"}],
            "line": [{"type": "hover", "handler": "show_tooltip", "constraint": "default"}],
            "scatter": [{"type": "hover", "handler": "show_tooltip", "constraint": "default"},
                       {"type": "brush", "handler": "select_region", "constraint": "default"}],
            "heatmap": [{"type": "hover", "handler": "show_tooltip", "constraint": "default"}],
        }

        defaults = default_interactions.get(chart.chart_type, [])

        # Merge without duplicates
        existing_types = {i["type"] for i in injected}
        for d in defaults:
            if d["type"] not in existing_types:
                injected.append(d)

        final.interactions = injected
        return final

    def _fallback_cif(self, nl_query: str) -> CIFTriple:
        """Generate minimal CIF without LLM parser (for testing)."""
        from ..cif.schema import CIFTriple, VisualSpec, FieldSpec

        # Simple keyword-based guess
        query_lower = nl_query.lower()

        chart_type = "bar"  # default
        for ct in ["bar", "line", "scatter", "area", "pie", "heatmap",
                    "radar", "sankey", "treemap", "boxplot", "gauge", "funnel"]:
            if ct in query_lower:
                chart_type = ct
                break

        fields = []
        if "del" in query_lower or "数字" in nl_query:
            fields.append(FieldSpec("DEL", "quantitative", "y"))
        if "es" in query_lower or "能源" in nl_query:
            fields.append(FieldSpec("ES", "quantitative", "y"))
        if "省" in nl_query or "province" in query_lower:
            fields.append(FieldSpec("province", "nominal", "x"))
        if "年" in nl_query or "year" in query_lower:
            fields.append(FieldSpec("year", "temporal", "x"))

        if not fields:
            fields = [
                FieldSpec("DEL", "quantitative", "y"),
                FieldSpec("province", "nominal", "x"),
            ]

        return CIFTriple(
            data_semantics=fields,
            visual_encoding=VisualSpec(chart_type=chart_type, encoding_map={}),
            interaction_constraints=[],
            raw_query=nl_query,
            confidence=0.5,
        )
