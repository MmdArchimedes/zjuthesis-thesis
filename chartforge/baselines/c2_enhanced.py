"""
C²-Enhanced Baseline: LLM generation + auto-feedback refinement loop.

Inspired by Koh et al. (2025): "C²: Scalable auto-feedback for LLM-based chart generation."
Uses an LLM-based evaluator to provide feedback and iteratively refine charts.
"""

import json
import time
from typing import Dict, Any


EVALUATOR_PROMPT = """You are a chart quality evaluator. Review the chart specification below
and identify issues with:
1. Chart type appropriateness for the data and request
2. Data field mapping correctness
3. Visual encoding quality
4. Missing information

Chart request: {query}

Chart specification (Vega-Lite JSON):
{chart_json}

Output JSON with:
{{
  "issues": ["issue 1", "issue 2", ...],
  "overall_score": 0-100,
  "suggestions": ["suggestion 1", "suggestion 2", ...]
}}"""


REFINER_PROMPT = """You are a chart refinement expert. Given the original chart specification
and improvement suggestions, generate an improved Vega-Lite JSON specification.

Original chart:
{chart_json}

Issues found: {issues}

Suggestions: {suggestions}

Output ONLY the improved Vega-Lite JSON. Do not include explanations."""


class C2EnhancedGenerator:
    """C²-Enhanced baseline with auto-feedback refinement."""

    def __init__(self, client, model: str = "gpt-4o", max_iterations: int = 3):
        self.client = client
        self.model = model
        self.max_iterations = max_iterations

    def generate(self, nl_query: str) -> Dict[str, Any]:
        """Generate and iteratively refine chart using LLM self-feedback.

        Returns:
            dict with keys: chart_json, iterations, scores, latency_s
        """
        t_start = time.time()

        # Step 1: Initial generation
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Generate a Vega-Lite JSON chart specification."},
                {"role": "user", "content": nl_query},
            ],
            temperature=0.7,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        chart_json = json.loads(response.choices[0].message.content)

        scores = []
        iterations = 0

        # Step 2-N: Iterative refinement
        for iteration in range(self.max_iterations):
            # Evaluate current chart
            eval_prompt = EVALUATOR_PROMPT.format(
                query=nl_query,
                chart_json=json.dumps(chart_json, indent=2),
            )

            eval_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            eval_result = json.loads(eval_response.choices[0].message.content)
            score = eval_result.get("overall_score", 0)
            issues = eval_result.get("issues", [])
            suggestions = eval_result.get("suggestions", [])
            scores.append(score)

            # Stop if good enough
            if score >= 90 and len(issues) == 0:
                break

            iterations += 1

            # Refine
            if issues:
                refine_prompt = REFINER_PROMPT.format(
                    chart_json=json.dumps(chart_json, indent=2),
                    issues=issues,
                    suggestions=suggestions,
                )

                refine_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": refine_prompt}],
                    temperature=0.3,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )

                try:
                    chart_json = json.loads(refine_response.choices[0].message.content)
                except json.JSONDecodeError:
                    break  # can't parse refined output

        latency = time.time() - t_start

        return {
            "chart_json": chart_json,
            "iterations": iterations,
            "scores": scores,
            "latency_s": latency,
        }
