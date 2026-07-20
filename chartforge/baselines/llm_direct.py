"""
LLM-Direct Baseline: GPT-4o directly generates Vega-Lite JSON chart specification.

This is the simplest baseline — no intermediate representation, no refinement.
Represents the "naive" AIGC approach that ChartForge improves upon.
"""

import json
import time
from typing import Dict, Any, Optional


DIRECT_SYSTEM_PROMPT = """You are a data visualization expert. Given a natural language request,
generate a complete Vega-Lite JSON specification for the requested chart.

Available data fields: province (nominal), region (nominal), year (temporal),
DEL (quantitative, digital economy index), ES (quantitative, energy structure index),
PGDP (quantitative), URBAN (quantitative), INDS (quantitative), TEIN (quantitative).

Output ONLY valid Vega-Lite JSON. Do not include explanations."""


class LLMDirectGenerator:
    """Baseline: GPT-4o → Vega-Lite JSON directly."""

    def __init__(self, client, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    def generate(self, nl_query: str, available_fields: list = None) -> Dict[str, Any]:
        """Generate chart by directly prompting GPT-4o for Vega-Lite JSON.

        Returns:
            dict with keys: chart_json, latency_s, tokens_used
        """
        import openai

        t_start = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
                {"role": "user", "content": nl_query},
            ],
            temperature=0.7,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        latency = time.time() - t_start

        try:
            chart_json = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            chart_json = {"error": "Failed to parse JSON output"}

        return {
            "chart_json": chart_json,
            "latency_s": latency,
            "tokens_used": response.usage.total_tokens if response.usage else 0,
        }
