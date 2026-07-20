"""
ChartGPT Baseline: Step-by-step reasoning LLM chart generation.

Inspired by Tian et al. (2024): "ChartGPT: Leveraging LLMs to generate charts
from abstract natural language." Uses Chain-of-Thought prompting to decompose
chart generation into sequential steps.
"""

import json
import time
from typing import Dict, Any


CHARTGPT_SYSTEM_PROMPT = """You are a chart generation expert using step-by-step reasoning.
Follow these steps to generate a chart from a natural language request:

Step 1: Identify the chart type (bar/line/scatter/area/pie/heatmap/radar/sankey/treemap/boxplot/gauge/funnel)
Step 2: Identify data fields and their roles (x-axis, y-axis, color, size)
Step 3: Determine visual encoding parameters
Step 4: Generate the complete Vega-Lite JSON specification

Output format:
{
  "reasoning": "Step 1: ... Step 2: ... Step 3: ...",
  "chart_type": "...",
  "data_fields": [{"field": "...", "role": "x|y|color|size"}],
  "vega_lite_spec": {...}
}"""


class ChartGPTGenerator:
    """ChartGPT baseline with chain-of-thought prompting."""

    def __init__(self, client, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    def generate(self, nl_query: str) -> Dict[str, Any]:
        """Generate chart with step-by-step CoT reasoning.

        Returns:
            dict with keys: chart_json, reasoning, chart_type, latency_s
        """
        t_start = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CHARTGPT_SYSTEM_PROMPT},
                {"role": "user", "content": nl_query},
            ],
            temperature=0.3,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        latency = time.time() - t_start

        try:
            result = json.loads(response.choices[0].message.content)
            chart_json = result.get("vega_lite_spec", {})
            reasoning = result.get("reasoning", "")
            chart_type = result.get("chart_type", "unknown")
        except json.JSONDecodeError:
            chart_json = {"error": "Failed to parse"}
            reasoning = ""
            chart_type = "unknown"

        return {
            "chart_json": chart_json,
            "reasoning": reasoning,
            "chart_type": chart_type,
            "latency_s": latency,
        }
