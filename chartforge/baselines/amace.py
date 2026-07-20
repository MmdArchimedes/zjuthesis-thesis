"""
AMACE Baseline: Automatic Multi-Agent Chart Evolution.

Inspired by Namgoong et al. (2025): "AMACE: Automatic multi-agent chart evolution
for iteratively tailored chart generation."

Three specialized agents collaborate:
- Chart Code Generator: Creates chart specifications
- Chart Replier: Reviews and critiques generated charts
- Chart Quality Evaluator: Scores chart quality
"""

import json
import time
from typing import Dict, Any, List


GENERATOR_PROMPT = """You are a Chart Code Generator. Given a user's chart request,
generate a complete Vega-Lite JSON specification. Focus on accuracy and completeness.

User request: {query}

Output ONLY valid Vega-Lite JSON."""


REVIEWER_PROMPT = """You are a Chart Replier. Review this chart specification critically.
Identify ALL issues with data encoding, chart type choice, visual design, and completeness.

User request: {query}
Chart specification: {chart_json}

Output JSON: {{"issues": ["..."], "severity": {{"issue_name": "critical|major|minor"}}, "overall": "accept|revise|reject"}}"""


EVALUATOR_PROMPT = """You are a Chart Quality Evaluator. Score this chart on:
1. Data accuracy (0-100): Are fields correctly mapped?
2. Visual appropriateness (0-100): Is the chart type suitable?
3. Design quality (0-100): Is it well-designed?
4. Completeness (0-100): Are all requested elements present?

User request: {query}
Chart specification: {chart_json}

Output JSON: {{"data_accuracy": N, "visual_appropriateness": N, "design_quality": N, "completeness": N, "overall": N}}"""


INTEGRATOR_PROMPT = """You are a Chart Integrator. Given the original chart, reviewer feedback,
and quality scores, produce the final improved chart specification.

Original: {chart_json}
Review feedback: {feedback}
Quality scores: {scores}

Output ONLY the final improved Vega-Lite JSON."""


class AMACEGenerator:
    """AMACE multi-agent baseline."""

    def __init__(self, client, model: str = "gpt-4o", max_rounds: int = 2):
        self.client = client
        self.model = model
        self.max_rounds = max_rounds

    def generate(self, nl_query: str) -> Dict[str, Any]:
        """Multi-agent chart generation with evolution.

        Returns:
            dict with keys: chart_json, rounds, scores_history, latency_s
        """
        t_start = time.time()
        scores_history = []

        # Round 1: Initial generation
        gen_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": GENERATOR_PROMPT.format(query=nl_query)},
            ],
            temperature=0.7,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        chart_json = json.loads(gen_response.choices[0].message.content)

        for round_num in range(self.max_rounds):
            # Agent 2: Reviewer
            rev_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": REVIEWER_PROMPT.format(
                        query=nl_query,
                        chart_json=json.dumps(chart_json, indent=2),
                    ),
                }],
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            review = json.loads(rev_response.choices[0].message.content)

            # Agent 3: Evaluator
            eval_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": EVALUATOR_PROMPT.format(
                        query=nl_query,
                        chart_json=json.dumps(chart_json, indent=2),
                    ),
                }],
                temperature=0.2,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            scores = json.loads(eval_response.choices[0].message.content)
            scores_history.append(scores)

            # Stop if quality is sufficient
            if scores.get("overall", 0) >= 90:
                break

            # Agent 4: Integrator (refine chart)
            if review.get("overall") != "accept":
                int_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": INTEGRATOR_PROMPT.format(
                            chart_json=json.dumps(chart_json, indent=2),
                            feedback=json.dumps(review, indent=2),
                            scores=json.dumps(scores, indent=2),
                        ),
                    }],
                    temperature=0.4,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                try:
                    chart_json = json.loads(int_response.choices[0].message.content)
                except json.JSONDecodeError:
                    break

        latency = time.time() - t_start

        return {
            "chart_json": chart_json,
            "rounds": len(scores_history),
            "scores_history": scores_history,
            "latency_s": latency,
        }
