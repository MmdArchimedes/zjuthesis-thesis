"""
LLM Prompt Templates for CIF parsing.
GPT-4o uses these structured prompts to convert NL → CIF Triple.
"""

# Main CIF parsing system prompt
CIF_SYSTEM_PROMPT = """You are a Chart Intent Formalization (CIF) parser. Your task is to convert
natural language chart requests into structured CIF triples ⟨D, V, I⟩.

A CIF triple consists of:
- D (Data Semantics): list of fields with {field_name, field_type, suggested_channel, aggregation}
- V (Visual Encoding): {chart_type, encoding_map (field→channel), style_params}
- I (Interaction Constraints): list of {event_type, handler_spec, constraint}

Available chart types: bar, line, scatter, area, heatmap, pie, radar, sankey, treemap, boxplot, gauge, funnel
Available field types: nominal, ordinal, quantitative, temporal
Available visual channels: x, y, color, size, shape, facet, text, opacity
Available aggregations: sum, mean, count, none, min, max, median

Output ONLY valid JSON with this exact schema:
{
  "data_semantics": [{"field_name": "...", "field_type": "...", "suggested_channel": "...", "aggregation": "..."}],
  "visual_encoding": {"chart_type": "...", "encoding_map": {"field_name": "channel"}, "style_params": {}},
  "interaction_constraints": [{"event_type": "...", "handler_spec": "...", "constraint": "..."}]
}"""


def build_cif_parse_prompt(
    nl_query: str,
    available_fields: list = None,
    context: str = "",
) -> str:
    """Build a CIF parsing prompt with domain context.

    Args:
        nl_query: Natural language chart request
        available_fields: List of available data field names
        context: Additional domain context (e.g., "provincial economic data")

    Returns:
        Formatted prompt string
    """
    field_hint = ""
    if available_fields:
        field_hint = f"\nAvailable data fields: {', '.join(available_fields)}"

    context_hint = ""
    if context:
        context_hint = f"\nDomain context: {context}"

    return f"""Parse the following natural language chart request into a CIF triple.

User query: "{nl_query}"{field_hint}{context_hint}

Output the CIF triple as JSON:"""


def build_chart_evaluation_prompt(
    chart_spec: dict,
    cif: dict,
) -> str:
    """Build a prompt for LLM-based chart evaluation (used in SVAS fallback).

    Args:
        chart_spec: Generated chart specification (Vega-Lite JSON)
        cif: CIF triple dict

    Returns:
        Evaluation prompt
    """
    return f"""Evaluate how well this chart specification matches the user's intent.

Chart specification:
{chart_spec}

User intent (CIF):
{cif}

Rate the match on a scale of 0-100 for:
1. Data fidelity (are all required fields present and correctly mapped?)
2. Visual appropriateness (is the chart type appropriate? are visual encodings correct?)
3. Interaction completeness (are requested interactions implemented?)

Output JSON: {{"data_fidelity": N, "visual_appropriateness": N, "interaction_completeness": N, "overall": N, "issues": ["..."] }}"""


# Prompt for MS-GRP visual refinement stage
VISUAL_REFINEMENT_PROMPT = """You are a chart design expert. Given a chart specification and the user's intent,
suggest visual improvements:

1. Color palette optimization for the data type and audience
2. Layout adjustments to reduce clutter
3. Annotation placement for readability
4. Accessibility improvements (contrast, font size)

Output JSON: {"suggestions": [{"category": "...", "description": "...", "parameter": "...", "old_value": ..., "new_value": ...}]}"""
