"""
CIF Parser: Converts natural language queries into structured CIF triples.
Uses GPT-4o with structured output prompting for robust parsing.
"""

import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any

from .schema import CIFTriple, FieldSpec, VisualSpec, InteractionSpec
from .prompts import CIF_SYSTEM_PROMPT, build_cif_parse_prompt
from ..config import (
    LLM_MODEL, LLM_TEMPERATURE_CIF, LLM_MAX_TOKENS,
    DOMAIN_FIELDS, CHART_TYPES,
)


class CIFParser:
    """Natural Language → CIF Triple parser using LLM."""

    def __init__(
        self,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE_CIF,
        cache_path: str = None,
        openai_client=None,
    ):
        self.model = model
        self.temperature = temperature
        self.cache_path = cache_path or "data/cif_cache.db"
        self.client = openai_client
        self._init_cache()

    def _init_cache(self):
        """Initialize SQLite cache for parsed CIF results."""
        db_path = Path(self.cache_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cif_cache (
                query_hash TEXT PRIMARY KEY,
                nl_query TEXT,
                cif_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _hash_query(self, query: str) -> str:
        """Hash a query for cache lookup."""
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    def _cache_get(self, query: str) -> Optional[CIFTriple]:
        """Look up a parsed CIF from cache."""
        h = self._hash_query(query)
        row = self.conn.execute(
            "SELECT cif_json FROM cif_cache WHERE query_hash = ?", (h,)
        ).fetchone()
        if row:
            return CIFTriple.from_dict(json.loads(row[0]))
        return None

    def _cache_set(self, query: str, cif: CIFTriple):
        """Cache a parsed CIF result."""
        h = self._hash_query(query)
        self.conn.execute(
            "INSERT OR REPLACE INTO cif_cache (query_hash, nl_query, cif_json) VALUES (?, ?, ?)",
            (h, query, json.dumps(cif.to_dict(), ensure_ascii=False)),
        )
        self.conn.commit()

    def parse(
        self,
        nl_query: str,
        available_fields: List[str] = None,
        context: str = "",
        use_cache: bool = True,
    ) -> CIFTriple:
        """Parse a natural language query into a CIF triple.

        Args:
            nl_query: Natural language chart request
            available_fields: List of available data field names
            context: Domain context string
            use_cache: Whether to check cache first

        Returns:
            CIFTriple with parsed chart intent
        """
        # Check cache first
        if use_cache:
            cached = self._cache_get(nl_query)
            if cached is not None:
                return cached

        # Build prompt
        user_prompt = build_cif_parse_prompt(
            nl_query,
            available_fields=available_fields or list(DOMAIN_FIELDS.keys()),
            context=context,
        )

        # Call LLM
        try:
            cif = self._call_llm(user_prompt)
        except Exception as e:
            # Fallback: return low-confidence partial CIF
            return CIFTriple(
                data_semantics=[],
                visual_encoding=VisualSpec(chart_type="bar", encoding_map={}),
                interaction_constraints=[],
                raw_query=nl_query,
                confidence=0.0,
            )

        # Post-process and validate
        cif.raw_query = nl_query
        cif = self._validate_and_fix(cif, available_fields)

        # Cache result
        if use_cache:
            self._cache_set(nl_query, cif)

        return cif

    def _call_llm(self, user_prompt: str) -> CIFTriple:
        """Call GPT-4o API with structured output parsing."""
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")

        import openai
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CIF_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=LLM_MAX_TOKENS,
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content
        data = json.loads(raw_output)

        # Parse components
        fields = [
            FieldSpec(
                field_name=f["field_name"],
                field_type=f.get("field_type", "quantitative"),
                suggested_channel=f.get("suggested_channel", "y"),
                aggregation=f.get("aggregation", "none"),
            )
            for f in data.get("data_semantics", [])
        ]

        vis_data = data.get("visual_encoding", {})
        visual = VisualSpec(
            chart_type=vis_data.get("chart_type", "bar"),
            encoding_map=vis_data.get("encoding_map", {}),
            style_params=vis_data.get("style_params", {}),
        )

        interactions = [
            InteractionSpec(
                event_type=i.get("event_type", "click"),
                handler_spec=i.get("handler_spec", ""),
                constraint=i.get("constraint", ""),
            )
            for i in data.get("interaction_constraints", [])
        ]

        return CIFTriple(
            data_semantics=fields,
            visual_encoding=visual,
            interaction_constraints=interactions,
            confidence=0.85,
        )

    def _validate_and_fix(
        self, cif: CIFTriple, available_fields: List[str] = None
    ) -> CIFTriple:
        """Validate and fix CIF fields against domain vocabulary.

        Post-processing steps:
        1. Validate chart type is in known set
        2. Validate field names against available_fields (if provided)
        3. Infer missing field types from DOMAIN_FIELDS
        4. Validate channel assignments against field type compatibility
        """
        # Fix chart type
        if cif.visual_encoding.chart_type not in CHART_TYPES:
            # Find closest match
            from difflib import get_close_matches
            matches = get_close_matches(cif.visual_encoding.chart_type, CHART_TYPES, n=1)
            if matches:
                cif.visual_encoding.chart_type = matches[0]
            else:
                cif.visual_encoding.chart_type = "bar"
                cif.confidence *= 0.5

        # Validate fields
        valid_fields = set(available_fields) if available_fields else set(DOMAIN_FIELDS.keys())
        valid_field_specs = []
        for f in cif.data_semantics:
            # Check field name exists
            if f.field_name not in valid_fields:
                # Try fuzzy match
                from difflib import get_close_matches
                matches = get_close_matches(f.field_name, list(valid_fields), n=1)
                if matches:
                    f.field_name = matches[0]
                else:
                    cif.confidence *= 0.7
                    continue

            # Infer type from domain knowledge
            if f.field_name in DOMAIN_FIELDS:
                domain_info = DOMAIN_FIELDS[f.field_name]
                if f.field_type == "quantitative" and domain_info["type"] != "quantitative":
                    f.field_type = domain_info["type"]
                # Validate channel
                if f.suggested_channel not in domain_info.get("channels", []):
                    # Assign first compatible channel
                    f.suggested_channel = domain_info["channels"][0]

            valid_field_specs.append(f)

        cif.data_semantics = valid_field_specs

        # Clamp confidence
        cif.confidence = max(0.0, min(1.0, cif.confidence))

        return cif

    def batch_parse(
        self,
        queries: List[str],
        available_fields: List[str] = None,
        context: str = "",
    ) -> List[CIFTriple]:
        """Parse multiple queries in batch."""
        return [
            self.parse(q, available_fields=available_fields, context=context)
            for q in queries
        ]

    def close(self):
        """Close cache database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
