"""
ChartForge FastAPI Server — REST API for AR system integration.

Endpoints:
  POST /parse-intent      NL → CIF
  POST /generate-chart    CIF → ChartSpec
  POST /refine-chart      ChartSpec → Refined ChartSpec
  POST /svas-score        Chart × CIF → SVAS score
  POST /full-pipeline     NL → Final Chart (all stages)
  GET  /health            Health check

Usage:
  python -m chartforge.main --task serve --port 8001
"""

import json
import time
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Lazy-loaded components (initialized on first request)
app = FastAPI(
    title="ChartForge API",
    description="Declarative AIGC Chart Widget Generation Framework",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response Models ───────────────────────────────────────

class ParseIntentRequest(BaseModel):
    query: str
    available_fields: Optional[List[str]] = None
    context: Optional[str] = ""

class ParseIntentResponse(BaseModel):
    cif: Dict[str, Any]
    confidence: float
    latency_ms: float

class GenerateChartRequest(BaseModel):
    cif: Dict[str, Any]
    beam_size: Optional[int] = 5

class GenerateChartResponse(BaseModel):
    chart_spec: Dict[str, Any]
    chart_type: str
    svas_score: Optional[float] = None
    latency_ms: float

class FullPipelineRequest(BaseModel):
    query: str
    available_fields: Optional[List[str]] = None
    context: Optional[str] = ""
    output_format: Optional[str] = "ar"  # "vegalite" | "echarts" | "ar"

class FullPipelineResponse(BaseModel):
    chart: Dict[str, Any]
    chart_type: str
    svas_score: float
    svas_breakdown: Dict[str, float]
    stage_times: Dict[str, float]
    total_latency_ms: float

class SVASScoreRequest(BaseModel):
    chart_spec: Dict[str, Any]
    cif: Dict[str, Any]

class SVASScoreResponse(BaseModel):
    phi_sem: float
    phi_vis: float
    phi_int: float
    svas: float
    passed_filter: bool

# ── Lazy component initialization ─────────────────────────────────

_grammar = None
_pipeline = None
_cif_parser = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from .pcg.grammar import ChartGrammar
        from .msgrp.pipeline import MSGRPPipeline
        _grammar = ChartGrammar()
        try:
            from .pcg.probability import PCFGProbabilityLearner
            learner = PCFGProbabilityLearner(_grammar)
            learner.load()
        except Exception:
            pass
        _pipeline = MSGRPPipeline(grammar=_grammar)
    return _pipeline

def get_cif_parser():
    global _cif_parser
    if _cif_parser is None:
        from .cif.parser import CIFParser
        _cif_parser = CIFParser()
    return _cif_parser


# ── API Endpoints ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "ChartForge",
        "version": "1.0.0",
        "components": {
            "cif": True,
            "pcg": True,
            "svas": True,
            "msgrp": True,
        },
    }


@app.post("/parse-intent", response_model=ParseIntentResponse)
async def parse_intent(req: ParseIntentRequest):
    """Parse natural language query into CIF triple."""
    t_start = time.time()
    parser = get_cif_parser()

    try:
        cif = parser.parse(
            req.query,
            available_fields=req.available_fields,
            context=req.context or "",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CIF parsing failed: {e}")

    latency_ms = (time.time() - t_start) * 1000

    return ParseIntentResponse(
        cif=cif.to_dict(),
        confidence=cif.confidence,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/generate-chart", response_model=GenerateChartResponse)
async def generate_chart(req: GenerateChartRequest):
    """Generate chart specification from CIF."""
    t_start = time.time()

    from .cif.schema import CIFTriple
    from .svas.scorer import SVASScorer

    cif = CIFTriple.from_dict(req.cif)
    pipeline = get_pipeline()

    # Stage 1: Coarse generation
    candidates = pipeline.sampler.sample(cif, beam_size=req.beam_size or 5)
    if not candidates:
        raise HTTPException(status_code=500, detail="No chart candidates generated")

    best = candidates[0]

    # Quick SVAS evaluation
    scorer = SVASScorer()
    svas_result = scorer.score(best, cif)

    latency_ms = (time.time() - t_start) * 1000

    return GenerateChartResponse(
        chart_spec=best.to_dict(),
        chart_type=best.chart_type,
        svas_score=svas_result["svas"],
        latency_ms=round(latency_ms, 2),
    )


@app.post("/full-pipeline", response_model=FullPipelineResponse)
async def full_pipeline(req: FullPipelineRequest):
    """Complete pipeline: NL → CIF → PCG → Verify → Refine → Inject → Render."""
    t_start = time.time()

    pipeline = get_pipeline()
    result = pipeline.generate(
        req.query,
        available_fields=req.available_fields,
        context=req.context or "",
    )

    # Render to requested format
    fmt = req.output_format or "ar"
    if fmt == "vegalite":
        from .rendering.spec_to_vegalite import to_vegalite
        chart_json = to_vegalite(result.chart_spec)
    elif fmt == "echarts":
        from .rendering.spec_to_echarts import to_echarts
        chart_json = to_echarts(result.chart_spec)
    else:  # ar
        from .rendering.ar_adapter import to_ar_format
        chart_json = to_ar_format(result.chart_spec)

    total_latency_ms = (time.time() - t_start) * 1000

    return FullPipelineResponse(
        chart=chart_json,
        chart_type=result.chart_spec.chart_type,
        svas_score=result.svas_score,
        svas_breakdown=result.svas_breakdown,
        stage_times=result.stage_times,
        total_latency_ms=round(total_latency_ms, 2),
    )


@app.post("/svas-score", response_model=SVASScoreResponse)
async def svas_score(req: SVASScoreRequest):
    """Evaluate SVAS score for a chart-CIF pair."""
    from .cif.schema import CIFTriple
    from .pcg.sampler import ChartSpec
    from .svas.scorer import SVASScorer

    cif = CIFTriple.from_dict(req.cif)
    chart = ChartSpec(
        chart_type=req.chart_spec.get("chart_type", "bar"),
        data_bindings=req.chart_spec.get("data_bindings", {}),
        encoding_map=req.chart_spec.get("encoding_map", {}),
        layout=req.chart_spec.get("layout", {}),
        glyph_type=req.chart_spec.get("glyph_type", "BarGlyph"),
    )

    scorer = SVASScorer()
    result = scorer.score(chart, cif)

    return SVASScoreResponse(**result)


# ── Startup ───────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-load components on server start."""
    print("ChartForge API starting...")
    get_pipeline()
    print("  PCG grammar loaded")
    print("ChartForge API ready")
