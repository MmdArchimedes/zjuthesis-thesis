"""
ChartForge Configuration — all tunable parameters in one place for reproducibility.
Follows the same pattern as gesture_nn/config.py.
"""

# ── Chart Types ──────────────────────────────────────────────────
CHART_TYPES = [
    "bar", "line", "scatter", "area", "heatmap",
    "pie", "radar", "sankey", "treemap",
    "boxplot", "gauge", "funnel"
]

# Chart type complexity tiers (for fine-grained analysis)
CHART_TIERS = {
    "basic":  ["bar", "line", "scatter", "pie"],
    "medium": ["area", "radar", "boxplot", "gauge", "funnel"],
    "complex": ["heatmap", "sankey", "treemap"],
}

# ── CIF Configuration ────────────────────────────────────────────
FIELD_TYPES = ["nominal", "ordinal", "quantitative", "temporal"]
VISUAL_CHANNELS = ["x", "y", "color", "size", "shape", "facet", "text", "opacity"]
AGG_FUNCTIONS = ["sum", "mean", "count", "none", "min", "max", "median"]

# Domain-specific field vocabulary (matching thesis Ch5 data)
DOMAIN_FIELDS = {
    "province": {"type": "nominal", "channels": ["x", "color", "facet"]},
    "region": {"type": "nominal", "channels": ["color", "facet"]},
    "year": {"type": "temporal", "channels": ["x", "facet"]},
    "DEL": {"type": "quantitative", "channels": ["y", "color", "size"]},
    "DEL_sq": {"type": "quantitative", "channels": ["y"]},
    "ES": {"type": "quantitative", "channels": ["y", "color", "size"]},
    "PGDP": {"type": "quantitative", "channels": ["x", "y", "color"]},
    "URBAN": {"type": "quantitative", "channels": ["x", "y"]},
    "INDS": {"type": "quantitative", "channels": ["x", "y"]},
    "LEOP": {"type": "quantitative", "channels": ["x", "y"]},
    "FDE": {"type": "quantitative", "channels": ["x", "y"]},
    "TEIN": {"type": "quantitative", "channels": ["x", "y"]},
    "DEN": {"type": "quantitative", "channels": ["x", "y"]},
    "DEL_coefficient": {"type": "quantitative", "channels": ["y", "text"]},
    "DEL_sq_coefficient": {"type": "quantitative", "channels": ["y", "text"]},
    "ES_predicted": {"type": "quantitative", "channels": ["y"]},
}

# Province names (30 provinces, matching thesis)
PROVINCES = [
    "北京", "天津", "河北", "山西", "内蒙古",
    "辽宁", "吉林", "黑龙江",
    "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "广西", "海南",
    "重庆", "四川", "贵州", "云南",
    "陕西", "甘肃", "青海", "宁夏", "新疆",
]

REGIONS = ["东部", "中部", "西部", "东北"]
YEARS = list(range(2014, 2023))  # 2014-2022

# ── PCG Configuration ────────────────────────────────────────────
BEAM_SIZE = 5
MAX_DEPTH = 10
NUM_PRODUCTIONS = 200  # approximate |R|

# Grammar non-terminals
NON_TERMINALS = [
    "Chart", "Layout", "Glyph", "Axis",
    "Scale", "Legend", "Guide", "Facet",
    "Layer", "Annotation"
]

# Grammar terminals
TERMINALS = [
    "BarGlyph", "PointGlyph", "LinePath", "AreaPath",
    "ArcGlyph", "RectCell", "TextLabel", "TickMark",
    "GridLine", "ColorScale", "SizeScale", "OpacityScale",
    "ShapeScale", "OrdinalAxis", "QuantitativeAxis",
    "TemporalAxis", "LegendDef", "GuideLine", "FacetGrid",
    "AnnotationText", "AnnotationRegion"
]

# ── SVAS Weights ─────────────────────────────────────────────────
ALPHA_SEM = 0.40      # semantic fidelity weight
BETA_VIS = 0.40       # visual completeness weight
GAMMA_INT = 0.20      # interaction accessibility weight
TAU_SEM = 0.70        # semantic threshold for filtering

# Visual loss weights for MS-GRP Stage 3
LAMBDA_PALETTE = 0.4
LAMBDA_CLUTTER = 0.35
LAMBDA_ACCESSIBILITY = 0.25

# ── MS-GRP Configuration ─────────────────────────────────────────
N_VISUAL_REFINE_STEPS = 50
REINFORCE_N_SAMPLES = 16
REINFORCE_SIGMA_INIT = 0.1
REINFORCE_SIGMA_DECAY = 0.95
REINFORCE_LR = 0.01
CHAOS_INJECTION_RATE = 0.05  # probability of injecting exploration noise

# ── Dataset ──────────────────────────────────────────────────────
DATASET_SIZE = 10000
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Synthetic data generation
N_SYNTHETIC_TEMPLATES = 7000  # template-driven
N_LLM_ENHANCED = 2000         # LLM-varied
N_EXPERT_ANNOTATED = 1000     # human-expert

# ── LLM ──────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE_CIF = 0.3
LLM_TEMPERATURE_GEN = 0.7
LLM_MAX_TOKENS = 2048

# ── Visual Refinement Model (~4M params) ─────────────────────────
VISUAL_REFINER_D_MODEL = 64
VISUAL_REFINER_N_LAYERS = 4
VISUAL_REFINER_DROPOUT = 0.1

# ── API ──────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8001
API_WORKERS = 1

# ── Paths ────────────────────────────────────────────────────────
from pathlib import Path
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR.parent / "checkpoints" / "chartforge"
RESULTS_DIR = BASE_DIR.parent / "experiment_results_cf"

# ── Rendering ────────────────────────────────────────────────────
# Default style for generated charts
DEFAULT_STYLE = {
    "color_scheme": "tableau10",
    "font_family": "sans-serif",
    "font_size": 12,
    "title_font_size": 16,
    "background": "#ffffff",
    "grid": True,
    "grid_alpha": 0.3,
}
