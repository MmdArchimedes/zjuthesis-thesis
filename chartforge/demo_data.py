"""
Synthetic provincial economic data matching thesis Chapter 5.
30 provinces × 9 years (2014–2022), with DEL, ES, PGDP, URBAN, INDS, TEIN metrics.
"""
import numpy as np

# ── Province & region definitions ──
PROVINCES = [
    "北京","天津","河北","山西","内蒙古","辽宁","吉林","黑龙江",
    "上海","江苏","浙江","安徽","福建","江西","山东","河南",
    "湖北","湖南","广东","广西","海南","重庆","四川","贵州",
    "云南","陕西","甘肃","青海","宁夏","新疆",
]

REGION_MAP = {
    "北京":"东部","天津":"东部","河北":"东部","上海":"东部","江苏":"东部",
    "浙江":"东部","福建":"东部","山东":"东部","广东":"东部","海南":"东部",
    "山西":"中部","安徽":"中部","江西":"中部","河南":"中部","湖北":"中部","湖南":"中部",
    "内蒙古":"西部","广西":"西部","重庆":"西部","四川":"西部","贵州":"西部",
    "云南":"西部","陕西":"西部","甘肃":"西部","青海":"西部","宁夏":"西部","新疆":"西部",
    "辽宁":"东北","吉林":"东北","黑龙江":"东北",
}

YEARS = list(range(2014, 2023))
N_PROVINCES = 30
N_YEARS = 9

np.random.seed(42)

# ── Generate realistic economic data ──
def _gen_metric(base, trend, spread, spatial_noise=True):
    """Generate province×year panel with realistic variation."""
    data = {}
    for i, prov in enumerate(PROVINCES):
        prov_base = base * (0.5 + 0.5 * (i / N_PROVINCES)) if spatial_noise else base
        vals = []
        for t, year in enumerate(YEARS):
            val = prov_base + trend * t + spread * np.random.randn()
            vals.append(max(0.01, round(val, 4)))
        data[prov] = vals
    return data

# DEL: Digital Economy Level index (0~1 scale)
DEL_DATA = _gen_metric(base=0.25, trend=0.055, spread=0.04)

# ES: Energy Structure index (0~1 scale, reverse-U relationship)
ES_DATA = _gen_metric(base=0.35, trend=0.04, spread=0.05)

# PGDP: Per capita GDP (万元)
PGDP_DATA = _gen_metric(base=3.5, trend=0.45, spread=0.8)

# URBAN: Urbanization rate (0~1)
URBAN_DATA = _gen_metric(base=0.48, trend=0.025, spread=0.03)

# INDS: Industrial structure (secondary industry share)
INDS_DATA = {}
for prov in PROVINCES:
    base = 0.42 - 0.15 * (np.random.rand())
    vals = [max(0.15, min(0.55, base - 0.012 * t + 0.02 * np.random.randn()))
            for t in range(N_YEARS)]
    INDS_DATA[prov] = [round(v, 4) for v in vals]

# TEIN: Technology innovation index
TEIN_DATA = _gen_metric(base=0.15, trend=0.04, spread=0.035)

# ── Derived fields ──
DEL_SQ = {p: [round(v**2, 6) for v in DEL_DATA[p]] for p in PROVINCES}

# ES predicted (for scatter/regression display)
ES_PRED = {}
for p in PROVINCES:
    del_vals = DEL_DATA[p]
    es_vals = ES_DATA[p]
    ES_PRED[p] = [round(0.15 + 1.2 * d - 0.8 * d**2 + 0.02 * np.random.randn(), 4)
                  for d in del_vals]

# Region-level aggregated data
def _yearly_avg(predicate):
    """Compute yearly mean across provinces matching predicate."""
    provs = [p for p in PROVINCES if predicate(p)]
    return [round(np.mean([DEL_DATA[p][t] for p in provs]), 4) for t in range(N_YEARS)]

REGION_DATA = {
    "东部": {y: _yearly_avg_multi(y) for y in range(N_YEARS)},
}

def _yearly_avg_multi(yr_idx):
    """Return dict of region→DEL avg for a given year index."""
    result = {}
    for region in ["东部","中部","西部","东北"]:
        provs = [p for p in PROVINCES if REGION_MAP[p] == region]
        result[region] = round(np.mean([DEL_DATA[p][yr_idx] for p in provs]), 4)
    return result

# Simple region-level yearly DEL
REGION_DEL = {}
for region in ["东部","中部","西部","东北"]:
    REGION_DEL[region] = _yearly_avg(lambda p: REGION_MAP[p] == region)

REGION_ES = {}
for region in ["东部","中部","西部","东北"]:
    provs = [p for p in PROVINCES if REGION_MAP[p] == region]
    REGION_ES[region] = [round(np.mean([ES_DATA[p][t] for p in provs]), 4)
                         for t in range(N_YEARS)]

# ── Unified data access ──
ALL_DATA = {
    "DEL": DEL_DATA,
    "ES": ES_DATA,
    "PGDP": PGDP_DATA,
    "URBAN": URBAN_DATA,
    "INDS": INDS_DATA,
    "TEIN": TEIN_DATA,
    "DEL_sq": DEL_SQ,
    "ES_predicted": ES_PRED,
}

def get_series(metric: str, provinces=None, year=None):
    """Get data series for chart rendering.

    Returns:
        If year specified: list of (province, value) pairs
        If provinces specified: list of (year, value) pairs
        If both: single value
    """
    data = ALL_DATA.get(metric, DEL_DATA)

    if year is not None and provinces is not None:
        # Single value
        yr_idx = YEARS.index(year) if year in YEARS else year - 2014
        if isinstance(provinces, str):
            return data[provinces][yr_idx]
        return {p: data[p][yr_idx] for p in provinces}

    if year is not None:
        # All provinces, one year
        yr_idx = YEARS.index(year) if year in YEARS else year - 2014
        return [(p, data[p][yr_idx]) for p in PROVINCES]

    if provinces is not None:
        # One province, all years
        if isinstance(provinces, str):
            return [(y, data[provinces][YEARS.index(y)]) for y in YEARS]
        # Multiple provinces, all years
        result = {}
        for p in provinces:
            result[p] = [(y, data[p][YEARS.index(y)]) for y in YEARS]
        return result

    # Default: all provinces, latest year
    return [(p, data[p][-1]) for p in PROVINCES]
