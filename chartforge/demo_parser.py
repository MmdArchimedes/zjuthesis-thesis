"""
Template-based NL→Chart parser for ChartForge demo.
No LLM required — uses regex matching against common query patterns.
"""
import re
from dataclasses import dataclass, field
from typing import Optional, List

from .config import CHART_TYPES, PROVINCES, REGIONS, YEARS, DOMAIN_FIELDS

# ── Query templates ──
QUERY_TEMPLATES = [
    # Pattern: "[chart] of [metric] by [dimension] in [year]"
    (r"(?P<chart>柱状图|bar|line|折线图|散点图|scatter|面积图|area|热力图|heatmap|饼图|pie|雷达图|radar|桑基图|sankey|树图|treemap|箱线图|boxplot|仪表盘|gauge|漏斗图|funnel).*?"
     r"(?P<metric>DEL|ES|PGDP|URBAN|INDS|TEIN|数字经济|能源结构).*?"
     r"(?:按|by|按照|分)?(?P<dimension>省份|province|区域|region|年份|year)?.*?"
     r"(?:在|in|于)?(?P<year>\d{4})?"),
]

CHART_NAME_MAP = {
    "柱状图":"bar","bar":"bar",
    "折线图":"line","line":"line",
    "散点图":"scatter","scatter":"scatter",
    "面积图":"area","area":"area",
    "热力图":"heatmap","heatmap":"heatmap",
    "饼图":"pie","pie":"pie",
    "雷达图":"radar","radar":"radar",
    "桑基图":"sankey","sankey":"sankey",
    "矩形树图":"treemap","树图":"treemap","treemap":"treemap",
    "箱线图":"boxplot","boxplot":"boxplot",
    "仪表盘":"gauge","gauge":"gauge",
    "漏斗图":"funnel","funnel":"funnel",
}

METRIC_NAME_MAP = {
    "DEL":"DEL","数字经济":"DEL",
    "ES":"ES","能源结构":"ES",
    "PGDP":"PGDP","人均GDP":"PGDP",
    "URBAN":"URBAN","城镇化":"URBAN",
    "INDS":"INDS","产业结构":"INDS",
    "TEIN":"TEIN","技术创新":"TEIN",
}

DIM_MAP = {"省份":"province","province":"province","区域":"region","region":"region","年份":"year","year":"year"}


@dataclass
class ParsedQuery:
    """Result of template-based NL parsing."""
    chart_type: str = "bar"
    metrics: List[str] = field(default_factory=lambda: ["DEL"])
    dimension: str = "province"  # province, region, year
    year: Optional[int] = None
    provinces: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    confidence: float = 0.5
    raw_query: str = ""


def parse_query(nl_query: str) -> ParsedQuery:
    """Parse NL query into structured chart request using template matching.

    Supports patterns:
      - "Show bar chart of DEL by province in 2022"
      - "各省DEL柱状图 2022"
      - "浙江省2014-2022年DEL变化趋势折线图"
      - "DEL vs ES scatter by region"
      - "Compare DEL across eastern provinces as pie chart"
    """
    result = ParsedQuery(raw_query=nl_query)
    q = nl_query.lower().replace(" ", "")

    # ── Detect chart type ──
    for name, ctype in CHART_NAME_MAP.items():
        if name in q or name in nl_query:
            result.chart_type = ctype
            result.confidence += 0.15
            break

    # ── Detect metrics ──
    metrics_found = []
    for name, metric in METRIC_NAME_MAP.items():
        if name.lower() in q:
            if metric not in metrics_found:
                metrics_found.append(metric)
    if metrics_found:
        result.metrics = metrics_found
        result.confidence += 0.15

    # ── Detect dimension ──
    for name, dim in DIM_MAP.items():
        if name in q:
            result.dimension = dim
            result.confidence += 0.1
            break

    # ── Detect year ──
    year_match = re.search(r'(?:19|20)(\d{2})', nl_query)
    if year_match:
        yr = int(year_match.group(0))
        if 2014 <= yr <= 2022:
            result.year = yr
            result.confidence += 0.1

    # ── Detect specific provinces ──
    provs = []
    for p in PROVINCES:
        if p in nl_query:
            provs.append(p)
    if provs:
        result.provinces = provs
        result.confidence += 0.05

    # ── Detect regions ──
    regs = []
    for r in REGIONS:
        if r in nl_query:
            regs.append(r)
    if regs:
        result.regions = regs
        result.confidence += 0.05

    # ── Smart defaults based on chart type ──
    if result.chart_type == "line" and result.year is None:
        result.dimension = "year"  # line charts default to time series
    if result.chart_type == "pie" and result.dimension == "province":
        result.dimension = "region"  # pie better with fewer categories
    if result.chart_type == "scatter" and len(result.metrics) == 1:
        result.metrics = ["DEL", "ES"]  # scatter needs 2 metrics

    return result


# ── Preset queries for the demo ──
PRESET_QUERIES = [
    ("bar",    "📊 2022年各省数字经济指数(DEL)柱状图", "2022年各省DEL柱状图"),
    ("line",   "📈 浙江省2014-2022年DEL与ES变化趋势", "浙江省DEL和ES历年变化折线图"),
    ("scatter","🟢 各省DEL与ES散点图（按区域着色）", "各省DEL与ES散点图按区域着色"),
    ("area",   "🏔️ 2014-2022年全国数字经济与能源结构面积图", "DEL和ES年度趋势面积图"),
    ("heatmap","🔥 各省份各年份DEL热力图", "各省各年份DEL热力图"),
    ("pie",    "🥧 2022年各区域DEL占比饼图", "2022年四大区域DEL饼图"),
    ("radar",  "🎯 浙江省2022年综合指标雷达图", "浙江省DEL ES PGDP URBAN INDS TEIN雷达图"),
    ("sankey", "🌊 能源结构区域流向桑基图", "能源结构从区域到类型的桑基图"),
    ("treemap","🗺️ 2022年各省DEL矩形树图", "各省DEL矩形树图"),
    ("boxplot","📦 各区域DEL分布箱线图（2014-2022）", "各区域历年DEL箱线图"),
    ("gauge",  "⏱️ 2022年全国平均DEL仪表盘", "全国DEL均值仪表盘"),
    ("funnel", "🔻 能源结构转化漏斗图", "能源结构各环节漏斗图"),
]

# Chart type Chinese labels
CHART_LABELS_CN = {
    "bar":"柱状图","line":"折线图","scatter":"散点图","area":"面积图",
    "heatmap":"热力图","pie":"饼图","radar":"雷达图","sankey":"桑基图",
    "treemap":"矩形树图","boxplot":"箱线图","gauge":"仪表盘","funnel":"漏斗图",
}
