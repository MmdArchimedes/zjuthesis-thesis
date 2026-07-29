"""
ChartForge Interactive Demo Server — AI chart generation from NL queries.
No LLM API key required. Supports all 12 chart types with real provincial data.

Usage:
  python chartforge/demo_server.py [--port 8765]

Then open: http://localhost:8765
"""
import json
import sys
import os
import re
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local constants (avoid broken imports from demo_data)
PROVINCES = ["北京","天津","河北","山西","内蒙古","辽宁","吉林","黑龙江","上海","江苏","浙江","安徽","福建","江西","山东","河南","湖北","湖南","广东","广西","海南","重庆","四川","贵州","云南","陕西","甘肃","青海","宁夏","新疆"]
REGIONS = ["东部","中部","西部","东北"]
YEARS = list(range(2014, 2023))
CHART_LABELS_CN = {"bar":"柱状图","line":"折线图","scatter":"散点图","area":"面积图","heatmap":"热力图","pie":"饼图","radar":"雷达图","sankey":"桑基图","treemap":"矩形树图","boxplot":"箱线图","gauge":"仪表盘","funnel":"漏斗图"}

# ── Load real panel data ──
def load_panel_data() -> Dict:
    """Load panel_data.csv."""
    candidates = [
        "workspace/SDCR_Vis_System/Unity/Assets/Resources/Data/panel_data.csv",
        "../workspace/SDCR_Vis_System/Unity/Assets/Resources/Data/panel_data.csv",
    ]
    base = Path(__file__).parent.parent
    csv_path = None
    for c in candidates:
        p = base / c
        if p.exists():
            csv_path = str(p)
            break

    if csv_path is None:
        raise FileNotFoundError("panel_data.csv not found")

    import csv
    data = {}
    region_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prov = row["province_name"]
            year = int(row["year"])
            for metric in ["DEL", "ES", "PGDP", "URBAN", "INDS", "LEOP", "FDE", "DEN", "TEIN"]:
                if metric not in data:
                    data[metric] = {}
                if prov not in data[metric]:
                    data[metric][prov] = {}
                data[metric][prov][year] = float(row[metric])
            region_map[prov] = row["region_tag"]
    data["_region"] = region_map
    return data


# ── Enhanced NL Parser ──
def smart_parse(query: str) -> Dict:
    """Enhanced parser: returns dict with chart_type, metrics, dimension, year, provinces, regions, confidence."""
    result = {
        "chart_type": "bar",
        "metrics": ["DEL"],
        "dimension": "province",
        "year": None,
        "provinces": [],
        "regions": [],
        "confidence": 0.5,
    }
    q = query.replace(" ", "").lower()
    raw = query

    # ── Chart type detection (expanded) ──
    chart_patterns = [
        (["柱状图","柱形图","bar","条形图","直方图"], "bar"),
        (["折线图","趋势图","line","曲线图","走势"], "line"),
        (["散点图","scatter","相关性","相关图"], "scatter"),
        (["面积图","area","区域图","堆积"], "area"),
        (["热力图","heatmap","热图","矩阵"], "heatmap"),
        (["饼图","pie","占比","比例","份额"], "pie"),
        (["雷达图","radar","蜘蛛图","多维"], "radar"),
        (["桑基图","sankey","流向","桑葚"], "sankey"),
        (["树图","treemap","矩形树","层级"], "treemap"),
        (["箱线图","boxplot","箱型","分布"], "boxplot"),
        (["仪表盘","gauge","仪表","表盘"], "gauge"),
        (["漏斗图","funnel","漏斗","转化"], "funnel"),
    ]
    for keywords, ctype in chart_patterns:
        if any(kw in q for kw in keywords):
            result["chart_type"] = ctype
            result["confidence"] += 0.2
            break

    # ── Metric detection ──
    metric_patterns = [
        (["del","数字经济","数字"], "DEL"),
        (["es","能源结构","能源"], "ES"),
        (["pgdp","人均gdp","人均生产总值"], "PGDP"),
        (["urban","城镇化","城市化"], "URBAN"),
        (["inds","产业结构","工业"], "INDS"),
        (["tein","技术创新","研发","科技"], "TEIN"),
    ]
    metrics = []
    for keywords, metric in metric_patterns:
        if any(kw in q for kw in keywords):
            if metric not in metrics:
                metrics.append(metric)
    if metrics:
        result["metrics"] = metrics
        result["confidence"] += 0.15

    # ── Dimension detection ──
    if any(w in raw for w in ["省份","各省","各省份","分省","province","哪个省"]):
        result["dimension"] = "province"
    elif any(w in raw for w in ["区域","地区","东部","西部","中部","东北","region"]):
        result["dimension"] = "region"
    elif any(w in raw for w in ["年份","年度","历年","year","趋势","时间","变化"]):
        result["dimension"] = "year"
        result["confidence"] += 0.05

    # ── Year detection ──
    year_match = re.search(r'(20[12]\d)', raw)
    if year_match:
        yr = int(year_match.group(1))
        if 2014 <= yr <= 2022:
            result["year"] = yr
            result["confidence"] += 0.1

    # ── Province detection ──
    for p in PROVINCES:
        if p in raw:
            result["provinces"].append(p)
    if result["provinces"]:
        result["confidence"] += 0.1

    # ── Region detection ──
    for r in REGIONS:
        if r in raw:
            result["regions"].append(r)
    if result["regions"]:
        result["confidence"] += 0.05

    # ── Smart defaults ──
    ct = result["chart_type"]
    if ct == "line" and result["dimension"] == "province" and not result["year"]:
        result["dimension"] = "year"
    if ct in ("pie", "treemap") and result["dimension"] == "province" and len(result["metrics"]) == 1:
        pass  # keep provinces
    if ct == "scatter" and len(result["metrics"]) < 2:
        if "ES" not in result["metrics"]:
            result["metrics"].append("ES")
        elif "DEL" not in result["metrics"]:
            result["metrics"].append("DEL")

    result["confidence"] = min(result["confidence"], 1.0)
    return result


# ── Chart Renderers (same as dashboard, adapted for server) ──

def render_to_echarts(parsed: Dict, data: Dict) -> Dict:
    """Generate ECharts option from parsed query."""
    ct = parsed["chart_type"]

    if ct == "bar":
        return _render_bar(parsed, data)
    elif ct == "line":
        return _render_line(parsed, data)
    elif ct == "scatter":
        return _render_scatter(parsed, data)
    elif ct == "area":
        return _render_area(parsed, data)
    elif ct == "heatmap":
        return _render_heatmap(parsed, data)
    elif ct == "pie":
        return _render_pie(parsed, data)
    elif ct == "radar":
        return _render_radar(parsed, data)
    elif ct == "sankey":
        return _render_sankey(parsed, data)
    elif ct == "treemap":
        return _render_treemap(parsed, data)
    elif ct == "boxplot":
        return _render_boxplot(parsed, data)
    elif ct == "gauge":
        return _render_gauge(parsed, data)
    elif ct == "funnel":
        return _render_funnel(parsed, data)
    return _render_bar(parsed, data)


def _render_bar(p, data):
    metric = p["metrics"][0]
    year = p["year"] or 2022
    metric_data = data.get(metric, data.get("DEL", {}))
    provinces = p["provinces"] if p["provinces"] else PROVINCES
    values, labels = [], []
    for prov in provinces:
        if prov in metric_data and year in metric_data[prov]:
            values.append(round(metric_data[prov][year], 4))
            labels.append(prov)
    return {
        "title": {"text": f"{year}年各省{metric}", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": labels, "axisLabel": {"rotate": 45, "fontSize": 10}},
        "yAxis": {"type": "value", "name": metric},
        "series": [{"type": "bar", "data": values,
            "itemStyle": {"color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                "colorStops": [{"offset": 0, "color": "#667eea"}, {"offset": 1, "color": "#764ba2"}]}}}],
        "grid": {"bottom": 100},
        "dataZoom": [{"type": "slider", "start": 0, "end": 100}],
    }

def _render_line(p, data):
    provinces = p["provinces"] if p["provinces"] else ["浙江", "江苏", "广东", "山东", "河南"]
    colors = ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de"]
    series = []
    for i, prov in enumerate(provinces[:5]):
        vals = [round(data.get(p["metrics"][0], {}).get(prov, {}).get(y, 0), 4) for y in YEARS]
        series.append({"name": prov, "type": "line", "data": vals, "smooth": True,
                        "itemStyle": {"color": colors[i % 5]}})
    return {
        "title": {"text": f"{'、'.join(p['metrics'])}变化趋势", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0, "type": "scroll"},
        "xAxis": {"type": "category", "data": [str(y) for y in YEARS]},
        "yAxis": {"type": "value", "name": p["metrics"][0]},
        "series": series,
        "grid": {"bottom": 80},
    }

def _render_scatter(p, data):
    m1, m2 = p["metrics"][0], p["metrics"][1] if len(p["metrics"]) > 1 else "ES"
    year = p["year"] or 2022
    region_map = data.get("_region", {})
    region_colors = {"东部": "#5470c6", "中部": "#91cc75", "西部": "#fac858", "东北": "#ee6666"}
    series_list = []
    for region, color in region_colors.items():
        pts = []
        for prov in PROVINCES:
            if region_map.get(prov) == region:
                v1 = data.get(m1, {}).get(prov, {}).get(year)
                v2 = data.get(m2, {}).get(prov, {}).get(year)
                if v1 and v2:
                    pts.append([round(v1, 4), round(v2, 4), prov])
        if pts:
            series_list.append({"name": region, "type": "scatter", "data": pts, "symbolSize": 12,
                                 "itemStyle": {"color": color}})
    return {
        "title": {"text": f"{year}年{m1} vs {m2}", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {"trigger": "item",
            "formatter": "function(p){return p.value[2]+'<br/>'+'{m1}: '+p.value[0].toFixed(4)+'<br/>'+'{m2}: '+p.value[1].toFixed(4);}"},
        "legend": {"bottom": 0},
        "xAxis": {"type": "value", "name": m1}, "yAxis": {"type": "value", "name": m2},
        "series": series_list,
    }

def _render_area(p, data):
    colors = ["#5470c6", "#91cc75"]
    series = []
    for i, metric in enumerate(p["metrics"][:2]):
        vals = [round(float(np.mean([data.get(metric, {}).get(prov, {}).get(y, 0)
                    for prov in PROVINCES])), 4) for y in YEARS]
        series.append({"name": metric, "type": "line", "data": vals, "smooth": True,
                        "areaStyle": {"opacity": 0.6}, "stack": "total",
                        "itemStyle": {"color": colors[i]}})
    return {
        "title": {"text": "全国年度趋势面积图", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0},
        "xAxis": {"type": "category", "data": [str(y) for y in YEARS], "boundaryGap": False},
        "yAxis": {"type": "value"},
        "series": series,
    }

def _render_heatmap(p, data):
    metric = p["metrics"][0]
    hdata, xl, yl = [], [str(y) for y in YEARS], PROVINCES
    for pi, prov in enumerate(PROVINCES):
        for yi, year in enumerate(YEARS):
            v = data.get(metric, {}).get(prov, {}).get(year)
            if v:
                hdata.append([yi, pi, round(v, 4)])
    return {
        "title": {"text": f"{metric}热力图", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {"position": "top"},
        "xAxis": {"type": "category", "data": xl}, "yAxis": {"type": "category", "data": yl, "axisLabel": {"fontSize": 8}},
        "visualMap": {"min": 0.1, "max": 0.8, "calculable": True, "orient": "horizontal", "left": "center", "bottom": 0,
                       "inRange": {"color": ["#f0f9e8", "#bae4bc", "#7bccc4", "#43a2ca", "#0868ac"]}},
        "series": [{"type": "heatmap", "data": hdata, "label": {"show": False}}],
        "grid": {"bottom": "15%", "left": "12%", "right": "5%", "top": "10%"},
    }

def _render_pie(p, data):
    metric = p["metrics"][0]; year = p["year"] or 2022
    region_map = data.get("_region", {})
    region_vals = {}
    for prov in PROVINCES:
        region = region_map.get(prov, "其他")
        v = data.get(metric, {}).get(prov, {}).get(year, 0)
        region_vals[region] = region_vals.get(region, 0) + v
    pie_data = [{"name": r, "value": round(v, 2)} for r, v in region_vals.items()]
    return {
        "title": {"text": f"{year}年各区域{metric}占比", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0},
        "series": [{"type": "pie", "radius": ["40%", "70%"], "data": pie_data,
                     "label": {"formatter": "{b}\n{d}%"}}],
    }

def _render_radar(p, data):
    year = p["year"] or 2022; prov = p["provinces"][0] if p["provinces"] else "浙江"
    metrics = ["DEL", "ES", "PGDP", "URBAN", "INDS", "TEIN"]
    max_vals = {"DEL": 0.8, "ES": 0.8, "PGDP": 12.0, "URBAN": 1.0, "INDS": 4.5, "TEIN": 6.0}
    vals = [round(data.get(m, {}).get(prov, {}).get(year, 0) / max_vals.get(m, 1.0), 4) for m in metrics]
    return {
        "title": {"text": f"{prov}{year}年综合指标", "left": "center", "textStyle": {"fontSize": 16}},
        "radar": {"indicator": [{"name": m, "max": 1.0} for m in metrics]},
        "series": [{"type": "radar", "data": [{"value": vals, "name": prov}], "areaStyle": {"opacity": 0.3}}],
    }

def _render_sankey(p, data):
    year = p["year"] or 2022; region_map = data.get("_region", {})
    nodes = [{"name": r} for r in REGIONS] + [{"name": t} for t in ["煤炭","石油","天然气","可再生","电力"]]
    links = []
    for region in REGIONS:
        provs = [pr for pr in PROVINCES if region_map.get(pr) == region]
        d = float(np.mean([data.get("DEL", {}).get(pr, {}).get(year, 0.2) for pr in provs]))
        e = float(np.mean([data.get("ES", {}).get(pr, {}).get(year, 0.3) for pr in provs]))
        links.append({"source": region, "target": "煤炭", "value": round(max(0.5 - d * 0.4, 0.1) * 30, 1)})
        links.append({"source": region, "target": "石油", "value": round(max(0.2 - e * 0.15, 0.05) * 25, 1)})
        links.append({"source": region, "target": "天然气", "value": round(e * 3, 1)})
        links.append({"source": region, "target": "可再生", "value": round(d * 5, 1)})
        links.append({"source": region, "target": "电力", "value": round(d * 4, 1)})
    return {
        "title": {"text": f"{year}年能源流向", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {"trigger": "item"},
        "series": [{"type": "sankey", "layout": "none", "data": nodes, "links": links,
                     "lineStyle": {"color": "gradient", "curveness": 0.5}}],
    }

def _render_treemap(p, data):
    metric = p["metrics"][0]; year = p["year"] or 2022
    region_map = data.get("_region", {})
    td = []
    for region in REGIONS:
        children = []
        for prov in PROVINCES:
            if region_map.get(prov) == region:
                v = data.get(metric, {}).get(prov, {}).get(year)
                if v: children.append({"name": prov, "value": round(v, 4)})
        if children: td.append({"name": region, "children": children})
    return {
        "title": {"text": f"{year}年{metric}树图", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {},
        "series": [{"type": "treemap", "data": td, "label": {"show": True, "formatter": "{b}"},
                     "levels": [{"itemStyle": {"borderColor": "#555"}}, {"colorMappingBy": "id", "itemStyle": {"gapWidth": 1}}]}],
    }

def _render_boxplot(p, data):
    metric = p["metrics"][0]; region_map = data.get("_region", {})
    xl, bd = [], []
    for region in REGIONS:
        all_vals = [data.get(metric, {}).get(pr, {}).get(y, 0)
                    for pr in PROVINCES if region_map.get(pr) == region
                    for y in YEARS]
        if all_vals:
            xl.append(region); arr = np.array(all_vals)
            bd.append([round(float(np.min(arr)),4), round(float(np.percentile(arr,25)),4),
                        round(float(np.median(arr)),4), round(float(np.percentile(arr,75)),4),
                        round(float(np.max(arr)),4)])
    return {
        "title": {"text": f"各区域{metric}分布", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {}, "xAxis": {"type": "category", "data": xl}, "yAxis": {"type": "value"},
        "series": [{"type": "boxplot", "data": bd, "itemStyle": {"color": "#5470c6"}}],
    }

def _render_gauge(p, data):
    metric = p["metrics"][0]; year = p["year"] or 2022
    vals = [data.get(metric, {}).get(pr, {}).get(year, 0) for pr in PROVINCES]
    return {
        "title": {"text": f"{year}年全国{metric}", "left": "center", "textStyle": {"fontSize": 16}},
        "series": [{"type": "gauge", "startAngle": 210, "endAngle": -30, "radius": "80%",
                     "min": 0.0, "max": 0.8,
                     "axisLine": {"lineStyle": {"width": 20,
                         "color": [[0.3, "#ee6666"], [0.6, "#fac858"], [1, "#91cc75"]]}},
                     "detail": {"fontSize": 24},
                     "data": [{"value": round(float(np.mean(vals)), 4), "name": metric}]}],
    }

def _render_funnel(p, data):
    year = p["year"] or 2022
    d = float(np.mean([data.get("DEL", {}).get(pr, {}).get(year, 0.2) for pr in PROVINCES]))
    stages = ["传统能源消费","能效提升","清洁替代","碳达峰","碳中和"]
    values = [100, round(100-d*30,1), round(70-d*40,1), round(50-d*35,1), round(30-d*20,1)]
    return {
        "title": {"text": "能源结构转化漏斗", "left": "center", "textStyle": {"fontSize": 16}},
        "tooltip": {"trigger": "item"},
        "series": [{"type": "funnel", "left": "15%", "right": "15%", "top": 60, "bottom": 60,
                     "min": 0, "max": 100, "sort": "descending", "gap": 2,
                     "label": {"show": True, "position": "inside", "fontSize": 14},
                     "data": [{"name": s, "value": v} for s, v in zip(stages, values)]}],
    }


# ── HTTP Server ──
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ChartForge AI — 智能图表生成</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f0f2f5}
header{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:20px;text-align:center}
header h1{font-size:22px;margin-bottom:4px}
header p{font-size:13px;opacity:.85}
.container{max-width:1000px;margin:0 auto;padding:20px}
.query-box{background:#fff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.query-box input{width:100%;padding:14px 18px;font-size:15px;border:2px solid #e0e0e0;border-radius:8px;outline:none;transition:border .2s}
.query-box input:focus{border-color:#667eea}
.query-box .hints{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.query-box .hint{background:#f0f2ff;color:#667eea;padding:6px 14px;border-radius:20px;font-size:12px;cursor:pointer;transition:all .2s;border:1px solid transparent}
.query-box .hint:hover{background:#e0e4ff;border-color:#667eea}
.query-box .actions{display:flex;gap:10px;margin-top:14px;align-items:center}
.query-box .btn{padding:10px 28px;font-size:14px;border:none;border-radius:8px;cursor:pointer;transition:all .2s}
.btn-gen{background:#667eea;color:#fff}.btn-gen:hover{background:#5a6fd6}
.btn-preset{background:#f5f5f5;color:#555}.btn-preset:hover{background:#e8e8e8}
.result-box{background:#fff;border-radius:12px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.06);min-height:500px}
.result-header{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.result-badge{background:#667eea;color:#fff;padding:4px 14px;border-radius:14px;font-size:12px}
.result-svas{background:#e8f5e9;color:#2e7d32;padding:4px 10px;border-radius:8px;font-size:12px}
.result-query{color:#888;font-size:13px;font-style:italic;flex:1}
.chart-area{width:100%;height:500px}
.info-box{margin-top:12px;background:#fafafa;padding:12px;border-radius:8px;font-size:12px}
.loading{text-align:center;padding:60px;color:#aaa}
.error{color:#e74c3c;text-align:center;padding:40px}
.presets-section{margin-top:16px}
.presets-section h3{font-size:14px;color:#666;margin-bottom:8px}
.preset-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:6px}
.preset-btn{background:#f8f9fa;border:1px solid #e0e0e0;padding:8px 12px;border-radius:6px;font-size:12px;cursor:pointer;text-align:left;transition:all .2s}
.preset-btn:hover{background:#f0f2ff;border-color:#667eea}
footer{text-align:center;color:#aaa;font-size:12px;padding:20px}
</style>
</head>
<body>
<header>
<h1>ChartForge AI — 智能图表生成</h1>
<p>输入自然语言描述，AI自动生成数据图表 | 30省×9年真实经济数据 | 12种图表类型 | 无需API密钥</p>
</header>
<div class="container">
<div class="query-box">
    <input type="text" id="queryInput" placeholder="例如：显示2022年浙江省和广东省的DEL柱状图、ES与DEL的散点图按区域着色、东部省份的能源结构饼图..."
           onkeydown="if(event.key==='Enter')generate()">
    <div class="hints">
        <span class="hint" onclick="setQuery('显示2022年各省DEL柱状图')">📊 柱状图</span>
        <span class="hint" onclick="setQuery('浙江省2014到2022年DEL变化趋势折线图')">📈 折线图</span>
        <span class="hint" onclick="setQuery('各省DEL与ES散点图按区域着色')">🟢 散点图</span>
        <span class="hint" onclick="setQuery('DEL和ES年度趋势面积图')">🏔️ 面积图</span>
        <span class="hint" onclick="setQuery('各省份各年份DEL热力图')">🔥 热力图</span>
        <span class="hint" onclick="setQuery('2022年各区域DEL占比饼图')">🥧 饼图</span>
        <span class="hint" onclick="setQuery('浙江省2022年综合指标雷达图')">🎯 雷达图</span>
        <span class="hint" onclick="setQuery('能源结构区域流向桑基图')">🌊 桑基图</span>
        <span class="hint" onclick="setQuery('2022年各省DEL矩形树图')">🗺️ 树图</span>
        <span class="hint" onclick="setQuery('各区域DEL分布箱线图')">📦 箱线图</span>
        <span class="hint" onclick="setQuery('2022年全国平均DEL仪表盘')">⏱️ 仪表盘</span>
        <span class="hint" onclick="setQuery('能源结构转化漏斗图')">🔻 漏斗图</span>
    </div>
    <div class="actions">
        <button class="btn btn-gen" onclick="generate()">⚡ 生成图表</button>
        <span style="color:#aaa;font-size:12px">支持中英文混合查询，可指定省份、年份、指标和图表类型</span>
    </div>
</div>
<div class="result-box" id="resultBox">
    <div class="loading" id="placeholder">👆 输入查询或点击上方标签开始生成图表</div>
</div>
<details style="margin-top:16px">
    <summary style="cursor:pointer;color:#667eea;font-size:13px">📋 所有预设查询</summary>
    <div class="preset-grid" style="margin-top:8px">
        <button class="preset-btn" onclick="setQuery('2022年各省DEL柱状图')">📊 各省DEL柱状图</button>
        <button class="preset-btn" onclick="setQuery('广东江苏浙江山东河南DEL历年变化折线图')">📈 五省DEL趋势</button>
        <button class="preset-btn" onclick="setQuery('各省DEL与ES散点图按区域着色2022')">🟢 DEL vs ES散点</button>
        <button class="preset-btn" onclick="setQuery('全国DEL和ES年度趋势面积图')">🏔️ 全国趋势面积</button>
        <button class="preset-btn" onclick="setQuery('各省各年份DEL热力图')">🔥 DEL热力图</button>
        <button class="preset-btn" onclick="setQuery('2022年各区域ES占比饼图')">🥧 区域ES饼图</button>
        <button class="preset-btn" onclick="setQuery('上海2022年综合指标雷达图')">🎯 上海雷达图</button>
        <button class="preset-btn" onclick="setQuery('2022年能源流向桑基图')">🌊 能源桑基图</button>
        <button class="preset-btn" onclick="setQuery('2022年各省ES矩形树图')">🗺️ ES树图</button>
        <button class="preset-btn" onclick="setQuery('各区域ES分布箱线图')">📦 区域ES箱线</button>
        <button class="preset-btn" onclick="setQuery('2022年全国平均ES仪表盘')">⏱️ 全国ES仪表</button>
        <button class="preset-btn" onclick="setQuery('能源结构转化漏斗图2022')">🔻 能源漏斗图</button>
        <button class="preset-btn" onclick="setQuery('北京天津河北DEL对比柱状图2020')">📊 京津冀DEL</button>
        <button class="preset-btn" onclick="setQuery('东部省份INDS变化趋势折线图')">📈 东部产业结构</button>
        <button class="preset-btn" onclick="setQuery('各省PGDP与URBAN散点图')">🟢 PGDP vs URBAN</button>
        <button class="preset-btn" onclick="setQuery('西部省份TEIN柱状图2021')">📊 西部TEIN</button>
    </div>
</details>
</div>
<footer>ChartForge AI © 2026 | 基于关键词+规则引擎的智能NL解析 | 30省份×9年面板数据 | ECharts渲染</footer>
<script>
var chart = null;
var chartDom = null;

function setQuery(q) {
    document.getElementById('queryInput').value = q;
    generate();
}

async function generate() {
    var query = document.getElementById('queryInput').value.trim();
    if (!query) return;

    var box = document.getElementById('resultBox');
    box.innerHTML = '<div class="loading">⏳ 正在生成图表...</div>';

    try {
        var resp = await fetch('/api/generate?query=' + encodeURIComponent(query));
        var data = await resp.json();

        if (data.error) {
            box.innerHTML = '<div class="error">❌ ' + data.error + '</div>';
            return;
        }

        var html = '<div class="result-header">' +
            '<span class="result-badge">' + data.chart_label + '</span>' +
            '<span class="result-query">"' + query + '"</span>' +
            '<span class="result-svas">置信度: ' + (data.confidence * 100).toFixed(0) + '%</span>' +
            '</div>' +
            '<div class="chart-area" id="liveChart"></div>' +
            '<div class="info-box">' +
            '<strong>CIF解析:</strong> 图表=' + data.chart_type +
            ' | 指标=' + data.metrics.join(', ') +
            ' | 维度=' + data.dimension +
            (data.year ? ' | 年份=' + data.year : '') +
            (data.provinces.length ? ' | 省份=' + data.provinces.join(', ') : '') +
            '</div>';

        box.innerHTML = html;

        // Render chart
        setTimeout(function() {
            chartDom = document.getElementById('liveChart');
            if (chartDom && data.option) {
                if (chart) chart.dispose();
                chart = echarts.init(chartDom);
                chart.setOption(data.option);
                window.addEventListener('resize', function() { chart.resize(); });
            }
        }, 50);

    } catch(e) {
        box.innerHTML = '<div class="error">❌ 网络错误: ' + e.message + '</div>';
    }
}

// Auto-generate on page load if there's a query param
window.onload = function() {
    var params = new URLSearchParams(window.location.search);
    var q = params.get('query');
    if (q) {
        document.getElementById('queryInput').value = q;
        generate();
    }
};
</script>
</body>
</html>"""


class ChartForgeHandler(BaseHTTPRequestHandler):
    panel_data = None

    @classmethod
    def init_data(cls):
        if cls.panel_data is None:
            cls.panel_data = load_panel_data()
            print(f"Loaded {len(PROVINCES)} provinces × {len(YEARS)} years panel data")

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_html(HTML_PAGE)
        elif parsed.path == "/api/generate":
            self._handle_generate(parsed)
        elif parsed.path == "/api/health":
            self._serve_json({"status": "ok", "charts": len(CHART_LABELS_CN)})
        else:
            self.send_error(404)

    def _serve_html(self, html: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _serve_json(self, data: Dict):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _handle_generate(self, parsed):
        params = parse_qs(parsed.query)
        query = params.get("query", [""])[0].strip()

        if not query:
            self._serve_json({"error": "请提供查询参数 ?query=..."})
            return

        try:
            # Parse NL query
            parsed_q = smart_parse(query)

            # Render chart
            option = render_to_echarts(parsed_q, self.panel_data)

            self._serve_json({
                "chart_type": parsed_q["chart_type"],
                "chart_label": CHART_LABELS_CN.get(parsed_q["chart_type"], parsed_q["chart_type"]),
                "metrics": parsed_q["metrics"],
                "dimension": parsed_q["dimension"],
                "year": parsed_q["year"],
                "provinces": parsed_q["provinces"],
                "regions": parsed_q["regions"],
                "confidence": parsed_q["confidence"],
                "option": option,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._serve_json({"error": f"生成失败: {str(e)}"})

    def log_message(self, format, *args):
        print(f"[{args[0]}] {args[1]} {args[2]}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ChartForge Interactive Demo Server")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    args = parser.parse_args()

    # Load data
    ChartForgeHandler.init_data()

    server = HTTPServer((args.host, args.port), ChartForgeHandler)
    print(f"\n{'='*60}")
    print(f"  ChartForge AI Demo Server")
    print(f"  Open: http://localhost:{args.port}")
    print(f"  12 chart types | 30 provinces | Real economic data")
    print(f"  Type NL queries to generate charts!")
    print(f"{'='*60}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

if __name__ == "__main__":
    main()
