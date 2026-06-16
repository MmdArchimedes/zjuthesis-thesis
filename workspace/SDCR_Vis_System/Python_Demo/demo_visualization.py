"""
SDCR--Vis Python Demo — Interactive China Provincial Data Visualization
=====================================================================
Implements the thesis SDCR--Vis pipeline for immediate desktop demo.
State-driven conditional refresh: year, indicator, region filter,
province selection → synchronized map, timeline, result panels.

Run: pip install plotly pandas kaleido ; python demo_visualization.py
"""

import json
import math
import webbrowser
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ═══════════════════════════════════════════════════════════════
# 1. DATA LOADING (mirrors Unity DataManager)
# ═══════════════════════════════════════════════════════════════

def load_panel_data(csv_path: str) -> pd.DataFrame:
    """Load province-year panel data (mirrors DataManager.LoadPanelData)."""
    df = pd.read_csv(csv_path)
    return df


def load_regression_results(json_path: str) -> dict:
    """Load regression results (mirrors DataManager.LoadRegressionResults)."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# 2. PROVINCE COORDINATES
# ═══════════════════════════════════════════════════════════════

# Approximate centroid coordinates (longitude, latitude) for 30 Chinese provinces
PROVINCE_CENTROIDS: Dict[int, Tuple[float, float, str]] = {
    1:  (116.40, 39.90, "北京"), 2:  (117.20, 39.13, "天津"),
    3:  (114.50, 38.05, "河北"), 4:  (112.55, 37.87, "山西"),
    5:  (111.75, 40.83, "内蒙古"), 6:  (123.43, 41.80, "辽宁"),
    7:  (125.32, 43.88, "吉林"), 8:  (126.53, 45.80, "黑龙江"),
    9:  (121.47, 31.23, "上海"), 10: (118.80, 32.06, "江苏"),
    11: (120.15, 30.28, "浙江"), 12: (117.28, 31.86, "安徽"),
    13: (119.30, 26.08, "福建"), 14: (115.90, 28.68, "江西"),
    15: (117.00, 36.67, "山东"), 16: (113.65, 34.76, "河南"),
    17: (114.30, 30.60, "湖北"), 18: (112.98, 28.20, "湖南"),
    19: (113.28, 23.13, "广东"), 20: (108.33, 22.82, "广西"),
    21: (110.33, 20.03, "海南"), 22: (106.55, 29.57, "重庆"),
    23: (104.07, 30.67, "四川"), 24: (106.70, 26.60, "贵州"),
    25: (102.70, 25.05, "云南"), 26: (108.93, 34.27, "陕西"),
    27: (103.83, 36.07, "甘肃"), 28: (101.78, 36.62, "青海"),
    29: (106.27, 38.47, "宁夏"), 30: (87.63, 43.80, "新疆"),
}

def get_province_coords_df() -> pd.DataFrame:
    """Build DataFrame of province coordinates."""
    rows = []
    for pid, (lon, lat, name) in PROVINCE_CENTROIDS.items():
        rows.append({"province_id": pid, "province_name": name, "lon": lon, "lat": lat})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 3. SDCR PIPELINE CORE
# ═══════════════════════════════════════════════════════════════

class SDCRState:
    """State vector s_t (thesis Section 3.1)."""
    def __init__(self):
        self.year = 2022
        self.indicator = "ES"
        self.province_id = 0
        self.region_filter = "全部"

    def capture(self) -> dict:
        return {
            "year": self.year, "indicator": self.indicator,
            "province_id": self.province_id, "region_filter": self.region_filter,
        }


class SDCRPipeline:
    """
    State-driven Conditional Refresh pipeline (thesis Chapter 4).

    Normalization (Eq 4-4) → Channel Mapping (Eq 4-5) → State Submission (Eq 4-6)
    """

    # Color maps (matching Unity ProvinceShader)
    ES_CMIN = (0.85, 0.15, 0.15)   # warm red
    ES_CMAX = (0.15, 0.75, 0.25)   # cool green
    DEL_CMIN = (0.65, 0.75, 0.95)  # light blue
    DEL_CMAX = (0.35, 0.25, 0.75)  # deep purple

    def __init__(self, data: pd.DataFrame, reg_results: dict):
        self.data = data
        self.reg_results = reg_results
        self.state = SDCRState()
        self.height_amplifier = 0.4

    # ── Data Query ──

    def get_year_data(self) -> pd.DataFrame:
        """Query records for current year + region filter (thesis Layer 3)."""
        df = self.data[self.data["year"] == self.state.year]
        if self.state.region_filter != "全部":
            df = df[df["region_tag"] == self.state.region_filter]
        return df

    def get_value_range(self) -> Tuple[float, float]:
        """Get min/max of current indicator for normalization (Eq 4-4)."""
        col = self.state.indicator
        df = self.data.copy()
        if self.state.region_filter != "全部":
            df = df[df["region_tag"] == self.state.region_filter]
        return float(df[col].min()), float(df[col].max())

    # ── Visual Encoding (Eq 4-4, 4-5) ──

    def normalize_and_map(self) -> pd.DataFrame:
        """
        Execute full normalization → channel mapping pipeline.

        Returns DataFrame with per-province: n, r, g, b, height, color_hex
        """
        df_year = self.get_year_data()
        vmin, vmax = self.get_value_range()
        vrange = max(vmax - vmin, 1e-6)

        cmap = self.ES_CMIN if self.state.indicator == "ES" else self.DEL_CMIN
        cmax = self.ES_CMAX if self.state.indicator == "ES" else self.DEL_CMAX

        results = []
        for _, row in df_year.iterrows():
            val = float(row[self.state.indicator])
            n = (val - vmin) / vrange
            n = max(0.0, min(1.0, n))

            # Eq 4-5: c_i = Lerp(c_min, c_max, n_i)
            r = cmap[0] + n * (cmax[0] - cmap[0])
            g = cmap[1] + n * (cmax[1] - cmap[1])
            b = cmap[2] + n * (cmax[2] - cmap[2])

            results.append({
                "province_id": int(row["province_id"]),
                "province_name": row["province_name"],
                "value": val,
                "n": round(n, 4),
                "r": round(r, 3),
                "g": round(g, 3),
                "b": round(b, 3),
                "color_hex": f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}",
                "region": row["region_tag"],
            })

        return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# 4. INTERACTIVE VISUALIZATION (Plotly)
# ═══════════════════════════════════════════════════════════════

class SDCRVisualizer:
    """Plotly-based interactive visualization matching Unity SDCR--Vis views."""

    def __init__(self, pipeline: SDCRPipeline):
        self.pipeline = pipeline
        self.coords_df = get_province_coords_df()

    def build_figure(self) -> go.Figure:
        """Build the complete multi-view figure."""

        # Create subplot layout (4 views matching thesis Layer 4)
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "scattergeo", "colspan": 1}, {"type": "table"}],
                [{"type": "scatter", "colspan": 1}, {"type": "scatter"}],
            ],
            subplot_titles=(
                "省域地图 (Map View)",
                "回归结果 (Result Panel)",
                "时序演化 (Timeline)",
                "机制路径 (Mechanism Graph)",
            ),
            column_widths=[0.55, 0.45],
            row_heights=[0.55, 0.45],
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
        )

        self._add_map_view(fig)
        self._add_result_panel(fig)
        self._add_timeline_view(fig)
        self._add_mechanism_view(fig)

        # Update layout
        indicator_label = "能源结构优化指数 (ES)" if self.pipeline.state.indicator == "ES" else "数字经济发展水平 (DEL)"
        year = self.pipeline.state.year
        region = self.pipeline.state.region_filter

        fig.update_layout(
            title=dict(
                text=f"<b>SDCR--Vis 省域数字经济与能源结构AR沉浸式分析系统</b><br>"
                     f"<sup>状态: {year}年 | {indicator_label} | 区域: {region} | "
                     f"Δs_t ≠ 0 → RenderUpdate(s_t)</sup>",
                font=dict(size=18),
            ),
            height=950,
            showlegend=False,
            geo=dict(
                scope="asia",
                showframe=False,
                projection_type="mercator",
                center=dict(lat=35, lon=105),
                lataxis=dict(range=[15, 55]),
                lonaxis=dict(range=[73, 135]),
                resolution=50,
            ),
        )

        return fig

    def _add_map_view(self, fig: go.Figure):
        """Add 3D-ish provincial map (coloring by indicator via SDCR pipeline)."""
        mapped = self.pipeline.normalize_and_map()

        # Merge with coordinates
        plot_data = self.coords_df.merge(mapped, on="province_id", how="left")
        plot_data = plot_data.fillna({"value": 0, "color_hex": "#888888", "province_name_y": ""})

        # Color: dimension by indicator value, map via SDCR encoding
        fig.add_trace(
            go.Scattergeo(
                lon=plot_data["lon"],
                lat=plot_data["lat"],
                mode="markers+text",
                marker=dict(
                    size=plot_data["value"] * 30 + 8,
                    color=plot_data["color_hex"],
                    line=dict(width=1, color="white"),
                    sizemode="area",
                    sizeref=0.02,
                ),
                text=plot_data["province_name_y"] if "province_name_y" in plot_data.columns else plot_data["province_name_x"],
                textfont=dict(size=8, color="white"),
                textposition="middle center",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "ID: %{customdata[0]}<br>"
                    f"{self.pipeline.state.indicator}: %{{customdata[1]:.4f}}<br>"
                    "归一化: %{customdata[2]:.3f}<br>"
                    "区域: %{customdata[3]}<br>"
                    "<extra></extra>"
                ),
                customdata=plot_data[["province_id", "value", "n", "region"]].values,
            ),
            row=1, col=1,
        )

    def _add_result_panel(self, fig: go.Figure):
        """Add regression result table (mirrors ResultPanelController)."""
        baseline = self.pipeline.reg_results.get("baseline", {})
        models = baseline.get("models", [])

        if not models:
            return

        # Build table from model coefficients
        model2 = models[1] if len(models) > 1 else models[0]  # col (2) with controls

        header = ["变量", "系数", "SE", "显著性"]
        cells = [[], [], [], []]

        for c in model2.get("coefficients", []):
            cells[0].append(c["variable"])
            cells[1].append(f"{c['estimate']:.4f}")
            cells[2].append(f"{c['se']:.4f}")
            cells[3].append(c.get("significance", ""))

        fig.add_trace(
            go.Table(
                header=dict(
                    values=header,
                    fill_color="#4A90D9",
                    font=dict(color="white", size=12),
                    align="center",
                ),
                cells=dict(
                    values=cells,
                    fill_color=[["#f5f5f5", "white"] * 4],
                    font=dict(size=11),
                    align="center",
                ),
            ),
            row=1, col=2,
        )

    def _add_timeline_view(self, fig: go.Figure):
        """Add time evolution plot (mirrors TimelineController + timeline panel)."""
        data = self.pipeline.data

        # Compute national mean by year
        annual = data.groupby("year").agg(
            ES_mean=("ES", "mean"),
            DEL_mean=("DEL", "mean"),
            ES_std=("ES", "std"),
            DEL_std=("DEL", "std"),
        ).reset_index()

        indicator = self.pipeline.state.indicator
        mean_col = f"{indicator}_mean"
        std_col = f"{indicator}_std"

        # Line with error band
        fig.add_trace(
            go.Scatter(
                x=annual["year"],
                y=annual[mean_col],
                mode="lines+markers",
                line=dict(width=3, color="#4A90D9"),
                marker=dict(size=8),
                name=f"全国均值 ({indicator})",
                hovertemplate="%{x}年: %{y:.4f}",
            ),
            row=2, col=1,
        )

        # Error band
        fig.add_trace(
            go.Scatter(
                x=list(annual["year"]) + list(annual["year"])[::-1],
                y=list(annual[mean_col] + annual[std_col]) + list((annual[mean_col] - annual[std_col])[::-1]),
                fill="toself",
                fillcolor="rgba(74, 144, 217, 0.15)",
                line=dict(width=0),
                name="±1σ",
                showlegend=False,
            ),
            row=2, col=1,
        )

        # Highlight current year
        current_val = annual[annual["year"] == self.pipeline.state.year][mean_col]
        if len(current_val) > 0:
            fig.add_trace(
                go.Scatter(
                    x=[self.pipeline.state.year],
                    y=[float(current_val.iloc[0])],
                    mode="markers",
                    marker=dict(size=14, color="red", symbol="diamond"),
                    name=f"当前: {self.pipeline.state.year}",
                ),
                row=2, col=1,
            )

        fig.update_xaxes(title_text="年份", row=2, col=1)
        fig.update_yaxes(title_text=indicator, row=2, col=1)

    def _add_mechanism_view(self, fig: go.Figure):
        """Add simplified mechanism pathway diagram."""
        # Node positions (matching mechanism_paths.json layout)
        nodes = {
            "DEL": (0.0, 0.5),
            "直接效应": (0.3, 0.75),
            "间接效应": (0.3, 0.25),
            "资源配置": (0.6, 0.85),
            "消费变革": (0.6, 0.65),
            "技术创新\n(12.33%)": (0.6, 0.45),
            "产业升级": (0.6, 0.25),
            "绿色治理": (0.6, 0.05),
            "ES优化": (0.9, 0.5),
        }

        edges = [
            ("DEL", "直接效应"), ("DEL", "间接效应"),
            ("直接效应", "资源配置"), ("直接效应", "消费变革"),
            ("间接效应", "技术创新\n(12.33%)"), ("间接效应", "产业升级"), ("间接效应", "绿色治理"),
            ("资源配置", "ES优化"), ("消费变革", "ES优化"),
            ("技术创新\n(12.33%)", "ES优化"), ("产业升级", "ES优化"), ("绿色治理", "ES优化"),
        ]

        # Plot edges as lines
        for src, dst in edges:
            if src in nodes and dst in nodes:
                fig.add_trace(
                    go.Scatter(
                        x=[nodes[src][0], nodes[dst][0]],
                        y=[nodes[src][1], nodes[dst][1]],
                        mode="lines",
                        line=dict(width=1.5, color="rgba(100,100,100,0.5)"),
                        showlegend=False,
                        hoverinfo="none",
                    ),
                    row=2, col=2,
                )

        # Plot nodes
        node_labels = list(nodes.keys())
        node_x = [nodes[n][0] for n in node_labels]
        node_y = [nodes[n][1] for n in node_labels]

        # Color by type
        node_colors = []
        for n in node_labels:
            if n == "DEL":
                node_colors.append("#4A90D9")
            elif n == "ES优化":
                node_colors.append("#E74C3C")
            elif "直接" in n:
                node_colors.append("#50C878")
            elif "间接" in n:
                node_colors.append("#FFB347")
            else:
                node_colors.append("#95A5A6")

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(size=30, color=node_colors, line=dict(width=2, color="white")),
                text=[n.replace("\\n", "<br>") for n in node_labels],
                textfont=dict(size=9, color="white"),
                textposition="middle center",
                showlegend=False,
                hoverinfo="text",
                hovertext=node_labels,
            ),
            row=2, col=2,
        )

        fig.update_xaxes(range=[-0.1, 1.1], showgrid=False, zeroline=False,
                         showticklabels=False, row=2, col=2)
        fig.update_yaxes(range=[-0.1, 1.1], showgrid=False, zeroline=False,
                         showticklabels=False, row=2, col=2)


# ═══════════════════════════════════════════════════════════════
# 5. STATIC EXPORT (for thesis figures)
# ═══════════════════════════════════════════════════════════════

def export_thesis_figures(pipeline: SDCRPipeline, output_dir: str = "thesis_figures"):
    """Export static PNG figures for each key thesis view."""
    output = Path(output_dir)
    output.mkdir(exist_ok=True)

    viz = SDCRVisualizer(pipeline)

    # Figure 1: ES coloring 2022 national view
    pipeline.state.year = 2022
    pipeline.state.indicator = "ES"
    pipeline.state.region_filter = "全部"
    fig1 = viz.build_figure()
    fig1.write_image(str(output / "fig_es_2022_national.png"), width=1600, height=950)
    print(f"  Saved: fig_es_2022_national.png")

    # Figure 2: DEL coloring 2022 national view
    pipeline.state.indicator = "DEL"
    fig2 = viz.build_figure()
    fig2.write_image(str(output / "fig_del_2022_national.png"), width=1600, height=950)
    print(f"  Saved: fig_del_2022_national.png")

    # Figure 3: ES 2014 (start of sample)
    pipeline.state.indicator = "ES"
    pipeline.state.year = 2014
    fig3 = viz.build_figure()
    fig3.write_image(str(output / "fig_es_2014_national.png"), width=1600, height=950)
    print(f"  Saved: fig_es_2014_national.png")

    # Figure 4: Regional heterogeneity - Central region
    pipeline.state.year = 2022
    pipeline.state.indicator = "ES"
    for region in ["东部", "中部", "西部"]:
        pipeline.state.region_filter = region
        fig4 = viz.build_figure()
        fig4.write_image(str(output / f"fig_es_2022_{region}.png"), width=1600, height=950)
        print(f"  Saved: fig_es_2022_{region}.png")

    print(f"\nAll figures saved to: {output_dir}/")


# ═══════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SDCR--Vis Python Demo")
    parser.add_argument("--data_dir", type=str,
                        default="../Unity/Assets/Resources/Data",
                        help="Path to data directory")
    parser.add_argument("--export_figures", action="store_true",
                        help="Export static PNG figures for thesis")
    parser.add_argument("--open_browser", action="store_true", default=True,
                        help="Open interactive visualization in browser")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    csv_path = data_dir / "panel_data.csv"
    json_path = data_dir / "regression_results.json"

    if not csv_path.exists():
        print(f"ERROR: panel_data.csv not found at {csv_path}")
        print("Run from SDCR_Vis_System/Python_Demo/ directory")
        return

    # Load data
    print("Loading data...")
    data = load_panel_data(str(csv_path))
    reg_results = load_regression_results(str(json_path))
    print(f"  Loaded {len(data)} panel records, {data['province_id'].nunique()} provinces")

    # Initialize SDCR pipeline
    pipeline = SDCRPipeline(data, reg_results)

    # Export thesis figures if requested
    if args.export_figures:
        print("\nExporting thesis figures...")
        try:
            export_thesis_figures(pipeline)
        except Exception as e:
            print(f"  Figure export failed (install kaleido: pip install kaleido): {e}")

    # Build and display interactive visualization
    print("\nBuilding interactive visualization...")
    viz = SDCRVisualizer(pipeline)
    fig = viz.build_figure()

    # Add control buttons via updatemenus (state-driven interaction)
    fig.update_layout(
        updatemenus=[
            # Year selector
            dict(
                type="buttons",
                direction="left",
                x=0.0, y=1.08,
                buttons=[
                    dict(label=str(y), method="update", args=[{}])  # placeholder
                    for y in range(2014, 2023)
                ],
                pad=dict(r=5),
                showactive=True,
                active=8,  # 2022 is index 8 (default)
            ),
        ],
    )

    # Save HTML
    html_path = "sdcr_vis_demo.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"\nSaved interactive demo to: {html_path}")

    if args.open_browser:
        webbrowser.open(f"file://{Path(html_path).absolute()}")
        print("Opened in browser.")


if __name__ == "__main__":
    main()
