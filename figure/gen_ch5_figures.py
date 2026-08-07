#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Chapter 5 data visualization figures.
Three charts: spatial distribution, nonlinear verification, regional heterogeneity.
Uses actual panel_data.csv and matches the thesis academic color palette.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression

# ═══════════════════════════════════════════════════════
# Global style — matching regenerate_figures.py
# ═══════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.4,
    'legend.framealpha': 0.85,
    'legend.edgecolor': '#CCCCCC',
})

PALETTE = {
    'blue':   '#2B579A',
    'orange': '#D35400',
    'green':  '#27AE60',
    'red':    '#C0392B',
    'purple': '#7D3C98',
    'teal':   '#1ABC9C',
    'grey':   '#7F8C8D',
    'gold':   '#F39C12',
}

REGION_COLORS = {
    '东部': PALETTE['blue'],
    '中部': PALETTE['orange'],
    '西部': PALETTE['green'],
    '东北': PALETTE['purple'],
}

DATA_PATH = Path(__file__).resolve().parent.parent / 'workspace' / 'SDCR_Vis_System' / 'Unity' / 'Assets' / 'Resources' / 'Data' / 'panel_data.csv'
OUTPUT_DIR = Path(__file__).resolve().parent


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


# ═══════════════════════════════════════════════════════
# Figure 1: Spatial Distribution — Province DEL & ES (2022)
# ═══════════════════════════════════════════════════════
def fig_spatial_distribution(df):
    d2022 = df[df['year'] == 2022].copy()
    d2022 = d2022.sort_values('ES', ascending=True)

    # Select ~10 representative provinces (every 3rd from sorted list)
    idxs = list(range(0, len(d2022), 3))
    d10 = d2022.iloc[idxs].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    provinces = d10['province_name'].tolist()
    y = np.arange(len(provinces))
    es_vals = d10['ES'].values
    del_vals = d10['DEL'].values

    bar_h = 0.32
    ax.barh(y - bar_h/2, es_vals, bar_h,
            color=PALETTE['blue'], alpha=0.82, label='ES 能源结构',
            edgecolor='white', linewidth=0.3, zorder=3)
    ax.barh(y + bar_h/2, del_vals, bar_h,
            color=PALETTE['orange'], alpha=0.82, label='DEL 数字经济',
            edgecolor='white', linewidth=0.3, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(provinces, fontsize=10)
    ax.set_xlabel('综合指数值（2022）', fontsize=11)
    ax.set_title('省域 DEL 与 ES 综合指数分布（2022）', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.85)
    ax.set_xlim(0, 0.78)
    ax.grid(axis='x', alpha=0.22, linewidth=0.4)

    plt.tight_layout()
    out = OUTPUT_DIR / 'ch5_spatial_distribution'
    fig.savefig(str(out) + '.pdf', dpi=300)
    fig.savefig(str(out) + '.png', dpi=300)
    plt.close(fig)
    print(f'  [OK] {out}.pdf + .png')


# ═══════════════════════════════════════════════════════
# Figure 2: Nonlinear Verification — Scatter + Quadratic Fit
# ═══════════════════════════════════════════════════════
def fig_nonlinear_verification(df):
    d2022 = df[df['year'] == 2022].copy()

    fig, ax = plt.subplots(figsize=(10, 7.2))

    del_vals = d2022['DEL'].values
    es_vals = d2022['ES'].values
    regions = d2022['region_tag'].values
    provinces = d2022['province_name'].values

    # ── Panel FE regression coefficients (Table 5.5 col 3) ──
    # ES_it = mu_i + 2.131*DEL_it - 2.103*DEL_it^2 + controls + e_it
    panel_alpha1, panel_alpha2 = 2.131, -2.103
    # Intercept calibrated to 2022 cross-section mean (for visualization only)
    mean_es = np.mean(es_vals)
    mean_del = np.mean(del_vals)
    mean_del2 = np.mean(del_vals**2)
    intercept = mean_es - panel_alpha1 * mean_del - panel_alpha2 * mean_del2

    # Generate predicted curve from panel coefficients
    x_curve = np.linspace(0.02, 0.66, 300)
    y_curve = intercept + panel_alpha1 * x_curve + panel_alpha2 * x_curve**2

    # Turning point from panel coefficients
    x_turn = -panel_alpha1 / (2 * panel_alpha2)  # = 0.507
    y_turn = intercept + panel_alpha1 * x_turn + panel_alpha2 * x_turn**2

    # ── Scatter: colored by region ──
    region_order = ['东部', '中部', '西部', '东北']
    marker_styles = {'东部': 'o', '中部': 's', '西部': '^', '东北': 'D'}
    for region in region_order:
        mask = regions == region
        if mask.sum() == 0:
            continue
        ax.scatter(del_vals[mask], es_vals[mask],
                   c=REGION_COLORS[region], label=region,
                   s=64, edgecolors='white', linewidth=0.6,
                   zorder=5, alpha=0.88,
                   marker=marker_styles.get(region, 'o'))

    # ── Quadratic fit curve from PANEL coefficients (dashed red) ──
    ax.plot(x_curve, y_curve, '--', color=PALETTE['red'], linewidth=2.2,
            label='面板FE二次拟合', zorder=4)

    # ── Turning point vertical line ──
    ax.axvline(x=x_turn, color=PALETTE['grey'], linewidth=1.2,
               linestyle='-', alpha=0.50, zorder=3)
    # Light shade for post-inflection zone
    ax.axvspan(x_turn, 0.68, alpha=0.04, color=PALETTE['red'], zorder=2)

    # ── Annotate turning point ──
    ax.annotate(f'拐点 DEL* = {x_turn:.3f}',
                xy=(x_turn, y_turn), xytext=(x_turn + 0.10, y_turn + 0.04),
                fontsize=9, fontweight='bold', color=PALETTE['grey'],
                arrowprops=dict(arrowstyle='->', color=PALETTE['grey'],
                               lw=1.0, connectionstyle='arc3,rad=-0.2'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         alpha=0.85, edgecolor=PALETTE['grey']))

    # ── Label key provinces (name only) ──
    for prov_name in ['广东', '北京', '浙江', '青海']:
        row = d2022[d2022['province_name'] == prov_name]
        if len(row) == 0:
            continue
        ox, oy = {'广东': (-50, 10), '北京': (12, 15), '浙江': (10, -12), '青海': (-50, 8)}.get(prov_name, (10, 10))
        ax.annotate(prov_name,
                    (row['DEL'].values[0], row['ES'].values[0]),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=9, fontweight='bold', color='#333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.82, edgecolor=PALETTE['grey'], linewidth=0.5),
                    arrowprops=dict(arrowstyle='->', color=PALETTE['grey'],
                                    lw=0.7))

    # ── Axis labels & title ──
    ax.set_xlabel('DEL 数字经济水平', fontsize=12)
    ax.set_ylabel('ES 能源结构优化指数', fontsize=12)
    ax.set_title('DEL–ES 非线性关系：倒U型验证（2022年截面）', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlim(0.02, 0.66)
    ax.set_ylim(0.32, 0.78)
    ax.grid(alpha=0.20, linewidth=0.4)

    # ── Legend ──
    ax.legend(loc='lower right', fontsize=9, ncol=1,
              framealpha=0.88, edgecolor='#ccc')

    plt.tight_layout()
    out = OUTPUT_DIR / 'ch5_nonlinear_scatter'
    fig.savefig(str(out) + '.pdf', dpi=300)
    fig.savefig(str(out) + '.png', dpi=300)
    plt.close(fig)
    print(f'  [OK] {out}.pdf + .png')
    print(f'      Panel-based curve: ES = {intercept:.4f} + {panel_alpha1}*DEL + {panel_alpha2}*DEL^2')
    print(f'      Turning point DEL* = {x_turn:.3f}, ES* = {y_turn:.3f}')


# ═══════════════════════════════════════════════════════
# Figure 3: Regional Heterogeneity — Coefficient Bar Chart
# ═══════════════════════════════════════════════════════
def fig_regional_heterogeneity(df):
    region_coefs_zh = ['东部', '中部', '西部', '东北']
    coefs  = [0.336, 1.994, 1.100, 0.255]
    errors = [0.0862, 0.340, 0.218, 0.776]
    sigs   = ['***', '***', '***', 'n.s.']

    region_del_means = df[df['year'] == 2022].groupby('region_tag')['DEL'].mean()
    region_es_means = df[df['year'] == 2022].groupby('region_tag')['ES'].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))

    # --- Left: Coefficient bar chart ---
    colors = [REGION_CODES[r] for r in region_coefs_zh]
    bars = ax1.bar(region_coefs_zh, coefs, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5,
                   yerr=errors, capsize=5, error_kw={'linewidth': 1.3, 'color': PALETTE['grey']})

    for bar, sig, err in zip(bars, sigs, errors):
        if sig == '***':
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.06,
                     '***', ha='center', fontsize=13, fontweight='bold', color=PALETTE['red'])
        elif sig == 'n.s.':
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.06,
                     'n.s.', ha='center', fontsize=9, color=PALETTE['grey'], style='italic')

    # National baseline
    ax1.axhline(y=0.575, color=PALETTE['grey'], linestyle='--', linewidth=1.0, alpha=0.6)
    ax1.text(3.3, 0.575 + 0.05, '全国基准 0.575***', fontsize=8,
             color=PALETTE['grey'], ha='left', va='bottom')

    ax1.set_ylabel('DEL 对 ES 的回归系数', fontsize=11)
    ax1.set_title('区域异质性：DEL→ES 分组回归系数', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 2.7)
    ax1.grid(axis='y', alpha=0.20, linewidth=0.4)

    # --- Right: DEL vs ES by region (2022 means) ---
    x_pos = np.arange(len(region_coefs_zh))
    bar_w = 0.32

    del_by_region = [region_del_means.get(r, 0) for r in region_coefs_zh]
    es_by_region = [region_es_means.get(r, 0) for r in region_coefs_zh]

    bars_del = ax2.bar(x_pos - bar_w/2, del_by_region, bar_w, color=PALETTE['orange'], alpha=0.85,
                       label='DEL 均值', edgecolor='white', linewidth=0.3)
    bars_es  = ax2.bar(x_pos + bar_w/2, es_by_region, bar_w, color=PALETTE['blue'], alpha=0.85,
                       label='ES 均值', edgecolor='white', linewidth=0.3)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(region_coefs_zh, fontsize=11)
    ax2.set_ylabel('综合指数均值（2022）', fontsize=11)
    ax2.set_title('各区域 DEL 与 ES 均值（2022）', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.85)
    ax2.grid(axis='y', alpha=0.20, linewidth=0.4)

    plt.tight_layout()
    out = OUTPUT_DIR / 'ch5_regional_heterogeneity'
    fig.savefig(str(out) + '.pdf', dpi=300)
    fig.savefig(str(out) + '.png', dpi=300)
    plt.close(fig)
    print(f'  [OK] {out}.pdf + .png')


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
REGION_CODES = {
    '东部': PALETTE['blue'],
    '中部': PALETTE['orange'],
    '西部': PALETTE['green'],
    '东北': PALETTE['purple'],
}

if __name__ == '__main__':
    print('Loading data...')
    df = load_data()
    print(f'  {len(df)} rows, provinces: {df["province_name"].nunique()}, years: {df["year"].min()}-{df["year"].max()}')

    print('\nGenerating Figure 1: Spatial Distribution...')
    fig_spatial_distribution(df)

    print('\nGenerating Figure 2: Nonlinear Scatter + Quadratic Fit...')
    fig_nonlinear_verification(df)

    print('\nGenerating Figure 3: Regional Heterogeneity...')
    fig_regional_heterogeneity(df)

    print('\nDone. All figures saved to figure/')
