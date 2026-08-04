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
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
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

    fig, ax = plt.subplots(figsize=(10, 7))

    provinces = d2022['province_name'].tolist()
    y = np.arange(len(provinces))

    es_vals = d2022['ES'].values
    del_vals = d2022['DEL'].values

    # Horizontal bars
    bar_h = 0.35
    bars_es = ax.barh(y - bar_h/2, es_vals, bar_h,
                       color=PALETTE['blue'], alpha=0.85, label='ES (Energy Structure)',
                       edgecolor='white', linewidth=0.3)
    bars_del = ax.barh(y + bar_h/2, del_vals, bar_h,
                       color=PALETTE['orange'], alpha=0.85, label='DEL (Digital Economy)',
                       edgecolor='white', linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(provinces, fontsize=7)
    ax.set_xlabel('Index Value (2022)', fontsize=10)
    ax.set_title('Provincial DEL and ES Composite Index (2022)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 0.85)
    ax.grid(axis='x', alpha=0.3, linewidth=0.4)

    # Annotations
    ax.annotate('East-high, West-low DEL gradient', xy=(0.62, 27), fontsize=8,
                color=PALETTE['orange'], style='italic',
                xytext=(0.45, 25), arrowprops=dict(arrowstyle='->', color=PALETTE['grey'], lw=0.8))
    ax.annotate('ES not fully aligned\nwith DEL', xy=(0.45, 8), fontsize=8,
                color=PALETTE['blue'], style='italic',
                xytext=(0.25, 4), arrowprops=dict(arrowstyle='->', color=PALETTE['grey'], lw=0.8))

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

    fig, ax = plt.subplots(figsize=(8, 6))

    del_vals = d2022['DEL'].values.reshape(-1, 1)
    es_vals = d2022['ES'].values

    # Quadratic fit using thesis coefficients (from Table baseline col 3)
    alpha1, alpha2 = 2.131, -2.103
    x_fit = np.linspace(0, 0.65, 200)
    y_fit = 1.283 + alpha1 * x_fit + alpha2 * x_fit**2  # constant from thesis

    # Also fit from data for comparison
    poly = np.polynomial.Polynomial.fit(del_vals.flatten(), es_vals, 2)
    x_poly = np.linspace(0, 0.65, 200)
    y_poly = poly(x_poly)

    # Inflection point
    x_infl = -alpha1 / (2 * alpha2)  # = 0.507

    # Scatter
    regions = d2022['region_tag'].values
    for region, color in REGION_COLORS.items():
        mask = regions == region
        ax.scatter(del_vals[mask], es_vals[mask], c=color, label=region,
                   s=50, edgecolors='white', linewidth=0.4, zorder=5, alpha=0.85)

    # Quadratic fit line
    ax.plot(x_fit, y_fit, '--', color=PALETTE['red'], linewidth=1.8,
            label='Quadratic fit (DEL$^2$: −2.103***)', zorder=4)

    # Inflection line
    y_infl_line = np.linspace(0.25, 0.80, 100)
    ax.plot([x_infl, x_infl], [0.25, 0.80], '-', color=PALETTE['grey'],
            linewidth=1.2, alpha=0.7, zorder=3)
    ax.annotate(f'Inflection\nDEL={x_infl:.3f}', xy=(x_infl, 0.76), fontsize=8,
                color=PALETTE['grey'], ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=PALETTE['grey']))

    # Label key provinces
    key_provinces = {
        '广东': (0.58, 0.62, 'Guangdong'),
        '北京': (0.52, 0.64, 'Beijing'),
        '浙江': (0.48, 0.55, 'Zhejiang'),
        '青海': (0.08, 0.48, 'Qinghai'),
    }
    for name, (dx, dy, _) in key_provinces.items():
        row = d2022[d2022['province_name'] == name]
        if len(row) > 0:
            ax.annotate(name, (row['DEL'].values[0], row['ES'].values[0]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8,
                        fontweight='bold', color=PALETTE['grey'],
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75))

    ax.set_xlabel('DEL (Digital Economy Level)', fontsize=11)
    ax.set_ylabel('ES (Energy Structure)', fontsize=11)
    ax.set_title('DEL–ES Nonlinear Relationship: Inverted U-Shape (2022)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8, ncol=2)
    ax.set_xlim(-0.01, 0.68)
    ax.set_ylim(0.22, 0.82)
    ax.grid(alpha=0.25, linewidth=0.4)

    # R² annotation
    ax.text(0.97, 0.06, 'Panel FE (col 3):\nDEL 2.131***, DEL² −2.103***\nR² = 0.570, N = 270',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor=PALETTE['grey']))

    plt.tight_layout()
    out = OUTPUT_DIR / 'ch5_nonlinear_scatter'
    fig.savefig(str(out) + '.pdf', dpi=300)
    fig.savefig(str(out) + '.png', dpi=300)
    plt.close(fig)
    print(f'  [OK] {out}.pdf + .png')


# ═══════════════════════════════════════════════════════
# Figure 3: Regional Heterogeneity — Coefficient Bar Chart
# ═══════════════════════════════════════════════════════
def fig_regional_heterogeneity(df):
    # Coefficients from thesis Table: Regional heterogeneity test
    region_coefs = {
        'Eastern (东部)':    (0.336, 0.0862, '***'),
        'Central (中部)':    (1.994, 0.340,  '***'),
        'Western (西部)':    (1.100, 0.218,  '***'),
        'Northeast (东北)':  (0.255, 0.776,  'ns'),
    }

    # Also compute from actual data for verification
    region_del_means = df[df['year'] == 2022].groupby('region_tag')['DEL'].mean()
    region_es_means = df[df['year'] == 2022].groupby('region_tag')['ES'].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Coefficient bar chart ---
    names = list(region_coefs.keys())
    coefs = [v[0] for v in region_coefs.values()]
    errors = [v[1] for v in region_coefs.values()]
    sigs = [v[2] for v in region_coefs.values()]
    colors = [REGION_CODES['东部'], REGION_CODES['中部'], REGION_CODES['西部'], REGION_CODES['东北']]

    bars = ax1.bar(names, coefs, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5,
                   yerr=errors, capsize=4, error_kw={'linewidth': 1.2, 'color': PALETTE['grey']})

    # Significance stars
    for i, (bar, sig) in enumerate(zip(bars, sigs)):
        if sig == '***':
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[i] + 0.08,
                     '***', ha='center', fontsize=12, fontweight='bold', color=PALETTE['red'])
        elif sig == 'ns':
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[i] + 0.08,
                     'n.s.', ha='center', fontsize=9, color=PALETTE['grey'], style='italic')

    # National baseline line
    ax1.axhline(y=0.575, color=PALETTE['grey'], linestyle='--', linewidth=1.0, alpha=0.7)
    ax1.text(3.5, 0.575 + 0.03, 'National baseline\nDEL=0.575***', fontsize=7,
             color=PALETTE['grey'], ha='left', va='bottom')

    ax1.set_ylabel('DEL Coefficient on ES', fontsize=10)
    ax1.set_title('Regional Heterogeneity: DEL→ES Coefficients', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 2.8)
    ax1.grid(axis='y', alpha=0.25, linewidth=0.4)

    # --- Right: DEL vs ES by region (2022 means) ---
    region_order = ['东部', '东北', '中部', '西部']
    x_pos = np.arange(len(region_order))
    bar_w = 0.3

    del_by_region = [region_del_means.get(r, 0) for r in region_order]
    es_by_region = [region_es_means.get(r, 0) for r in region_order]

    ax2.bar(x_pos - bar_w/2, del_by_region, bar_w, color=PALETTE['orange'], alpha=0.85,
            label='DEL (mean)', edgecolor='white', linewidth=0.3)
    ax2.bar(x_pos + bar_w/2, es_by_region, bar_w, color=PALETTE['blue'], alpha=0.85,
            label='ES (mean)', edgecolor='white', linewidth=0.3)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(region_order, fontsize=10)
    ax2.set_ylabel('Mean Index Value (2022)', fontsize=10)
    ax2.set_title('Regional Mean DEL & ES (2022)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.25, linewidth=0.4)

    # Annotation
    ax2.annotate('Central: high DEL→ES elasticity\nbut moderate DEL level',
                 xy=(1, 0.25), fontsize=7.5, color=PALETTE['orange'], style='italic',
                 xytext=(0.5, 0.12), arrowprops=dict(arrowstyle='->', color=PALETTE['grey'], lw=0.7))

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
