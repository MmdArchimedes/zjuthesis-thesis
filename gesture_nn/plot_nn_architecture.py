#!/usr/bin/env python3
"""
Generate DBEW-NN architecture diagram in NN-SVG style.
Produces a clean, publication-quality figure with tensor shapes.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Style settings (NN-SVG aesthetic) ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 9,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# ── Color palette (NN-SVG inspired) ──
C_INPUT    = '#E8F5E9'  # light green
C_CONV     = '#FFF3E0'  # light orange
C_POOL     = '#E3F2FD'  # light blue
C_ATTN     = '#FCE4EC'  # light pink
C_FC       = '#F3E5F5'  # light purple
C_OUTPUT   = '#E0F2F1'  # light teal
C_EDGE     = '#37474F'  # dark gray
C_ARROW    = '#546E7A'
C_TEXT     = '#263238'

def draw_3d_block(ax, x, y, w, h, depth=0.08, color=C_CONV, label='', sublabels=None, fontsize=8):
    """Draw a 3D block similar to NN-SVG style."""
    # Main face
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=C_EDGE, linewidth=1.0, zorder=3)
    ax.add_patch(rect)
    # 3D depth effect (right and top edges)
    poly = plt.Polygon([
        (x + w, y), (x + w + depth, y + depth),
        (x + w + depth, y + h + depth), (x + w, y + h)
    ], facecolor=color, edgecolor=C_EDGE, linewidth=0.8, alpha=0.6, zorder=2)
    ax.add_patch(poly)
    poly2 = plt.Polygon([
        (x, y + h), (x + depth, y + h + depth),
        (x + w + depth, y + h + depth), (x + w, y + h)
    ], facecolor=color, edgecolor=C_EDGE, linewidth=0.8, alpha=0.4, zorder=1)
    ax.add_patch(poly2)
    # Label
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=C_TEXT, zorder=4)
    if sublabels:
        for i, sl in enumerate(sublabels):
            ax.text(x + w/2, y + h/2 - (i+1)*0.035, sl, ha='center', va='top',
                    fontsize=fontsize-2, color='#546E7A', zorder=4)

def draw_arrow(ax, x1, y1, x2, y2):
    """Draw a connecting arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.5,
                               connectionstyle='arc3,rad=0'))

def main():
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Layout parameters ──
    y_center = 3.0
    bw, bh = 1.8, 0.9   # block width, height
    gap = 0.55           # gap between blocks
    depth = 0.12

    # ── Stage labels ──
    stage_y = 5.5
    stages = [
        (1.1, 'Stage 1\nSpatial Embed'),
        (4.0, 'Stage 2\nDilated CNN'),
        (7.1, 'Stage 3\nSelf-Attention'),
        (10.5, 'Stage 4\nClassifier'),
    ]
    for sx, slabel in stages:
        ax.text(sx, stage_y, slabel, ha='center', va='top', fontsize=8,
                fontweight='bold', color='#455A64')

    # ── Input block ──
    x0 = 0.3
    draw_3d_block(ax, x0, y_center - bh/2, bw, bh, depth, C_INPUT,
                  'Input\n$B\\times32\\times26\\times3$', fontsize=8)

    # ── Stage 1: Spatial Embedding ──
    x1 = x0 + bw + gap
    draw_3d_block(ax, x1, y_center - bh/2 + 0.7, bw, bh*0.85, depth*0.8, C_CONV,
                  '1×1 Conv\n$3\\to64$', ['LayerNorm'], fontsize=7)

    # Joint pool
    x1p = x1
    draw_3d_block(ax, x1p, y_center - bh/2 - 0.6, bw*0.85, bh*0.7, depth*0.6, C_POOL,
                  'Joint Pool\nmean($J$)', ['$26\\to1$'], fontsize=7)

    # Position encoding
    x1pe = x1
    draw_3d_block(ax, x1pe, y_center - bh/2 - 1.3, bw*0.7, bh*0.55, depth*0.5, C_POOL,
                  'PosEnc\nsin/cos', fontsize=7)

    # Shape annotation
    ax.text(x1 + bw/2, y_center - 2.2, '$B\\times32\\times64$', ha='center', va='top',
            fontsize=7, color='#00897B', style='italic')

    # Arrow from input to spatial
    draw_arrow(ax, x0 + bw, y_center, x1, y_center + 0.3)

    # ── Stage 2: Dilated CNN ──
    x2 = x1 + bw + gap
    cnn_colors = ['#FFE0B2', '#FFCC80', '#FFB74D']
    for i, (d, yoff) in enumerate([(1, 0.65), (2, -0.15), (4, -0.95)]):
        label = 'D-Conv1d $k=3,d=' + str(d) + '$';
        draw_3d_block(ax, x2, y_center + yoff, bw*1.2, bh*0.7, depth*0.7, cnn_colors[i],
                      label, ['BN + GELU'], fontsize=7)
    # Residual arrow
    ax.annotate('', xy=(x2 + bw*1.2, y_center - 0.9), xytext=(x2 + bw*1.2, y_center + 0.9),
                arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=1.2,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(x2 + bw*1.2 + 0.1, y_center, 'residual', fontsize=6, color='#FF6F00', rotation=90, va='center')

    draw_arrow(ax, x1 + bw, y_center, x2, y_center + 0.3)

    # ── Stage 3: Self-Attention ──
    x3 = x2 + bw*1.2 + gap + 0.4
    attn_blocks = [
        (0.8, 'Linear Proj\n$64\\to64\\times3$', C_ATTN),
        (0.0, '4-Head Split\n$4\\times(T\\times16)$', '#F8BBD0'),
        (-0.8, 'Scaled Dot-Prod\n$\\frac{QK^{\\top}}{\\sqrt{16}}V$', '#F48FB1'),
    ]
    for yoff, label, col in attn_blocks:
        draw_3d_block(ax, x3, y_center + yoff, bw*1.1, bh*0.65, depth*0.6, col,
                      label, fontsize=7)
    # + Residual + LN annotation
    ax.text(x3 + bw*1.1/2, y_center - 1.5, '+ Residual + LN + Dropout(0.1)',
            ha='center', fontsize=6, color='#C62828')

    draw_arrow(ax, x2 + bw*1.2, y_center, x3, y_center + 0.3)

    # ── Stage 4: Classifier Head ──
    x4 = x3 + bw*1.1 + gap + 0.2
    draw_3d_block(ax, x4, y_center + 0.35, bw*0.9, bh*0.65, depth*0.5, C_POOL,
                  'GlobalAvgPool', ['$T\\to1$'], fontsize=7)
    draw_3d_block(ax, x4, y_center - 0.45, bw*0.8, bh*0.6, depth*0.5, C_FC,
                  'FC $64\\to32$', ['GELU'], fontsize=7)
    draw_3d_block(ax, x4, y_center - 1.15, bw*0.7, bh*0.55, depth*0.5, C_FC,
                  'FC $32\\to7$', fontsize=7)
    # Shape
    ax.text(x4 + bw*0.45, y_center + 0.25, '$64$', fontsize=7, color='#00897B')
    ax.text(x4 + bw*0.4, y_center - 0.55, '$32$', fontsize=7, color='#00897B')

    draw_arrow(ax, x3 + bw*1.1, y_center, x4, y_center + 0.3)

    # ── Output ──
    x5 = x4 + bw*0.9 + gap + 0.2
    draw_3d_block(ax, x5, y_center - 0.3, bw*0.85, bh*0.85, depth*0.6, C_OUTPUT,
                  'Softmax\n$\\mathbf{p}\\in\\mathbb{R}^{7}$', fontsize=8)
    ax.text(x5 + bw*0.85/2, y_center - 1.2, '$g{=}\\arg\\max\\mathbf{p}$\n$s{=}\\max\\mathbf{p}$',
            ha='center', fontsize=7, color='#00695C')

    draw_arrow(ax, x4 + bw*0.9, y_center, x5, y_center)

    # ── Parameter summary at bottom ──
    ax.text(8.0, 0.4, 'Total params: 56,711 (full)  /  40,199 (CNN-only ablation)',
            ha='center', fontsize=9, color='#546E7A', fontweight='bold')
    ax.text(8.0, 0.1, 'Memory: 221.5 KB (float32)  |  Inference: 1.32 ms on Snapdragon XR2 (Unity Barracuda)',
            ha='center', fontsize=7, color='#78909C')

    # ── Save ──
    outdir = 'experiment_results_v2/figures'
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f'{outdir}/fig_dbew_nn_architecture.pdf', dpi=300, bbox_inches='tight',
                pad_inches=0.1, facecolor='white', edgecolor='none')
    fig.savefig(f'{outdir}/fig_dbew_nn_architecture.png', dpi=300, bbox_inches='tight',
                pad_inches=0.1, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] Architecture diagram saved to {outdir}/fig_dbew_nn_architecture.pdf")

if __name__ == '__main__':
    main()
