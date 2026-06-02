#!/usr/bin/env python3
"""
Generate DBEW-NN architecture diagram for thesis.
Output: DBEW-NN_architecture.png (300dpi) + .pdf (vector)
Usage:  python plot_dbw_nn.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Use CJK-capable font ──
_cjk_fonts = [f for f in fm.fontManager.ttflist if 'Noto Sans SC' in f.name]
if _cjk_fonts:
    _font_prop = fm.FontProperties(fname=_cjk_fonts[0].fname)
    plt.rcParams['font.family'] = _cjk_fonts[0].name
else:
    _font_prop = None
    # Fallback: try SimHei, Microsoft YaHei, etc.
    for _name in ['SimHei', 'Microsoft YaHei', 'STSong', 'KaiTi']:
        _f = [f for f in fm.fontManager.ttflist if _name in f.name]
        if _f:
            plt.rcParams['font.family'] = _f[0].name
            break

plt.rcParams['axes.unicode_minus'] = False

# ── Color palette (ColorBrewer-inspired, print-friendly) ──
COLORS = {
    'input':    '#E8F0FE',
    'spatial':  '#D4E6F1',
    'pool':     '#D5F5E3',
    'pos':      '#FCF3CF',
    'cnn':      '#FADBD8',
    'attn':     '#EBDEF0',
    'cls':      '#F5B7B1',
    'output':   '#ABEBC6',
    'arrow':    '#555555',
    'stage':    '#777777',
    'dim':      '#888888',
}


def draw_block(ax, x, y, w, h, text, color, edge_color='#333333',
               title_size=9, body_size=7):
    """Draw a rounded rectangle block with title and body lines."""
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle="round,pad=0.1",
                         facecolor=color,
                         edgecolor=edge_color,
                         linewidth=0.8,
                         zorder=2)
    ax.add_patch(box)

    lines = text.strip().split('\n')
    if not lines:
        return

    # Title (first line)
    ax.text(x, y + h / 2 - 0.18, lines[0],
            ha='center', va='top',
            fontsize=title_size, fontweight='bold', zorder=3)

    # Body (italic, gray)
    for i, line in enumerate(lines[1:]):
        ax.text(x, y + h / 2 - 0.42 - i * 0.18, line,
                ha='center', va='top',
                fontsize=body_size, style='italic', color='#444444', zorder=3)


def draw_arrow(ax, x1, y1, x2, y2):
    """Draw arrow from (x1,y1) to (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'],
                                lw=1.5, connectionstyle='arc3,rad=0'))


def draw_stage_bracket(ax, x_center, w, y_top, y_bot, label):
    """Draw a bracket on the left side with a stage label."""
    bx = x_center - w / 2 - 0.30
    ax.plot([bx, bx], [y_top + 0.25, y_bot - 0.25],
            color=COLORS['stage'], lw=1.0)
    ax.plot([bx, bx + 0.10], [y_top + 0.25, y_top + 0.25],
            color=COLORS['stage'], lw=1.0)
    ax.plot([bx, bx + 0.10], [y_bot - 0.25, y_bot - 0.25],
            color=COLORS['stage'], lw=1.0)

    mid = (y_top + y_bot) / 2
    ax.text(x_center - w / 2 - 0.55, mid, label,
            ha='center', va='center',
            fontsize=7, color=COLORS['stage'], fontweight='bold')


def main():
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 13.5))

    # ── Canvas bounds ──
    ax.set_xlim(-4.5, 4.8)
    ax.set_ylim(-16.0, 2.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # ── Layout parameters ──
    w = 4.2       # block width
    h = 0.85      # standard block height
    gap = 1.35    # vertical spacing
    xc = 0.0      # horizontal center

    # ── Y coordinates (top to bottom) ──
    yi = 1.20     # input
    yn = yi - gap              # normalization
    ys = yn - gap              # spatial embedding
    yp = ys - gap              # joint pool
    ype = yp - gap             # pos encoding
    ycnn = ype - gap - 0.30   # dilated CNN (taller)
    yattn = ycnn - gap - 0.95 # self-attention
    ygap = yattn - gap         # GAP + head
    yo = ygap - gap            # output

    cnn_h = 1.9  # CNN block height

    # ── Title ──
    ax.text(xc, yi + 0.75, 'DBEW-NN: 轻量级手势识别神经网络架构',
            ha='center', fontsize=12, fontweight='bold')

    # ── Draw blocks ──
    draw_block(ax, xc, yi, w, h,
               '输入 (Input)\n'
               '[B, T=32, J=26, C=3] — 手部骨骼序列 (Rokid UXR 26关节 × 3维坐标)',
               COLORS['input'], title_size=9, body_size=7)

    draw_block(ax, xc, yn, w, h,
               '坐标归一化 (Preprocessing)\n'
               '腕部相对坐标: $x \\leftarrow x - x_{\\mathrm{wrist}}$ · 全局缩放至单位方差',
               COLORS['input'], title_size=9, body_size=7)

    draw_block(ax, xc, ys, w, h,
               'Stage 1: 空间嵌入 (SpatialEmbedding)\n'
               '1×1 Conv: (J=26, C=3) $\\rightarrow$ d$_{\\mathrm{model}}$=64 · LayerNorm',
               COLORS['spatial'], title_size=9, body_size=7)

    draw_block(ax, xc, yp, w, h,
               'Stage 2: 关节池化 (JointWisePool — Mean)\n'
               '对 26 关节做均值池化 $\\rightarrow$ [B, T, 64]',
               COLORS['pool'], title_size=9, body_size=7)

    draw_block(ax, xc, ype, w, h,
               'Stage 3: 位置编码 (PositionalEncoding)\n'
               '正弦位置编码 (Sinusoidal PE), max$_{\\mathrm{len}}$=64',
               COLORS['pos'], title_size=9, body_size=7)

    # CNN block (taller, 3 sub-layers)
    cnn_text = (
        'Stage 4: 膨胀时序卷积 (DilatedTemporalCNN)\n'
        'Conv1d(64,64, k=3, d=1) $\\rightarrow$ BatchNorm $\\rightarrow$ GELU $\\rightarrow$ Dropout(0.1)\n'
        'Conv1d(64,64, k=3, d=2) $\\rightarrow$ BatchNorm $\\rightarrow$ GELU $\\rightarrow$ Dropout(0.1)\n'
        'Conv1d(64,64, k=3, d=4) $\\rightarrow$ BatchNorm $\\rightarrow$ GELU $\\rightarrow$ Dropout(0.1)\n'
        '每层残差连接 + 全局残差连接 · 感受野: 3, 7, 15 帧'
    )
    draw_block(ax, xc, ycnn, w, cnn_h, cnn_text,
               COLORS['cnn'], title_size=9, body_size=6.5)

    draw_block(ax, xc, yattn, w, h,
               'Stage 5: 轻量自注意力 (LightweightSelfAttention)\n'
               '单层 4-Head MHA (head$_{\\mathrm{dim}}$=16) · LayerNorm · 残差连接',
               COLORS['attn'], title_size=9, body_size=7)

    draw_block(ax, xc, ygap, w, h + 0.15,
               'Stage 6: 全局平均池化 + 分类头 (Classifier Head)\n'
               'GAP over T $\\rightarrow$ [B,64] $\\rightarrow$ LN $\\rightarrow$ Linear(64$\\rightarrow$32) '
               '$\\rightarrow$ GELU $\\rightarrow$ Dropout $\\rightarrow$ Linear(32$\\rightarrow$7)',
               COLORS['cls'], title_size=9, body_size=7)

    draw_block(ax, xc, yo, w, h,
               '输出 (Output)\n'
               '[B, 7] logits $\\rightarrow$ Softmax $\\rightarrow$ 7 类手势 + 置信度',
               COLORS['output'], title_size=9, body_size=7)

    # ── Arrows ──
    y_positions = [yi, yn, ys, yp, ype, ycnn, yattn, ygap, yo]
    heights = [h, h, h, h, h, cnn_h, h, h + 0.15, h]
    for i in range(len(y_positions) - 1):
        y1 = y_positions[i] - heights[i] / 2 + 0.02
        y2 = y_positions[i + 1] + heights[i + 1] / 2 - 0.02
        draw_arrow(ax, xc, y1, xc, y2)

    # ── Right side: tensor shapes ──
    shapes = [
        (yi, '[B, 32, 26, 3]'),
        (yn, '[B, 32, 26, 3]'),
        (ys, '[B, 32, 26, 64]'),
        (yp, '[B, 32, 64]'),
        (ype, '[B, 32, 64]'),
        (ycnn, '[B, 32, 64]'),
        (yattn, '[B, 32, 64]'),
        (ygap, '[B, 64]'),
        (yo, '[B, 7]'),
    ]
    for y, label in shapes:
        ax.text(xc + w / 2 + 0.35, y, label,
                va='center', fontsize=7.5, color=COLORS['dim'])

    # ── Left side: stage brackets ──
    stages = [
        (yi + 0.1, yn - 0.1,          '预处理\nPreprocess'),
        (ys + 0.1, yp - 0.1,          '特征提取\nFeature'),
        (ype + 0.1, ycnn - cnn_h/2 + 0.3, '时序建模\nTemporal'),
        (yattn - 0.05, yattn + 0.05,  '全局建模\nGlobal'),
        (ygap - 0.1, yo + 0.1,        '分类输出\nClassify'),
    ]
    for y_top, y_bot, label in stages:
        draw_stage_bracket(ax, xc, w, y_top, y_bot, label)

    # ── Bottom: parameter summary ──
    ax.text(xc, yo - gap + 0.15,
            '总参数量: ~57K   |   CNN: ~37K   |   Attention: ~16K   |   目标推理延迟: <2ms (Snapdragon XR2)',
            ha='center', fontsize=7.5, color='#999999')

    # ── Ablation note ──
    ax.text(xc + w / 2 + 0.35, yo - gap - 0.2,
            '消融变体\nCNN-only\n(移除 Stage 5)',
            fontsize=7, color='#aaaaaa', va='top')

    # ── Save ──
    for fmt, dpi in [('png', 300), ('pdf', None)]:
        fname = f'DBEW-NN_architecture.{fmt}'
        kwargs = dict(bbox_inches='tight', facecolor='white', edgecolor='none')
        if dpi:
            kwargs['dpi'] = dpi
        fig.savefig(fname, **kwargs)
        print(f'Saved: {fname}')

    plt.close(fig)


if __name__ == '__main__':
    main()
