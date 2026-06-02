#!/usr/bin/env python3
"""
DBEW-NN 架构图 — 学术论文规范版
按照 PlotNeuralNet / 顶会论文配色规范绘制

规范来源（搜索聚合）:
  - 配色: 淡蓝/淡红/淡黄/淡绿四主色 + 紫色模块 + 灰色辅助
  - 原则: 低饱和度, 全文配色统一, 矢量PDF导出
  - 参考: Rougier et al. (2014) 10条科研作图法则
  - 风格: PlotNeuralNet 同类 (TikZ-based), 本脚本用 matplotlib 复现

Usage:  python plot_dbw_nn_academic.py
Output: DBEW-NN_academic.png (300dpi) + DBEW-NN_academic.pdf (vector)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# 字体配置 (CJK support)
# ══════════════════════════════════════════════════════════════════════
_font_name = 'DejaVu Sans'  # fallback
for _candidate in ['Noto Sans SC', 'SimHei', 'Microsoft YaHei', 'STSong']:
    _fonts = [f for f in fm.fontManager.ttflist if _candidate in f.name]
    if _fonts:
        _font_name = _fonts[0].name
        break

plt.rcParams.update({
    'font.family': _font_name,
    'font.size': 8,
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ══════════════════════════════════════════════════════════════════════
# 学术论文配色方案 (PlotNeuralNet风格 + 低饱和度)
# ══════════════════════════════════════════════════════════════════════
C = {
    # 主色调 — 低饱和度, 打印友好
    'bg':         '#FFFFFF',  # 纯白背景
    'input_l':    '#C6DBEF',  # 输入 → 淡蓝
    'input_d':    '#3182BD',  # 输入边框
    'conv_l':     '#FDD0A2',  # 卷积层 → 淡橙 (ConvColor)
    'conv_d':     '#D94801',  # 卷积层边框
    'pool_l':     '#C7E9C0',  # 池化层 → 淡绿 (PoolColor)
    'pool_d':     '#238B45',  # 池化层边框
    'attn_l':     '#DADAEB',  # 注意力层 → 淡紫
    'attn_d':     '#6A51A3',  # 注意力层边框
    'fc_l':       '#F4CAE4',  # 全连接 → 淡粉 (FcColor)
    'fc_d':       '#C51B8A',  # 全连接边框
    'output_l':   '#C6DBEF',  # 输出 → 淡蓝
    'output_d':   '#08519C',  # 输出边框
    'pos_l':      '#FFF7BC',  # 位置编码 → 淡黄
    'pos_d':      '#D9A404',  # 位置编码边框
    'norm_l':     '#D9D9D9',  # 归一化 → 淡灰
    'norm_d':     '#636363',  # 归一化边框
    # 辅助色
    'arrow':      '#555555',
    'text':       '#333333',
    'dim_text':   '#888888',
    'bracket':    '#999999',
    'skip_line':  '#BDBDBD',
}

# ══════════════════════════════════════════════════════════════════════
# 绘图辅助函数
# ══════════════════════════════════════════════════════════════════════

def rounded_box(ax, x, y, w, h, facecolor, edgecolor, linewidth=1.0, zorder=2):
    """绘制圆角矩形块"""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def block_text(ax, x, y, title, subtitle=None, title_size=8, sub_size=6.5,
               title_color='#222222', sub_color='#555555', line_spacing=1.0):
    """在块内居中写标题和副标题"""
    if subtitle:
        y_offset = 0.08
        ax.text(x, y + y_offset, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=title_color, zorder=3)
        ax.text(x, y - 0.16 - (line_spacing - 1) * 0.05, subtitle, ha='center', va='top',
                fontsize=sub_size, style='italic', color=sub_color, zorder=3)
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=title_color, zorder=3)


def arrow(ax, x1, y1, x2, y2, color='#666666', lw=1.2):
    """绘制带箭头的连线"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0'))


def bracket_left(ax, x, y_top, y_bot, label, color='#999999'):
    """左侧大括号标注阶段"""
    bx = x - 0.35
    ax.plot([bx, bx], [y_top, y_bot], color=color, lw=0.8)
    ax.plot([bx, bx + 0.12], [y_top, y_top], color=color, lw=0.8)
    ax.plot([bx, bx + 0.12], [y_bot, y_bot], color=color, lw=0.8)
    mid = (y_top + y_bot) / 2
    ax.text(bx - 0.18, mid, label, ha='right', va='center',
            fontsize=6.5, color=color, fontweight='bold')


def shape_annot_right(ax, x, y, text, color='#888888'):
    """右侧张量形状标注"""
    ax.text(x + 0.35, y, text, ha='left', va='center',
            fontsize=6.5, color=color, fontfamily='monospace')


# ══════════════════════════════════════════════════════════════════════
# 主图绘制
# ══════════════════════════════════════════════════════════════════════

def main():
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7.5))

    # Canvas
    ax.set_xlim(-5.5, 6.0)
    ax.set_ylim(-10.5, 1.0)
    ax.axis('off')
    ax.set_aspect('equal')

    # ── Layout params (垂直流向, 从上到下) ──
    bw = 5.0         # 块宽度
    bh = 0.72        # 标准块高度
    gap = 0.18       # 块间距
    xc = 0.0         # 水平中心

    # ── 每块的Y坐标 ──
    y = {}
    _top = 0.3

    y['input']   = _top
    y['norm']    = y['input']  - bh - gap
    y['spatial'] = y['norm']   - bh - gap
    y['pool']    = y['spatial']- bh - gap
    y['pos']     = y['pool']   - bh - gap
    y['cnn']     = y['pos']    - 1.8 - gap  # 3层CNN taller block
    y['attn']    = y['cnn']    - 1.8 - gap
    y['gap']     = y['attn']   - bh - gap
    y['cls']     = y['gap']    - bh - gap
    y['output']  = y['cls']    - bh - gap

    cnn_h = 1.8   # CNN 3层叠块高度

    # ── 标题 ──
    ax.text(xc, y['input'] + 0.55,
            'DBEW-NN: Lightweight Skeleton-Based Gesture Recognition Network',
            ha='center', fontsize=11, fontweight='bold', color='#111111')

    # ══════════════════════════════════════════════════════════════════
    # 绘制每个块
    # ══════════════════════════════════════════════════════════════════

    # ── Input ──
    rounded_box(ax, xc, y['input'], bw, bh, C['input_l'], C['input_d'])
    block_text(ax, xc, y['input'],
               'Input Skeleton Sequence',
               '[ B, 32 frames, 26 joints, 3 coords ] · Rokid UXR Hand Tracking')
    shape_annot_right(ax, xc + bw/2, y['input'], '[B, 32, 26, 3]')

    # ── Normalization ──
    rounded_box(ax, xc, y['norm'], bw, bh, C['norm_l'], C['norm_d'])
    block_text(ax, xc, y['norm'],
               'Coordinate Normalization',
               'Wrist-relative: x - x_wrist · Global scale to unit variance')
    shape_annot_right(ax, xc + bw/2, y['norm'], '[B, 32, 26, 3]')

    # ── Spatial Embedding ──
    rounded_box(ax, xc, y['spatial'], bw, bh, C['conv_l'], C['conv_d'])
    block_text(ax, xc, y['spatial'],
               'Stage 1: SpatialEmbedding',
               '1×1 Conv: (J=26, C=3) → d_model=64  ·  LayerNorm')
    shape_annot_right(ax, xc + bw/2, y['spatial'], '[B, 32, 26, 64]')

    # ── Joint Pooling ──
    rounded_box(ax, xc, y['pool'], bw, bh, C['pool_l'], C['pool_d'])
    block_text(ax, xc, y['pool'],
               'Stage 2: JointWisePool (Mean)',
               'Mean over 26 joints → [B, T, 64]')
    shape_annot_right(ax, xc + bw/2, y['pool'], '[B, 32, 64]')

    # ── Positional Encoding ──
    rounded_box(ax, xc, y['pos'], bw, bh, C['pos_l'], C['pos_d'])
    block_text(ax, xc, y['pos'],
               'Stage 3: PositionalEncoding',
               'Sinusoidal PE · max_len=64')
    shape_annot_right(ax, xc + bw/2, y['pos'], '[B, 32, 64]')

    # ── Dilated Temporal CNN (taller, 3 sub-layers) ──
    rounded_box(ax, xc, y['cnn'], bw, cnn_h, C['conv_l'], C['conv_d'], linewidth=1.2)
    block_text(ax, xc, y['cnn'] + cnn_h/2 - 0.22,
               'Stage 4: DilatedTemporalCNN',
               '3-layer dilated 1D-CNN with GELU activation', title_size=8)

    # CNN 子层 (inside the large block)
    sub_names = [
        ('Conv1d(d=1, k=3)', 'Receptive field: 3 frames'),
        ('Conv1d(d=2, k=3)', 'Receptive field: 7 frames'),
        ('Conv1d(d=4, k=3)', 'Receptive field: 15 frames'),
    ]
    sub_y_start = y['cnn'] + cnn_h/2 - 0.58
    sub_h = 0.32
    sub_w = bw - 1.4
    for i, (title, sub) in enumerate(sub_names):
        sy = sub_y_start - i * (sub_h + 0.06)
        rounded_box(ax, xc, sy, sub_w, sub_h, '#FDE0C8', '#E87B2B', linewidth=0.6, zorder=3)
        block_text(ax, xc, sy, title, sub, title_size=6.5, sub_size=5.5,
                   title_color='#333333', sub_color='#777777')

    # CNN batch info
    cnn_bottom = y['cnn'] - cnn_h/2
    ax.text(xc, cnn_bottom + 0.10,
            'BatchNorm · Dropout(0.1) · Residual (per-layer + global)',
            ha='center', fontsize=5.8, style='italic', color='#666666', zorder=3)

    shape_annot_right(ax, xc + bw/2, y['cnn'], '[B, 32, 64]')

    # ── Self-Attention ──
    rounded_box(ax, xc, y['attn'], bw, bh + 0.1, C['attn_l'], C['attn_d'])
    block_text(ax, xc, y['attn'],
               'Stage 5: LightweightSelfAttention',
               'Single-layer 4-Head MHA · head_dim=16 · LayerNorm · Residual')
    shape_annot_right(ax, xc + bw/2, y['attn'], '[B, 32, 64]')

    # ── GAP ──
    rounded_box(ax, xc, y['gap'], bw, bh, C['pool_l'], C['pool_d'])
    block_text(ax, xc, y['gap'],
               'Stage 6a: Global Average Pooling',
               'GAP over temporal dim → [B, 64] · LayerNorm')
    shape_annot_right(ax, xc + bw/2, y['gap'], '[B, 64]')

    # ── Classifier Head ──
    rounded_box(ax, xc, y['cls'], bw, bh, C['fc_l'], C['fc_d'])
    block_text(ax, xc, y['cls'],
               'Stage 6b: Classifier Head',
               'Linear(64→32) → GELU → Dropout(0.1) → Linear(32→7)')
    shape_annot_right(ax, xc + bw/2, y['cls'], '[B, 7]')

    # ── Output ──
    rounded_box(ax, xc, y['output'], bw, bh, C['output_l'], C['output_d'], linewidth=1.2)
    block_text(ax, xc, y['output'],
               'Output: Gesture Logits',
               'Softmax → {NONE, Index←, Index→, TwoFinger·Palm, TwoFinger·Back, FourFinger, Fist}')
    shape_annot_right(ax, xc + bw/2, y['output'], '[B, 7]')

    # ══════════════════════════════════════════════════════════════════
    # 箭头连接
    # ══════════════════════════════════════════════════════════════════
    block_y = [y['input'], y['norm'], y['spatial'], y['pool'], y['pos'],
               y['cnn'], y['attn'], y['gap'], y['cls'], y['output']]
    block_h = [bh, bh, bh, bh, bh, cnn_h, bh + 0.1, bh, bh, bh]

    for i in range(len(block_y) - 1):
        y1 = block_y[i] - block_h[i]/2
        y2 = block_y[i+1] + block_h[i+1]/2
        arrow(ax, xc, y1, xc, y2, C['arrow'], lw=1.2)

    # ══════════════════════════════════════════════════════════════════
    # 左侧阶段标注
    # ══════════════════════════════════════════════════════════════════
    stages = [
        (y['input'] + bh/2, y['norm'] - bh/2,   'Preprocessing'),
        (y['spatial'] + bh/2, y['pool'] - bh/2,  'Spatial\nFeature'),
        (y['pos'] + bh/2, cnn_bottom + 0.05,     'Temporal\nModeling'),
        (y['attn'] + (bh+0.1)/2, y['attn'] - (bh+0.1)/2, 'Global\nAttention'),
        (y['gap'] + bh/2, y['output'] - bh/2,    'Classification'),
    ]
    for yt, yb, label in stages:
        bracket_left(ax, xc - bw/2, yt, yb, label, C['bracket'])

    # ══════════════════════════════════════════════════════════════════
    # 消融标注 (虚线指向)
    # ══════════════════════════════════════════════════════════════════
    # CNN-only ablation: skip arrow bypassing Stage 5
    bypass_start_x = xc + bw/2 + 0.65
    bypass_top_y = y['cnn'] - cnn_h/2
    bypass_bot_y = y['attn'] + (bh + 0.1)/2

    ax.annotate('', xy=(bypass_start_x, y['gap'] + bh/2),
                xytext=(bypass_start_x, bypass_top_y),
                arrowprops=dict(arrowstyle='->', color=C['skip_line'],
                                lw=1.0, linestyle='dashed',
                                connectionstyle='arc3,rad=0'))

    ax.text(bypass_start_x + 0.35, (bypass_top_y + y['gap'] + bh/2) / 2,
            'CNN-only\nablation\n(remove\nStage 5)',
            ha='left', va='center', fontsize=5.5, color='#aaaaaa', style='italic')

    # ══════════════════════════════════════════════════════════════════
    # 底部图例
    # ══════════════════════════════════════════════════════════════════
    legend_y = y['output'] - bh/2 - 0.6
    legend_items = [
        (C['input_l'], C['input_d'], 'Input/Output'),
        (C['conv_l'], C['conv_d'], 'Conv / Embed'),
        (C['pool_l'], C['pool_d'], 'Pooling'),
        (C['pos_l'], C['pos_d'], 'Positional Enc.'),
        (C['attn_l'], C['attn_d'], 'Self-Attention'),
        (C['fc_l'], C['fc_d'], 'FC / Classifier'),
        (C['norm_l'], C['norm_d'], 'Normalization'),
    ]

    lx = -3.2
    lw_small = 0.25
    for i, (face, edge, label) in enumerate(legend_items):
        ix = lx + i * 1.05
        rounded_box(ax, ix, legend_y, lw_small * 2, lw_small * 1.2, face, edge,
                    linewidth=0.5, zorder=3)
        ax.text(ix, legend_y - 0.22, label, ha='center', va='top',
                fontsize=5.5, color='#555555')

    # ══════════════════════════════════════════════════════════════════
    # 底部参数信息
    # ══════════════════════════════════════════════════════════════════
    ax.text(xc, legend_y - 0.7,
            'Total params: ~57K    |    CNN: ~37K    |    Attention: ~16K    '
            '|    Target latency: < 2ms (Snapdragon XR2)    |    ONNX → Unity Barracuda',
            ha='center', fontsize=6, color='#999999')

    # ══════════════════════════════════════════════════════════════════
    # 保存
    # ══════════════════════════════════════════════════════════════════
    for fmt, dpi in [('png', 300), ('pdf', None)]:
        fname = f'DBEW-NN_academic.{fmt}'
        kwargs = dict(bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.15)
        if dpi:
            kwargs['dpi'] = dpi
        fig.savefig(fname, **kwargs)
        print(f'Saved: {fname}')

    plt.close(fig)
    print('Done. Academic-style diagram generated successfully.')


if __name__ == '__main__':
    main()
