#!/usr/bin/env python3
"""
Generate DBEW-NN architecture diagram in NN-SVG style (SVG output).

NN-SVG (alexlenail.me/NN-SVG) offers three styles:
  - FCNN:  fully-connected layers, vertical rectangles
  - LeNet: CNN blocks (conv→pool→conv→pool→fc→fc), horizontal flow
  - AlexNet: deeper CNN with parallel branches

DBEW-NN is a 1D-CNN + Self-Attention pipeline on skeleton data.
This script generates SVG in the "LeNet-style" horizontal block flow,
which is the closest match among NN-SVG's three supported formats.

Usage:  python plot_dbw_nn_svg.py
Output: DBEW-NN_nn_svg_style.svg
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════
# NN-SVG 配色表 (from NN-SVG source, low-saturation academic palette)
# ══════════════════════════════════════════════════════════════════════
COLORS = {
    # Conv layers → warm orange/yellow (NN-SVG ConvColor)
    'conv_fill':   '#FDD0A2',
    'conv_stroke': '#D94801',
    # Pool layers → muted red (NN-SVG PoolColor)
    'pool_fill':   '#C7E9C0',
    'pool_stroke': '#238B45',
    # FC / Embed / Classifier → blue-red (NN-SVG FcColor)
    'fc_fill':     '#F4CAE4',
    'fc_stroke':   '#C51B8A',
    # Attention / Transformer modules → purple
    'attn_fill':   '#DADAEB',
    'attn_stroke': '#6A51A3',
    # Input / Output / Normalization → light blue
    'io_fill':     '#C6DBEF',
    'io_stroke':   '#3182BD',
    # Positional encoding → light yellow
    'pe_fill':     '#FFF7BC',
    'pe_stroke':   '#D9A404',
    # Structural
    'bg':          '#FFFFFF',
    'arrow':       '#666666',
    'text':        '#222222',
    'subtext':     '#666666',
    'dim_text':    '#999999',
    'dash_line':   '#BBBBBB',
    'edge_dim':    '#E0E0E0',  # subtle dimension box edge
}

WIDTH = 960
HEIGHT = 1280
LEFT_MARGIN = 80
RIGHT_MARGIN = 80
TOP_MARGIN = 60
BOTTOM_MARGIN = 80

# Block geometry
BLOCK_W = 520   # main block width
BLOCK_H = 56    # standard block height
BLOCK_H_TALL = 180  # CNN tall block
GAP = 24        # vertical gap
CENTER_X = LEFT_MARGIN + 60 + BLOCK_W / 2  # visual center

# Left label area
LABEL_X = LEFT_MARGIN
LABEL_W = 48

# Right dimension area
DIM_X = LEFT_MARGIN + 60 + BLOCK_W + 30


def svg_rect(x, y, w, h, rx, fill, stroke, sw=1.0, cls=""):
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw:.1f}" class="{cls}"/>')


def svg_text(x, y, text, size=13, color=COLORS['text'], anchor="middle",
             weight="normal", italic=False, ff="sans-serif", cls=""):
    fw = f'font-weight="{weight}" ' if weight != "normal" else ''
    fs = f'font-style="italic" ' if italic else ''
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{ff}" '
            f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'{fw}{fs}class="{cls}">{text}</text>')


def svg_arrow(x1, y1, x2, y2, color=COLORS['arrow'], sw=1.5, dash=False):
    dash_str = 'stroke-dasharray="6,4" ' if dash else ''
    marker = 'url(#arrowhead)' if not dash else 'url(#arrowhead_dash)'
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{sw:.1f}" {dash_str}marker-end="{marker}"/>')


def svg_bracket(x, y_top, y_bot, label, color='#999999'):
    """Left-side bracket annotation."""
    parts = []
    # vertical line
    parts.append(f'<line x1="{x}" y1="{y_top}" x2="{x}" y2="{y_bot}" '
                 f'stroke="{color}" stroke-width="1.0"/>')
    # top tick
    parts.append(f'<line x1="{x}" y1="{y_top}" x2="{x+10}" y2="{y_top}" '
                 f'stroke="{color}" stroke-width="1.0"/>')
    # bottom tick
    parts.append(f'<line x1="{x}" y1="{y_bot}" x2="{x+10}" y2="{y_bot}" '
                 f'stroke="{color}" stroke-width="1.0"/>')
    # label
    mid = (y_top + y_bot) / 2
    parts.append(svg_text(x - 8, mid + 4, label, size=12, color=color,
                          anchor="end", weight="bold"))
    return '\n'.join(parts)


def make_block(x, y, w, h, title, subtitle, fill, stroke, sw=1.2):
    """Generate a rounded-rect block with title + subtitle."""
    parts = []
    parts.append(svg_rect(x - w/2, y - h/2, w, h, 6, fill, stroke, sw))
    parts.append(svg_text(x, y - 4, title, size=14, weight="bold"))
    if subtitle:
        parts.append(svg_text(x, y + 17, subtitle, size=11, color=COLORS['subtext'],
                              italic=True))
    return '\n'.join(parts)


def build_svg():
    """Build the complete SVG document."""
    cy = TOP_MARGIN  # current y position (top of next block)

    def next_block(h=BLOCK_H):
        nonlocal cy
        top = cy
        cy += h + GAP
        return top + h / 2  # return block center Y

    def block_center(h=BLOCK_H):
        return cy - GAP - h / 2

    # ── Collect all blocks ──
    blocks = []
    dims = []  # right-side dimension annotations
    arrows_y = []  # y positions for down arrows
    brackets = []  # left-side bracket annotations

    # ── Title ──
    cy += 20
    blocks.append(
        svg_text(CENTER_X, cy - 10,
                 'DBEW-NN: Lightweight 1D-CNN + Self-Attention Gesture Classifier',
                 size=18, color='#111111', weight='bold'))
    cy += 36

    # ── Input ──
    by = next_block()
    blocks.append(make_block(CENTER_X, by, BLOCK_W, BLOCK_H,
        'Input Skeleton Sequence',
        '[B, 32 frames, 26 joints, 3 coords]',
        COLORS['io_fill'], COLORS['io_stroke']))
    dims.append((DIM_X, by, '[B, 32, 26, 3]'))
    arrows_y.append(cy - GAP)

    # ── Normalization ──
    by = next_block()
    blocks.append(make_block(CENTER_X, by, BLOCK_W, BLOCK_H,
        'Coordinate Normalization',
        'Wrist-relative coords · Unit-variance global scaling',
        COLORS['io_fill'], COLORS['io_stroke'], sw=0.8))
    dims.append((DIM_X, by, '[B, 32, 26, 3]'))
    arrows_y.append(cy - GAP)

    # ── Stage 1: SpatialEmbedding ──
    by = next_block()
    blocks.append(make_block(CENTER_X, by, BLOCK_W, BLOCK_H,
        'Stage 1: SpatialEmbedding (1×1 Conv)',
        '(J=26, C=3) → d_model=64  +  LayerNorm',
        COLORS['conv_fill'], COLORS['conv_stroke']))
    dims.append((DIM_X, by, '[B, 32, 26, 64]'))
    arrows_y.append(cy - GAP)

    # ── Stage 2: JointWisePool ──
    by = next_block()
    blocks.append(make_block(CENTER_X, by, BLOCK_W, BLOCK_H,
        'Stage 2: JointWisePool (Mean)',
        'Mean over 26 joints → [B, 32, 64]',
        COLORS['pool_fill'], COLORS['pool_stroke']))
    dims.append((DIM_X, by, '[B, 32, 64]'))
    arrows_y.append(cy - GAP)

    # ── Stage 3: PositionalEncoding ──
    by = next_block()
    blocks.append(make_block(CENTER_X, by, BLOCK_W, BLOCK_H,
        'Stage 3: PositionalEncoding',
        'Sinusoidal PE  ·  max_len=64',
        COLORS['pe_fill'], COLORS['pe_stroke']))
    dims.append((DIM_X, by, '[B, 32, 64]'))
    arrows_y.append(cy - GAP)

    # ── Stage 4: Dilated Temporal CNN (TALL BLOCK) ──
    cnn_top = cy
    by = next_block(h=BLOCK_H_TALL)
    cnn_center = by

    blocks.append(svg_rect(CENTER_X - BLOCK_W/2, cnn_top, BLOCK_W, BLOCK_H_TALL,
                           6, COLORS['conv_fill'], COLORS['conv_stroke'], sw=1.5))
    blocks.append(svg_text(CENTER_X, cnn_top + 22, 'Stage 4: DilatedTemporalCNN',
                           size=14, weight="bold"))

    # Three sub-layers inside CNN block
    sub_w = BLOCK_W - 80
    sub_h = 38
    sub_gap = 12
    sub_start_y = cnn_top + 52
    sub_labels = [
        ('Conv1d  d=1, k=3', 'Receptive field: 3 frames · GELU'),
        ('Conv1d  d=2, k=3', 'Receptive field: 7 frames · GELU'),
        ('Conv1d  d=4, k=3', 'Receptive field: 15 frames · GELU'),
    ]
    for i, (title, sub) in enumerate(sub_labels):
        sy = sub_start_y + i * (sub_h + sub_gap)
        blocks.append(svg_rect(CENTER_X - sub_w/2, sy, sub_w, sub_h, 4,
                               '#FDE0C8', '#E87B2B', sw=0.7))
        blocks.append(svg_text(CENTER_X, sy + 16, title, size=12, weight="bold",
                               color='#333333'))
        blocks.append(svg_text(CENTER_X, sy + 31, sub, size=10,
                               color='#777777', italic=True))

    # CNN bottom note
    blocks.append(svg_text(CENTER_X, cnn_top + BLOCK_H_TALL - 14,
                           'BatchNorm · Dropout(0.1) · Per-layer + Global Residual',
                           size=10, color=COLORS['subtext'], italic=True))

    dims.append((DIM_X, by, '[B, 32, 64]'))
    arrows_y.append(cy - GAP)

    # ── Stage 5: Self-Attention ──
    by_top = cy
    by = next_block(h=BLOCK_H + 8)
    blocks.append(make_block(CENTER_X, by, BLOCK_W, BLOCK_H + 8,
        'Stage 5: LightweightSelfAttention',
        'Single-layer 4-Head MHA  ·  head_dim=16  ·  LayerNorm  ·  Residual',
        COLORS['attn_fill'], COLORS['attn_stroke'], sw=1.2))
    dims.append((DIM_X, by, '[B, 32, 64]'))
    arrows_y.append(cy - GAP)

    # ── Stage 6a: GAP ──
    by = next_block()
    blocks.append(make_block(CENTER_X, by, BLOCK_W, BLOCK_H,
        'Stage 6a: Global Average Pooling',
        'GAP over temporal dim → [B, 64]  +  LayerNorm',
        COLORS['pool_fill'], COLORS['pool_stroke']))
    dims.append((DIM_X, by, '[B, 64]'))
    arrows_y.append(cy - GAP)

    # ── Stage 6b: Classifier Head ──
    by = next_block()
    blocks.append(make_block(CENTER_X, by, BLOCK_W, BLOCK_H,
        'Stage 6b: Classifier Head',
        'Linear(64→32) → GELU → Dropout(0.1) → Linear(32→7)',
        COLORS['fc_fill'], COLORS['fc_stroke']))
    dims.append((DIM_X, by, '[B, 7]'))
    arrows_y.append(cy - GAP)

    # ── Output ──
    by = next_block()
    blocks.append(make_block(CENTER_X, by, BLOCK_W, BLOCK_H,
        'Output: Gesture Logits',
        'Softmax → 7 gesture classes (NONE + 6 gestures)',
        COLORS['io_fill'], COLORS['io_stroke'], sw=1.5))
    dims.append((DIM_X, by, '[B, 7]'))

    # ── Arrows between blocks ──
    block_centers_y = [
        arrows_y[0] - GAP/2,  # after input
    ]
    # We need actual block centers
    y_positions = []
    block_heights = [BLOCK_H, BLOCK_H, BLOCK_H, BLOCK_H, BLOCK_H, BLOCK_H_TALL,
                     BLOCK_H + 8, BLOCK_H, BLOCK_H, BLOCK_H]
    current = TOP_MARGIN + 20 + 36  # after title
    for h in block_heights:
        y_positions.append(current + h/2)
        current += h + GAP

    for i in range(len(y_positions) - 1):
        y1 = y_positions[i] + block_heights[i] / 2
        y2 = y_positions[i+1] - block_heights[i+1] / 2
        blocks.append(svg_arrow(CENTER_X, y1, CENTER_X, y2))

    # ── Left-side brackets ──
    bx = LEFT_MARGIN + 14  # bracket x position
    bh = block_heights
    stages = [
        (y_positions[0] + bh[0]/2, y_positions[1] - bh[1]/2, 'Preprocessing'),
        (y_positions[2] + bh[2]/2, y_positions[3] - bh[3]/2, 'Spatial'),
        (y_positions[4] + bh[4]/2, y_positions[5] - bh[5]/2, 'Temporal'),
        (y_positions[6] + bh[6]/2, y_positions[6] - bh[6]/2, 'Attention'),
        (y_positions[7] + bh[7]/2, y_positions[9] - bh[9]/2, 'Classification'),
    ]
    for yt, yb, label in stages:
        blocks.append(svg_bracket(bx, yt, yb, label))

    # ── Right-side dimension labels ──
    for dx, dy, dlabel in dims:
        blocks.append(svg_text(dx, dy + 4, dlabel, size=12,
                               color=COLORS['dim_text'], anchor="start",
                               ff="monospace"))

    # ── Ablation: CNN-only bypass dashed line ──
    bypass_x = CENTER_X + BLOCK_W/2 + 44
    bypass_y1 = y_positions[5] - BLOCK_H_TALL/2   # below CNN
    bypass_y2 = y_positions[7] + BLOCK_H/2         # above GAP
    blocks.append(svg_arrow(bypass_x, bypass_y1, bypass_x, bypass_y2,
                            color=COLORS['dash_line'], sw=1.2, dash=True))
    blocks.append(svg_text(bypass_x + 10, (bypass_y1 + bypass_y2)/2 + 4,
                           'CNN-only\nablation\n(remove Stage 5)',
                           size=10, color='#BBBBBB', anchor="start", italic=True))

    # ── Bottom legend ──
    legend_y = current + 36
    legend_items = [
        (COLORS['io_fill'], COLORS['io_stroke'], 'Input / Output'),
        (COLORS['conv_fill'], COLORS['conv_stroke'], 'Conv / Embedding'),
        (COLORS['pool_fill'], COLORS['pool_stroke'], 'Pooling'),
        (COLORS['pe_fill'], COLORS['pe_stroke'], 'Positional Enc.'),
        (COLORS['attn_fill'], COLORS['attn_stroke'], 'Self-Attention'),
        (COLORS['fc_fill'], COLORS['fc_stroke'], 'FC / Classifier'),
    ]
    lx_start = CENTER_X - (len(legend_items) - 1) * 65
    for i, (fill, stroke, label) in enumerate(legend_items):
        lx = lx_start + i * 130
        blocks.append(svg_rect(lx - 12, legend_y - 8, 24, 16, 3, fill, stroke, sw=0.6))
        blocks.append(svg_text(lx, legend_y + 22, label, size=10,
                               color=COLORS['subtext']))

    # ── Bottom stats ──
    blocks.append(svg_text(CENTER_X, legend_y + 48,
        'Total: ~57K params  |  CNN: ~37K  |  Attention: ~16K  |  '
        'Inference: <2ms (Snapdragon XR2)  |  Export: ONNX → Unity Barracuda',
        size=10, color=COLORS['dim_text']))

    # ── Assemble full SVG ──
    total_h = legend_y + 70
    _arrow_color = COLORS['arrow']
    _dash_color = COLORS['dash_line']
    _bg_color = COLORS['bg']
    _nl = '\n'

    svg_parts = []
    svg_parts.append(f'<?xml version="1.0" encoding="UTF-8"?>')
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{total_h}" viewBox="0 0 {WIDTH} {total_h}">')
    svg_parts.append('  <defs>')
    svg_parts.append(f'    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">')
    svg_parts.append(f'      <polygon points="0 0, 8 3, 0 6" fill="{_arrow_color}"/>')
    svg_parts.append(f'    </marker>')
    svg_parts.append(f'    <marker id="arrowhead_dash" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">')
    svg_parts.append(f'      <polygon points="0 0, 8 3, 0 6" fill="{_dash_color}"/>')
    svg_parts.append(f'    </marker>')
    svg_parts.append(f"    <style>")
    svg_parts.append(f"      text {{ font-family: 'Noto Sans SC', 'DejaVu Sans', 'Arial', sans-serif; }}")
    svg_parts.append(f"    </style>")
    svg_parts.append(f'  </defs>')
    svg_parts.append(f'  <!-- Background -->')
    svg_parts.append(f'  <rect width="100%" height="100%" fill="{_bg_color}"/>')
    svg_parts.append(f'  <!-- All elements -->')
    svg_parts.extend(blocks)
    svg_parts.append('</svg>')
    svg = _nl.join(svg_parts)

    return svg


def main():
    svg = build_svg()
    out_path = 'DBEW-NN_nn_svg_style.svg'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(svg)
    print(f'Saved: {out_path} ({len(svg):,} bytes)')

    # Also write a compact version
    import re
    svg_compact = re.sub(r'\n\s+', '\n', svg)
    out_compact = 'DBEW-NN_nn_svg_style.min.svg'
    with open(out_compact, 'w', encoding='utf-8') as f:
        f.write(svg_compact)
    print(f'Saved: {out_compact} ({len(svg_compact):,} bytes)')


if __name__ == '__main__':
    main()
