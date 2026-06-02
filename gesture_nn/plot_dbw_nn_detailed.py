#!/usr/bin/env python3
"""
DBEW-NN 详细网络结构图 — 横向流式布局，论文风格
严格计算所有坐标，避免重叠和比例失调

Layout (left → right, 1600×800):
  Col 0: Hand Skeleton Input
  Col 1: SpatialEmbedding (1×1 Conv)
  Col 2: JointPool + PositionalEncoding
  Col 3: DilatedTemporalCNN (3 layers detailed)
  Col 4: Self-Attention (QKV + Attention Matrix)
  Col 5: GAP + Classifier + Output

Usage: python plot_dbw_nn_detailed.py
Output: DBEW-NN_detailed_architecture.svg
"""

# ══════════════════════════════════════════════════════════════════════
# Canvas & layout constants
# ══════════════════════════════════════════════════════════════════════
W, H = 1600, 780
MARGIN_L = 30
MARGIN_R = 20
MARGIN_TOP = 50
MARGIN_BOT = 40

# 6 columns, equally spaced
N_COLS = 6
AVAIL_W = W - MARGIN_L - MARGIN_R
COL_W = AVAIL_W / N_COLS
COL_GAP = 8
INNER_W = COL_W - COL_GAP * 2

# Column center X positions
def col_cx(i):
    return MARGIN_L + COL_W * i + COL_W / 2

# Vertical zones
Y_TITLE = 28
Y_STAGE_LABEL = 50        # stage title
Y_BLOCK_TOP = 70          # main block top
BLOCK_H = 220             # main visualization block height
Y_BLOCK_MID = Y_BLOCK_TOP + BLOCK_H / 2  # 180
Y_BLOCK_BOT = Y_BLOCK_TOP + BLOCK_H       # 290
Y_DIMS = Y_BLOCK_BOT + 16                 # dimension label
Y_ARROW = Y_BLOCK_MID                      # horizontal arrow y
Y_PARAMS = 340            # detail params
Y_BOTTOM_LINE = 390       # bottom annotation area

# Bottom detail areas
Y_DETAIL_TOP = 420
DETAIL_H = 280
Y_DETAIL_MID = Y_DETAIL_TOP + DETAIL_H / 2

Y_LEGEND = 730
Y_FOOTER = 760

# ══════════════════════════════════════════════════════════════════════
# Colors
# ══════════════════════════════════════════════════════════════════════
C = {
    'bg':            '#FFFFFF',
    'fill_blue':     '#D6EAF8',
    'stroke_blue':   '#2E86C1',
    'fill_orange':   '#FDEBD0',
    'stroke_orange': '#E67E22',
    'fill_green':    '#D5F5E3',
    'stroke_green':  '#27AE60',
    'fill_purple':   '#E8DAEF',
    'stroke_purple': '#8E44AD',
    'fill_pink':     '#FADBD8',
    'stroke_pink':   '#E74C3C',
    'fill_yellow':   '#FCF3CF',
    'stroke_yellow': '#F1C40F',
    'fill_gray':     '#EAECEE',
    'stroke_gray':   '#99A3A4',
    'joint':         '#E74C3C',
    'bone':          '#85929E',
    'arrow':         '#616A6B',
    'attn_heat':     '#AF7AC5',
    'text':          '#1C2833',
    'sub':           '#566573',
    'dim':           '#99A3A4',
    'head_c':        ['#E74C3C', '#2E86C1', '#27AE60', '#F39C12'],
}

# ══════════════════════════════════════════════════════════════════════
# SVG helpers
# ══════════════════════════════════════════════════════════════════════
def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

def el(tag, attrs=None, content='', close=True):
    a = ''
    if attrs:
        a = ' ' + ' '.join(f'{k}="{esc(v)}"' for k, v in attrs.items() if v is not None)
    if close:
        return f'<{tag}{a}>{content}</{tag}>'
    return f'<{tag}{a}/>'

def T(x, y, s, size=11, color=None, bold=False, italic=False, anchor='middle'):
    return el('text', {'x': f'{x:.1f}', 'y': f'{y:.1f}', 'font-size': size,
               'fill': color or C['text'], 'text-anchor': anchor,
               'font-weight': 'bold' if bold else None,
               'font-style': 'italic' if italic else None}, esc(s))

def R(x, y, w, h, rx=4, fill='#fff', stroke='#333', sw=1.0, dash=None, opacity=None):
    return el('rect', {'x': f'{x:.1f}', 'y': f'{y:.1f}', 'width': f'{w:.1f}',
               'height': f'{h:.1f}', 'rx': f'{rx:.1f}', 'fill': fill,
               'stroke': stroke, 'stroke-width': f'{sw:.1f}',
               'stroke-dasharray': dash, 'opacity': opacity})

def L(x1, y1, x2, y2, color=None, sw=1.2, dash=None, end=False, opacity=None):
    return el('line', {'x1': f'{x1:.1f}', 'y1': f'{y1:.1f}', 'x2': f'{x2:.1f}',
               'y2': f'{y2:.1f}', 'stroke': color or C['arrow'],
               'stroke-width': f'{sw:.1f}', 'stroke-dasharray': dash,
               'marker-end': 'url(#ah)' if end else None, 'opacity': opacity})

def Cir(cx, cy, r, fill='#fff', stroke='#333', sw=1.0, opacity=None):
    return el('circle', {'cx': f'{cx:.1f}', 'cy': f'{cy:.1f}', 'r': f'{r:.1f}',
               'fill': fill, 'stroke': stroke, 'stroke-width': f'{sw:.1f}',
               'opacity': opacity})

# ══════════════════════════════════════════════════════════════════════
# Component builders — each returns list of SVG strings
# ══════════════════════════════════════════════════════════════════════

def skeleton_viz(cx, cy):
    """Draw hand skeleton with 26 joints in anatomically plausible layout."""
    out = []
    # Background
    out.append(R(cx-65, cy-95, 130, 190, 8, C['fill_blue'], '#AED6F1', 0.8))
    out.append(T(cx, cy-82, 'Input: Hand Skeleton', 10, C['stroke_blue'], True))
    out.append(T(cx, cy+100, '[B, T=32, J=26, C=3]', 9, C['dim']))
    out.append(T(cx, cy+113, 'Rokid UXR 26 joints × 3 coords', 8, C['dim'], italic=True))

    # Joint layout — scaled to fit 130×190 region
    # Wrist at bottom center
    wx, wy = cx, cy+70
    # Palm
    px, py = cx, cy+35
    # Finger base positions spread across palm
    fingers = [
        # (name, mcp_x_offset, joints: [(rel_x, rel_y), ...], color_hint)
        # Thumb — goes left-down
        ('Thumb',  [
            (cx-12, cy+12),  # CMC
            (cx-25, cy-2),   # MCP
            (cx-34, cy-22),  # IP
            (cx-38, cy-40),  # Tip
        ]),
        # Index — goes up-right
        ('Index',  [
            (cx+8, cy+5),    # MCP
            (cx+12, cy-20),  # PIP
            (cx+14, cy-40),  # DIP
            (cx+15, cy-58),  # Tip
        ]),
        # Middle — goes straight up
        ('Middle', [
            (cx+1, cy+7),    # MCP
            (cx+0, cy-18),   # PIP
            (cx-1, cy-42),   # DIP
            (cx-1, cy-62),   # Tip
        ]),
        # Ring — goes up-left
        ('Ring',   [
            (cx-8, cy+8),    # MCP
            (cx-13, cy-15),  # PIP
            (cx-16, cy-35),  # DIP
            (cx-17, cy-52),  # Tip
        ]),
        # Pinky — far left-up
        ('Pinky',  [
            (cx-18, cy+12),  # MCP
            (cx-26, cy-2),   # PIP
            (cx-30, cy-18),  # DIP
            (cx-32, cy-32),  # Tip
        ]),
    ]

    # Draw bones from wrist to palm
    out.append(L(wx, wy, px, py, C['bone'], 2.5))
    # Palm to each MCP
    for _, joints in fingers:
        out.append(L(px, py, joints[0][0], joints[0][1], C['bone'], 1.8))
    # Within each finger
    for _, joints in fingers:
        for i in range(len(joints)-1):
            out.append(L(joints[i][0], joints[i][1], joints[i+1][0], joints[i+1][1], '#BDC3C7', 1.0))

    # Draw joint circles
    out.append(Cir(wx, wy, 5, C['joint'], '#C0392B', 1.5))
    out.append(T(wx+12, wy+3, 'Wrist', 7, C['sub'], anchor='start'))
    out.append(Cir(px, py, 3.5, '#F5B041', '#D68910', 1.0))
    for _, joints in fingers:
        for jx, jy in joints:
            out.append(Cir(jx, jy, 2.5, C['joint'], '#C0392B', 0.6))

    return out


def conv1d_viz(cx, cy, kernel_size, dilation, label, rf_label):
    """Visualize a single 1D dilated conv layer: input steps → kernel → output steps."""
    out = []
    step_w, step_h = 10, 18
    step_gap = 3
    n_steps = 10  # show partial time dimension

    total_h = n_steps * (step_h + step_gap) - step_gap
    top = cy - total_h / 2

    # Input column (left)
    ix = cx - 55
    for t in range(n_steps):
        sy = top + t * (step_h + step_gap)
        out.append(R(ix - step_w/2, sy, step_w, step_h, 1, C['fill_blue'], C['stroke_blue'], 0.4))

    # Output column (right)
    ox = cx + 55
    kernel_span = (kernel_size - 1) * dilation
    out_steps = n_steps - kernel_span
    for t in range(out_steps):
        sy = top + t * (step_h + step_gap) + kernel_span * (step_h + step_gap) / 2
        out.append(R(ox - step_w/2, sy, step_w, step_h, 1, C['fill_orange'], C['stroke_orange'], 0.5))

    # Kernel overlay (highlighted input region)
    k_start = n_steps // 2 - 1
    for k in range(kernel_size):
        ky = top + (k_start + k * dilation) * (step_h + step_gap)
        out.append(R(ix - step_w/2 - 3, ky, step_w + 6, step_h, 2, '#F5B041', '#E67E22', 1.2, opacity='0.5'))

    # Connecting lines (kernel → input)
    for k in range(kernel_size):
        ky = top + (k_start + k * dilation) * (step_h + step_gap) + step_h / 2
        out.append(L(ix + step_w/2, ky, ox - step_w/2, ky, '#E67E22', 0.5, dash='3,2'))

    # Labels
    out.append(T(cx, top - 10, label, 9, C['text'], True))
    out.append(T(cx, cy + total_h/2 + 12, rf_label, 8, C['dim'], italic=True))
    out.append(T(ix, cy + total_h/2 + 4, 'Input', 7, C['dim']))
    out.append(T(ox, cy + total_h/2 + 4, 'Output', 7, C['dim']))

    return out


def attention_viz(cx, cy):
    """Self-Attention: Q/K/V projections → scaled dot-product → output."""
    out = []
    w, h = 200, 210
    out.append(R(cx-w/2, cy-h/2, w, h, 6, C['fill_purple'], C['stroke_purple'], 0.5, opacity='0.15'))
    out.append(T(cx, cy-h/2-6, 'Stage 5: LightweightSelfAttention', 10, C['stroke_purple'], True))

    # Q, K, V blocks
    qkv_y = [cy-50, cy, cy+50]
    qkv_names = ['Q', 'K', 'V']
    for i, (qy, name) in enumerate(zip(qkv_y, qkv_names)):
        col = C['head_c'][i]
        out.append(R(cx-75, qy-16, 42, 32, 3, col, col, 0.6, opacity='0.2'))
        out.append(T(cx-54, qy+4, name, 14, col, True))
        out.append(T(cx-75+42+8, qy+4, '[T×16]', 8, C['dim'], anchor='start'))

    # Attention matrix (center)
    am_x, am_y = cx-5, cy-62
    am_sz = 56
    out.append(R(am_x, am_y, am_sz, am_sz, 3, '#F4ECF7', C['attn_heat'], 1.0))
    grid = 5
    for gi in range(grid+1):
        gx = am_x + gi * am_sz / grid
        gy = am_y + gi * am_sz / grid
        out.append(L(gx, am_y, gx, am_y+am_sz, C['attn_heat'], 0.3, opacity='0.4'))
        out.append(L(am_x, gy, am_x+am_sz, gy, C['attn_heat'], 0.3, opacity='0.4'))
    # Diagonal dots
    for gi in range(grid):
        out.append(Cir(am_x + gi*am_sz/grid + am_sz/(2*grid),
                       am_y + gi*am_sz/grid + am_sz/(2*grid),
                       3, C['attn_heat'], C['attn_heat'], 0.5, opacity='0.7'))
    out.append(T(am_x + am_sz/2, am_y-8, 'Softmax', 7, C['sub'], italic=True))
    out.append(T(am_x + am_sz/2, am_y+am_sz+10, 'Q·Kᵀ [T×T]', 7, C['attn_heat'], True))

    # Output block
    out.append(R(cx+70, cy-16, 40, 32, 3, C['head_c'][3], C['head_c'][3], 0.6, opacity='0.25'))
    out.append(T(cx+90, cy+4, 'Out', 11, C['head_c'][3], True))
    out.append(T(cx+90, cy+20, '[T×64]', 8, C['dim']))

    # Arrows connecting
    # Q → Attn
    out.append(L(cx-33, qkv_y[0], am_x, am_y+am_sz/2, C['arrow'], 0.8, end=True))
    # K → Attn
    out.append(L(cx-33, qkv_y[1], am_x, am_y+am_sz/2, C['arrow'], 0.8, end=True))
    # V → Out
    out.append(L(cx-33, qkv_y[2], cx+70, cy, C['arrow'], 0.8, end=True))
    # Attn → Out
    out.append(L(am_x+am_sz, am_y+am_sz/2, cx+70, cy, C['arrow'], 0.8, end=True))

    # Info
    out.append(T(cx, cy+h/2-6, 'Single-layer 4-Head MHA · head_dim=16', 8, C['dim'], italic=True))
    out.append(T(cx, cy+h/2+6, '+ LayerNorm + Residual', 8, C['dim'], italic=True))

    return out


def classifier_viz(cx, cy):
    """GAP → FC → Output bar chart."""
    out = []
    w, h = 170, 180
    out.append(R(cx-w/2, cy-h/2, w, h, 6, C['fill_pink'], C['stroke_pink'], 0.5, opacity='0.10'))
    out.append(T(cx, cy-h/2-6, 'Stage 6: GAP + Classifier', 10, C['stroke_pink'], True))

    # GAP: tall tensor collapsing to flat
    gap_x = cx - 40
    gap_h = 80
    out.append(R(gap_x-7, cy-gap_h/2, 14, gap_h, 3, C['fill_green'], C['stroke_green'], 1.0))
    out.append(T(gap_x, cy-gap_h/2-10, '[T,64]', 7, C['dim']))
    # Converging lines
    for i in range(4):
        ly = cy - gap_h/2 + gap_h/6 + i * gap_h/5
        out.append(L(gap_x+7, ly, cx-8, cy, C['stroke_green'], 0.5, end=True))

    # 1D vector after GAP
    out.append(R(cx-3, cy-10, 24, 20, 2, C['fill_green'], C['stroke_green'], 1.2))
    out.append(T(cx+9, cy+4, '64', 10, C['text'], True))

    # FC layers
    # FC1
    fc1_x = cx + 30
    out.append(R(fc1_x-10, cy-18, 20, 36, 3, C['fill_pink'], C['stroke_pink'], 1.0))
    out.append(T(fc1_x, cy-22, '64→32', 7, C['text'], True))
    out.append(T(fc1_x, cy+2, 'GELU', 7, C['sub'], italic=True))
    out.append(L(cx+21, cy, fc1_x-10, cy, C['arrow'], 0.8, end=True))

    # FC2
    fc2_x = fc1_x + 30
    out.append(R(fc2_x-8, cy-12, 16, 24, 3, C['fill_pink'], C['stroke_pink'], 1.0))
    out.append(T(fc2_x, cy-16, '32→7', 7, C['text'], True))
    out.append(L(fc1_x+10, cy, fc2_x-8, cy, C['arrow'], 0.8, end=True))

    # Output bar chart
    bar_x = fc2_x + 20
    bar_w = 7
    bar_gap = 3
    bar_hs = [8, 18, 18, 14, 16, 22, 12]  # illustrative heights
    bar_ls = ['N', '←', '→', '2P', '2B', '4F', 'F']
    for i, (bh_val, bl) in enumerate(zip(bar_hs, bar_ls)):
        bx = bar_x + i * (bar_w + bar_gap)
        out.append(R(bx, cy - bh_val, bar_w, bh_val*2, 1, C['head_c'][i%4], C['head_c'][i%4], 0.6, opacity='0.6'))
        out.append(T(bx + bar_w/2, cy + bh_val + 10, bl, 6, C['sub']))
    out.append(T(bar_x + 3.5*(bar_w+bar_gap), cy + 30, '7-Class Softmax', 7, C['dim'], italic=True))

    return out


# ══════════════════════════════════════════════════════════════════════
# Main assembly
# ══════════════════════════════════════════════════════════════════════

def main():
    parts = []

    # ── Title ──
    parts.append(T(W/2, 24, 'DBEW-NN: Lightweight Skeleton-Based Gesture Recognition Network', 17, '#111', True))
    parts.append(T(W/2, 44, '1D Dilated CNN + Self-Attention  |  57K params  |  &lt;2ms latency  |  ONNX &rarr; Unity Barracuda',
                  10, C['dim'], italic=True))

    # ════════════════════════════════════════════════════════════════
    # Top row: 6 columns showing each stage
    # ════════════════════════════════════════════════════════════════
    stage_names = [
        'Input Skeleton',
        'SpatialEmbedding',
        'JointPool + PosEnc',
        'DilatedTemporalCNN',
        'SelfAttention',
        'GAP + Classifier',
    ]
    stage_colors = [
        (C['fill_blue'], C['stroke_blue']),
        (C['fill_orange'], C['stroke_orange']),
        (C['fill_green'], C['stroke_green']),
        (C['fill_orange'], C['stroke_orange']),
        (C['fill_purple'], C['stroke_purple']),
        (C['fill_pink'], C['stroke_pink']),
    ]

    # ── Column 0: Skeleton Input ──
    cx0 = col_cx(0)
    parts.append(R(cx0-INNER_W/2, Y_BLOCK_TOP, INNER_W, BLOCK_H + 30, 6, C['fill_blue'], C['stroke_blue'], 1.2, opacity='0.15'))
    parts.append(T(cx0, Y_STAGE_LABEL, 'Input Skeleton', 10, C['stroke_blue'], True))
    parts.extend(skeleton_viz(cx0, Y_BLOCK_TOP + BLOCK_H/2 + 10))
    parts.append(T(cx0, Y_DIMS, '[B, 32, 26, 3]', 9, C['dim']))
    # Normalization note
    parts.append(R(cx0-55, Y_BLOCK_BOT+34, 110, 22, 3, C['fill_gray'], C['stroke_gray'], 0.5))
    parts.append(T(cx0, Y_BLOCK_BOT+48, 'Wrist-relative · Unit-scale', 7.5, C['sub'], italic=True))

    # ── Column 1: SpatialEmbedding ──
    cx1 = col_cx(1)
    parts.append(R(cx1-INNER_W/2, Y_BLOCK_TOP, INNER_W, BLOCK_H + 30, 6, C['fill_orange'], C['stroke_orange'], 1.0, opacity='0.12'))
    parts.append(T(cx1, Y_STAGE_LABEL, 'SpatialEmbedding', 10, C['stroke_orange'], True))

    # 3D tensor blocks showing the transformation
    by = Y_BLOCK_TOP + 40
    # Input tensor
    parts.append(R(cx1-40, by, 80, 55, 4, C['fill_blue'], C['stroke_blue'], 0.8))
    parts.append(T(cx1, by+18, 'Skeleton', 10, C['text'], True))
    parts.append(T(cx1, by+38, '[J=26, C=3]', 8, C['dim']))

    # 1×1 Conv symbol
    parts.append(T(cx1, by+75, '↓ 1×1 Conv', 9, C['stroke_orange'], True))
    parts.append(T(cx1, by+90, '(26,3) → d_model=64', 8, C['dim'], italic=True))
    parts.append(T(cx1, by+103, '+ LayerNorm', 8, C['dim'], italic=True))

    # Output tensor
    parts.append(R(cx1-40, by+120, 80, 55, 4, C['fill_orange'], C['stroke_orange'], 0.8))
    parts.append(T(cx1, by+138, 'Embedded', 10, C['text'], True))
    parts.append(T(cx1, by+158, '[J=26, D=64]', 8, C['dim']))

    parts.append(T(cx1, Y_DIMS, '[B, 32, 26, 64]', 9, C['dim']))

    # ── Column 2: JointPool + PosEnc ──
    cx2 = col_cx(2)
    parts.append(R(cx2-INNER_W/2, Y_BLOCK_TOP, INNER_W, BLOCK_H + 30, 6, C['fill_green'], C['stroke_green'], 1.0, opacity='0.12'))
    parts.append(T(cx2, Y_STAGE_LABEL, 'JointPool + PosEnc', 10, C['stroke_green'], True))

    by = Y_BLOCK_TOP + 35
    # Joint array → Pool → Single vector
    parts.append(R(cx2-45, by, 90, 48, 4, C['fill_blue'], C['stroke_blue'], 0.7))
    parts.append(T(cx2, by+16, '26 joints', 10, C['text'], True))
    parts.append(T(cx2, by+34, '[26, 64] per frame', 8, C['dim']))

    # Pool symbol
    parts.append(T(cx2, by+68, '↓ Mean Pool', 9, C['stroke_green'], True))
    parts.append(T(cx2, by+80, 'over 26 joints', 7, C['dim'], italic=True))

    # After pool
    parts.append(R(cx2-35, by+92, 70, 30, 3, C['fill_green'], C['stroke_green'], 0.8))
    parts.append(T(cx2, by+110, '[64] vector', 10, C['text'], True))

    # PE
    parts.append(R(cx2-45, by+140, 90, 36, 4, C['fill_yellow'], C['stroke_yellow'], 0.8))
    parts.append(T(cx2, by+155, '+ Sinusoidal PE', 9, C['text'], True))
    parts.append(T(cx2, by+170, 'max_len=64', 7, C['dim'], italic=True))

    parts.append(T(cx2, Y_DIMS, '[B, 32, 64]', 9, C['dim']))

    # ── Column 3: Dilated Temporal CNN ──
    cx3 = col_cx(3)
    parts.append(R(cx3-INNER_W/2, Y_BLOCK_TOP, INNER_W, BLOCK_H + 30, 6, C['fill_orange'], C['stroke_orange'], 1.5, opacity='0.12'))
    parts.append(T(cx3, Y_STAGE_LABEL, 'DilatedTemporalCNN', 10, C['stroke_orange'], True))
    parts.append(T(cx3, Y_STAGE_LABEL+12, '3-layer · GELU · BN · Dropout', 7, C['dim'], italic=True))

    # 3 conv layers detailed
    conv_start_y = Y_BLOCK_TOP + 55
    conv_h = 55
    conv_gap = 14
    for i, (d, rf) in enumerate([(1, 'RF=3 frames'), (2, 'RF=7 frames'), (4, 'RF=15 frames')]):
        ly = conv_start_y + i * (conv_h + conv_gap)
        parts.extend(conv1d_viz(cx3, ly + conv_h/2, 3, d,
                                f'Conv1d  d={d}, k=3', rf))
        # Per-layer residual (except first)
        if i > 0:
            rx = cx3 + 62
            parts.append(L(rx, ly-8, rx, ly+conv_h, C['arrow'], 0.6, dash='4,3', end=True))
            parts.append(T(rx+8, ly+conv_h/2-4, 'Res.', 6, C['dim'], anchor='start'))

    # Global residual
    grx = cx3 - INNER_W/2 + 6
    parts.append(L(grx, conv_start_y, grx, conv_start_y + 2*(conv_h+conv_gap)+20, C['arrow'], 0.7, dash='6,4', end=True))
    parts.append(T(grx-6, conv_start_y + conv_h+conv_gap, 'Global\nRes.', 6, C['dim'], anchor='end'))

    parts.append(T(cx3, Y_DIMS, '[B, 32, 64]', 9, C['dim']))

    # ── Column 4: Self-Attention ──
    cx4 = col_cx(4)
    parts.append(R(cx4-INNER_W/2, Y_BLOCK_TOP, INNER_W, BLOCK_H + 30, 6, C['fill_purple'], C['stroke_purple'], 1.2, opacity='0.10'))
    parts.append(T(cx4, Y_STAGE_LABEL, 'SelfAttention', 10, C['stroke_purple'], True))
    parts.extend(attention_viz(cx4, Y_BLOCK_TOP + BLOCK_H/2))
    parts.append(T(cx4, Y_DIMS, '[B, 32, 64]', 9, C['dim']))

    # ── Column 5: GAP + Classifier ──
    cx5 = col_cx(5)
    parts.append(R(cx5-INNER_W/2, Y_BLOCK_TOP, INNER_W, BLOCK_H + 30, 6, C['fill_pink'], C['stroke_pink'], 1.2, opacity='0.10'))
    parts.append(T(cx5, Y_STAGE_LABEL, 'GAP + Classifier', 10, C['stroke_pink'], True))
    parts.extend(classifier_viz(cx5, Y_BLOCK_TOP + BLOCK_H/2))
    parts.append(T(cx5, Y_DIMS, '[B, 7]', 9, C['dim']))

    # ════════════════════════════════════════════════════════════════
    # Horizontal arrows between columns (top of blocks)
    # ════════════════════════════════════════════════════════════════
    for i in range(N_COLS - 1):
        x1 = col_cx(i) + INNER_W/2 + COL_GAP/2
        x2 = col_cx(i+1) - INNER_W/2 - COL_GAP/2
        # Main flow arrow
        parts.append(L(x1, Y_ARROW, x2, Y_ARROW, C['arrow'], 1.6, end=True))
        # Dimension label above arrow
        dims_flow = ['[B,32,26,3]', '[B,32,26,64]', '[B,32,64]', '[B,32,64]', '[B,32,64]', '[B,7]']
        mx = (x1 + x2) / 2
        if i < len(dims_flow) - 1:
            parts.append(T(mx, Y_ARROW - 12, dims_flow[i+1], 7, C['dim']))

    # ════════════════════════════════════════════════════════════════
    # CNN-only ablation dashed bypass
    # ════════════════════════════════════════════════════════════════
    # Bypass from after CNN (col 3 right) directly to Self-Attention output (col 4 right) → GAP
    bypass_x = col_cx(3) + INNER_W/2 + COL_GAP + 4
    bypass_y1 = Y_BLOCK_BOT + 6
    bypass_y2 = Y_DIMS + 8
    parts.append(L(bypass_x, bypass_y1, bypass_x, bypass_y2, C['dim'], 1.0, dash='8,5', end=True))
    parts.append(T(bypass_x+12, (bypass_y1+bypass_y2)/2, 'CNN-only\nAblation', 7, C['dim'], anchor='start', italic=True))

    # ════════════════════════════════════════════════════════════════
    # Key architectural annotations
    # ════════════════════════════════════════════════════════════════
    anno_y = Y_BLOCK_BOT + 60
    annotations = [
        ('SpatialEmbedding', '1×1 Conv projects (J,C)→D, no spatial mixing'),
        ('JointPool (Mean)', 'Collapse joint dim, preserve temporal sequence'),
        ('Dilated Conv [1,2,4]', '3 layers cover 15-frame receptive field'),
        ('Self-Attention', '4-head MHA over 32 time steps, global context'),
        ('Classifier Head', 'GAP → FC(64→32) → GELU → FC(32→7)'),
    ]
    anno_start_x = col_cx(1) - INNER_W/2
    for i, (title, desc) in enumerate(annotations):
        ax = anno_start_x + i * COL_W
        parts.append(T(ax, anno_y, title, 8, C['text'], True))
        parts.append(T(ax, anno_y + 12, desc, 7, C['dim'], italic=True))

    # ════════════════════════════════════════════════════════════════
    # Legend
    # ════════════════════════════════════════════════════════════════
    legend_items = [
        (C['fill_blue'], C['stroke_blue'], 'Input / Embed'),
        (C['fill_orange'], C['stroke_orange'], 'Conv / CNN'),
        (C['fill_green'], C['stroke_green'], 'Pooling'),
        (C['fill_yellow'], C['stroke_yellow'], 'Positional Enc.'),
        (C['fill_purple'], C['stroke_purple'], 'Self-Attention'),
        (C['fill_pink'], C['stroke_pink'], 'FC / Classifier'),
        (C['fill_gray'], C['stroke_gray'], 'Normalization'),
    ]
    lx = W/2 - (len(legend_items) - 1) * 75
    for fill, stroke, label in legend_items:
        parts.append(R(lx-9, Y_LEGEND-7, 18, 14, 3, fill, stroke, 0.7))
        parts.append(T(lx, Y_LEGEND+14, label, 8, C['sub']))
        lx += 150

    # Footer stats
    parts.append(T(W/2, Y_FOOTER,
        'Total: ~57K params  |  CNN: ~37K  |  Attention: ~16K  |  Inference: &lt;2ms (Snapdragon XR2)  |  Export: ONNX opset=14',
        9, C['dim'], italic=True))

    # ════════════════════════════════════════════════════════════════
    # Assemble SVG
    # ════════════════════════════════════════════════════════════════
    defs = '''  <defs>
    <marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#616A6B"/>
    </marker>
    <style>
      text { font-family: 'Noto Sans SC', 'DejaVu Sans', 'Arial', sans-serif; }
    </style>
  </defs>'''

    nl = '\n'
    svg = nl.join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        defs,
        f'  <rect width="100%" height="100%" fill="{C["bg"]}"/>',
    ] + parts + ['</svg>'])

    path = 'DBEW-NN_detailed_architecture.svg'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(svg)
    print(f'Saved: {path} ({len(svg):,} bytes)')
    print('Layout: 6-column horizontal flow, 1600×780, all coords calculated.')


if __name__ == '__main__':
    main()
