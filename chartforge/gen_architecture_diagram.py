"""
生成 ChartForge + SDCR-Vis 融合系统架构图
匹配 drawio 设计: ch4-1-architecture.drawio
四层递进架构 + 层间数据流 + 右侧 s_t 状态同步
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.set_aspect('equal')
ax.axis('off')

# ═══════════════════════════════════════════════════════════
# 配色 (与 drawio 一致)
# ═══════════════════════════════════════════════════════════
C = {
    'L1_bg':    '#1a3a5c',  # 意图层深蓝
    'L1_fill':  '#dae8fc',
    'L1_stroke':'#6c8ebf',
    'L1_text':  '#1a3a5c',
    'L2_bg':    '#2d6a9f',  # 生成精炼层
    'L2_fill':  '#b0d4f1',
    'L2_stroke':'#4a90c4',
    'L2_subfill':'#d0e4f8',
    'L2_text':  '#2d6a9f',
    'L3_bg':    '#4a90c4',  # 渲染适配层
    'L3_fill':  '#b8daf5',
    'L3_stroke':'#6cb4d9',
    'L4_bg':    '#6cb4d9',  # 交互运行时层
    'L4_fill':  '#c5e4f7',
    'L4_stroke':'#6cb4d9',
    'accent':   '#e8a820',  # s_t 金色
    'arrow':    '#555555',
    'domain':   '#e1d5e7',
    'domain_s': '#9673a6',
}

# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════
def round_box(ax, x, y, w, h, face, edge, lw=1.5, alpha=1.0, z=2, ls='-'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=face, edgecolor=edge, linewidth=lw,
                          alpha=alpha, linestyle=ls, zorder=z)
    ax.add_patch(box)
    return box

def comp_box(ax, x, y, w, h, face, edge, title, desc_lines, lw=1.5, ts=10, tc='#333333'):
    """绘制组件框 + 标题 + 描述"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=face, edgecolor=edge, linewidth=lw, zorder=4)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.15, title, fontsize=ts, fontweight='bold',
            color=tc, ha='center', va='top', zorder=5)
    for i, line in enumerate(desc_lines):
        ax.text(x + w/2, y + h - 0.42 - i*0.22, line, fontsize=7,
                color='#555555', ha='center', va='top', zorder=5)

def layer_bg(ax, x, y, w, h, color, alpha=0.10):
    """半透明层背景"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                          facecolor=color, edgecolor=color, linewidth=2,
                          alpha=alpha, zorder=1)
    ax.add_patch(box)

def layer_label(ax, x, y, title, subtitle, color):
    """层左侧标注"""
    ax.text(x, y, title, fontsize=12, fontweight='bold', color=color,
            ha='center', va='center', zorder=6)
    ax.text(x + 0.02, y - 0.28, subtitle, fontsize=7, color=color,
            ha='center', va='top', alpha=0.7, zorder=6)

def arrow_h(ax, x1, x2, y, color='#555555', lw=1.5):
    """水平箭头"""
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=5)

def arrow_v(ax, x, y1, y2, label, color='#555555', lw=2):
    """垂直箭头 + 标签"""
    ax.annotate('', xy=(x, y2 + 0.05), xytext=(x, y1 - 0.05),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=5)
    if label:
        ax.text(x + 0.18, (y1 + y2) / 2, label, fontsize=7.5, color=color,
                va='center', fontweight='bold', zorder=5)

# ═══════════════════════════════════════════════════════════
# 布局参数
# ═══════════════════════════════════════════════════════════
LX = 0.8          # 左侧标签 X
LBX = 1.6         # 层背景 X
LBW = 13.2        # 层背景宽度

# 各层 Y 坐标 (从底到顶), 层高
LAYERS = [
    {'y': 0.8,  'h': 2.0, 'bg': C['L1_bg'], 'fill': C['L1_fill'], 'stroke': C['L1_stroke'],
     'title': '意图层', 'sub': 'Intent Layer', 'tc': C['L1_text']},
    {'y': 3.1,  'h': 2.2, 'bg': C['L2_bg'], 'fill': C['L2_fill'], 'stroke': C['L2_stroke'],
     'title': '生成精炼层', 'sub': 'Generation & Refinement', 'tc': C['L2_text']},
    {'y': 5.65, 'h': 1.7, 'bg': C['L3_bg'], 'fill': C['L3_fill'], 'stroke': C['L3_stroke'],
     'title': '渲染适配层', 'sub': 'Render & Adaptation', 'tc': C['L3_bg']},
    {'y': 7.65, 'h': 2.1, 'bg': C['L4_bg'], 'fill': C['L4_fill'], 'stroke': C['L4_stroke'],
     'title': '交互运行时层', 'sub': 'Interaction Runtime', 'tc': C['L4_bg']},
]

# ═══════════════════════════════════════════════════════════
# 绘制各层
# ═══════════════════════════════════════════════════════════
for L in LAYERS:
    layer_bg(ax, LBX, L['y'], LBW, L['h'], L['bg'])
    layer_label(ax, LX, L['y'] + L['h']/2, L['title'], L['sub'], L['tc'])

# ═══════════════════════════════════════════════════════════
# L1: 意图层 — 组件
# ═══════════════════════════════════════════════════════════
y1m = LAYERS[0]['y'] + LAYERS[0]['h']/2  # 层1中线

comp_box(ax, 1.9, y1m - 0.38, 2.0, 0.9, C['L1_fill'], C['L1_stroke'],
    '多模态NL查询', ['手势 / 语音 / 射线', '来自第三章通道'], ts=10)
comp_box(ax, 4.5, y1m - 0.38, 2.5, 0.9, C['L1_fill'], C['L1_stroke'],
    'CIF 解析器', ['GPT-4o + 结构化提示词', '字段验证 · 类型推断补充'], ts=10)
comp_box(ax, 7.7, y1m - 0.38, 2.5, 0.9, C['L1_fill'], C['L1_stroke'],
    '结构化图表意图', ['<D, V, I>', '与 TSTQ-Fusion 共享时基'], ts=10)
comp_box(ax, 11.5, y1m - 0.3, 1.4, 0.75, C['domain'], C['domain_s'],
    '领域知识库', ['DOMAIN_FIELDS'], ts=9, lw=1)

arrow_h(ax, 3.9, 4.5, y1m + 0.07, C['L1_stroke'])
arrow_h(ax, 7.0, 7.7, y1m + 0.07, C['L1_stroke'])
# 领域库虚线箭头
ax.annotate('', xy=(7.0, y1m), xytext=(12.9, y1m),
            arrowprops=dict(arrowstyle='->', color=C['domain_s'], lw=1,
                           linestyle='dashed'), zorder=4)

# ═══════════════════════════════════════════════════════════
# L2: 生成精炼层 — 组件
# ═══════════════════════════════════════════════════════════
y2m = LAYERS[1]['y'] + LAYERS[1]['h']/2  # 层2中线

comp_box(ax, 1.9, y2m - 0.42, 2.1, 1.05, C['L2_fill'], C['L2_stroke'],
    'PCG 束搜索', ['概率图表语法派生', '束宽 k=5, 深度<=10', '单次采样 < 1 ms'], ts=10)

arrow_h(ax, 4.0, 4.7, y2m + 0.1, C['L2_stroke'])

# MS-GRP 子管线标题条
comp_box(ax, 4.7, y2m + 0.2, 5.6, 0.35, C['L2_stroke'], C['L2_stroke'],
    'MS-GRP 管线', [], ts=9, tc='white')

# 四个子阶段
for i, (name, desc) in enumerate([
    ('粗生成', '生成候选\n图表规约'),
    ('语义验证', 'SVAS 评分\nsem >= 0.7'),
    ('视觉精炼', 'vis 优化\n配色/布局'),
    ('交互注入', '绑定事件\n处理器'),
]):
    comp_box(ax, 4.85 + i*1.33, y2m - 0.42, 1.15, 0.52, C['L2_subfill'], C['L2_stroke'],
        name, desc.split('\n'), ts=9, lw=1, tc='#2d6a9f')

# MS-GRP 阶段间箭头
for i in range(3):
    arrow_h(ax, 6.0 + i*1.33, 6.18 + i*1.33, y2m - 0.16, C['L2_stroke'], lw=1)

# MS-GRP 输出箭头
arrow_h(ax, 10.3, 10.8, y2m - 0.16, C['L2_stroke'])

comp_box(ax, 10.8, y2m - 0.42, 2.2, 1.05, C['L2_fill'], C['L2_stroke'],
    '优化 ChartSpec', ['平台无关图表规约', 'JSON 中间表示'], ts=10)

# 迭代反馈回路
ax.annotate('', xy=(7.3, y2m + 0.7), xytext=(7.3, y2m + 0.55),
            arrowprops=dict(arrowstyle='->', color='#d6b656', lw=1,
                           connectionstyle='arc3,rad=0'), zorder=5)
ax.annotate('迭代精炼', xy=(8.0, y2m + 0.75), fontsize=7,
            color='#996600', ha='center', va='center', zorder=5)
ax.annotate('', xy=(6.3, y2m + 0.7), xytext=(8.3, y2m + 0.7),
            arrowprops=dict(arrowstyle='->', color='#d6b656', lw=0.8),
            zorder=4)

# ═══════════════════════════════════════════════════════════
# L3: 渲染适配层 — 组件
# ═══════════════════════════════════════════════════════════
y3m = LAYERS[2]['y'] + LAYERS[2]['h']/2

comp_box(ax, 1.9, y3m - 0.33, 1.8, 0.78, C['L3_fill'], C['L3_stroke'],
    'ChartSpec', ['平台无关', '中间表示'], ts=10)
comp_box(ax, 4.3, y3m - 0.33, 2.8, 0.78, C['L3_fill'], C['L3_stroke'],
    '多目标格式转换', [], ts=10)

arrow_h(ax, 3.7, 4.3, y3m + 0.06, C['L3_stroke'])

# 三个目标
targets = [
    (7.7, y3m + 0.15, 'AR 端', ['Unity ChartRenderer', 'JSON 格式']),
    (7.7, y3m - 0.42, 'Web / 离线', ['Vega-Lite / ECharts', '基线对比评测']),
    (10.0, y3m - 0.15, '桌面端', ['Plotly / Matplotlib']),
]
for tx, ty, tname, tdesc in targets:
    w = 1.5 if tname != 'Web / 离线' else 1.5
    comp_box(ax, tx, ty, 1.6, 0.48, C['L3_fill'], C['L3_stroke'],
        tname, tdesc, ts=9, lw=1)

# L3 分叉箭头
for tx, ty in [(7.7, y3m + 0.39), (7.7, y3m - 0.18), (10.0, y3m + 0.09)]:
    ax.annotate('', xy=(tx + 0.05, ty + 0.01), xytext=(7.1, y3m + 0.06),
                arrowprops=dict(arrowstyle='->', color=C['L3_stroke'], lw=1.2,
                               connectionstyle='arc3,rad=0'), zorder=4)

# ═══════════════════════════════════════════════════════════
# L4: 交互运行时层 — 组件
# ═══════════════════════════════════════════════════════════
y4m = LAYERS[3]['y'] + LAYERS[3]['h']/2

# ChartStateSync
comp_box(ax, 1.9, y4m - 0.4, 2.1, 1.0, C['L4_fill'], C['L4_stroke'],
    'ChartStateSync', ['状态同步协调器', '与 SDCR-Vis 共享 s_t'], ts=10)

# 四个视图面板 (2x2)
views = [
    (4.5, y4m + 0.15, '省域地图', ['热力图 · 分级着色', '区域下钻 · 空间筛选']),
    (4.5, y4m - 0.42, '时间轴', ['时序折线 · 年度对比', '时间窗滑动']),
    (6.8, y4m + 0.15, 'ChartForge 图表', ['动态生成 · CCA 组合', '12类图表类型']),
    (6.8, y4m - 0.42, '结果面板', ['回归表 · 机制图', '计量分析呈现']),
]
for vx, vy, vname, vdesc in views:
    comp_box(ax, vx, vy, 1.8, 0.48, C['L4_fill'], C['L4_stroke'],
        vname, vdesc, ts=9, lw=1, tc='#2d6a9f')

# 交互事件
comp_box(ax, 9.2, y4m - 0.28, 1.8, 0.72, C['L4_fill'], C['L4_stroke'],
    '交互事件处理', ['点击 · 悬停 · 刷选', '缩放 · 筛选 · 下钻'], ts=9, lw=1)

# L4 连接箭头
arrow_h(ax, 4.0, 4.5, y4m + 0.39, C['L4_stroke'], lw=1.2)
arrow_h(ax, 4.0, 4.5, y4m - 0.18, C['L4_stroke'], lw=1.2)
arrow_h(ax, 6.3, 6.8, y4m + 0.39, C['L4_stroke'], lw=1.2)
arrow_h(ax, 6.3, 6.8, y4m - 0.18, C['L4_stroke'], lw=1.2)
arrow_h(ax, 8.6, 9.2, y4m + 0.08, C['L4_stroke'], lw=1.2)

# ═══════════════════════════════════════════════════════════
# 层间垂直箭头 (数据流)
# ═══════════════════════════════════════════════════════════
ax_center = 9.0
arrow_v(ax, ax_center, LAYERS[0]['y'] + LAYERS[0]['h'], LAYERS[1]['y'], 'CIF <D,V,I>')
arrow_v(ax, ax_center, LAYERS[1]['y'] + LAYERS[1]['h'], LAYERS[2]['y'], 'ChartSpec')
arrow_v(ax, ax_center, LAYERS[2]['y'] + LAYERS[2]['h'], LAYERS[3]['y'], '渲染指令')

# ═══════════════════════════════════════════════════════════
# 右侧: s_t 状态向量
# ═══════════════════════════════════════════════════════════
st_x = 14.5
ax.plot([st_x, st_x], [LAYERS[0]['y'] + 0.3, LAYERS[3]['y'] + LAYERS[3]['h'] - 0.3],
        color=C['accent'], lw=2.5, ls='--', dashes=(8, 5), zorder=3)
ax.text(st_x + 0.15, 5.5, 'st\n\n统一\n状态\n向量', fontsize=9,
        color=C['accent'], ha='left', va='center', fontweight='bold', zorder=6)

# s_t 到各层的双向连接
for L in LAYERS:
    ly = L['y'] + L['h']/2
    ax.annotate('', xy=(LBX + LBW - 0.1, ly), xytext=(st_x - 0.1, ly),
                arrowprops=dict(arrowstyle='<->', color=C['accent'], lw=1,
                               linestyle='dashed'), zorder=3)

# ═══════════════════════════════════════════════════════════
# 标题与脚注
# ═══════════════════════════════════════════════════════════
ax.text(8.2, 10.3, 'ChartForge + SDCR–Vis 融合系统架构', fontsize=16,
        fontweight='bold', color='#333333', ha='center', zorder=6)

ax.text(8.2, 0.2,
    '数据源: 第五章省域面板数据 (DEL, ES, PGDP, URBAN, INDS, TEIN 等)  '
    '|  自底向上数据流: 意图 -> 生成 -> 适配 -> 呈现  '
    '|  右侧虚线: st 状态同步 (ChartStateSync <-> SDCR-Vis)',
    fontsize=7, color='#999999', ha='center', style='italic', zorder=6)

# ═══════════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════════
out_base = r'C:\Users\12078\Documents\thesis\figure\ch4-1-architecture'
plt.tight_layout(pad=0.5)
plt.savefig(out_base + '.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(out_base + '.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Done: ch4-1-architecture.png + .pdf')
