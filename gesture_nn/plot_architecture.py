"""
Professional training pipeline architecture diagram for DBEW-NN gesture classifier.
Thesis-quality figure: data generation → preprocessing → model → training → output.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Polygon
from matplotlib.path import Path
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')

# ── color palette ──
C_BLUE    = '#2563EB'   # data
C_CYAN    = '#0891B2'   # preprocessing
C_ORANGE  = '#EA580C'   # model
C_GREEN   = '#16A34A'   # training
C_PURPLE  = '#7C3AED'   # output
C_RED     = '#DC2626'   # loss
C_GRAY    = '#6B7280'
C_BG1     = '#EFF6FF'   # light blue bg
C_BG2     = '#FEF3C7'   # light amber bg
C_BG3     = '#ECFDF5'   # light green bg
C_BG4     = '#F5F3FF'   # light purple bg
C_BORDER  = '#374151'
C_WHITE   = '#FFFFFF'
C_BLACK   = '#1F2937'

# ── helper: rounded box ──
def rbox(x, y, w, h, color=C_WHITE, edge=C_BORDER, lw=1.2, alpha=1.0, zorder=2):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=edge, linewidth=lw,
                          alpha=alpha, zorder=zorder)
    ax.add_patch(rect)

# ── helper: filled area background ──
def section_bg(x, y, w, h, color, alpha=0.25, zorder=0):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                          facecolor=color, edgecolor='none', linewidth=0,
                          alpha=alpha, zorder=zorder)
    ax.add_patch(rect)

# ── helper: arrow ──
def arrow(x1, y1, x2, y2, color=C_BLACK, lw=1.5, style='simple', zorder=3):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0'), zorder=zorder)

def arrow_down(x1, y1, x2, y2, color=C_BLACK, lw=1.5, zorder=3):
    arrow(x1, y1, x2, y2, color, lw, zorder=zorder)

# ── helper: text in box ──
def txt(x, y, w, h, text, fontsize=8, bold=False, color=C_BLACK, ha='center', va='center'):
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, fontsize=fontsize, fontweight=weight,
            color=color, ha=ha, va=va, zorder=5)

def txt_left(x, y, text, fontsize=8, bold=False, color=C_BLACK):
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, fontsize=fontsize, fontweight=weight, color=color, zorder=5)

# ── helper: section label ──
def section_label(x, y, text, color, fontsize=11):
    ax.text(x, y, text, fontsize=fontsize, fontweight='bold', color=color, zorder=6)

# ══════════════════════════════════════════════════════════════════════
# SECTION 1: DATA GENERATION (left side)
# ══════════════════════════════════════════════════════════════════════

section_bg(0.2, 2.5, 5.0, 10.8, C_BLUE, 0.08)
section_label(0.4, 13.0, '① 合成数据生成', C_BLUE, 12)

# box: canonical hand
rbox(0.5, 11.2, 4.4, 1.5, C_WHITE, C_BLUE)
txt(0.5, 11.2, 4.4, 1.5, '标准右手骨架 (Canonical Hand)\n26关节 × 3D坐标 (x,y,z)\nRokid UXR 格式', 8, color=C_BLUE)

arrow(2.7, 11.2, 2.7, 10.85, C_BLUE)

# box: gesture deformer
rbox(0.5, 9.2, 4.4, 1.65, C_WHITE, C_BLUE)
txt(0.5, 9.2, 4.4, 1.65, '手势变形器 (GestureDeformer)\n手指弯曲角度 → 各关节旋转\n拇指对掌 · 5指独立控制', 8, color=C_BLUE)

arrow(2.7, 9.2, 2.7, 8.85, C_BLUE)

# box: 6 gesture classes
rbox(0.5, 7.0, 4.4, 1.85, C_WHITE, C_BLUE)
txt(0.5, 7.0, 4.4, 1.85, '6类手势模板\nindex_left | index_right\n two_finger_palm | two_finger_back\n four_finger_palm | fist', 7.5, color=C_BLUE)

arrow(2.7, 7.0, 2.7, 6.65, C_BLUE)

# box: variation injection
rbox(0.5, 4.8, 4.4, 1.85, C_WHITE, C_BLUE)
txt(0.5, 4.8, 4.4, 1.85, '随机化注入\n· 手指角度 ±20° 偏差\n· 手腕旋转/平移变化\n· 2mm 高斯传感器噪声\n· 10人 × 3次 × 10重复', 7.5, color=C_BLUE)

arrow(2.7, 4.8, 2.7, 4.45, C_BLUE)

# box: output
rbox(0.5, 2.8, 4.4, 1.65, C_BLUE, C_BLUE)
txt(0.5, 2.8, 4.4, 1.65, '合成数据集\n~25万帧 | 6类手势 + NONE\nsequences.pkl + labels.pkl', 8, color=C_WHITE, bold=True)

# ══════════════════════════════════════════════════════════════════════
# ARROW: Data → Preprocess
# ══════════════════════════════════════════════════════════════════════
arrow(4.95, 3.6, 5.9, 3.6, C_GRAY, 2.5)

# ══════════════════════════════════════════════════════════════════════
# SECTION 2: DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

section_bg(5.8, 2.5, 3.8, 10.8, C_CYAN, 0.08)
section_label(6.0, 13.0, '② 数据预处理', C_CYAN, 12)

# box: split
rbox(6.1, 11.2, 3.2, 1.5, C_WHITE, C_CYAN)
txt(6.1, 11.2, 3.2, 1.5, '参与者级数据划分\nTrain 70% | Val 15% | Test 15%\n同一人不跨集合', 8, color=C_CYAN)

arrow(7.7, 11.2, 7.7, 10.85, C_CYAN)

# box: sliding window
rbox(6.1, 9.4, 3.2, 1.45, C_WHITE, C_CYAN)
txt(6.1, 9.4, 3.2, 1.45, '滑动窗口切分\n窗口: 32帧 (≈0.5s)\n步长: 4帧 | 多数投票标签', 8, color=C_CYAN)

arrow(7.7, 9.4, 7.7, 9.05, C_CYAN)

# box: normalize
rbox(6.1, 7.6, 3.2, 1.45, C_WHITE, C_CYAN)
txt(6.1, 7.6, 3.2, 1.45, '归一化\n· 减手腕坐标 → 位置无关\n· 除全局标准差 → 尺度无关', 8, color=C_CYAN)

arrow(7.7, 7.6, 7.7, 7.25, C_CYAN)

# box: augment (2 boxes side by side in this area)
rbox(6.1, 5.2, 3.2, 2.05, C_WHITE, C_CYAN)
txt(6.1, 5.2, 3.2, 2.05, '数据增强 (仅训练集)\n· 时间偏移 ±5帧\n· 坐标噪声 2mm\n· 50%概率左右镜像\n· WeightedRandomSampler\n  类别平衡采样', 7.5, color=C_CYAN)

# batch arrow
arrow(7.7, 5.2, 7.7, 4.85, C_CYAN)

# box: batch
rbox(6.1, 3.8, 3.2, 1.05, C_CYAN, C_CYAN)
txt(6.1, 3.8, 3.2, 1.05, 'DataLoader\n[batch, 32, 26, 3]\nbatch_size=64', 8, color=C_WHITE, bold=True)

# ══════════════════════════════════════════════════════════════════════
# ARROW: Preprocess → Model
# ══════════════════════════════════════════════════════════════════════
arrow(9.65, 3.6, 10.55, 3.6, C_GRAY, 2.5)

# ══════════════════════════════════════════════════════════════════════
# SECTION 3: MODEL ARCHITECTURE (center-right)
# ══════════════════════════════════════════════════════════════════════

section_bg(10.4, 2.5, 5.4, 10.8, C_ORANGE, 0.08)
section_label(10.6, 13.0, '③ 模型架构 (DBEW-NN)', C_ORANGE, 12)

# Input
rbox(10.7, 11.6, 4.8, 0.7, C_WHITE, C_ORANGE)
txt(10.7, 11.6, 4.8, 0.7, 'Input: [B, 32, 26, 3]  骨骼序列', 8.5, bold=True, color=C_ORANGE)

arrow(13.1, 11.6, 13.1, 11.35, C_ORANGE)

# Spatial Embedding
rbox(10.7, 10.4, 4.8, 0.95, C_WHITE, C_ORANGE)
txt(10.7, 10.4, 4.8, 0.95, 'SpatialEmbedding\n26×3=78 → Linear(78, 64) + ReLU\n每帧独立 → [B, 32, 64]', 7.5, color=C_ORANGE)

arrow(13.1, 10.4, 13.1, 10.1, C_ORANGE)

# 3x Dilated CNN layers
rbox(10.7, 8.0, 4.8, 2.1, C_WHITE, C_ORANGE)
txt(10.7, 8.0, 4.8, 2.1, 'DilatedTemporalCNN (3层)\n· Conv1d(d=1, k=3) → ReLU → Dropout\n· Conv1d(d=2, k=3) → ReLU → Dropout\n· Conv1d(d=4, k=3) → ReLU → Dropout\n感受野: 1+2+4+6=13帧  ≈ 217ms @60fps', 7.5, color=C_ORANGE)

arrow(13.1, 8.0, 13.1, 7.7, C_ORANGE)

# Self-Attention
rbox(10.7, 6.0, 4.8, 1.7, C_ORANGE, C_RED)
txt(10.7, 6.0, 4.8, 1.7, 'LightweightSelfAttention (4头)\n· Multi-Head Attention (d=64, h=4)\n· 残差连接 (Residual)\n· Feed-Forward + LayerNorm\n· 前5轮冻结 → 之后解冻', 7.5, color=C_ORANGE)

arrow(13.1, 6.0, 13.1, 5.7, C_ORANGE)

# Classifier Head
rbox(10.7, 4.7, 4.8, 1.0, C_WHITE, C_ORANGE)
txt(10.7, 4.7, 4.8, 1.0, 'ClassifierHead\nGlobalAvgPool → Linear(64,7)\n输出: [B, 7]  logits (未归一化)', 7.5, color=C_ORANGE)

arrow(13.1, 4.7, 13.1, 4.35, C_ORANGE)

# Softmax output
rbox(10.7, 3.5, 4.8, 0.85, C_ORANGE, C_ORANGE)
txt(10.7, 3.5, 4.8, 0.85, 'Softmax → 7类概率\nNONE · index_L/R · two_finger_P/B · four_finger · fist', 7.5, bold=True, color=C_WHITE)

# ── param count annotation ──
txt_left(15.6, 12.2, '参数量', 8, bold=True, color=C_GRAY)
txt_left(15.6, 11.6, 'SpatialEmb: ~5K', 7.5, color=C_GRAY)
txt_left(15.6, 11.0, 'CNN: ~30K', 7.5, color=C_GRAY)
txt_left(15.6, 10.4, 'Attention: ~16K', 7.5, color=C_GRAY)
txt_left(15.6, 9.8, 'Head: ~5K', 7.5, color=C_GRAY)
txt_left(15.6, 9.0, 'Total: ~57K', 8, bold=True, color=C_GRAY)

# ── receptive field annotation ──
txt_left(15.6, 7.8, '时序感受野', 8, bold=True, color=C_GRAY)
txt_left(15.6, 7.2, 'd=1 → ±1帧', 7.5, color=C_GRAY)
txt_left(15.6, 6.6, 'd=2 → ±3帧', 7.5, color=C_GRAY)
txt_left(15.6, 6.0, 'd=4 → ±7帧', 7.5, color=C_GRAY)
txt_left(15.6, 5.2, '总计: 13帧 ≈ 217ms', 7.5, color=C_GRAY)

# ══════════════════════════════════════════════════════════════════════
# SECTION 4: TRAINING LOOP (right)
# ══════════════════════════════════════════════════════════════════════

section_bg(16.2, 2.5, 5.4, 10.8, C_GREEN, 0.08)
section_label(16.4, 13.0, '④ 训练循环', C_GREEN, 12)

# Loss functions
rbox(16.5, 11.2, 4.8, 1.6, C_WHITE, C_GREEN)
txt(16.5, 11.2, 4.8, 1.6, '损失函数\n· Focal Loss (γ=2.0) → 聚焦难样本\n· 类别权重 → 解决不平衡\n· 时序平滑损失 (λ=0.1)\n  惩罚相邻窗口预测突变', 7.5, color=C_GREEN)

arrow(18.9, 11.2, 18.9, 10.85, C_GREEN)

# Optimizer
rbox(16.5, 9.35, 4.8, 1.5, C_WHITE, C_GREEN)
txt(16.5, 9.35, 4.8, 1.5, '优化策略\n· AdamW (lr=1e-3, wd=1e-4)\n· Cosine退火 → min 1e-6\n· 混合精度训练 (AMP)\n· Batch Size = 64', 7.5, color=C_GREEN)

arrow(18.9, 9.35, 18.9, 9.0, C_GREEN)

# Two-stage training
rbox(16.5, 6.85, 4.8, 2.15, C_WHITE, C_GREEN)
txt(16.5, 6.85, 4.8, 2.15, '两阶段训练\n\n阶段1: Warmup (epoch 0–4)\n  └ 冻结 Self-Attention\n  └ 仅训练 CNN + SpatialEmbed\n\n阶段2: 全模型 (epoch 5+)\n  └ 解冻 Self-Attention\n  └ 端到端联合优化', 7.5, color=C_GREEN)

arrow(18.9, 6.85, 18.9, 6.5, C_GREEN)

# Early stopping
rbox(16.5, 5.1, 4.8, 1.4, C_WHITE, C_GREEN)
txt(16.5, 5.1, 4.8, 1.4, '早停策略\n· 监控: 验证集 Macro F1\n· 耐心值: 10 epochs\n· 保存最佳模型权重', 7.5, color=C_GREEN)

arrow(18.9, 5.1, 18.9, 4.75, C_GREEN)

# Training loop box
rbox(16.5, 2.8, 4.8, 1.95, C_GREEN, C_GREEN)
txt(16.5, 2.8, 4.8, 1.95, '训练循环 (max 80 epochs)\nfor epoch in range(N_EPOCHS):\n  train_epoch() → 验证 → 早停检查\n  scheduler.step()  # 学习率衰减\n输出: best_model.pt + ONNX', 7.5, color=C_WHITE, bold=True)

# ══════════════════════════════════════════════════════════════════════
# SECTION 5: OUTPUTS & EXPERIMENTS (bottom)
# ══════════════════════════════════════════════════════════════════════

section_bg(0.2, 0.2, 21.3, 2.1, C_PURPLE, 0.08)
section_label(0.4, 2.0, '⑤ 输出与实验评估', C_PURPLE, 12)

# Outputs
rbox(0.5, 0.4, 3.5, 1.3, C_PURPLE, C_PURPLE)
txt(0.5, 0.4, 3.5, 1.3, '训练产出\nbest_model.pt (PyTorch)\ngesture_classifier.onnx\n  → Unity Barracuda部署', 8, color=C_WHITE, bold=True)

arrow(4.0, 1.05, 4.8, 1.05, C_PURPLE, 2)

# Experiment 1
rbox(4.9, 0.4, 3.8, 1.3, C_WHITE, C_PURPLE)
txt(4.9, 0.4, 3.8, 1.3, '实验一: 精度对比\n规则方法 vs CNN vs CNN+Attn\n指标: Acc + 逐类F1', 7.5, color=C_PURPLE)

arrow(8.7, 1.05, 9.5, 1.05, C_PURPLE, 2)

# Experiment 2
rbox(9.6, 0.4, 3.8, 1.3, C_WHITE, C_PURPLE)
txt(9.6, 0.4, 3.8, 1.3, '实验二: 鲁棒性\nnormal / low_light /\npartial_occlusion / severe', 7.5, color=C_PURPLE)

arrow(13.4, 1.05, 14.2, 1.05, C_PURPLE, 2)

# Experiment 3
rbox(14.3, 0.4, 3.8, 1.3, C_WHITE, C_PURPLE)
txt(14.3, 0.4, 3.8, 1.3, '实验三: 端侧性能\n时延 · FPS · 参数量\n内存占用 · FPS稳定性', 7.5, color=C_PURPLE)

arrow(18.1, 1.05, 18.9, 1.05, C_PURPLE, 2)

# Final
rbox(19.0, 0.4, 2.5, 1.3, C_PURPLE, C_PURPLE)
txt(19.0, 0.4, 2.5, 1.3, '论文\n§3.6 实验\n评估', 8, color=C_WHITE, bold=True)


# ══════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════
ax.text(11, 13.85, 'DBEW-NN 手势识别模型 — 完整训练流程图', fontsize=16,
        fontweight='bold', color=C_BLACK, ha='center', zorder=10)

plt.tight_layout(pad=0.5)
plt.savefig('architecture_diagram.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Saved: architecture_diagram.png')
