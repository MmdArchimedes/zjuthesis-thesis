# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

电子信息专业硕士毕业论文 — 省域数字经济与能源结构的AR沉浸式分析系统研究。LaTeX项目（中文），使用`ctexbook`文档类。

## Build

```bash
# 编译论文 (xelatex + bibtex)
xelatex content.tex && bibtex content && xelatex content.tex && xelatex content.tex

# Python手势识别实验
cd gesture_nn && python main.py --device cpu            # 完整管线（数据生成+训练+实验）
cd gesture_nn && python main.py --device cpu --skip_gen --skip_train  # 仅实验
```

## File Architecture

| 文件 | 用途 |
|------|------|
| `content.tex` | 主论文（6章），`\chapter{}`驱动，图表用`\captionof` |
| `abstract.tex` | 独立摘要页，中英文双版本 |
| `ref.bib` | BibTeX参考文献，需同时被content.tex `\bibliography`引用 |
| `修改说明.md` | 逐条对应评审意见的修改记录，每次修改content.tex后同步更新 |
| `修改意见.txt` | 原始评审意见（三组，含格式、逻辑、内容、参考文献等意见） |

### content.tex 章节结构
- 第1章 引言 — 背景、文献综述、技术路线、创新点、论文结构
- 第2章 理论基础与概念框架 — 经济学理论（非核心，为AR系统服务）
- **第3章** 面向省域数据场景的AR多模态交互设计与实现（核心技术章）
  - §3.3 手势交互（DBEW--Gesture几何规则管线 + DBEW-NN轻量NN增强）
  - §3.4 语音识别与大语言模型增强
  - §3.5 多模态融合与冲突消解（TSTQ--Fusion）
  - §3.6 实验评估（手势+语音+融合+NN对比）
- **第4章** 状态驱动的AR沉浸式可视化系统设计与实现（核心技术章）
  - SDCR--Vis管线、四层架构、Unity部署
- 第5章 系统应用：省域数字经济与能源结构分析（应用验证，非核心创新）
- 第6章 总结与展望

### 核心技术命名体系
- **DBEW--Gesture**: 深度骨骼约束下的边沿触发—时间窗离散化手势管线（几何规则版）
- **DBEW-NN**: 同上管线 + 轻量1D-CNN+Self-Attention神经网络前端
- **TSTQ--Fusion**: 通道内稳定化—跨通道时间窗仲裁—队列串行提交（多模态融合）
- **SDCR--Vis**: 状态驱动的条件刷新多视图可视化管线

### gesture_nn/ 目录
- `config.py` — 所有可调参数（模型架构、训练超参、DBEW触发参数）
- `data_generator.py` — 合成手部骨骼数据（26关节×3坐标，6类手势，累计~25万帧）
- `model.py` — GestureClassifier：SpatialEmbedding → DilatedTemporalCNN → LightweightSelfAttention → ClassifierHead（~57K参数）
- `train.py` — Focal Loss + warmup + 早停 + ONNX导出（dynamo=False兼容Unity Barracuda）
- `experiments.py` — 三组对比实验：精度/鲁棒性/性能，含RuleBasedClassifier几何规则baseline
- `unity/` — C#脚本：GestureClassifierNN.cs（主分类器）、RuleBasedGestureClassifier.cs（对比）、RokidHandProvider.cs（SDK抽象）、HandDataCollector.cs（真实数据采集）

## Editing Conventions

- **论文定位**：电子信息技术专业，AR系统实现是核心贡献，经济学实证是应用验证。修改时优先凸显技术方案，缩减经济学理论篇幅
- **术语规范**：英文缩写首次出现加中文全称，如"增强现实（Augmented Reality，AR）"；核心方法用粗体标注
- **图表编号**：content.tex使用`\captionof{table/figure}`而非浮动体，标签格式`\label{tab:xxx}`/`\label{fig:xxx}`
- **交叉引用**：使用`\ref{sec:xxx}`（节）、`\ref{tab:xxx}`（表）、`\ref{fig:xxx}`（图）、`\ref{ch:xxx}`（章）
- **修改说明同步**：每次修改content.tex后须在`修改说明.md`中记录修改位置、内容与对应评审意见编号
- **评审意见优先级**：第三组（重点调整）> 第二组 > 第一组（格式规范）

## Key Constraints

- 第5章的经济学理论不是核心创新，保持压缩后的篇幅（已从长篇经济学论述重构为"数据基础—计量摘要—AR呈现"三层）
- 第3章和第4章是论文核心，创新性论证需要充分的对比实验支撑
- 技术文献与经济政策文献需合理搭配，近3年高水平技术文献（CCF-A/B、SCI一区）优先
- 系统平台：Unity + Rokid AR Studio（Station Pro + Max Pro）

## GitHub & Version Control

- **论文仓库**: `C:\Users\12078\Documents\thesis` → https://github.com/MmdArchimedes/zjuthesis-thesis
- **每次修改论文后必须**:
  1. 更新 `修改说明.md`（日期 + 修改位置 + 内容 + 对应评审意见编号）
  2. 编译 PDF: `cd C:\Users\12078\Documents\thesis && xelatex→biber→xelatex→xelatex`
  3. 复制 PDF: `cp out/zjuthesis.pdf ./论文.pdf`
  4. 清理根目录编译产物: `rm -f zjuthesis.aux zjuthesis.bbl ...`
  5. `git add -A && git commit -m "描述" && git push`
- Git credential 已配置，push 无需密码
