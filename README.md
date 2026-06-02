# 面向计量经济分析的AR可视化交互系统

> 电子信息专业硕士毕业论文  
> 省域数字经济与能源结构的AR沉浸式分析系统研究

## 项目结构

```
├── zjuthesis.tex          # 主文件（编译入口）
├── zjuthesis.cls          # 文档类
├── body/
│   ├── graduate/
│   │   └── content.tex    # ★ 论文正文（6章）
│   └── ref.bib            # ★ 参考文献 (BibTeX)
├── page/graduate/         # 封面/摘要/声明等前置页面
├── figure/                # 论文插图
├── experiment_results_v2/ # ★ 实验图表（自动生成）
│   ├── figures/           #   7张PDF图
│   └── tables/            #   6张LaTeX表
├── gesture_nn/            # ★ 手势识别实验代码
│   ├── config.py          #   参数配置
│   ├── data_generator.py  #   合成数据生成
│   ├── model.py           #   模型架构 (1D-CNN+Self-Attention)
│   ├── train.py           #   训练管线 (Focal Loss + Warmup)
│   ├── experiments.py     #   基础实验脚本
│   ├── experiments_v2.py  #   增强实验脚本（图表生成）
│   └── unity/             #   Unity C# 部署脚本
├── gesture_related_work.tex       # 相关工作综述
├── gesture_experiments_figures.tex # 实验图表LaTeX
└── 修改说明.md            # ★ 修改记录
```

## 编译

```bash
# 需要 TeX Live (XeLaTeX + BibTeX)
latexmk -xelatex -interaction=nonstopmode zjuthesis.tex
# 或手动编译
xelatex zjuthesis && bibtex zjuthesis && xelatex zjuthesis && xelatex zjuthesis
```

## 手势识别实验

```bash
cd gesture_nn

# 完整管线（数据生成 + 训练 + 实验）
python main.py --device cpu

# 仅运行增强实验（需已有训练好的模型）
python experiments_v2.py --device cpu --output_dir experiment_results_v2
```

### 核心技术

| 方法 | 缩写 | 类型 | 参数量 | 宏平均F1 |
|------|------|------|--------|----------|
| 深度骨骼约束边沿触发手势管线 | DBEW--Gesture | 几何规则 | 0 | 0.676 |
| 轻量神经网络增强版 | DBEW-NN | 1D-CNN+Self-Attention | 56,711 | 0.964 |

### 实验环境

- **部署平台**: Rokid AR Studio (Station Pro + Max Pro, 骁龙XR2)
- **推理引擎**: Unity Barracuda (ONNX Runtime)
- **训练环境**: PyTorch 2.x, Python 3.10+

## 章节结构

| 章节 | 内容 |
|------|------|
| 第1章 | 引言（背景、文献综述、技术路线、创新点） |
| 第2章 | 理论基础与概念框架 |
| **第3章** | **面向省域数据场景的AR多模态交互设计与实现**（核心技术） |
| **第4章** | **状态驱动的AR沉浸式可视化系统设计与实现**（核心技术） |
| 第5章 | 系统应用：省域数字经济与能源结构分析 |
| 第6章 | 总结与展望 |

## 开源许可

本项目的 LaTeX 模板基于 [zjuthesis](https://github.com/TheNetAdmin/zjuthesis) 修改，原始模板采用 MIT 协议。

论文内容版权归作者所有。
