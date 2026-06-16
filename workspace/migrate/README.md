# 论文工作环境一键迁移指南

纯网络方式：所有软件通过网络下载安装，项目文件通过 GitHub 同步，**不需要 U 盘/移动硬盘**。

---

## 第一步（在原电脑上）：推送工作文件到 GitHub

```powershell
# 以管理员身份打开 PowerShell
cd C:\Users\<用户名>\Claudecode\migrate
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\push-workspace.ps1
```

这一步将 Claudecode 工作目录中的关键文件（CLAUDE.md、修改说明、chartforge、SDCR_Vis_System 等）推送到论文 GitHub 仓库的 `workspace/` 目录。

## 第二步（在新电脑上）：一键部署环境

```powershell
# 以管理员身份打开 PowerShell
cd <任意目录>  # 不需要任何文件，纯网络安装

# 直接运行（从 GitHub 获取脚本）：
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
iwr -Uri "https://raw.githubusercontent.com/MmdArchimedes/zjuthesis-thesis/main/workspace/migrate/restore.ps1" -OutFile restore.ps1
.\restore.ps1
```

脚本会自动完成以下所有步骤：

| 步骤 | 内容 | 方式 |
|------|------|------|
| WSL2 + Ubuntu | 虚拟化环境 | `wsl --install` |
| Git + GitHub CLI | 版本控制 & 认证 | `winget install` |
| Python 3.11 | 运行手势识别实验 | `winget install` + pip |
| TeX Live 2026 | LaTeX 编译 | 手动下载 ISO（约 8GB） |
| Node.js | Claude Code 运行环境 | `winget install` |
| Claude Code | AI 编程助手 | `npm install -g` |
| 论文项目 | LaTeX 源码 + 手势识别 | `git clone` |
| 工作文件 | CLAUDE.md 等 | 从论文仓库 `workspace/` 还原 |
| SSH 密钥 | GitHub 免密访问 | 自动生成 + 提示添加 |
| Claude 设置 | DeepSeek API 配置 | 交互式输入 / 模板 |

### TeX Live 手动安装

TeX Live 无法通过包管理器安装，脚本会给出下载链接：

- **清华镜像（推荐）**：https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/Images/texlive2026.iso
- **官方**：https://mirror.ctan.org/systems/texlive/Images/texlive2026.iso

挂载 ISO → 运行 `install-tl-windows.bat` → 完整安装 → 路径选择 `C:\texlive\2026`。

安装完成后重新运行 `restore.ps1` 继续后续步骤。

---

## 脚本文件说明

```
migrate/
├── README.md                ← 本文件
├── push-workspace.ps1       ← 步骤1：推工作文件到 GitHub
├── restore.ps1              ← 步骤2：新电脑一键部署（也可从 GitHub 直接下载）
├── requirements_win.txt     ← Python 依赖清单（restore.ps1 自动使用）
├── .gitignore               ← 防止敏感文件误提交
└── 启动ClaudeCode.bat       ← 快捷启动脚本（部署后复制到 Claudecode 目录）
```

## 部署后验证

### 编译论文
```bash
cd C:\Users\<用户名>\Documents\thesis
xelatex zjuthesis.tex
bibtex zjuthesis
xelatex zjuthesis.tex
xelatex zjuthesis.tex
```

### 测试手势识别
```bash
cd C:\Users\<用户名>\Documents\thesis\gesture_nn
python main.py --device cpu --skip_gen --skip_train
```

### 启动 Claude Code
```bash
cd C:\Users\<用户名>\Claudecode
claude
```

## 环境组件一览

| 组件 | 版本 | 用途 |
|------|------|------|
| WSL2 | 最新 | Linux 虚拟化层 |
| Ubuntu (WSL) | 26.04 LTS | Linux 环境 |
| TeX Live | 2026 完整版 (~8GB) | LaTeX 论文编译 |
| Python | 3.11.9 | 手势识别神经网络 |
| PyTorch | 2.12.0+cpu | 深度学习框架 |
| scikit-learn | 1.8.0 | 机器学习对比实验 |
| transformers | 5.8.1 | 大语言模型 |
| Node.js | 最新 LTS | Claude Code 运行环境 |
| Claude Code | 最新 | AI 编程助手 |
| Git | 最新 | 版本控制 |
| GitHub CLI | 最新 | GitHub 认证 |

## 常见问题

### Q: 脚本提示"无法加载，因为在此系统上禁止运行脚本"
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Q: winget 安装软件失败
手动从官网下载安装：
- Python 3.11: https://www.python.org/downloads/
- Node.js: https://nodejs.org/
- Git: https://git-scm.com/download/win

### Q: GitHub clone 失败
```powershell
# 检查 SSH 密钥是否已添加到 GitHub
gh auth login
# 或手动测试
ssh -T git@github.com
```

### Q: 论文编译报 "font not found"
TeX Live 未完整安装中文字体：
```bash
tlmgr install ctex xeCJK zhnumber fontspec
```

### Q: Claude Code 提示 API 错误
编辑 `C:\Users\<用户名>\.claude\settings.json`，确认 API Key 正确。
