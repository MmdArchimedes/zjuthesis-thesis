# ============================================================================
# 一键环境部署脚本 — 在新 Windows 电脑上通过网络安装完整工作环境
# ============================================================================
# 前提:
#   1. Windows 10/11，有网络连接
#   2. 以管理员身份运行 PowerShell
#   3. Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   4. .\restore.ps1
#
# 所有软件通过网络下载安装，项目文件从 GitHub clone
# 不需要任何本地备份文件
# ============================================================================

#Requires -RunAsAdministrator

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

# === 用户配置 ===
$GITHUB_REPO = "https://github.com/MmdArchimedes/zjuthesis-thesis.git"
$GIT_USER_NAME = "MmdArchimedes"
$GIT_USER_EMAIL = "1207890259@qq.com"
$PYTHON_VERSION = "3.11"
$TEXLIVE_YEAR = "2026"
$TEXLIVE_URL = "https://mirror.ctan.org/systems/texlive/Images/texlive2026.iso"

$UserProfile = $env:USERPROFILE
$Username = $env:USERNAME

# === 颜色函数 ===
function Write-Step { Write-Host "`n============================================================" -ForegroundColor Cyan; Write-Host "  $args" -ForegroundColor Cyan; Write-Host "============================================================" -ForegroundColor Cyan }
function Write-OK { Write-Host "    [OK] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "    [--] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "    [ER] $args" -ForegroundColor Red }
function Write-Ask { Write-Host "    [??] $args" -ForegroundColor Magenta }

# === 刷新 PATH ===
function Refresh-Path {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}

# ============================================================================
# 0. 欢迎
# ============================================================================
Clear-Host
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       论文工作环境一键部署 — 纯网络安装                     ║" -ForegroundColor Cyan
Write-Host "║       所有软件通过网络下载 | 项目从 GitHub clone            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  目标计算机: $env:COMPUTERNAME"
Write-Host "  用户: $Username"
Write-Host ""
Write-Host "  将安装以下组件:" -ForegroundColor White
Write-Host "    • WSL2 + Ubuntu" -ForegroundColor Gray
Write-Host "    • Git + GitHub CLI" -ForegroundColor Gray
Write-Host "    • Python $PYTHON_VERSION + 全部依赖包" -ForegroundColor Gray
Write-Host "    • TeX Live $TEXLIVE_YEAR (需手动步骤)" -ForegroundColor Gray
Write-Host "    • Node.js + Claude Code" -ForegroundColor Gray
Write-Host "    • 论文项目 + 工作文件 (从 GitHub)" -ForegroundColor Gray
Write-Host ""
Write-Ask "按 Enter 开始部署，Ctrl+C 取消..."
Read-Host

# ============================================================================
# 1. WSL2 + Ubuntu
# ============================================================================
Write-Step "[1/8] 安装 WSL2 + Ubuntu"

$wslOk = $false
try { $null = wsl --status 2>$null; $wslOk = ($LASTEXITCODE -eq 0) } catch {}

if (-not $wslOk) {
    Write-Warn "正在安装 WSL2 (需要重启)..."
    wsl --install --no-distribution 2>$null
    Write-Warn "=============================================="
    Write-Warn "  WSL 安装完成，请重启电脑!"
    Write-Warn "  重启后重新运行此脚本继续安装"
    Write-Warn "=============================================="
    Read-Host "按 Enter 键退出"
    exit 0
}
Write-OK "WSL2 已就绪"

# 检查 Ubuntu
$ubuntuOk = $false
try { $null = wsl --list --quiet 2>$null | Select-String "Ubuntu"; $ubuntuOk = $? } catch {}
if (-not $ubuntuOk) {
    Write-Warn "正在安装 Ubuntu..."
    wsl --install -d Ubuntu
    Write-Warn "Ubuntu 安装完成。请在 Ubuntu 窗口中创建用户后按 Enter 继续..."
    Read-Host
}
Write-OK "Ubuntu 已就绪"

# ============================================================================
# 2. 通过 winget 安装开发工具
# ============================================================================
Write-Step "[2/8] 安装开发工具 (winget)"

# --- Git ---
if (Get-Command git.exe -ErrorAction SilentlyContinue) {
    Write-OK "Git 已安装: $(git --version 2>&1)"
} else {
    Write-Warn "正在安装 Git..."
    winget install --id Git.Git --silent --accept-source-agreements --accept-package-agreements 2>$null
    Refresh-Path
    Write-OK "Git 安装完成"
}

# --- GitHub CLI ---
if (Get-Command gh.exe -ErrorAction SilentlyContinue) {
    Write-OK "GitHub CLI 已安装"
} else {
    Write-Warn "正在安装 GitHub CLI..."
    winget install --id GitHub.cli --silent --accept-source-agreements --accept-package-agreements 2>$null
    Refresh-Path
    Write-OK "GitHub CLI 安装完成"
}

# --- Python 3.11 ---
$pythonExe = "C:\Python311\python.exe"
if (Test-Path $pythonExe) {
    Write-OK "Python 3.11 已安装"
} else {
    Write-Warn "正在安装 Python 3.11..."
    winget install --id Python.Python.3.11 --silent --accept-source-agreements --accept-package-agreements 2>$null
    Refresh-Path
    Write-OK "Python 3.11 安装完成"
}

# --- Node.js ---
if (Get-Command node.exe -ErrorAction SilentlyContinue) {
    Write-OK "Node.js 已安装: $(node --version 2>&1)"
} else {
    Write-Warn "正在安装 Node.js..."
    winget install --id OpenJS.NodeJS --silent --accept-source-agreements --accept-package-agreements 2>$null
    Refresh-Path
    Write-OK "Node.js 安装完成"
}

Refresh-Path

# ============================================================================
# 3. TeX Live (手动步骤)
# ============================================================================
Write-Step "[3/8] TeX Live 2026"

$texliveBin = "C:\texlive\2026\bin\windows\xelatex.exe"
if (Test-Path $texliveBin) {
    Write-OK "TeX Live 2026 已安装"
} else {
    Write-Warn "TeX Live 需要手动安装 (ISO 约 8GB):"
    Write-Host ""
    Write-Host "    方法1: 从镜像站下载 ISO" -ForegroundColor White
    Write-Host "      清华: https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/Images/texlive2026.iso" -ForegroundColor Gray
    Write-Host "      官方: $TEXLIVE_URL" -ForegroundColor Gray
    Write-Host "      挂载 ISO → 运行 install-tl-windows.bat → 完整安装 → 路径 C:\texlive\2026" -ForegroundColor Gray
    Write-Host ""
    Write-Host "    方法2: 使用在线安装器" -ForegroundColor White
    Write-Host "      下载: https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe" -ForegroundColor Gray
    Write-Host "      运行 → 完整安装 → 路径 C:\texlive\2026" -ForegroundColor Gray
    Write-Host ""
    $skipTex = Read-Host "    是否跳过 TeX Live，继续后续步骤? (y/n)"
    if ($skipTex -ne 'y' -and $skipTex -ne 'Y') {
        Write-Warn "请安装完 TeX Live 后重新运行本脚本"
        Read-Host "按 Enter 键退出"
        exit 0
    }
    Write-Warn "已跳过 TeX Live，稍后可手动安装"
}

# 将 TeX Live 加入 PATH
$texBinDir = "C:\texlive\2026\bin\windows"
$userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($userPath -notlike "*$texBinDir*") {
    [Environment]::SetEnvironmentVariable("PATH", "$userPath;$texBinDir", "User")
    Write-OK "TeX Live 已加入 PATH"
}

# ============================================================================
# 4. GitHub 认证 + 项目 Clone
# ============================================================================
Write-Step "[4/8] GitHub 认证 & 克隆项目"

# 检查是否已有 SSH 密钥
$sshKey = "$UserProfile\.ssh\id_rsa"
$needsSsh = -not (Test-Path $sshKey)

if ($needsSsh) {
    Write-Warn "未检测到 SSH 密钥，正在生成..."
    New-Item -ItemType Directory -Force -Path "$UserProfile\.ssh" | Out-Null
    ssh-keygen -t rsa -b 4096 -f $sshKey -N '""' -C $GIT_USER_EMAIL 2>$null
    Write-OK "SSH 密钥已生成: $sshKey"
    Write-Warn "请将以下公钥添加到 GitHub:"
    Write-Host ""
    Get-Content "$sshKey.pub"
    Write-Host ""
    Write-Warn "1. 打开 https://github.com/settings/ssh/new"
    Write-Warn "2. 粘贴上面的公钥，保存"
    Write-Warn "3. 按 Enter 继续..."
    Read-Host
}

# 尝试 GitHub CLI 登录
$ghOk = $false
try { $null = gh auth status 2>$null; $ghOk = ($LASTEXITCODE -eq 0) } catch {}
if (-not $ghOk) {
    Write-Warn "正在启动 GitHub CLI 登录 (浏览器)..."
    gh auth login --hostname github.com --web --git-protocol https --scopes repo,workflow 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-OK "GitHub 登录成功"
    } else {
        Write-Warn "GitHub CLI 登录跳过 (不影响后续，使用 SSH)"
    }
}

# Clone 论文仓库
Write-Step "克隆论文仓库..."

$thesisDir = "$UserProfile\Documents\thesis"
if (Test-Path "$thesisDir\.git") {
    Write-OK "论文仓库已存在，执行 git pull..."
    Push-Location $thesisDir
    git pull origin main 2>$null
    Pop-Location
} else {
    Write-Warn "正在克隆论文仓库..."
    New-Item -ItemType Directory -Force -Path "$UserProfile\Documents" | Out-Null
    git clone $GITHUB_REPO $thesisDir 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-OK "论文仓库克隆完成"
    } else {
        Write-Err "克隆失败，请检查网络和 GitHub 认证"
        Write-Err "手动重试: git clone $GITHUB_REPO $thesisDir"
    }
}

# ============================================================================
# 5. 还原工作目录
# ============================================================================
Write-Step "[5/8] 还原 Claudecode 工作目录"

$claudeDir = "$UserProfile\Claudecode"
$workspaceSrc = "$thesisDir\workspace"

New-Item -ItemType Directory -Force -Path $claudeDir | Out-Null

if (Test-Path $workspaceSrc) {
    Write-Warn "从论文仓库还原工作文件..."
    robocopy "$workspaceSrc" "$claudeDir" /E /NFL /NDL /NP /NS /NC /XO `
        /XD ".git" "__pycache__" 2>$null
    Write-OK "工作文件已还原到 $claudeDir"
} else {
    Write-Warn "论文仓库中未找到 workspace/ 目录"
    Write-Warn "请在原电脑上运行 push-workspace.ps1 推送工作文件"
    Write-Warn "或手动创建 CLAUDE.md"
}

# 确保 migrate 目录存在
New-Item -ItemType Directory -Force -Path "$claudeDir\migrate" | Out-Null

# ============================================================================
# 6. 安装 Python 依赖
# ============================================================================
Write-Step "[6/8] 安装 Python 依赖包"

if (Test-Path $pythonExe) {
    Write-Warn "升级 pip..."
    & $pythonExe -m pip install --upgrade pip 2>$null

    $reqFile = "$claudeDir\requirements_win.txt"
    if (-not (Test-Path $reqFile)) {
        $reqFile = "$claudeDir\migrate\requirements_win.txt"
    }

    if (Test-Path $reqFile) {
        Write-Warn "安装 Python 包 (可能需要几分钟)..."
        & $pythonExe -m pip install -r $reqFile 2>$null
        Write-OK "Python 依赖安装完成"
    } else {
        Write-Warn "未找到 requirements_win.txt，安装核心包..."

        & $pythonExe -m pip install torch --index-url https://download.pytorch.org/whl/cpu 2>$null
        & $pythonExe -m pip install numpy scipy scikit-learn matplotlib transformers sentence-transformers 2>$null
        & $pythonExe -m pip install pandas plotly openpyxl pytesseract PyMuPDF pypdf PyPDF2 reportlab python-docx 2>$null
        & $pythonExe -m pip install onnx onnxscript safetensors tokenizers 2>$null
        Write-OK "核心 Python 包安装完成"
    }

    # WSL Python 包
    Write-Warn "安装 WSL Python 包..."
    try {
        wsl -d Ubuntu -e bash -c "pip3 install --upgrade pip 2>/dev/null; pip3 install numpy scipy matplotlib pandas scikit-learn 2>/dev/null" 2>$null
        Write-OK "WSL Python 包安装完成"
    } catch {
        Write-Warn "WSL Python 包安装跳过"
    }
} else {
    Write-Err "Python 3.11 未找到，跳过包安装"
}

# ============================================================================
# 7. Claude Code + Git 配置
# ============================================================================
Write-Step "[7/8] 安装 Claude Code & 配置 Git"

# --- Claude Code ---
if (Get-Command claude -ErrorAction SilentlyContinue) {
    Write-OK "Claude Code 已安装: $(claude --version 2>&1)"
} else {
    Write-Warn "安装 Claude Code..."
    npm install -g @anthropic-ai/claude-code 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-OK "Claude Code 安装完成"
    } else {
        Write-Err "Claude Code 安装失败，请手动: npm install -g @anthropic-ai/claude-code"
    }
}

# --- Git 配置 ---
Write-Warn "设置 Git 全局配置..."
git config --global user.name $GIT_USER_NAME
git config --global user.email $GIT_USER_EMAIL
git config --global init.defaultbranch main
Write-OK "Git 配置完成 ($GIT_USER_NAME <$GIT_USER_EMAIL>)"

# WSL Git 配置
try {
    wsl -d Ubuntu -e bash -c "
        git config --global user.name '$GIT_USER_NAME'
        git config --global user.email '$GIT_USER_EMAIL'
        git config --global init.defaultbranch main
    " 2>$null
    Write-OK "WSL Git 配置完成"
} catch {}

# --- Claude Code 设置模板 ---
Write-Step "Claude Code API 配置"

$claudeSettingsDir = "$UserProfile\.claude"
$claudeSettingsFile = "$claudeSettingsDir\settings.json"

if (Test-Path $claudeSettingsFile) {
    Write-OK "Claude Code 设置已存在"
} else {
    New-Item -ItemType Directory -Force -Path $claudeSettingsDir | Out-Null

    Write-Warn "创建 Claude Code 设置模板..."
    Write-Ask "请输入 DeepSeek API Key (留空则稍后手动编辑 settings.json):"
    $apiKey = Read-Host

    if ($apiKey) {
        $settingsJson = @"
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "$apiKey",
    "ANTHROPIC_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro[1m]",
    "CLAUDE_CODE_EFFORT_LEVEL": "max"
  },
  "includeCoAuthoredBy": false,
  "theme": "dark"
}
"@
        $settingsJson | Out-File -Encoding UTF8 $claudeSettingsFile
        Write-OK "Claude Code 设置已创建"
    } else {
        # 创建模板文件
        @"
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "<填入你的 DeepSeek API Key>",
    "ANTHROPIC_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro[1m]",
    "CLAUDE_CODE_EFFORT_LEVEL": "max"
  },
  "includeCoAuthoredBy": false,
  "theme": "dark"
}
"@ | Out-File -Encoding UTF8 $claudeSettingsFile
        Write-Warn "模板已创建，请编辑 $claudeSettingsFile 填入 API Key"
    }
}

# ============================================================================
# 8. 验证
# ============================================================================
Write-Step "[8/8] 环境验证"

Write-Host ""
Write-Host "  Windows 工具:" -ForegroundColor White

function Test-Cmd {
    param($Name, $Cmd)
    try {
        $result = Invoke-Expression "$Cmd 2>&1" | Select-Object -First 1
        Write-Host "  [$([char]0x2713)] $Name — $result" -ForegroundColor Green
    } catch {
        Write-Host "  [$([char]0x2717)] $Name — 未找到" -ForegroundColor Red
    }
}

Test-Cmd "TeX Live" 'xelatex --version 2>$null | Select-Object -First 1'
Test-Cmd "Python 3.11" '& C:\Python311\python.exe --version'
Test-Cmd "Node.js" 'node --version'
Test-Cmd "npm" 'npm --version'
Test-Cmd "Git" 'git --version'
Test-Cmd "GitHub CLI" 'gh --version 2>$null | Select-Object -First 1'
Test-Cmd "Claude Code" 'claude --version'

Write-Host ""
Write-Host "  项目文件:" -ForegroundColor White
if (Test-Path "$UserProfile\Claudecode\CLAUDE.md") { Write-Host "  [OK] Claudecode 目录" -ForegroundColor Green } else { Write-Host "  [--] Claudecode 目录 — 请运行 push-workspace.ps1 推送" -ForegroundColor Yellow }
if (Test-Path "$UserProfile\Documents\thesis\zjuthesis.tex") { Write-Host "  [OK] 论文目录" -ForegroundColor Green } else { Write-Host "  [ER] 论文目录 — 请检查 GitHub clone" -ForegroundColor Red }
if (Test-Path "$UserProfile\.ssh\id_rsa") { Write-Host "  [OK] SSH 密钥" -ForegroundColor Green } else { Write-Host "  [--] SSH 密钥" -ForegroundColor Yellow }
if (Test-Path "$UserProfile\.claude\settings.json") { Write-Host "  [OK] Claude Code 设置" -ForegroundColor Green } else { Write-Host "  [--] Claude Code 设置" -ForegroundColor Yellow }

Write-Host ""
Write-Host "  WSL:" -ForegroundColor White
try {
    wsl --version 2>$null | Select-Object -First 1 | ForEach-Object { Write-Host "  [OK] $_" -ForegroundColor Green }
} catch { Write-Host "  [ER] WSL 未就绪" -ForegroundColor Red }

# ============================================================================
# 完成
# ============================================================================
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              环境部署完成!                                  ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$hasAll = $true

if (-not (Test-Path $texliveBin)) {
    Write-Warn "TeX Live 未安装 — 请手动安装后重新运行本脚本"
    $hasAll = $false
}
if (-not (Test-Path "$UserProfile\Documents\thesis\zjuthesis.tex")) {
    Write-Err "论文仓库未克隆 — 请检查 GitHub 认证"
    $hasAll = $false
}

if ($hasAll) {
    Write-Host "  快速测试:" -ForegroundColor White
    Write-Host ""
    Write-Host "  # 编译论文" -ForegroundColor Gray
    Write-Host "  cd $UserProfile\Documents\thesis" -ForegroundColor Gray
    Write-Host "  xelatex zjuthesis.tex && bibtex zjuthesis && xelatex zjuthesis.tex && xelatex zjuthesis.tex" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  # 测试手势识别" -ForegroundColor Gray
    Write-Host "  cd $UserProfile\Documents\thesis\gesture_nn" -ForegroundColor Gray
    Write-Host "  python main.py --device cpu --skip_gen --skip_train" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  # 启动 Claude Code" -ForegroundColor Gray
    Write-Host "  cd $UserProfile\Claudecode && claude" -ForegroundColor Gray
} else {
    Write-Warn "部分组件未安装完成，修复后重新运行本脚本"

    if (-not (Test-Path $texliveBin)) {
        Write-Host "    → TeX Live 2026: $TEXLIVE_URL" -ForegroundColor Yellow
    }
}

Write-Host ""
Read-Host "按 Enter 键退出"
