# ============================================================================
# push-workspace.ps1
# Push Claudecode workspace files to the thesis GitHub repo (workspace/ dir)
# Run on OLD machine before migrating
# ============================================================================

$ErrorActionPreference = "Stop"

$ClaudecodeDir = "$env:USERPROFILE\Claudecode"
$ThesisDir = "$env:USERPROFILE\Documents\thesis"
$WorkspaceDir = "$ThesisDir\workspace"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Push Workspace Files to GitHub" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check thesis repo exists
if (-not (Test-Path $ThesisDir)) {
    Write-Host "[ERROR] Thesis repo not found: $ThesisDir" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path "$ThesisDir\.git")) {
    Write-Host "[ERROR] Thesis dir is not a Git repo" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Create workspace directory
New-Item -ItemType Directory -Force -Path $WorkspaceDir | Out-Null

# ============================================================================
# Copy Claudecode files to thesis repo workspace/ directory
# ============================================================================
Write-Host "[1/3] Copying workspace files..." -ForegroundColor Green

$filesToCopy = @(
    "CLAUDE.md",
    "AIGC_Chart_Paper.md",
    "chartforge.tex",
    "basketball_game.html",
    [char]0x4FEE + [char]0x6539 + [char]0x610F + [char]0x89C1 + ".txt",
    [char]0x4FEE + [char]0x6539 + [char]0x8BF4 + [char]0x660E + ".md",
    "SDCR_Vis_System",
    "migrate"
)

foreach ($item in $filesToCopy) {
    $src = Join-Path $ClaudecodeDir $item
    if (Test-Path $src) {
        if (Test-Path $src -PathType Container) {
            robocopy "$src" "$WorkspaceDir\$item" /E /NFL /NDL /NP /NS /NC `
                /XD "__pycache__" ".git" ".claude" "node_modules" 2>$null
        } else {
            Copy-Item $src "$WorkspaceDir\$item" -Force
        }
        Write-Host "  OK: $item" -ForegroundColor Gray
    } else {
        Write-Host "  SKIP: $item (not found)" -ForegroundColor DarkGray
    }
}

# Copy requirements_win.txt
if (Test-Path "$ClaudecodeDir\migrate\requirements_win.txt") {
    Copy-Item "$ClaudecodeDir\migrate\requirements_win.txt" "$WorkspaceDir\" -Force
}

# Copy launcher batch file template
@"
@echo off
REM Launch Claude Code
wsl -e bash -c "cd /mnt/c/Users/%USERNAME%/Claudecode && claude"
"@ | Out-File -Encoding ASCII "$WorkspaceDir\launch_claude.bat"

Write-Host "  Workspace files copied to $WorkspaceDir" -ForegroundColor Green

# ============================================================================
# Create .gitignore for workspace
# ============================================================================
Write-Host "[2/3] Creating workspace .gitignore..." -ForegroundColor Green

@"
# workspace .gitignore
*.pdf
*.png
*.jpg
__pycache__/
*.pyc
"@ | Out-File -Encoding ASCII "$WorkspaceDir\.gitignore"

# ============================================================================
# Git commit & push
# ============================================================================
Write-Host "[3/3] Commit & push to GitHub..." -ForegroundColor Green

Push-Location $ThesisDir

$status = git status --porcelain workspace/ 2>$null
if (-not $status) {
    Write-Host "  No changes, skip commit" -ForegroundColor Yellow
} else {
    git add workspace/

    $commitMsg = "chore: update workspace files ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))"
    git commit -m $commitMsg

    Write-Host "  Pushing to GitHub..." -ForegroundColor Gray
    git push origin main 2>$null

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Pushed successfully" -ForegroundColor Green
    } else {
        Write-Host "  Push failed. Check network/GitHub auth" -ForegroundColor Red
        Write-Host "  Retry: cd $ThesisDir && git push origin main" -ForegroundColor Yellow
    }
}

Pop-Location

# ============================================================================
# Done
# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Done!" -ForegroundColor Green
Write-Host ""
Write-Host "  On new machine, run restore.ps1 to set up everything" -ForegroundColor White
Write-Host "  GitHub repo: https://github.com/MmdArchimedes/zjuthesis-thesis" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Cyan

Read-Host "`nPress Enter to exit"
