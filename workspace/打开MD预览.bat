@echo off
chcp 65001 >nul
echo.
echo   ┌─────────────────────────────────────────┐
echo   │     Markdown 离线预览 — 双击 .md 查看    │
echo   └─────────────────────────────────────────┘
echo.
echo   拖放 .md 文件到浏览器窗口即可渲染
echo.
start "" "%~dp0预览Markdown.html"
exit
