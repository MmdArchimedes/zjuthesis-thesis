@echo off
cd /d "%~dp0"

echo ========================================
echo   论文编译脚本
echo ========================================
echo.

echo [1/4] xelatex (第一次)...
xelatex -halt-on-error -interaction=nonstopmode zjuthesis.tex
if errorlevel 1 (
    echo.
    echo [错误] 编译失败，请检查上方错误信息
    pause
    exit /b 1
)
echo [1/4] 完成
echo.

echo [2/4] biber (参考文献)...
biber zjuthesis
if errorlevel 1 (
    echo.
    echo [错误] biber 失败
    pause
    exit /b 1
)
echo [2/4] 完成
echo.

echo [3/4] xelatex (第二次)...
xelatex -halt-on-error -interaction=nonstopmode zjuthesis.tex
if errorlevel 1 (
    echo.
    echo [错误] 编译失败
    pause
    exit /b 1
)
echo [3/4] 完成
echo.

echo [4/4] xelatex (第三次)...
xelatex -halt-on-error -interaction=nonstopmode zjuthesis.tex
if errorlevel 1 (
    echo.
    echo [错误] 编译失败
    pause
    exit /b 1
)
echo [4/4] 完成
echo.

echo ========================================
echo   编译成功！
echo ========================================
start "" "%~dp0zjuthesis.pdf"
pause
