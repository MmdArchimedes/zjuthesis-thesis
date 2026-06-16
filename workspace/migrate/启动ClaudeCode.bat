@echo off
REM 启动 Claude Code — 迁移后在新电脑上的快捷启动脚本
REM 将此文件复制到 C:\Users\<用户名>\Claudecode\ 目录下运行

wsl -e bash -c "cd /mnt/c/Users/%USERNAME%/Claudecode && claude"
