@echo off
chcp 65001 >nul
title DFL 自动切脸插件 - 处理 目标图片 (DST)

set "PLUGIN_DIR=%~dp0"
cd /d "%PLUGIN_DIR%"
set "VENV_DIR=..\plugins_env"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [错误] 找不到虚拟环境！路径不存在: %VENV_DIR%\Scripts\activate.bat
    pause
    exit /b
)

echo [INFO] 正在激活虚拟环境...
call "%VENV_DIR%\Scripts\activate.bat"

echo [INFO] 开始处理 data_dst...
python main.py --target dst

echo.
echo [INFO] 目标图片 (DST) 切脸任务结束。
pause