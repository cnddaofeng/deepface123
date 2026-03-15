@echo off
:: 1. 切换到项目目录
cd /d "D:/DeepFaceLab-Torch/plugins/"

:: 2. 使用 call 激活环境，这样脚本执行完 activate 后会继续往下走
call D:\DeepFaceLab-Torch\plugins\plugins_env\Scripts\activate.bat


:: 3. 使用 /k 保持 CMD 窗口开启，并留在虚拟环境中
cmd /k