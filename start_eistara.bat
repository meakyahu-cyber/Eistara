@echo off
chcp 65001 >nul 2>&1
cd /D "%~dp0"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore
set FORCE_COLOR=1
set TTY_COMPATIBLE=1
set TERM=xterm-256color
if exist "%~dp0tools\ffmpeg\bin\ffmpeg.exe" set PATH=%~dp0tools\ffmpeg\bin;%PATH%

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found.
    echo Please run: python setup_env.py
    pause
    exit /b 1
)

echo Eistara starting...
echo Python: .venv\Scripts\python.exe
echo URL: http://localhost:10127
echo.

".venv\Scripts\python.exe" -X utf8 launch.py

echo.
echo Eistara stopped.
pause
