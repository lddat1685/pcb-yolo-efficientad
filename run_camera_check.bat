@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
echo Checking MV Viewer SDK and GigE cameras...
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_env.bat first.
    if not defined PCB_NO_PAUSE pause
    exit /b 1
)
".venv\Scripts\python.exe" -u scripts/camera_probe.py %*
set "result=%ERRORLEVEL%"
if not defined PCB_NO_PAUSE pause
exit /b %result%
