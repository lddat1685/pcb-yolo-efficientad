@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_env.bat first.
    if not defined PCB_NO_PAUSE pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 exit /b 1
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
".venv\Scripts\python.exe" scripts/run_verify.py
set "result=%ERRORLEVEL%"
if not defined PCB_NO_PAUSE pause
exit /b %result%
