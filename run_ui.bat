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
rem Streamlit opens the default browser when headless is false.
".venv\Scripts\python.exe" -m streamlit run src/pcb_lab/ui/app.py --server.port 8501 --server.address localhost --server.headless false --browser.gatherUsageStats false
set "result=%ERRORLEVEL%"
if not "%result%"=="0" if not defined PCB_NO_PAUSE pause
exit /b %result%
