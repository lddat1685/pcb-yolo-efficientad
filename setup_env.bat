@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" if exist ".venv\Scripts\activate.bat" goto install
if defined PCB_PYTHON (
    "%PCB_PYTHON%" -m venv .venv
) else (
    py -3.11 -m venv .venv 2>nul
    if errorlevel 1 py -3.12 -m venv .venv 2>nul
    if errorlevel 1 python -m venv .venv
)
if errorlevel 1 goto fail
:install
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto fail
set "PATH=%CD%\.venv\Scripts;%PATH%"
".venv\Scripts\python.exe" -c "import sys; assert sys.prefix != sys.base_prefix, 'A virtual environment is required'; assert (3,11) <= sys.version_info[:2] < (3,13), 'Use Python 3.11 or 3.12; rename an incompatible .venv before setup.'"
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install --no-deps -e .
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto fail
".venv\Scripts\python.exe" scripts/verify_reproducibility.py
if errorlevel 1 goto fail
echo Setup complete. Run run_ui.bat to open the demo.
if not defined PCB_NO_PAUSE pause
exit /b 0
:fail
echo Setup failed. See the error above and README.md.
if not defined PCB_NO_PAUSE pause
exit /b 1
