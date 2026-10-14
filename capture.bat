@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if "%~1"=="" (
    echo Dung: capture ^<ten_anh^>
    echo Vi du: capture giay_dki_xe_bus
    echo Anh luu tai: exports\camera\^<ten_anh^>.png
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_env.bat first.
    exit /b 1
)
set "NAME=%*"
set "NAME=%NAME: =_%"
if not defined MVSDK_LIBRARY set "MVSDK_LIBRARY=D:\MV Viewer\Runtime\x64\MVSDKmd.dll"
".venv\Scripts\python.exe" -u scripts/camera_probe.py --capture "Dahua Technology:6E00F58PAK00020" --output "exports/camera/%NAME%.png"
set "result=%ERRORLEVEL%"
if not defined PCB_NO_PAUSE pause
exit /b %result%
