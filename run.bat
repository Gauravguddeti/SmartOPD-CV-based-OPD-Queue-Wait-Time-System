@echo off
rem ============================================================
rem  SmartOPD - one-click launcher
rem  Everything (venv, caches, model weights, data) stays inside
rem  this project folder. Nothing is installed globally.
rem ============================================================
setlocal
cd /d "%~dp0"

set "PIP_USER="
set "PIP_CACHE_DIR=%CD%\.pip-cache"
set "TMP=%CD%\.tmp"
set "TEMP=%CD%\.tmp"
set "YOLO_CONFIG_DIR=%CD%\.ultralytics"
set "TORCH_HOME=%CD%\.torch"
set "MPLCONFIGDIR=%CD%\.mpl"
set "PYTHONIOENCODING=utf-8"

if not exist ".tmp" mkdir ".tmp" >nul 2>&1

py -3.11 ".\scripts\ensure_ready.py"
if errorlevel 1 (
  echo.
  echo Setup failed - see the message above.
  pause
  exit /b 1
)

echo.
echo   SmartOPD starting on http://127.0.0.1:8000
echo   Monitor .... http://127.0.0.1:8000/
echo   Nurse desk . http://127.0.0.1:8000/nurse
echo   Patient .... http://127.0.0.1:8000/patient
echo.
echo   Press Ctrl+C here to stop the server.
echo.
start "" http://127.0.0.1:8000/

".venv\Scripts\python.exe" -m uvicorn server:app --app-dir app --host 127.0.0.1 --port 8000
pause
