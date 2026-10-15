@echo off
REM ============================================================================
REM  Start the pyPhotometry analysis app on this computer (Windows).
REM  Double-click this file. The app opens in your browser at
REM  http://localhost:8501 . Close this window to stop it.
REM ============================================================================
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (
  echo Python 3 is not installed.
  echo Install Python 3.12 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" on the first installer screen.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo First run: setting up the app ^(takes 1-3 minutes, needs internet^)...
  %PY% -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
)
set "VPY=.venv\Scripts\python.exe"

fc /b requirements.txt .venv\requirements_installed.txt >nul 2>nul
if errorlevel 1 (
  echo Installing / updating packages...
  "%VPY%" -m pip install --upgrade pip -q
  "%VPY%" -m pip install -r requirements.txt -q || (echo Package installation failed. & pause & exit /b 1)
  copy /y requirements.txt .venv\requirements_installed.txt >nul
)

echo.
echo Starting the app... it opens in your browser at http://localhost:8501
echo Keep this window open while you use it. Close it to stop.
echo.
"%VPY%" -m streamlit run app.py --server.maxUploadSize 4096 --browser.gatherUsageStats false
pause
