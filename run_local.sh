#!/bin/bash
# =============================================================================
#  Start the pyPhotometry analysis app on this computer (macOS / Linux).
#  macOS: double-click this file. If macOS says it "cannot be opened",
#  right-click it -> Open -> Open (once), or run in Terminal:
#        bash run_local.command
#  The app opens in your browser at http://localhost:8501
#  Close this window (or press Ctrl+C) to stop it.
# =============================================================================
cd "$(dirname "$0")" || exit 1

pause() { read -r -p "Press Enter to close this window." _; }

PY=""
for c in python3.12 python3.11 python3.13 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3 is not installed."
  echo "Install Python 3.12 from https://www.python.org/downloads/ and run this again."
  pause; exit 1
fi

# A private environment in .venv, so nothing on the system is changed.
if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up the app (takes 1-3 minutes, needs internet)..."
  "$PY" -m venv .venv || { echo "Could not create the environment."; pause; exit 1; }
fi
VPY=.venv/bin/python

# Reinstall packages only when requirements.txt has changed.
STAMP=.venv/.requirements_installed
if [ ! -f "$STAMP" ] || ! cmp -s requirements.txt "$STAMP"; then
  echo "Installing / updating packages..."
  "$VPY" -m pip install --upgrade pip -q
  "$VPY" -m pip install -r requirements.txt -q || { echo "Package installation failed (see above)."; pause; exit 1; }
  cp requirements.txt "$STAMP"
fi

echo ""
echo "Starting the app... it opens in your browser at http://localhost:8501"
echo "Keep this window open while you use it. Close it (or press Ctrl+C) to stop."
echo ""
# Locally there is no 2.7 GB cloud limit, so allow large uploads.
"$VPY" -m streamlit run app.py --server.maxUploadSize 4096 --browser.gatherUsageStats false
pause
