#!/bin/bash
# Double-click in Finder to play the card games: starts the local game server and opens the lobby in your browser.
# Close this window, or press Ctrl+C in it, to stop the server. If the server is already running, it just opens it.
cd "$(dirname "$0")/.." || exit 1  # the project root, one up from viz/

pause() { read -r -n 1 -p "Press any key to close this window." _; echo; }

if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "Python 3 was not found. Install it, then run: python3 -m pip install -r requirements.txt"
  pause; exit 1
fi

if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
  echo "The project's Python packages are missing. From this folder, run:"
  echo "  $PY -m pip install -r requirements.txt"
  pause; exit 1
fi

"$PY" -m src.CardGames.Play "$@"
status=$?
[ "$status" -ne 0 ] && pause
exit "$status"
