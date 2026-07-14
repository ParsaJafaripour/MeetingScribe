#!/bin/bash
# Launch MeetingScribe. Installs dependencies automatically on first run.
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ] || ! ./venv/bin/python -c "import faster_whisper, flask, sherpa_onnx" >/dev/null 2>&1; then
  echo "Installing dependencies (first run or upgrade)…"
  bash "./Install MeetingScribe.command" --no-pause || exit 1
fi

exec ./venv/bin/python app.py
