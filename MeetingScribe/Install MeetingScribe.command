#!/bin/bash
# MeetingScribe installer — creates a private Python environment in this folder.
set -e
cd "$(dirname "$0")"

echo "────────────────────────────────────"
echo "  MeetingScribe installer"
echo "────────────────────────────────────"

if ! command -v python3 >/dev/null 2>&1; then
  echo ""
  echo "Python 3 was not found."
  echo "Run this in Terminal first:   xcode-select --install"
  echo "(or install Python from python.org), then run this installer again."
  exit 1
fi

echo "Using $(python3 --version)"
echo "Creating environment…"
python3 -m venv venv
./venv/bin/python -m pip install --quiet --upgrade pip
echo "Installing faster-whisper, sherpa-onnx and flask (a few minutes on first install)…"
./venv/bin/python -m pip install faster-whisper flask sherpa-onnx numpy

echo ""
echo "✓ Installed."
echo "Double-click “Start MeetingScribe.command” to open the app."

if [ "$1" != "--no-pause" ]; then
  echo ""
  read -n 1 -s -r -p "Press any key to close this window…"
  echo ""
fi
