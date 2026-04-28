#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -f "brain/data/english_core.jsonl" ]; then
  echo "[first run] building dataset..."
  python3 tools/build_dataset.py
  python3 tools/rebuild_indexes.py
  python3 tools/validate_brain.py
fi
python3 runtime/main.py
