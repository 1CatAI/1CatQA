#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist}"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/build/pyinstaller/work}"
SPEC_DIR="${SPEC_DIR:-$ROOT_DIR/build/pyinstaller/spec}"

"$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --console \
  --name "1Cat-V100-QA" \
  --distpath "$DIST_DIR" \
  --workpath "$WORK_DIR" \
  --specpath "$SPEC_DIR" \
  --paths "$ROOT_DIR/src" \
  --add-data "$ROOT_DIR/src/gpuqa/vendor:gpuqa/vendor" \
  --add-data "$ROOT_DIR/cuda/p2p_bandwidth_matrix.cu:cuda" \
  "$ROOT_DIR/scripts/launch_gui.py"

printf 'Built text UI bundle: %s\n' "$DIST_DIR/1Cat-V100-QA"
