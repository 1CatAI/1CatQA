#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:?usage: install_desktop_source_app.sh /path/to/project-root [desktop-dir]}"
APP_NAME="1Cat-V100-QA"

strip_cr() {
  printf '%s' "$1" | tr -d '\r'
}

resolve_desktop_dir() {
  if command -v xdg-user-dir >/dev/null 2>&1; then
    local dir
    dir="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
    if [ -n "$dir" ]; then
      printf '%s\n' "$(strip_cr "$dir")"
      return
    fi
  fi

  if [ -d "$HOME/桌面" ]; then
    printf '%s\n' "$HOME/桌面"
    return
  fi

  printf '%s\n' "$HOME/Desktop"
}

PROJECT_ROOT="$(strip_cr "$PROJECT_ROOT")"
DESKTOP_DIR="$(strip_cr "${2:-$(resolve_desktop_dir)}")"
APP_DIR="$DESKTOP_DIR/$APP_NAME"
DESKTOP_FILE="$DESKTOP_DIR/$APP_NAME.desktop"
LAUNCHER="$APP_DIR/$APP_NAME"

mkdir -p "$APP_DIR"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude '.git/' \
    --exclude '.codex-tmp/' \
    --exclude 'remote_reports/' \
    --exclude '__pycache__/' \
    "$PROJECT_ROOT"/src "$PROJECT_ROOT"/scripts "$PROJECT_ROOT"/cuda \
    "$PROJECT_ROOT"/pyproject.toml "$PROJECT_ROOT"/README.md \
    "$APP_DIR"/
else
  rm -rf "$APP_DIR/src" "$APP_DIR/scripts" "$APP_DIR/cuda"
  cp -a "$PROJECT_ROOT"/src "$PROJECT_ROOT"/scripts "$PROJECT_ROOT"/cuda "$APP_DIR"/
  cp -a "$PROJECT_ROOT"/pyproject.toml "$PROJECT_ROOT"/README.md "$APP_DIR"/
fi

cat > "$LAUNCHER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m gpuqa.gui_entry "$@"
EOF

chmod u=rwx,go=rx "$LAUNCHER"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=$APP_NAME
Name[zh_CN]=$APP_NAME
Comment=V100-specialized GPU QA runner optimized for CUDA 12
Comment[zh_CN]=面向 V100 的 CUDA 12 专项质检工具
Exec=$LAUNCHER
Terminal=true
Categories=Utility;System;
StartupNotify=true
Icon=utilities-terminal
EOF

chown "$(id -un):$(id -gn)" "$DESKTOP_FILE" 2>/dev/null || true
chgrp "$(id -gn)" "$DESKTOP_FILE" 2>/dev/null || true
chmod u=rwx,go=rx "$DESKTOP_FILE"
if command -v gio >/dev/null 2>&1; then
  gio set "$DESKTOP_FILE" metadata::trusted true >/dev/null 2>&1 || true
fi

printf 'Installed desktop source app: %s\n' "$APP_DIR"
printf 'Installed desktop entry: %s\n' "$DESKTOP_FILE"
