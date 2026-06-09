#!/usr/bin/env bash
set -euo pipefail

APP_SOURCE="${1:?usage: install_desktop_entry.sh /path/to/1Cat-V100-QA[/bundle] [desktop-dir|desktop-file]}"
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

APP_SOURCE="$(strip_cr "$APP_SOURCE")"
DESKTOP_TARGET="$(strip_cr "${2:-$(resolve_desktop_dir)}")"
if [[ "$DESKTOP_TARGET" == *.desktop ]]; then
  DESKTOP_FILE="$DESKTOP_TARGET"
  DESKTOP_DIR="$(dirname "$DESKTOP_FILE")"
else
  DESKTOP_DIR="$DESKTOP_TARGET"
  DESKTOP_FILE="$DESKTOP_DIR/$APP_NAME.desktop"
fi

mkdir -p "$DESKTOP_DIR"

if [ -d "$APP_SOURCE" ]; then
  APP_DIR="$DESKTOP_DIR/$APP_NAME"
  mkdir -p "$APP_DIR"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "$APP_SOURCE"/ "$APP_DIR"/
  else
    cp -a "$APP_SOURCE"/. "$APP_DIR"/
  fi
  APP_BIN="$APP_DIR/$APP_NAME"
else
  APP_BIN="$APP_SOURCE"
fi

if [ ! -x "$APP_BIN" ]; then
  printf 'Executable not found: %s\n' "$APP_BIN" >&2
  exit 1
fi

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=$APP_NAME
Name[zh_CN]=$APP_NAME
Comment=V100-specialized GPU QA runner optimized for CUDA 12
Comment[zh_CN]=面向 V100 的 CUDA 12 专项质检工具
Exec=$APP_BIN
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
printf 'Installed desktop entry: %s\n' "$DESKTOP_FILE"
if [ -n "${APP_DIR:-}" ]; then
  printf 'Installed desktop app bundle: %s\n' "$APP_DIR"
fi
