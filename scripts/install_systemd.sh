#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${1:-/opt/gpuqa}"
OUTPUT_DIR="${2:-/var/lib/gpuqa/latest}"
UNIT_PATH="/etc/systemd/system/gpuqa.service"

install -d "${WORKDIR}"
install -d "${OUTPUT_DIR}"
install -m 0644 "${WORKDIR}/systemd/gpuqa.service" "${UNIT_PATH}"
systemctl daemon-reload
systemctl enable gpuqa.service

printf 'Installed %s\n' "${UNIT_PATH}"
printf 'Working directory: %s\n' "${WORKDIR}"
printf 'Output directory: %s\n' "${OUTPUT_DIR}"
