#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDORED_BIN="${SCRIPT_DIR}/../src/gpuqa/vendor/gpu-burn/ubuntu24.04-cuda12/usr/sbin/gpu-burn"
VENDORED_COMPARE="${SCRIPT_DIR}/../src/gpuqa/vendor/gpu-burn/ubuntu24.04-cuda12/usr/share/gpu-burn/compare.ptx"
GPU_BURN_MEMORY_ARG="${GPU_BURN_MEMORY_ARG:-100%}"

if [[ -f "${VENDORED_BIN}" && -f "${VENDORED_COMPARE}" ]]; then
  chmod 755 "${VENDORED_BIN}" 2>/dev/null || true
  exec "${VENDORED_BIN}" -c "${VENDORED_COMPARE}" -m "${GPU_BURN_MEMORY_ARG}" "$@"
fi

if [[ -x /usr/sbin/gpu-burn && -f /usr/share/gpu-burn/compare.ptx ]]; then
  exec /usr/sbin/gpu-burn -c /usr/share/gpu-burn/compare.ptx -m "${GPU_BURN_MEMORY_ARG}" "$@"
fi

exec /opt/gpu-burn/bin/gpu_burn -c /opt/gpu-burn/bin/compare.ptx -m "${GPU_BURN_MEMORY_ARG}" "$@"
