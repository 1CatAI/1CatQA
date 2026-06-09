#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

PKG_URL="${PKG_URL:-https://mirrors.kernel.org/ubuntu/pool/multiverse/g/gpu-burn/gpu-burn_0+git20240115+ds-2_amd64.deb}"
PKG_SHA256="${PKG_SHA256:-633e659402eee7907d5009e94b2413457eca944df3a0483d55663a599f7fcada}"
PKG_FILE="${WORK_DIR}/gpu-burn.deb"
OUT_DIR="${REPO_ROOT}/src/gpuqa/vendor/gpu-burn/ubuntu24.04-cuda12"

curl -L --fail --output "${PKG_FILE}" "${PKG_URL}"
echo "${PKG_SHA256}  ${PKG_FILE}" | sha256sum -c -

mkdir -p "${WORK_DIR}/deb" "${WORK_DIR}/rootfs"
tar -xf "${PKG_FILE}" -C "${WORK_DIR}/deb"
tar -xf "${WORK_DIR}/deb/data.tar.zst" -C "${WORK_DIR}/rootfs"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}/usr/sbin" "${OUT_DIR}/usr/share/gpu-burn" "${OUT_DIR}/usr/share/doc/gpu-burn"
install -m 0755 "${WORK_DIR}/rootfs/usr/sbin/gpu-burn" "${OUT_DIR}/usr/sbin/gpu-burn"
install -m 0644 "${WORK_DIR}/rootfs/usr/share/gpu-burn/compare.ptx" "${OUT_DIR}/usr/share/gpu-burn/compare.ptx"
install -m 0644 "${WORK_DIR}/rootfs/usr/share/doc/gpu-burn/README.md" "${OUT_DIR}/usr/share/doc/gpu-burn/README.md"
install -m 0644 "${WORK_DIR}/rootfs/usr/share/doc/gpu-burn/copyright" "${OUT_DIR}/usr/share/doc/gpu-burn/copyright"

cat > "${OUT_DIR}/METADATA.json" <<EOF
{
  "package": "gpu-burn",
  "package_version": "0+git20240115+ds-2",
  "distribution": "Ubuntu 24.04 (noble) amd64",
  "cuda_line": "CUDA 12 runtime via Ubuntu noble package build",
  "homepage": "https://github.com/wilicc/gpu-burn",
  "download_url": "${PKG_URL}",
  "download_sha256": "${PKG_SHA256}",
  "installed_files": [
    "usr/sbin/gpu-burn",
    "usr/share/gpu-burn/compare.ptx",
    "usr/share/doc/gpu-burn/README.md",
    "usr/share/doc/gpu-burn/copyright"
  ]
}
EOF

echo "Vendored gpu-burn refreshed under ${OUT_DIR}"
