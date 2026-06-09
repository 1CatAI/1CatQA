from __future__ import annotations

from dataclasses import dataclass
import argparse
import gzip
import io
import os
from pathlib import Path, PurePosixPath
import tarfile
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_NAME = "1cat-v100-qa"
DEFAULT_INSTALL_ROOT = PurePosixPath("/opt/1cat-v100-qa")
DEFAULT_DESKTOP_FILE = PurePosixPath("/usr/share/applications/1Cat-V100-QA.desktop")
DEFAULT_BIN_DIR = PurePosixPath("/usr/local/bin")
DEFAULT_SERVICE_DIR = PurePosixPath("/lib/systemd/system")
DEFAULT_EXCLUDES = {".git", ".codex-tmp", "remote_reports", "__pycache__", "build", "dist"}
DEFAULT_EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


@dataclass(slots=True)
class PackageMetadata:
    package: str
    version: str
    architecture: str
    maintainer: str
    description: str
    depends: list[str]


def parse_version() -> str:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for line in pyproject.splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("无法从 pyproject.toml 读取版本号")


def infer_file_mode(path: Path) -> int:
    if path.suffix == ".sh":
        return 0o755
    if path.name in {"gpu-burn", "1Cat-V100-QA"}:
        return 0o755
    return 0o644


def iter_repo_files() -> list[tuple[Path, PurePosixPath, int]]:
    includes = [
        ("src", 0o644),
        ("scripts", 0o644),
        ("cuda", 0o644),
        ("systemd", 0o644),
        ("README.md", 0o644),
        ("requirements.txt", 0o644),
        ("pyproject.toml", 0o644),
        ("LICENSE", 0o644),
    ]
    files: list[tuple[Path, PurePosixPath, int]] = []
    for relative, default_mode in includes:
        source = REPO_ROOT / relative
        arcbase = DEFAULT_INSTALL_ROOT / "app" / relative
        if source.is_dir():
            for path in sorted(source.rglob("*")):
                if any(part in DEFAULT_EXCLUDES for part in path.parts):
                    continue
                if path.suffix in DEFAULT_EXCLUDE_SUFFIXES:
                    continue
                rel = path.relative_to(source)
                arcname = arcbase / rel.as_posix()
                if path.is_file():
                    mode = infer_file_mode(path)
                    files.append((path, arcname, mode))
        else:
            files.append((source, arcbase, infer_file_mode(source)))
    return files


def build_wrapper(module: str) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        APP_ROOT="{DEFAULT_INSTALL_ROOT.as_posix()}/app"
        export PYTHONPATH="$APP_ROOT/src${{PYTHONPATH:+:$PYTHONPATH}}"
        exec python3 -m {module} "$@"
        """
    )


def build_desktop_entry() -> str:
    return textwrap.dedent(
        """\
        [Desktop Entry]
        Type=Application
        Version=1.0
        Name=1Cat-V100-QA
        Name[zh_CN]=1Cat-V100-QA
        Comment=V100-specialized GPU QA runner optimized for CUDA 12
        Comment[zh_CN]=面向 V100 的 CUDA 12 专项质检工具
        Exec=/usr/local/bin/1cat-v100-qa-gui
        Terminal=true
        Categories=Utility;System;
        StartupNotify=true
        Icon=utilities-terminal
        """
    )


def build_service_unit() -> str:
    return textwrap.dedent(
        f"""\
        [Unit]
        Description=1Cat-V100-QA automatic validation for V100 (CUDA 12 optimized)
        After=multi-user.target nvidia-persistenced.service
        Wants=nvidia-persistenced.service

        [Service]
        Type=oneshot
        User=root
        WorkingDirectory={DEFAULT_INSTALL_ROOT.as_posix()}/app
        Environment=PYTHONUNBUFFERED=1
        Environment=PYTHONPATH={DEFAULT_INSTALL_ROOT.as_posix()}/app/src
        ExecStart=/usr/bin/env python3 -m gpuqa run --output-dir /var/lib/gpuqa/latest --burn-seconds 600 --nvlink-seconds 10 --sample-interval 5.0 --wait-for-driver-seconds 180 --max-temperature-c 85.0
        StandardOutput=journal
        StandardError=journal

        [Install]
        WantedBy=multi-user.target
        """
    )


def build_control(metadata: PackageMetadata) -> str:
    depends = ", ".join(metadata.depends)
    return textwrap.dedent(
        f"""\
        Package: {metadata.package}
        Version: {metadata.version}
        Section: utils
        Priority: optional
        Architecture: {metadata.architecture}
        Maintainer: {metadata.maintainer}
        Depends: {depends}
        Homepage: https://github.com/1CatAI/1CatAI-GPUQA
        Description: {metadata.description}
         1Cat-V100-QA is a V100-specialized GPU QA suite optimized for CUDA 12.
         It installs the CLI, text-mode desktop launcher, vendored gpu-burn assets,
         report generation pipeline, and service template under /opt/1cat-v100-qa.
        """
    )


def build_postinst() -> str:
    return textwrap.dedent(
        """\
        #!/bin/sh
        set -e
        if command -v update-desktop-database >/dev/null 2>&1; then
          update-desktop-database /usr/share/applications >/dev/null 2>&1 || true
        fi
        if command -v systemctl >/dev/null 2>&1; then
          systemctl daemon-reload >/dev/null 2>&1 || true
        fi
        exit 0
        """
    )


def build_prerm() -> str:
    return textwrap.dedent(
        """\
        #!/bin/sh
        set -e
        if command -v systemctl >/dev/null 2>&1; then
          systemctl daemon-reload >/dev/null 2>&1 || true
        fi
        exit 0
        """
    )


def normalized_tarinfo(name: PurePosixPath, mode: int, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name.as_posix().lstrip("/"))
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    info.size = size
    return info


def add_bytes_to_tar(tf: tarfile.TarFile, dest: PurePosixPath, content: bytes, mode: int) -> None:
    info = normalized_tarinfo(dest, mode, len(content))
    tf.addfile(info, io.BytesIO(content))


def build_control_tar(metadata: PackageMetadata) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        add_bytes_to_tar(tf, PurePosixPath("control"), build_control(metadata).encode("utf-8"), 0o644)
        add_bytes_to_tar(tf, PurePosixPath("postinst"), build_postinst().encode("utf-8"), 0o755)
        add_bytes_to_tar(tf, PurePosixPath("prerm"), build_prerm().encode("utf-8"), 0o755)
    return buffer.getvalue()


def build_data_tar() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        for source, dest, mode in iter_repo_files():
            add_bytes_to_tar(tf, dest, source.read_bytes(), mode)

        wrappers = {
            DEFAULT_BIN_DIR / "1cat-v100-qa": build_wrapper("gpuqa"),
            DEFAULT_BIN_DIR / "1cat-v100-qa-gui": build_wrapper("gpuqa.gui_entry"),
            DEFAULT_BIN_DIR / "gpuqa": build_wrapper("gpuqa"),
        }
        for dest, script in wrappers.items():
            add_bytes_to_tar(tf, dest, script.encode("utf-8"), 0o755)

        add_bytes_to_tar(tf, DEFAULT_DESKTOP_FILE, build_desktop_entry().encode("utf-8"), 0o644)
        add_bytes_to_tar(tf, DEFAULT_SERVICE_DIR / "gpuqa.service", build_service_unit().encode("utf-8"), 0o644)
    return buffer.getvalue()


def write_ar_member(fp: io.BufferedWriter, name: str, content: bytes, mode: int = 0o100644) -> None:
    if len(name) > 15:
        raise ValueError(f"ar 成员名过长: {name}")
    header = (
        f"{name}/".ljust(16)
        + f"{0}".rjust(12)
        + f"{0}".rjust(6)
        + f"{0}".rjust(6)
        + f"{oct(mode)[2:]}".rjust(8)
        + f"{len(content)}".rjust(10)
        + "`\n"
    )
    fp.write(header.encode("ascii"))
    fp.write(content)
    if len(content) % 2 == 1:
        fp.write(b"\n")


def build_deb(output_path: Path) -> Path:
    metadata = PackageMetadata(
        package=DEFAULT_PACKAGE_NAME,
        version=parse_version(),
        architecture="all",
        maintainer="1CatAI",
        description="1Cat-V100-QA Debian package",
        depends=["python3 (>= 3.11)", "xdg-utils"],
    )
    control_tar = build_control_tar(metadata)
    data_tar = build_data_tar()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fp:
        fp.write(b"!<arch>\n")
        write_ar_member(fp, "debian-binary", b"2.0\n")
        write_ar_member(fp, "control.tar.gz", control_tar)
        write_ar_member(fp, "data.tar.gz", data_tar)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Debian package for 1Cat-V100-QA.")
    parser.add_argument("--output", type=Path, default=Path.home() / "Desktop" / f"1cat-v100-qa_{parse_version()}_all.deb")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = build_deb(args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
