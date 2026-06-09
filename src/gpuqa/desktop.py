from __future__ import annotations

from pathlib import Path
import os
import shlex
import subprocess
import sys

from gpuqa.commands import which_first


TERMINAL_PREFIXES = [
    ["x-terminal-emulator", "-e"],
    ["gnome-terminal", "--"],
    ["kgx", "--"],
    ["konsole", "-e"],
    ["mate-terminal", "--"],
    ["xfce4-terminal", "-e"],
    ["tilix", "-e"],
    ["xterm", "-e"],
    ["alacritty", "-e"],
]


def has_graphical_session() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def launch_in_terminal(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[str] | None, str | None]:
    if not sys.platform.startswith("linux"):
        return None, "当前平台不是 Linux，无法自动打开终端窗口"
    if not has_graphical_session():
        return None, "未检测到图形会话，无法自动打开终端窗口"

    command = shlex.join([str(part) for part in argv])
    for prefix in TERMINAL_PREFIXES:
        terminal_path = which_first([prefix[0]])
        if not terminal_path:
            continue
        try:
            process = subprocess.Popen(
                [terminal_path, *prefix[1:], "/bin/sh", "-lc", command],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(cwd) if cwd else None,
                env=env,
                start_new_session=True,
            )
        except OSError:
            continue
        return process, None

    return None, "未找到可用终端程序，无法自动打开文本窗口"


def launch_nvtop() -> tuple[subprocess.Popen[str] | None, str | None]:
    nvtop_path = which_first(["nvtop"])
    if not nvtop_path:
        return None, "未找到 nvtop，已跳过自动打开 nvtop"
    return launch_in_terminal([nvtop_path])


def open_path(path: Path) -> str | None:
    target = path.expanduser().resolve()
    if sys.platform.startswith("linux"):
        opener = which_first(["xdg-open"])
        if not opener:
            return "未找到 xdg-open"
        try:
            subprocess.Popen(
                [opener, str(target)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            return f"打开 {target} 失败: {exc}"
        return None

    if sys.platform == "win32":
        try:
            os.startfile(target)  # type: ignore[attr-defined]
        except OSError as exc:
            return f"打开 {target} 失败: {exc}"
        return None

    return f"当前平台 {sys.platform} 不支持自动打开目录"
