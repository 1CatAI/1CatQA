from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from gpuqa.desktop import launch_in_terminal
from gpuqa.single_instance import SingleInstanceLock
from gpuqa.tui import main as run_text_ui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="1Cat-V100-QA 文本界面")
    parser.add_argument("--terminal-session", action="store_true", help=argparse.SUPPRESS)
    return parser


def ensure_terminal_session(terminal_session: bool) -> int | None:
    if terminal_session or sys.stdout.isatty():
        return None

    root_dir = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root_dir / "src") + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    process, warning = launch_in_terminal(
        [sys.executable, "-m", "gpuqa.gui_entry", "--terminal-session"],
        cwd=root_dir,
        env=env,
    )
    if process is not None:
        return 0
    print(warning or "无法自动打开文本终端窗口。")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    terminal_result = ensure_terminal_session(args.terminal_session)
    if terminal_result is not None:
        return terminal_result

    lock = SingleInstanceLock(app_name="1cat-v100-qa-tui")
    if not lock.acquire():
        return 0

    try:
        return run_text_ui()
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
