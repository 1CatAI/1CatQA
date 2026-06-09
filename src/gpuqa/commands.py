from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterable


@dataclass(slots=True)
class ExecResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(
    argv: Iterable[str],
    *,
    timeout: float | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> ExecResult:
    argv_list = [str(part) for part in argv]
    started = time.monotonic()
    completed = subprocess.run(
        argv_list,
        capture_output=True,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        timeout=timeout,
        check=False,
    )
    return ExecResult(
        argv=argv_list,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_sec=time.monotonic() - started,
    )


def which_first(candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value or value in {"N/A", "[N/A]", "[Not Supported]", "Not Supported"}:
        return None
    value = value.replace("%", "").replace("W", "").replace("MiB", "").strip()
    try:
        return float(value)
    except ValueError:
        return None


def parse_optional_int(raw: str | None) -> int | None:
    number = parse_optional_float(raw)
    if number is None:
        return None
    return int(number)


def clean_optional_text(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value or value in {"N/A", "[N/A]", "[Not Supported]", "Not Supported"}:
        return None
    return value
