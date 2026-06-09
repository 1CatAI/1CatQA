from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import io
import os
from pathlib import Path
import platform
import re
import signal
import shlex
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Callable, TypeVar

from gpuqa.commands import (
    ExecResult,
    clean_optional_text,
    parse_optional_float,
    parse_optional_int,
    run_command,
    which_first,
)
from gpuqa.models import GPUIdentity, GPUResult, MetricSample, RunReport


STATIC_FIELDS = ["index", "name", "uuid", "serial", "pci.bus_id", "driver_version"]
METRIC_FIELDS = [
    "index",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "utilization.gpu",
    "memory.used",
    "memory.total",
]
ECC_FIELD_GROUPS = [
    ("ecc.errors.corrected.volatile.total", "ecc.errors.uncorrected.volatile.total"),
    ("ecc.errors.corrected.aggregate.total", "ecc.errors.uncorrected.aggregate.total"),
]
ECC_MODE_FIELDS = ["index", "ecc.mode.current", "ecc.mode.pending"]
DEFAULT_P2P_CANDIDATES = [
    str(Path.home() / ".cache" / "1cat-v100-qa" / "bin" / "p2p_bandwidth_matrix"),
    "p2p_bandwidth_matrix",
    "p2pBandwidthLatencyTest",
    "/usr/local/cuda/extras/demo_suite/p2pBandwidthLatencyTest",
    "/usr/local/cuda/samples/1_Utilities/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest",
]
DEFAULT_GPU_BURN_MEMORY_PERCENT = "100%"
DEFAULT_NVLINK_TEST_SECONDS = 10
DEFAULT_NVLINK_EXPECTED_LINKS = 6
VENDORED_GPU_BURN_ROOT = (
    Path(__file__).resolve().parent / "vendor" / "gpu-burn" / "ubuntu24.04-cuda12" / "usr"
)
USER_GPU_BURN_ROOT = Path.home() / ".cache" / "1cat-v100-qa" / "gpu-burn"
P2P_CACHE_BINARY = Path.home() / ".cache" / "1cat-v100-qa" / "bin" / "p2p_bandwidth_matrix"
ProgressCallback = Callable[[str, str, dict[str, object] | None], None]
T = TypeVar("T")


class RunCancelled(RuntimeError):
    """Raised when the user stops an in-flight validation run."""


@dataclass(slots=True)
class BurnRunResult:
    samples: list[MetricSample] = field(default_factory=list)
    scores: dict[int, float] = field(default_factory=dict)
    issue_counts: dict[int, dict[str, int]] = field(default_factory=dict)
    output: str | None = None
    warning: str | None = None
    exit_ok: bool | None = None
    stop_reasons: dict[int, str] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
    **payload: object,
) -> None:
    if callback is None:
        return
    try:
        callback(stage, message, payload or None)
    except Exception:
        return


def raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RunCancelled("用户已手动停止检测")


def sleep_with_cancel(duration_sec: float, cancel_event: threading.Event | None) -> None:
    deadline = time.monotonic() + max(duration_sec, 0.0)
    while True:
        raise_if_cancelled(cancel_event)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.2, remaining))


def open_process(
    argv: list[str],
    *,
    stdout: int | None = None,
    stderr: int | None = None,
    text: bool = True,
    bufsize: int = -1,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    kwargs: dict[str, object] = {
        "stdout": stdout,
        "stderr": stderr,
        "text": text,
        "bufsize": bufsize,
        "cwd": str(cwd) if cwd else None,
        "env": env,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(argv, **kwargs)


def kill_processes_by_cmdline(
    match_tokens: list[str],
    *,
    exclude_pids: set[int] | None = None,
    grace_period_sec: float = 2.0,
) -> None:
    if os.name == "nt" or not match_tokens:
        return
    excluded = {os.getpid()}
    if exclude_pids:
        excluded.update(exclude_pids)

    matched_pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
        except OSError:
            continue
        if not cmdline or not all(token in cmdline for token in match_tokens):
            continue
        matched_pids.append(pid)

    for pid in matched_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue

    deadline = time.monotonic() + grace_period_sec
    while time.monotonic() < deadline:
        alive = []
        for pid in matched_pids:
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            alive.append(pid)
        if not alive:
            return
        time.sleep(0.1)

    for pid in matched_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue


def stop_process(
    process: subprocess.Popen[str],
    *,
    grace_period_sec: float = 5.0,
    match_tokens: list[str] | None = None,
) -> None:
    if process.poll() is not None:
        if match_tokens:
            kill_processes_by_cmdline(match_tokens, exclude_pids={process.pid})
        return
    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        except (AttributeError, OSError, ValueError):
            try:
                process.terminate()
            except OSError:
                return
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except OSError:
            try:
                process.terminate()
            except OSError:
                return
    try:
        process.wait(timeout=grace_period_sec)
        if match_tokens:
            kill_processes_by_cmdline(match_tokens, exclude_pids={process.pid})
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "nt":
        try:
            process.kill()
        except OSError:
            return
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except OSError:
            try:
                process.kill()
            except OSError:
                return
    try:
        process.wait(timeout=grace_period_sec)
    except subprocess.TimeoutExpired:
        pass
    if match_tokens:
        kill_processes_by_cmdline(match_tokens, exclude_pids={process.pid})


def run_command_with_cancel(
    argv: list[str],
    *,
    timeout: float | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    cancel_event: threading.Event | None = None,
) -> ExecResult:
    argv_list = [str(part) for part in argv]
    started = time.monotonic()
    process = open_process(
        argv_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
    )
    stdout = ""
    stderr = ""
    while True:
        raise_if_cancelled(cancel_event)
        try:
            stdout, stderr = process.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if timeout is not None and time.monotonic() - started > timeout:
                stop_process(process, match_tokens=argv_list[:1])
                raise subprocess.TimeoutExpired(argv_list, timeout)
            continue
        except Exception:
            stop_process(process, match_tokens=argv_list[:1])
            raise
    return ExecResult(
        argv=argv_list,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_sec=time.monotonic() - started,
    )


def resolve_resource_root() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parents[2]


def resolve_executable(candidate: str) -> str | None:
    path = Path(candidate)
    if path.is_absolute():
        try:
            if path.is_file() and path.stat().st_size > 0:
                return str(path)
        except OSError:
            return None
        return None

    resolved = which_first([candidate])
    if not resolved:
        return None
    try:
        if Path(resolved).stat().st_size > 0:
            return resolved
    except OSError:
        return None
    return None


def resolve_first_executable(candidates: list[str]) -> str | None:
    for candidate in candidates:
        resolved = resolve_executable(candidate)
        if resolved:
            return resolved
    return None


def ensure_local_executable(path: Path) -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if mode & 0o111:
        return
    try:
        path.chmod(mode | 0o755)
    except OSError:
        return


def build_gpu_burn_candidates() -> list[tuple[list[str], list[str], bool]]:
    candidates: list[tuple[list[str], list[str], bool]] = []
    if sys.platform.startswith("linux"):
        user_binary = USER_GPU_BURN_ROOT / "gpu_burn"
        user_compare = USER_GPU_BURN_ROOT / "compare.ptx"
        candidates.append(
            (
                [
                    str(user_binary),
                    "-c",
                    str(user_compare),
                    "-m",
                    DEFAULT_GPU_BURN_MEMORY_PERCENT,
                ],
                [str(user_compare)],
                True,
            )
        )
        vendored_binary = VENDORED_GPU_BURN_ROOT / "sbin" / "gpu-burn"
        vendored_compare = VENDORED_GPU_BURN_ROOT / "share" / "gpu-burn" / "compare.ptx"
        candidates.append(
            (
                [
                    str(vendored_binary),
                    "-c",
                    str(vendored_compare),
                    "-m",
                    DEFAULT_GPU_BURN_MEMORY_PERCENT,
                ],
                [str(vendored_compare)],
                True,
            )
        )
        candidates.append(
            (
                ["/usr/sbin/gpu-burn", "-c", "/usr/share/gpu-burn/compare.ptx", "-m", DEFAULT_GPU_BURN_MEMORY_PERCENT],
                ["/usr/share/gpu-burn/compare.ptx"],
                False,
            )
        )
        candidates.append(
            (
                [
                    "/opt/gpu-burn/bin/gpu_burn",
                    "-c",
                    "/opt/gpu-burn/bin/compare.ptx",
                    "-m",
                    DEFAULT_GPU_BURN_MEMORY_PERCENT,
                ],
                ["/opt/gpu-burn/bin/compare.ptx"],
                False,
            )
        )
    candidates.append((["gpu_burn", "-m", DEFAULT_GPU_BURN_MEMORY_PERCENT], [], False))
    candidates.append((["gpu-burn", "-m", DEFAULT_GPU_BURN_MEMORY_PERCENT], [], False))
    return candidates


def resolve_default_gpu_burn_command() -> list[str]:
    for argv, required_paths, should_ensure_executable in build_gpu_burn_candidates():
        resolved = resolve_executable(argv[0])
        if not resolved:
            continue
        if any(not Path(required).exists() for required in required_paths):
            continue
        if should_ensure_executable:
            ensure_local_executable(Path(resolved))
        return [resolved, *argv[1:]]
    return []


def resolve_bundled_p2p_source() -> Path | None:
    candidates = [
        resolve_resource_root() / "cuda" / "p2p_bandwidth_matrix.cu",
        Path(__file__).resolve().parents[2] / "cuda" / "p2p_bandwidth_matrix.cu",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def ensure_bundled_p2p_binary() -> tuple[str | None, str | None]:
    if not sys.platform.startswith("linux"):
        return None, None

    source = resolve_bundled_p2p_source()
    if source is None:
        return None, None

    if P2P_CACHE_BINARY.is_file() and P2P_CACHE_BINARY.stat().st_size > 0:
        try:
            if P2P_CACHE_BINARY.stat().st_mtime >= source.stat().st_mtime:
                return str(P2P_CACHE_BINARY), None
        except OSError:
            pass

    nvcc = resolve_first_executable(["nvcc", "/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"])
    host_cxx = resolve_first_executable(["g++-12", "g++", "c++"])
    if not nvcc or not host_cxx:
        return None, "检测到内置 p2p_bandwidth_matrix 源码，但未找到 nvcc 或 g++"

    try:
        P2P_CACHE_BINARY.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return None, f"准备内置 p2p 缓存目录失败: {exc}"

    result = run_command(
        [
            nvcc,
            "-ccbin",
            host_cxx,
            "-O3",
            "-std=c++14",
            str(source),
            "-o",
            str(P2P_CACHE_BINARY),
        ],
        timeout=900,
    )
    if result.ok and P2P_CACHE_BINARY.is_file() and P2P_CACHE_BINARY.stat().st_size > 0:
        return str(P2P_CACHE_BINARY), None

    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return None, f"编译内置 p2p_bandwidth_matrix 失败: {(output or '未知错误').strip()}"


def mean_or_none(values: list[float | None]) -> float | None:
    real_values = [value for value in values if value is not None]
    if not real_values:
        return None
    return statistics.fmean(real_values)


def max_or_none(values: list[float | None]) -> float | None:
    real_values = [value for value in values if value is not None]
    if not real_values:
        return None
    return max(real_values)


def nvidia_smi(argv: list[str], timeout: float = 30) -> ExecResult:
    return run_command(["nvidia-smi", *argv], timeout=timeout)


def wait_for_driver(
    timeout_sec: int,
    poll_interval: float = 5.0,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_sec
    last_error = "nvidia-smi did not become ready"
    emit_progress(progress_callback, "wait-driver", "正在等待 NVIDIA 驱动就绪")
    while time.monotonic() <= deadline:
        raise_if_cancelled(cancel_event)
        result = nvidia_smi(["-L"], timeout=15)
        if result.ok:
            emit_progress(progress_callback, "driver-ready", "NVIDIA 驱动已就绪")
            return True, "driver ready"
        last_error = (result.stderr or result.stdout or last_error).strip()
        sleep_with_cancel(poll_interval, cancel_event)
    emit_progress(progress_callback, "driver-timeout", last_error)
    return False, last_error


def parse_csv_query(fields: list[str], text: str) -> list[dict[str, str | None]]:
    reader = csv.reader(io.StringIO(text.strip()))
    rows: list[dict[str, str | None]] = []
    for raw_row in reader:
        if not raw_row:
            continue
        row = {}
        for index, field_name in enumerate(fields):
            value = raw_row[index].strip() if index < len(raw_row) else ""
            row[field_name] = clean_optional_text(value)
        rows.append(row)
    return rows


def query_gpu_fields(fields: list[str]) -> tuple[list[dict[str, str | None]], str | None]:
    result = nvidia_smi([f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if not result.ok:
        message = (result.stderr or result.stdout).strip()
        return [], message or "nvidia-smi query failed"
    return parse_csv_query(fields, result.stdout), None


def discover_gpus() -> tuple[list[GPUIdentity], list[str]]:
    warnings: list[str] = []
    rows, error = query_gpu_fields(STATIC_FIELDS)
    if error:
        warnings.append(error)
        return [], warnings

    gpus: list[GPUIdentity] = []
    for row in rows:
        index = parse_optional_int(row.get("index"))
        if index is None:
            continue
        gpus.append(
            GPUIdentity(
                index=index,
                name=row.get("name"),
                uuid=row.get("uuid"),
                serial=row.get("serial"),
                pci_bus_id=row.get("pci.bus_id"),
                driver_version=row.get("driver_version"),
            )
        )
    return gpus, warnings


def collect_metrics() -> tuple[dict[int, MetricSample], str | None]:
    rows, error = query_gpu_fields(METRIC_FIELDS)
    if error:
        return {}, error

    metrics: dict[int, MetricSample] = {}
    timestamp = utc_now()
    for row in rows:
        index = parse_optional_int(row.get("index"))
        if index is None:
            continue
        metrics[index] = MetricSample(
            timestamp=timestamp,
            gpu_index=index,
            temperature_c=parse_optional_float(row.get("temperature.gpu")),
            power_w=parse_optional_float(row.get("power.draw")),
            sm_clock_mhz=parse_optional_float(row.get("clocks.sm")),
            mem_clock_mhz=parse_optional_float(row.get("clocks.mem")),
            utilization_gpu_pct=parse_optional_float(row.get("utilization.gpu")),
            memory_used_mib=parse_optional_float(row.get("memory.used")),
            memory_total_mib=parse_optional_float(row.get("memory.total")),
        )
    return metrics, None


def collect_ecc_totals() -> tuple[dict[int, dict[str, int | None]], str | None]:
    merged: dict[int, dict[str, int | None]] = {}
    errors: list[str] = []
    for corrected_field, uncorrected_field in ECC_FIELD_GROUPS:
        rows, error = query_gpu_fields(["index", corrected_field, uncorrected_field])
        if error:
            errors.append(error)
            continue
        for row in rows:
            index = parse_optional_int(row.get("index"))
            if index is None:
                continue
            bucket = merged.setdefault(index, {})
            if bucket.get("corrected") is None:
                bucket["corrected"] = parse_optional_int(row.get(corrected_field))
            if bucket.get("uncorrected") is None:
                bucket["uncorrected"] = parse_optional_int(row.get(uncorrected_field))
        if merged:
            return merged, None
    if not merged:
        return {}, "; ".join(error for error in errors if error) or "ECC counters unavailable"
    return merged, None


def normalize_ecc_mode(raw: str | None) -> str | None:
    value = clean_optional_text(raw)
    if value is None:
        return None
    lowered = value.lower()
    if lowered.startswith("enabled"):
        return "Enabled"
    if lowered.startswith("disabled"):
        return "Disabled"
    return value


def collect_ecc_modes() -> tuple[dict[int, dict[str, str | None]], str | None]:
    rows, error = query_gpu_fields(ECC_MODE_FIELDS)
    if error:
        return {}, error or "ECC mode unavailable"

    modes: dict[int, dict[str, str | None]] = {}
    for row in rows:
        index = parse_optional_int(row.get("index"))
        if index is None:
            continue
        modes[index] = {
            "current": normalize_ecc_mode(row.get("ecc.mode.current")),
            "pending": normalize_ecc_mode(row.get("ecc.mode.pending")),
        }
    return modes, None


def parse_nvlink_error_output(text: str) -> dict[int, int]:
    totals: dict[int, int] = {}
    current_gpu: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.search(r"GPU\s+(\d+)", line, re.IGNORECASE)
        if match:
            current_gpu = int(match.group(1))
            totals.setdefault(current_gpu, 0)
            continue
        if current_gpu is None:
            continue
        if not re.search(r"error|replay|fatal|recovery", line, re.IGNORECASE):
            continue
        values = [int(number) for number in re.findall(r"\b\d+\b", line)]
        if re.search(r"^Link\s+\d+", line, re.IGNORECASE) and values:
            values = values[1:]
        totals[current_gpu] += sum(values)
    return totals


def collect_nvlink_errors() -> tuple[dict[int, int], str | None]:
    candidates = [
        ["nvlink", "-e"],
        ["nvlink", "--errors"],
    ]
    messages: list[str] = []
    for args in candidates:
        result = nvidia_smi(args, timeout=30)
        if not result.ok:
            messages.append((result.stderr or result.stdout).strip())
            continue
        parsed = parse_nvlink_error_output(result.stdout)
        if parsed:
            return parsed, None
        messages.append("NVLink error output was not parseable")
    return {}, "; ".join(message for message in messages if message) or "NVLink errors unavailable"


def parse_nvlink_crc_error_output(text: str) -> dict[int, int]:
    totals: dict[int, int] = {}
    current_gpu: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.search(r"GPU\s+(\d+)", line, re.IGNORECASE)
        if match:
            current_gpu = int(match.group(1))
            totals.setdefault(current_gpu, 0)
            continue
        if current_gpu is None or "crc" not in line.lower():
            continue
        crc_values = [int(number) for number in re.findall(r"crc[^0-9-]*([0-9]+)", line, re.IGNORECASE)]
        if crc_values:
            totals[current_gpu] += sum(crc_values)
            continue
        values = [int(number) for number in re.findall(r"\b\d+\b", line)]
        if re.search(r"^Link\s+\d+", line, re.IGNORECASE) and values:
            values = values[1:]
        totals[current_gpu] += sum(values)
    return totals


def collect_nvlink_crc_errors() -> tuple[dict[int, int], str | None]:
    candidates = [
        ["nvlink", "-ec"],
        ["nvlink", "--crcerrorcounters"],
    ]
    messages: list[str] = []
    for args in candidates:
        result = nvidia_smi(args, timeout=30)
        if not result.ok:
            messages.append((result.stderr or result.stdout).strip())
            continue
        parsed = parse_nvlink_crc_error_output(result.stdout)
        if parsed:
            return parsed, None
        messages.append("NVLink CRC error output was not parseable")
    return {}, "; ".join(message for message in messages if message) or "NVLink CRC errors unavailable"


def parse_nvlink_status_output(text: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    current_gpu: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.search(r"GPU\s+(\d+)", line, re.IGNORECASE)
        if match:
            current_gpu = int(match.group(1))
            counts.setdefault(current_gpu, 0)
            continue
        if current_gpu is None:
            continue
        link_match = re.match(r"Link\s+\d+:\s*(.+)$", line, re.IGNORECASE)
        if not link_match:
            continue
        status = link_match.group(1).strip().lower()
        if any(token in status for token in ("inactive", "disabled", "not connected", "n/a")):
            continue
        counts[current_gpu] += 1
    return counts


def collect_nvlink_link_counts() -> tuple[dict[int, int], str | None]:
    candidates = [
        ["nvlink", "-s"],
        ["nvlink", "--status"],
    ]
    messages: list[str] = []
    for args in candidates:
        result = nvidia_smi(args, timeout=30)
        if not result.ok:
            messages.append((result.stderr or result.stdout).strip())
            continue
        parsed = parse_nvlink_status_output(result.stdout)
        if parsed:
            return parsed, None
        messages.append("NVLink status output was not parseable")
    return {}, "; ".join(message for message in messages if message) or "NVLink status unavailable"


def filter_gpu_identities(
    gpus: list[GPUIdentity],
    target_gpu_index: int | None,
) -> list[GPUIdentity]:
    if target_gpu_index is None:
        return gpus
    return [gpu for gpu in gpus if gpu.index == target_gpu_index]


def filter_metric_map(
    metrics: dict[int, MetricSample],
    target_gpu_index: int | None,
) -> dict[int, MetricSample]:
    if target_gpu_index is None:
        return metrics
    if target_gpu_index not in metrics:
        return {}
    return {target_gpu_index: metrics[target_gpu_index]}


def filter_metric_samples(
    samples: list[MetricSample],
    target_gpu_index: int | None,
) -> list[MetricSample]:
    if target_gpu_index is None:
        return samples
    return [sample for sample in samples if sample.gpu_index == target_gpu_index]


def filter_index_map(
    values: dict[int, T],
    target_gpu_index: int | None,
) -> dict[int, T]:
    if target_gpu_index is None:
        return values
    if target_gpu_index not in values:
        return {}
    return {target_gpu_index: values[target_gpu_index]}


def remap_gpu_index_map(
    values: dict[int, T],
    visible_to_actual: dict[int, int],
) -> dict[int, T]:
    if not visible_to_actual:
        return values
    remapped: dict[int, T] = {}
    for visible_index, value in values.items():
        actual_index = visible_to_actual.get(visible_index, visible_index)
        remapped[actual_index] = value
    return remapped


def build_cuda_visible_devices_env(target_gpu_index: int | None) -> dict[str, str] | None:
    if target_gpu_index is None:
        return None
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(target_gpu_index)
    return env


def parse_p2p_bandwidth_matrix(text: str) -> dict[int, dict[int, float]]:
    matrices: list[dict[int, dict[int, float]]] = []
    current_rows: dict[int, dict[int, float]] = {}
    header_order: list[int] = []
    capture = False

    def finalize_current() -> None:
        if current_rows:
            matrices.append({gpu_index: dict(values) for gpu_index, values in current_rows.items()})

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if capture and current_rows:
                finalize_current()
                current_rows = {}
                header_order = []
                capture = False
            continue
        if "Bandwidth Matrix" in line:
            if capture and current_rows:
                finalize_current()
            capture = True
            current_rows = {}
            header_order = []
            continue
        if not capture:
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0].upper().startswith("D\\D") or parts[0].upper().startswith("D/D"):
            header_order = [int(token) for token in parts[1:] if token.isdigit()]
            continue
        if not parts[0].isdigit():
            continue
        gpu_index = int(parts[0])
        values: list[float] = []
        for token in parts[1:]:
            try:
                values.append(float(token))
            except ValueError:
                pass
        if not values:
            continue
        if header_order:
            limit = min(len(header_order), len(values))
            current_rows[gpu_index] = {header_order[position]: values[position] for position in range(limit)}
        else:
            current_rows[gpu_index] = {position: value for position, value in enumerate(values)}

    if capture and current_rows:
        finalize_current()

    if not matrices:
        return {}
    return max(matrices, key=lambda item: (len(item), sum(len(values) for values in item.values())))


def parse_p2p_bandwidth(text: str) -> dict[int, float]:
    matrix = parse_p2p_bandwidth_matrix(text)
    per_gpu: dict[int, float] = {}
    for gpu_index, peers_map in matrix.items():
        peers = [value for peer_index, value in peers_map.items() if peer_index != gpu_index and value > 0]
        if peers:
            per_gpu[gpu_index] = max(peers)
    return per_gpu


def collect_nvlink_bandwidth(
    command_override: str | None,
    nvlink_seconds: int = DEFAULT_NVLINK_TEST_SECONDS,
    target_gpu_index: int | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[dict[int, float], dict[int, dict[int, float]], str | None, str | None]:
    warning_parts: list[str] = []
    env = build_cuda_visible_devices_env(target_gpu_index)
    visible_to_actual = {0: target_gpu_index} if target_gpu_index is not None else {}
    if command_override:
        argv = shlex.split(command_override)
    else:
        resolved, compile_warning = ensure_bundled_p2p_binary()
        if compile_warning:
            warning_parts.append(compile_warning)
        if not resolved:
            resolved = resolve_first_executable(DEFAULT_P2P_CANDIDATES)
        argv = [resolved] if resolved else []
    if not argv:
        warning_parts.append("未找到 NVLink 带宽测试程序")
        return {}, {}, None, "; ".join(warning_parts)

    command_name = Path(argv[0]).name
    has_nvlink_seconds = any(
        token in {"--seconds", "-s"} or token.startswith("--seconds=")
        for token in argv[1:]
    )
    if command_name == "p2p_bandwidth_matrix" and not has_nvlink_seconds:
        argv = [*argv, "--seconds", str(nvlink_seconds)]

    try:
        result = run_command_with_cancel(argv, timeout=300, cancel_event=cancel_event, env=env)
    except subprocess.TimeoutExpired:
        warning_parts.append(f"{argv[0]} 执行超时")
        return {}, {}, None, "; ".join(warning_parts)
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    if not result.ok:
        warning_parts.append(f"{argv[0]} 退出码为 {result.returncode}")
        return {}, {}, output.strip() or None, "; ".join(warning_parts)
    parsed_matrix = parse_p2p_bandwidth_matrix(output)
    remapped_matrix = {
        visible_to_actual.get(gpu_index, gpu_index): {
            visible_to_actual.get(peer_index, peer_index): bandwidth
            for peer_index, bandwidth in peers_map.items()
        }
        for gpu_index, peers_map in parsed_matrix.items()
    } if visible_to_actual else parsed_matrix
    parsed = remap_gpu_index_map(parse_p2p_bandwidth(output), visible_to_actual)
    if not parsed:
        warning_parts.append("无法解析 NVLink 带宽测试输出")
        return {}, remapped_matrix, output.strip() or None, "; ".join(warning_parts)
    return parsed, remapped_matrix, output.strip() or None, "; ".join(warning_parts) or None


def parse_gpu_burn_output(text: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    gpu_order: list[int] = []
    patterns = [
        re.compile(r"GPU\s+(\d+).*?([0-9]+(?:\.[0-9]+)?)\s+GF/?s", re.IGNORECASE),
        re.compile(r"GPU\s+(\d+).*?([0-9]+(?:\.[0-9]+)?)\s+Gflop", re.IGNORECASE),
    ]
    progress_pattern = re.compile(r"\(([0-9]+(?:\.[0-9]+)?)\s+Gflop(?:/s)?\)", re.IGNORECASE)
    clean_progress_scores: dict[int, float] = {}
    warning_progress_scores: dict[int, float] = {}
    for line in text.splitlines():
        header_match = re.match(r"\s*GPU\s+(\d+):", line, re.IGNORECASE)
        if header_match:
            gpu_index = int(header_match.group(1))
            if gpu_index not in gpu_order:
                gpu_order.append(gpu_index)
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                scores[int(match.group(1))] = float(match.group(2))
                break
        progress_scores = [float(value) for value in progress_pattern.findall(line)]
        if progress_scores:
            target = warning_progress_scores if "warning" in line.lower() else clean_progress_scores
            for position, value in enumerate(progress_scores):
                gpu_index = gpu_order[position] if position < len(gpu_order) else position
                target[gpu_index] = value
    if scores:
        return scores
    if clean_progress_scores:
        return clean_progress_scores
    if warning_progress_scores:
        return warning_progress_scores
    return scores


def parse_gpu_burn_issue_counts(text: str) -> dict[int, dict[str, int]]:
    issue_counts: dict[int, dict[str, int]] = {}
    gpu_order: list[int] = []
    error_segment_pattern = re.compile(r"errors?\s*:\s*(.+?)(?:temps?:|$)", re.IGNORECASE)
    for line in text.splitlines():
        header_match = re.match(r"\s*GPU\s+(\d+):", line, re.IGNORECASE)
        if header_match:
            gpu_index = int(header_match.group(1))
            if gpu_index not in gpu_order:
                gpu_order.append(gpu_index)
        error_match = error_segment_pattern.search(line)
        if not error_match:
            continue
        counts = [int(value) for value in re.findall(r"\b\d+\b", error_match.group(1))]
        if not counts:
            continue
        warning_line = "warning" in line.lower()
        for position, count in enumerate(counts):
            gpu_index = gpu_order[position] if position < len(gpu_order) else position
            bucket = issue_counts.setdefault(gpu_index, {"warning_count": 0, "error_count": 0})
            bucket["error_count"] = max(bucket["error_count"], count)
            if warning_line:
                bucket["warning_count"] = max(bucket["warning_count"], count)
    return issue_counts


def detect_overtemperature(
    samples: list[MetricSample],
    max_temperature_c: float | None,
) -> dict[int, str]:
    if max_temperature_c is None:
        return {}
    reasons: dict[int, str] = {}
    for item in samples:
        if item.temperature_c is None or item.temperature_c <= max_temperature_c:
            continue
        reasons[item.gpu_index] = (
            f"GPU {item.gpu_index} 温度 {item.temperature_c:.2f}C 超过阈值 "
            f"{max_temperature_c:.2f}C，已自动停止测试"
        )
    return reasons


def start_gpu_burn(
    burn_seconds: int,
    sample_interval: float,
    max_temperature_c: float | None,
    command_override: str | None,
    target_gpu_index: int | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> BurnRunResult:
    env = build_cuda_visible_devices_env(target_gpu_index)
    visible_to_actual = {0: target_gpu_index} if target_gpu_index is not None else {}
    if command_override:
        base_argv = shlex.split(command_override)
    else:
        base_argv = resolve_default_gpu_burn_command()
    if not base_argv:
        return BurnRunResult(warning="未找到 gpu-burn 命令")

    argv = [*base_argv, str(burn_seconds)]
    emit_progress(
        progress_callback,
        "gpu-burn-start",
        f"开始执行 gpu-burn，时长 {burn_seconds} 秒",
        command=argv,
    )
    process = open_process(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    output_lines: list[str] = []

    def reader() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line)
            stripped = line.strip()
            if stripped:
                emit_progress(
                    progress_callback,
                    "gpu-burn-output",
                    "收到 GPU-Burn 实时输出",
                    line=stripped,
                )

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    collected: list[MetricSample] = []
    next_sample_at = time.monotonic()
    stop_reasons: dict[int, str] = {}
    try:
        while process.poll() is None:
            raise_if_cancelled(cancel_event)
            now = time.monotonic()
            if now >= next_sample_at:
                metrics, _ = collect_metrics()
                current_samples = list(filter_metric_map(metrics, target_gpu_index).values())
                collected.extend(current_samples)
                emit_progress(
                    progress_callback,
                    "gpu-burn-sample",
                    "已采集实时 GPU 指标",
                    samples=[asdict(item) for item in current_samples],
                )
                stop_reasons = detect_overtemperature(current_samples, max_temperature_c)
                if stop_reasons:
                    emit_progress(
                        progress_callback,
                        "gpu-burn-overheat",
                        "检测到 GPU 超过温度阈值，已自动停止压测",
                        reasons=stop_reasons,
                    )
                    stop_process(process, match_tokens=argv[:1])
                    break
                next_sample_at = now + sample_interval
            sleep_with_cancel(min(sample_interval, 1.0), cancel_event)
    except RunCancelled:
        stop_process(process, match_tokens=argv[:1])
        reader_thread.join(timeout=5)
        emit_progress(progress_callback, "gpu-burn-stop", "已停止 gpu-burn")
        raise

    reader_thread.join(timeout=5)
    output = "".join(output_lines).strip()
    scores = remap_gpu_index_map(parse_gpu_burn_output(output), visible_to_actual)
    issue_counts = remap_gpu_index_map(parse_gpu_burn_issue_counts(output), visible_to_actual)
    burn_issue_failed = any(
        bucket.get("warning_count", 0) > 0 or bucket.get("error_count", 0) > 0
        for bucket in issue_counts.values()
    )
    burn_exit_ok = process.returncode == 0 and not stop_reasons
    if not collected:
        metrics, _ = collect_metrics()
        current_samples = list(filter_metric_map(metrics, target_gpu_index).values())
        collected.extend(current_samples)
        emit_progress(
            progress_callback,
            "gpu-burn-sample",
            "已采集最终 GPU 指标",
            samples=[asdict(item) for item in current_samples],
        )
    emit_progress(
        progress_callback,
        "gpu-burn-finish",
        "gpu-burn 已结束",
        burn_ok=burn_exit_ok and not burn_issue_failed,
        burn_exit_ok=burn_exit_ok,
        scores=scores,
        issue_counts=issue_counts,
        stop_reasons=stop_reasons,
    )
    return BurnRunResult(
        samples=collected,
        scores=scores,
        issue_counts=issue_counts,
        output=output or None,
        exit_ok=burn_exit_ok,
        stop_reasons=stop_reasons,
    )


def collect_environment() -> dict[str, str | list[str] | None]:
    gpu_burn_argv = resolve_default_gpu_burn_command()
    p2p_path, _ = ensure_bundled_p2p_binary()
    if not p2p_path:
        p2p_path = resolve_first_executable(DEFAULT_P2P_CANDIDATES)
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "python_executable": None if getattr(sys, "frozen", False) else sys.executable,
        "frozen_build": "yes" if getattr(sys, "frozen", False) else "no",
        "nvidia_smi_path": which_first(["nvidia-smi"]),
        "gpu_burn_path": gpu_burn_argv[0] if gpu_burn_argv else None,
        "gpu_burn_command": gpu_burn_argv or None,
        "p2p_bandwidth_path": p2p_path,
    }


def build_gpu_results(
    gpus: list[GPUIdentity],
    samples: list[MetricSample],
    baseline_ecc_modes: dict[int, dict[str, str | None]],
    final_ecc_modes: dict[int, dict[str, str | None]],
    baseline_ecc: dict[int, dict[str, int | None]],
    final_ecc: dict[int, dict[str, int | None]],
    baseline_nvlink_links: dict[int, int],
    final_nvlink_links: dict[int, int],
    baseline_nvlink: dict[int, int],
    final_nvlink: dict[int, int],
    baseline_nvlink_crc: dict[int, int],
    final_nvlink_crc: dict[int, int],
    p2p_bandwidth: dict[int, float],
    burn_scores: dict[int, float],
    burn_issue_counts: dict[int, dict[str, int]],
    burn_exit_ok: bool | None,
    burn_stop_reasons: dict[int, str] | None,
    max_temperature_c: float,
) -> list[GPUResult]:
    grouped: dict[int, list[MetricSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.gpu_index].append(sample)

    results: list[GPUResult] = []
    for gpu in gpus:
        gpu_samples = grouped.get(gpu.index, [])
        avg_temp = mean_or_none([item.temperature_c for item in gpu_samples])
        max_temp = max_or_none([item.temperature_c for item in gpu_samples])
        avg_power = mean_or_none([item.power_w for item in gpu_samples])
        max_power = max_or_none([item.power_w for item in gpu_samples])
        avg_sm_clock = mean_or_none([item.sm_clock_mhz for item in gpu_samples])
        avg_mem_clock = mean_or_none([item.mem_clock_mhz for item in gpu_samples])
        avg_utilization = mean_or_none([item.utilization_gpu_pct for item in gpu_samples])
        baseline_mode = baseline_ecc_modes.get(gpu.index, {})
        final_mode = final_ecc_modes.get(gpu.index, {})
        ecc_mode_current = final_mode.get("current") or baseline_mode.get("current")
        ecc_mode_pending = final_mode.get("pending") or baseline_mode.get("pending")
        ecc_enabled = ecc_mode_current == "Enabled" if ecc_mode_current is not None else None

        baseline_corrected = baseline_ecc.get(gpu.index, {}).get("corrected")
        final_corrected = final_ecc.get(gpu.index, {}).get("corrected")
        baseline_uncorrected = baseline_ecc.get(gpu.index, {}).get("uncorrected")
        final_uncorrected = final_ecc.get(gpu.index, {}).get("uncorrected")

        corrected_delta = (
            final_corrected - baseline_corrected
            if baseline_corrected is not None and final_corrected is not None
            else None
        )
        uncorrected_delta = (
            final_uncorrected - baseline_uncorrected
            if baseline_uncorrected is not None and final_uncorrected is not None
            else None
        )
        nvlink_delta = (
            final_nvlink[gpu.index] - baseline_nvlink[gpu.index]
            if gpu.index in baseline_nvlink and gpu.index in final_nvlink
            else None
        )
        nvlink_crc_delta = (
            final_nvlink_crc[gpu.index] - baseline_nvlink_crc[gpu.index]
            if gpu.index in baseline_nvlink_crc and gpu.index in final_nvlink_crc
            else None
        )
        burn_issue = burn_issue_counts.get(gpu.index, {})
        burn_warning_count = burn_issue.get("warning_count")
        burn_error_count = burn_issue.get("error_count")
        burn_has_issue = (burn_warning_count or 0) > 0 or (burn_error_count or 0) > 0
        burn_passed = burn_exit_ok is True and not burn_has_issue
        stop_reason = (burn_stop_reasons or {}).get(gpu.index)
        run_stopped_for_overheat = bool(burn_stop_reasons)
        soft_nvlink_notes: list[str] = []

        result = GPUResult(
            index=gpu.index,
            name=gpu.name,
            serial=gpu.serial,
            uuid=gpu.uuid,
            pci_bus_id=gpu.pci_bus_id,
            ecc_mode_current=ecc_mode_current,
            ecc_mode_pending=ecc_mode_pending,
            ecc_enabled=ecc_enabled,
            avg_temp_c=avg_temp,
            max_temp_c=max_temp,
            avg_power_w=avg_power,
            max_power_w=max_power,
            avg_sm_clock_mhz=avg_sm_clock,
            avg_mem_clock_mhz=avg_mem_clock,
            avg_utilization_gpu_pct=avg_utilization,
            ecc_corrected_before=baseline_corrected,
            ecc_corrected_after=final_corrected,
            ecc_corrected_delta=corrected_delta,
            ecc_uncorrected_before=baseline_uncorrected,
            ecc_uncorrected_after=final_uncorrected,
            ecc_uncorrected_delta=uncorrected_delta,
            nvlink_link_count_before=baseline_nvlink_links.get(gpu.index),
            nvlink_link_count_after=final_nvlink_links.get(gpu.index),
            nvlink_error_delta=nvlink_delta,
            nvlink_crc_error_delta=nvlink_crc_delta,
            nvlink_bandwidth_gbps=p2p_bandwidth.get(gpu.index),
            burn_gflops=burn_scores.get(gpu.index),
            burn_warning_count=burn_warning_count,
            burn_error_count=burn_error_count,
            burn_ok=burn_passed,
        )

        result.result = "PASS"
        if not gpu_samples:
            result.result = "ERROR"
            result.notes.append("未采集到 GPU 指标")
        if stop_reason:
            result.result = "FAIL"
            result.notes.append(stop_reason)
        elif run_stopped_for_overheat:
            result.result = "FAIL"
            result.notes.append("测试因超温被提前终止")
        elif burn_exit_ok is False:
            result.result = "FAIL"
            result.notes.append("gpu-burn 非零退出")
        if burn_exit_ok is None:
            result.result = "NOT_RUN"
            result.notes.append("未执行 gpu-burn")
        if burn_has_issue:
            result.result = "FAIL"
            result.notes.append(
                f"gpu-burn WARNING/ERROR 计数为 {burn_warning_count or 0}/{burn_error_count or 0}"
            )
        if ecc_enabled is False:
            result.result = "FAIL"
            result.notes.append("ECC 模式未开启")
        if ecc_mode_pending and ecc_mode_current and ecc_mode_pending != ecc_mode_current:
            result.notes.append(
                f"ECC 待生效状态为 {ecc_mode_pending}，当前状态为 {ecc_mode_current}"
            )
        if max_temp is not None and max_temp > max_temperature_c:
            result.result = "FAIL"
            result.notes.append(
                f"最高温度 {max_temp:.2f}C 超过阈值 {max_temperature_c:.2f}C"
            )
        if corrected_delta is not None and corrected_delta > 0:
            result.result = "FAIL"
            result.notes.append(f"ECC 已纠正错误增量为 {corrected_delta}")
        if uncorrected_delta is not None and uncorrected_delta > 0:
            result.result = "FAIL"
            result.notes.append(f"ECC 未纠正错误增量为 {uncorrected_delta}")
        if nvlink_delta is not None and nvlink_delta > 0:
            soft_nvlink_notes.append(f"NVLink 错误增量为 {nvlink_delta}")
        if nvlink_crc_delta is not None and nvlink_crc_delta > 0:
            soft_nvlink_notes.append(f"NVLink CRC 错误增量为 {nvlink_crc_delta}")
        final_link_count = final_nvlink_links.get(gpu.index)
        baseline_link_count = baseline_nvlink_links.get(gpu.index)
        effective_link_count = final_link_count if final_link_count is not None else baseline_link_count
        if effective_link_count is not None and effective_link_count < DEFAULT_NVLINK_EXPECTED_LINKS:
            before_text = baseline_link_count if baseline_link_count is not None else "?"
            after_text = final_link_count if final_link_count is not None else "?"
            soft_nvlink_notes.append(
                f"NVLink 连接通道数 {before_text}/{DEFAULT_NVLINK_EXPECTED_LINKS} -> "
                f"{after_text}/{DEFAULT_NVLINK_EXPECTED_LINKS}"
            )
        if result.result == "PASS":
            if soft_nvlink_notes:
                result.assessment = "GOOD"
                result.notes.extend(f"{note}，标记为良好" for note in soft_nvlink_notes)
            else:
                result.assessment = "EXCELLENT"
        else:
            result.assessment = result.result
            result.notes.extend(soft_nvlink_notes)
        results.append(result)
    return results


def determine_overall_result(results: list[GPUResult], driver_ready: bool) -> str:
    if not driver_ready:
        return "ERROR"
    states = {item.result for item in results}
    if "FAIL" in states:
        return "FAIL"
    if "ERROR" in states:
        return "ERROR"
    if states == {"PASS"}:
        return "PASS"
    return "NOT_RUN"


def run_suite(
    *,
    burn_seconds: int,
    nvlink_seconds: int,
    sample_interval: float,
    wait_for_driver_seconds: int,
    max_temperature_c: float,
    gpu_burn_command: str | None,
    nvlink_bandwidth_command: str | None,
    target_gpu_index: int | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> RunReport:
    started_at = utc_now()
    started_monotonic = time.monotonic()
    host = socket.gethostname()
    warnings: list[str] = []
    errors: list[str] = []
    environment = collect_environment()
    environment.update(
        {
            "burn_seconds": str(burn_seconds),
            "nvlink_seconds": str(nvlink_seconds),
            "sample_interval": str(sample_interval),
            "wait_for_driver_seconds": str(wait_for_driver_seconds),
            "max_temperature_c": f"{max_temperature_c:.1f}",
            "target_gpu_index": "ALL" if target_gpu_index is None else str(target_gpu_index),
        }
    )
    driver_ready = False
    artifacts: dict[str, object] = {"run_status": "RUNNING"}
    gpus: list[GPUIdentity] = []
    samples: list[MetricSample] = []
    emit_progress(
        progress_callback,
        "suite-start",
        "开始执行 1Cat-V100-QA",
        host=host,
        burn_seconds=burn_seconds,
        nvlink_seconds=nvlink_seconds,
        sample_interval=sample_interval,
        max_temperature_c=max_temperature_c,
        target_gpu_index=target_gpu_index,
    )
    try:
        raise_if_cancelled(cancel_event)
        driver_ready, driver_message = wait_for_driver(
            wait_for_driver_seconds,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if not driver_ready:
            errors.append(driver_message)
            artifacts["run_status"] = "ERROR"
            finished_at = utc_now()
            emit_progress(progress_callback, "suite-error", driver_message, errors=errors)
            return RunReport(
                host=host,
                started_at=started_at,
                finished_at=finished_at,
                duration_sec=time.monotonic() - started_monotonic,
                overall_result="ERROR",
                driver_ready=False,
                gpu_count=0,
                errors=errors,
                warnings=warnings,
                environment=environment,
                results=[],
                samples=[],
                artifacts=artifacts,
            )

        discovered_gpus, discovery_warnings = discover_gpus()
        warnings.extend(discovery_warnings)
        if target_gpu_index is not None and not any(gpu.index == target_gpu_index for gpu in discovered_gpus):
            message = f"未找到指定的 GPU {target_gpu_index}"
            errors.append(message)
            artifacts["run_status"] = "ERROR"
            finished_at = utc_now()
            emit_progress(progress_callback, "suite-error", message, errors=errors)
            return RunReport(
                host=host,
                started_at=started_at,
                finished_at=finished_at,
                duration_sec=time.monotonic() - started_monotonic,
                overall_result="ERROR",
                driver_ready=driver_ready,
                gpu_count=0,
                errors=errors,
                warnings=warnings,
                environment=environment,
                results=[],
                samples=[],
                artifacts=artifacts,
            )
        gpus = filter_gpu_identities(discovered_gpus, target_gpu_index)
        emit_progress(
            progress_callback,
            "gpu-inventory",
            f"检测到 {len(gpus)} 张 GPU",
            gpus=[asdict(gpu) for gpu in gpus],
        )
        raise_if_cancelled(cancel_event)

        baseline_metrics, metric_error = collect_metrics()
        baseline_metrics = filter_metric_map(baseline_metrics, target_gpu_index)
        if metric_error:
            warnings.append(metric_error)
        samples = list(baseline_metrics.values())
        if samples:
            emit_progress(
                progress_callback,
                "baseline-metrics",
                "已采集基线 GPU 指标",
                samples=[asdict(item) for item in samples],
            )

        baseline_ecc_modes, ecc_mode_error = collect_ecc_modes()
        baseline_ecc_modes = filter_index_map(baseline_ecc_modes, target_gpu_index)
        if ecc_mode_error:
            warnings.append(ecc_mode_error)

        baseline_ecc, ecc_error = collect_ecc_totals()
        baseline_ecc = filter_index_map(baseline_ecc, target_gpu_index)
        if ecc_error:
            warnings.append(ecc_error)
        if baseline_ecc or baseline_ecc_modes:
            emit_progress(
                progress_callback,
                "baseline-ecc",
                "已读取启动 ECC 单比特 / 双比特计数",
                counters=baseline_ecc,
                modes=baseline_ecc_modes,
            )
        baseline_nvlink_links, nvlink_link_error = collect_nvlink_link_counts()
        baseline_nvlink_links = filter_index_map(baseline_nvlink_links, target_gpu_index)
        if nvlink_link_error:
            warnings.append(nvlink_link_error)
        if baseline_nvlink_links:
            emit_progress(
                progress_callback,
                "baseline-nvlink-links",
                "已读取启动 NVLink 连接通道数",
                counts=baseline_nvlink_links,
            )
        baseline_nvlink, nvlink_error = collect_nvlink_errors()
        baseline_nvlink = filter_index_map(baseline_nvlink, target_gpu_index)
        if nvlink_error:
            warnings.append(nvlink_error)
        baseline_nvlink_crc, nvlink_crc_error = collect_nvlink_crc_errors()
        baseline_nvlink_crc = filter_index_map(baseline_nvlink_crc, target_gpu_index)
        if nvlink_crc_error:
            warnings.append(nvlink_crc_error)
        raise_if_cancelled(cancel_event)

        p2p_bandwidth: dict[int, float] = {}
        p2p_bandwidth_matrix: dict[int, dict[int, float]] = {}
        p2p_output: str | None = None
        p2p_warning: str | None = None
        if len(gpus) >= 2:
            raise_if_cancelled(cancel_event)
            emit_progress(
                progress_callback,
                "nvlink-start",
                f"开始执行 NVLink 带宽测试，时长 {nvlink_seconds} 秒",
            )
            p2p_bandwidth, p2p_bandwidth_matrix, p2p_output, p2p_warning = collect_nvlink_bandwidth(
                nvlink_bandwidth_command,
                nvlink_seconds=nvlink_seconds,
                target_gpu_index=target_gpu_index,
                cancel_event=cancel_event,
            )
            if p2p_output:
                artifacts["p2p_bandwidth_output"] = p2p_output
            if p2p_bandwidth_matrix:
                artifacts["p2p_bandwidth_matrix"] = p2p_bandwidth_matrix
            artifacts["nvlink_test_status"] = "COMPLETED" if p2p_bandwidth else "FAILED"
        elif len(gpus) < 2:
            p2p_warning = "当前仅选择 1 张 GPU，已跳过 NVLink 带宽测试"
            artifacts["nvlink_test_status"] = "SKIPPED_SINGLE_GPU"
        if p2p_warning:
            warnings.append(p2p_warning)
        if len(gpus) >= 2:
            emit_progress(
                progress_callback,
                "nvlink-finish",
                "NVLink 带宽测试已结束",
                bandwidth=p2p_bandwidth,
            )

        burn_result = start_gpu_burn(
            burn_seconds=burn_seconds,
            sample_interval=sample_interval,
            max_temperature_c=max_temperature_c,
            command_override=gpu_burn_command,
            target_gpu_index=target_gpu_index,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        samples.extend(burn_result.samples)
        if burn_result.output:
            artifacts["gpu_burn_output"] = burn_result.output
        if burn_result.issue_counts:
            artifacts["gpu_burn_issue_counts"] = burn_result.issue_counts
        if burn_result.warning:
            warnings.append(burn_result.warning)
        if burn_result.stop_reasons:
            artifacts["burn_stop_reasons"] = burn_result.stop_reasons
            errors.extend(burn_result.stop_reasons.values())

        raise_if_cancelled(cancel_event)
        final_metrics, final_metric_error = collect_metrics()
        final_metrics = filter_metric_map(final_metrics, target_gpu_index)
        if final_metric_error:
            warnings.append(final_metric_error)
        samples.extend(final_metrics.values())

        final_ecc_modes, final_ecc_mode_error = collect_ecc_modes()
        final_ecc_modes = filter_index_map(final_ecc_modes, target_gpu_index)
        if final_ecc_mode_error:
            warnings.append(final_ecc_mode_error)
        final_ecc, final_ecc_error = collect_ecc_totals()
        final_ecc = filter_index_map(final_ecc, target_gpu_index)
        if final_ecc_error:
            warnings.append(final_ecc_error)
        if final_ecc or final_ecc_modes:
            emit_progress(
                progress_callback,
                "final-ecc",
                "已读取结束后的 ECC 单比特 / 双比特计数",
                counters=final_ecc,
                modes=final_ecc_modes,
            )
        final_nvlink_links, final_nvlink_link_error = collect_nvlink_link_counts()
        final_nvlink_links = filter_index_map(final_nvlink_links, target_gpu_index)
        if final_nvlink_link_error:
            warnings.append(final_nvlink_link_error)
        if final_nvlink_links:
            emit_progress(
                progress_callback,
                "final-nvlink-links",
                "已读取结束后的 NVLink 连接通道数",
                counts=final_nvlink_links,
            )
        final_nvlink, final_nvlink_error = collect_nvlink_errors()
        final_nvlink = filter_index_map(final_nvlink, target_gpu_index)
        if final_nvlink_error:
            warnings.append(final_nvlink_error)
        final_nvlink_crc, final_nvlink_crc_error = collect_nvlink_crc_errors()
        final_nvlink_crc = filter_index_map(final_nvlink_crc, target_gpu_index)
        if final_nvlink_crc_error:
            warnings.append(final_nvlink_crc_error)

        results = build_gpu_results(
            gpus=gpus,
            samples=samples,
            baseline_ecc_modes=baseline_ecc_modes,
            final_ecc_modes=final_ecc_modes,
            baseline_ecc=baseline_ecc,
            final_ecc=final_ecc,
            baseline_nvlink_links=baseline_nvlink_links,
            final_nvlink_links=final_nvlink_links,
            baseline_nvlink=baseline_nvlink,
            final_nvlink=final_nvlink,
            baseline_nvlink_crc=baseline_nvlink_crc,
            final_nvlink_crc=final_nvlink_crc,
            p2p_bandwidth=p2p_bandwidth,
            burn_scores=burn_result.scores,
            burn_issue_counts=burn_result.issue_counts,
            burn_exit_ok=burn_result.exit_ok,
            burn_stop_reasons=burn_result.stop_reasons,
            max_temperature_c=max_temperature_c,
        )
        for item in results:
            if (item.burn_warning_count or 0) > 0 or (item.burn_error_count or 0) > 0:
                warnings.append(
                    f"GPU {item.index} gpu-burn WARNING/ERROR 计数为 {item.burn_warning_count or 0}/{item.burn_error_count or 0}"
                )
        for item in results:
            link_count = item.nvlink_link_count_after
            if link_count is None:
                link_count = item.nvlink_link_count_before
            if link_count is not None and link_count < DEFAULT_NVLINK_EXPECTED_LINKS:
                warnings.append(
                    f"GPU {item.index} NVLink 连接通道数为 {link_count}/{DEFAULT_NVLINK_EXPECTED_LINKS}"
                )
        artifacts["run_status"] = "COMPLETED"
        artifacts["environment"] = environment
        artifacts["gpu_inventory"] = [asdict(gpu) for gpu in gpus]
        artifacts["target_gpu_index"] = target_gpu_index
        artifacts["baseline_ecc_modes"] = baseline_ecc_modes
        artifacts["final_ecc_modes"] = final_ecc_modes
        artifacts["baseline_ecc_counts"] = baseline_ecc
        artifacts["final_ecc_counts"] = final_ecc
        artifacts["baseline_nvlink_link_counts"] = baseline_nvlink_links
        artifacts["final_nvlink_link_counts"] = final_nvlink_links
        artifacts["baseline_nvlink_error_counts"] = baseline_nvlink
        artifacts["final_nvlink_error_counts"] = final_nvlink
        artifacts["baseline_nvlink_crc_error_counts"] = baseline_nvlink_crc
        artifacts["final_nvlink_crc_error_counts"] = final_nvlink_crc
        finished_at = utc_now()
        report = RunReport(
            host=host,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=time.monotonic() - started_monotonic,
            overall_result=determine_overall_result(results, driver_ready),
            driver_ready=driver_ready,
            gpu_count=len(gpus),
            errors=errors,
            warnings=warnings,
            environment=environment,
            results=results,
            samples=samples,
            artifacts=artifacts,
        )
        emit_progress(
            progress_callback,
            "suite-finish",
            f"测试完成，整体结果为 {report.overall_result}",
            overall_result=report.overall_result,
            warnings=warnings,
            errors=errors,
            results=[asdict(item) for item in results],
        )
        return report
    except RunCancelled as exc:
        cancelled_message = str(exc)
        warnings.append(cancelled_message)
        artifacts["run_status"] = "CANCELLED"
        artifacts["environment"] = environment
        if gpus:
            artifacts["gpu_inventory"] = [asdict(gpu) for gpu in gpus]
        results = [
            GPUResult(
                index=gpu.index,
                name=gpu.name,
                serial=gpu.serial,
                uuid=gpu.uuid,
                pci_bus_id=gpu.pci_bus_id,
                result="NOT_RUN",
                notes=[cancelled_message],
            )
            for gpu in gpus
        ]
        finished_at = utc_now()
        report = RunReport(
            host=host,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=time.monotonic() - started_monotonic,
            overall_result="NOT_RUN",
            driver_ready=driver_ready,
            gpu_count=len(gpus),
            errors=errors,
            warnings=warnings,
            environment=environment,
            results=results,
            samples=samples,
            artifacts=artifacts,
        )
        emit_progress(
            progress_callback,
            "suite-cancelled",
            cancelled_message,
            overall_result=report.overall_result,
            warnings=warnings,
            errors=errors,
            results=[asdict(item) for item in results],
        )
        return report
