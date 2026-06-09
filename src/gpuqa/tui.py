from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import queue
import textwrap
import threading
import time
import traceback
import unicodedata

from gpuqa.cli import build_default_output_dir
from gpuqa.core import (
    DEFAULT_NVLINK_EXPECTED_LINKS,
    DEFAULT_NVLINK_TEST_SECONDS,
    collect_ecc_modes,
    collect_ecc_totals,
    collect_metrics,
    collect_nvlink_link_counts,
    discover_gpus,
    run_suite,
)
from gpuqa.desktop import open_path
from gpuqa.models import RunReport, assessment_code_for_gpu, display_assessment, overall_assessment_code
from gpuqa.reporting import write_outputs

RESULT_LABELS = {
    "IDLE": "空闲",
    "RUNNING": "检测中",
    "EXCELLENT": "优秀",
    "GOOD": "良好",
    "PASS": "通过",
    "FAIL": "不通过",
    "ERROR": "错误",
    "NOT_RUN": "未运行",
}
ECC_MODE_LABELS = {
    "Enabled": "已开启",
    "Disabled": "未开启",
    "Unknown": "未知",
}
STAGE_LABELS = {
    "idle": "待命",
    "suite-start": "准备环境",
    "wait-driver": "等待驱动",
    "driver-ready": "驱动就绪",
    "driver-timeout": "驱动超时",
    "gpu-inventory": "识别 GPU",
    "baseline-metrics": "采集基线",
    "baseline-ecc": "ECC 启动快照",
    "baseline-nvlink-links": "NVLink 启动快照",
    "gpu-burn-overheat": "超温停测",
    "gpu-burn-start": "开始压测",
    "gpu-burn-sample": "压测采样",
    "gpu-burn-finish": "压测完成",
    "gpu-burn-stop": "停止压测",
    "nvlink-start": "NVLink 测试",
    "final-nvlink-links": "NVLink 结束复检",
    "nvlink-finish": "NVLink 完成",
    "final-ecc": "ECC 结束复检",
    "suite-finish": "测试完成",
    "suite-cancelled": "已手动停止",
    "suite-error": "测试异常",
}
SUMMARY_PLACEHOLDER = [
    "纯文字界面已就绪。",
    "按 R 开始检测，按 S 停止检测，按 O 打开结果目录，按 Q 退出。",
]
SPINNER_FRAMES = ["⠋", "⠙", "⠸", "⠴", "⠦", "⠇"]
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
BOX = {
    "tl": "╭",
    "tr": "╮",
    "bl": "╰",
    "br": "╯",
    "h": "─",
    "v": "│",
}


def display_result(value: str) -> str:
    return RESULT_LABELS.get(value, display_assessment(value))


def display_ecc_mode(value: str) -> str:
    return ECC_MODE_LABELS.get(value, value)


def format_float(value: float | None, digits: int = 1, fallback: str = "N/A") -> str:
    return fallback if value is None else f"{value:.{digits}f}"


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def cell_width(char: str) -> int:
    if not char:
        return 0
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1


def display_width(text: str) -> int:
    return sum(cell_width(char) for char in text)


def trim_to_width(text: str, width: int) -> str:
    if width <= 0:
        return ""
    current = 0
    parts: list[str] = []
    for char in text:
        next_width = cell_width(char)
        if current + next_width > width:
            break
        parts.append(char)
        current += next_width
    return "".join(parts)


def shorten(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    if width == 1:
        return trim_to_width(text, width)
    return trim_to_width(text, width - 1) + "…"


def pad_to_width(text: str, width: int) -> str:
    trimmed = shorten(text, width)
    return trimmed + (" " * max(width - display_width(trimmed), 0))


def format_bar(value: float | None, maximum: float, width: int) -> str:
    if width <= 0:
        return ""
    if value is None or maximum <= 0:
        return "·" * width
    ratio = clamp(value / maximum, 0.0, 1.0)
    filled = int(round(ratio * width))
    return "█" * filled + "░" * max(width - filled, 0)


def format_sparkline(history: list[float], width: int) -> str:
    if width <= 0:
        return ""
    if not history:
        return "·" * width
    values = history[-width:]
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return SPARKLINE_CHARS[0] * len(values)
    pieces = []
    for value in values:
        ratio = (value - minimum) / (maximum - minimum)
        index = int(round(ratio * (len(SPARKLINE_CHARS) - 1)))
        pieces.append(SPARKLINE_CHARS[index])
    return "".join(pieces).rjust(width, SPARKLINE_CHARS[0])


@dataclass(slots=True)
class LiveGPUState:
    index: int
    model: str = "未知型号"
    serial: str = "N/A"
    pci_bus_id: str = "N/A"
    uuid: str = "N/A"
    ecc: str = "未知"
    temperature: str = "N/A"
    temperature_value: float | None = None
    current_power: str = "N/A"
    current_power_value: float | None = None
    peak_power: str = "N/A"
    peak_power_value: float | None = None
    utilization: str = "N/A"
    utilization_value: float | None = None
    memory: str = "N/A"
    memory_used_mib: float | None = None
    memory_total_mib: float | None = None
    sm_clock: str = "N/A"
    sm_clock_value: float | None = None
    mem_clock: str = "N/A"
    mem_clock_value: float | None = None
    burn: str = "N/A"
    burn_value: float | None = None
    burn_warning_count: str = "N/A"
    burn_error_count: str = "N/A"
    nvlink: str = "N/A"
    nvlink_value: float | None = None
    nvlink_links_before: str = "N/A"
    nvlink_links_after: str = "N/A"
    crc_errors: str = "N/A"
    ecc_single_before: str = "N/A"
    ecc_single_after: str = "N/A"
    ecc_single_delta: str = "N/A"
    ecc_double_before: str = "N/A"
    ecc_double_after: str = "N/A"
    ecc_double_delta: str = "N/A"
    ecc_corrected_delta: str = "N/A"
    ecc_uncorrected_delta: str = "N/A"
    note: str = ""
    result: str = "运行中"
    result_code: str = "RUNNING"
    temperature_history: list[float] = field(default_factory=list)
    utilization_history: list[float] = field(default_factory=list)
    power_history: list[float] = field(default_factory=list)
    burn_history: list[float] = field(default_factory=list)


class GPUQATextApp:
    def __init__(self) -> None:
        self.queue: queue.Queue[dict[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event: threading.Event | None = None
        self.exit_after_finish = False
        self.latest_output_dir: Path | None = None
        self.latest_written: dict[str, Path] = {}
        self.gpus: dict[int, LiveGPUState] = {}
        self.logs: list[str] = []
        self.summary_lines = list(SUMMARY_PLACEHOLDER)
        self.status_text = "空闲，等待开始检测"
        self.result_code = "IDLE"
        self.warning_count = 0
        self.fail_count = 0
        self.started_at_monotonic: float | None = None
        self.last_duration_sec = 0.0
        self.output_dir = build_default_output_dir()
        self.current_stage = "idle"
        self.host = "N/A"
        self.gpu_burn_lines: list[str] = []
        self.burn_seconds = 600
        self.sample_interval = 5.0
        self.wait_driver_seconds = 180
        self.max_temp_c = 80.0
        self.nvlink_seconds = DEFAULT_NVLINK_TEST_SECONDS
        self.target_gpu_index: int | None = None

    def run(self) -> int:
        try:
            import curses
        except ImportError:
            print("当前 Python 缺少 curses，无法启动文本界面。")
            return 1
        return curses.wrapper(self._main)

    def _main(self, stdscr: object) -> int:
        import curses

        screen = stdscr
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        self._init_colors(curses)
        screen.nodelay(True)
        screen.timeout(150)
        self._append_log("文本界面已启动")
        self._append_log("按 R 开始检测，按 S 停止检测，按 T/B/N/G 修改配置，按 O 打开结果目录，按 Q 退出")
        self._load_idle_snapshot()

        while True:
            self._drain_queue()
            self._draw(screen)
            key = screen.getch()
            if key != -1:
                should_exit = self._handle_key(screen, key)
                if should_exit:
                    break
            if self.exit_after_finish and not self._is_running():
                break

        if self._is_running():
            self._request_stop()
            deadline = time.monotonic() + 10.0
            while self._is_running() and time.monotonic() < deadline:
                self._drain_queue()
                self._draw(screen)
                time.sleep(0.1)
        return 0

    def _is_running(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _append_log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{stamp}] {message}")
        if len(self.logs) > 300:
            self.logs = self.logs[-300:]

    def _push_history(self, history: list[float], value: float | None, limit: int = 36) -> None:
        if value is None:
            return
        history.append(value)
        if len(history) > limit:
            del history[:-limit]

    def _init_colors(self, curses_module: object) -> None:
        curses = curses_module
        self.colors = {
            "base": curses.A_NORMAL,
            "muted": curses.A_DIM,
            "panel": curses.A_NORMAL,
            "title": curses.A_BOLD,
            "accent": curses.A_BOLD,
            "success": curses.A_BOLD,
            "warning": curses.A_BOLD,
            "danger": curses.A_BOLD,
            "info": curses.A_BOLD,
            "header": curses.A_BOLD,
        }
        if not curses.has_colors():
            return
        pairs = [
            ("accent", curses.COLOR_CYAN, -1),
            ("success", curses.COLOR_GREEN, -1),
            ("warning", curses.COLOR_YELLOW, -1),
            ("danger", curses.COLOR_RED, -1),
            ("info", curses.COLOR_BLUE, -1),
            ("header", curses.COLOR_MAGENTA, -1),
            ("panel", curses.COLOR_WHITE, -1),
            ("title", curses.COLOR_CYAN, -1),
        ]
        for index, (name, fg, bg) in enumerate(pairs, start=1):
            curses.init_pair(index, fg, bg)
            self.colors[name] = curses.color_pair(index) | (
                curses.A_BOLD if name not in {"panel", "muted"} else curses.A_NORMAL
            )

    def _format_counter(self, value: int | None) -> str:
        return "N/A" if value is None else str(value)

    def _format_delta(self, value: int | None) -> str:
        return "N/A" if value is None else f"{value:+d}"

    def _format_nvlink_link_count(self, value: int | None) -> str:
        return "N/A" if value is None else f"{value}/{DEFAULT_NVLINK_EXPECTED_LINKS}"

    def _target_gpu_label(self) -> str:
        return "全部GPU" if self.target_gpu_index is None else f"GPU {self.target_gpu_index}"

    def _apply_ecc_snapshot(
        self,
        state: LiveGPUState,
        *,
        corrected_before: int | None = None,
        corrected_after: int | None = None,
        uncorrected_before: int | None = None,
        uncorrected_after: int | None = None,
    ) -> None:
        if corrected_before is not None:
            state.ecc_single_before = self._format_counter(corrected_before)
        if corrected_after is not None:
            state.ecc_single_after = self._format_counter(corrected_after)
        if corrected_before is not None or corrected_after is not None:
            before = corrected_before if corrected_before is not None else corrected_after
            after = corrected_after if corrected_after is not None else corrected_before
            if before is not None and after is not None:
                state.ecc_single_delta = self._format_delta(after - before)
            elif corrected_after is not None:
                state.ecc_single_delta = self._format_delta(0)

        if uncorrected_before is not None:
            state.ecc_double_before = self._format_counter(uncorrected_before)
        if uncorrected_after is not None:
            state.ecc_double_after = self._format_counter(uncorrected_after)
        if uncorrected_before is not None or uncorrected_after is not None:
            before = uncorrected_before if uncorrected_before is not None else uncorrected_after
            after = uncorrected_after if uncorrected_after is not None else uncorrected_before
            if before is not None and after is not None:
                state.ecc_double_delta = self._format_delta(after - before)
            elif uncorrected_after is not None:
                state.ecc_double_delta = self._format_delta(0)

        state.ecc_corrected_delta = state.ecc_single_delta
        state.ecc_uncorrected_delta = state.ecc_double_delta

    def _apply_nvlink_link_snapshot(
        self,
        state: LiveGPUState,
        *,
        before: int | None = None,
        after: int | None = None,
    ) -> None:
        if before is not None:
            state.nvlink_links_before = self._format_nvlink_link_count(before)
        if after is not None:
            state.nvlink_links_after = self._format_nvlink_link_count(after)
        elif before is not None and state.nvlink_links_after == "N/A":
            state.nvlink_links_after = self._format_nvlink_link_count(before)

    def _load_idle_snapshot(self, *, reset_display: bool = False) -> None:
        if reset_display:
            self.gpus.clear()
        self._append_log("正在读取启动时 ECC 快照")
        gpus, gpu_warnings = discover_gpus()
        for warning in gpu_warnings:
            self._append_log(warning)
        for gpu in gpus:
            if self.target_gpu_index is not None and gpu.index != self.target_gpu_index:
                continue
            state = self._ensure_gpu(gpu.index)
            state.model = str(gpu.name or state.model)
            state.serial = str(gpu.serial or state.serial)
            state.pci_bus_id = str(gpu.pci_bus_id or state.pci_bus_id)
            state.uuid = str(gpu.uuid or state.uuid)

        metrics, metric_error = collect_metrics()
        if metric_error:
            self._append_log(metric_error)
        else:
            filtered_metrics = {
                gpu_index: sample
                for gpu_index, sample in metrics.items()
                if self.target_gpu_index is None or gpu_index == self.target_gpu_index
            }
            self._handle_progress(
                "baseline-metrics",
                "已读取启动 GPU 指标",
                {"samples": [asdict(sample) for sample in filtered_metrics.values()]},
            )

        ecc_modes, mode_error = collect_ecc_modes()
        if mode_error:
            self._append_log(mode_error)
        ecc_counters, ecc_error = collect_ecc_totals()
        if ecc_error:
            self._append_log(ecc_error)
        if self.target_gpu_index is not None:
            ecc_modes = {
                gpu_index: values
                for gpu_index, values in ecc_modes.items()
                if gpu_index == self.target_gpu_index
            }
            ecc_counters = {
                gpu_index: values
                for gpu_index, values in ecc_counters.items()
                if gpu_index == self.target_gpu_index
            }
        if ecc_modes or ecc_counters:
            self._handle_progress(
                "baseline-ecc",
                "已读取启动 ECC 单比特 / 双比特计数",
                {"modes": ecc_modes, "counters": ecc_counters},
            )
        nvlink_links, nvlink_link_error = collect_nvlink_link_counts()
        if nvlink_link_error:
            self._append_log(nvlink_link_error)
        if self.target_gpu_index is not None:
            nvlink_links = {
                gpu_index: value
                for gpu_index, value in nvlink_links.items()
                if gpu_index == self.target_gpu_index
            }
        if nvlink_links:
            self._handle_progress(
                "baseline-nvlink-links",
                "已读取启动 NVLink 连接通道数",
                {"counts": nvlink_links},
            )
        self.current_stage = "idle"
        self.status_text = f"启动快照已就绪，等待开始检测，当前目标 {self._target_gpu_label()}"
        self.result_code = "IDLE"

    def _reset_state(self) -> None:
        self.latest_written = {}
        self.gpus.clear()
        self.logs.clear()
        self.gpu_burn_lines.clear()
        self.summary_lines = [
            "检测正在进行中。",
            "实时状态会显示在上方卡片区域。",
            f"当前配置：温度阈值 {self.max_temp_c:.1f}C | Burn {self.burn_seconds}s | NVLink {self.nvlink_seconds}s | {self._target_gpu_label()}",
            "按 S 停止检测。",
        ]
        self.status_text = "正在准备检测"
        self.result_code = "RUNNING"
        self.current_stage = "suite-start"
        self.warning_count = 0
        self.fail_count = 0
        self.started_at_monotonic = time.monotonic()
        self.last_duration_sec = 0.0

    def _start_run(self) -> None:
        if self._is_running():
            return
        self.output_dir = build_default_output_dir()
        self._reset_state()
        self.cancel_event = threading.Event()
        self._append_log(f"输出目录: {self.output_dir}")
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()

    def _request_stop(self) -> None:
        if not self._is_running() or self.cancel_event is None:
            self._append_log("当前没有正在运行的检测任务")
            return
        if self.cancel_event.is_set():
            return
        self.cancel_event.set()
        self.status_text = "正在停止检测，请等待 gpu-burn 退出"
        self._append_log("已请求停止当前检测")

    def _run_worker(self) -> None:
        try:
            report = run_suite(
                burn_seconds=self.burn_seconds,
                nvlink_seconds=self.nvlink_seconds,
                sample_interval=self.sample_interval,
                wait_for_driver_seconds=self.wait_driver_seconds,
                max_temperature_c=self.max_temp_c,
                gpu_burn_command=None,
                nvlink_bandwidth_command=None,
                target_gpu_index=self.target_gpu_index,
                progress_callback=self._queue_progress,
                cancel_event=self.cancel_event,
            )
            written = write_outputs(report, self.output_dir)
        except Exception:
            self.queue.put({"type": "error", "message": traceback.format_exc()})
            return
        self.queue.put(
            {
                "type": "finished",
                "report": report,
                "written": {name: str(path) for name, path in written.items()},
                "output_dir": str(self.output_dir),
            }
        )

    def _queue_progress(self, stage: str, message: str, payload: dict[str, object] | None) -> None:
        self.queue.put({"type": "progress", "stage": stage, "message": message, "payload": payload or {}})

    def _ensure_gpu(self, gpu_index: int) -> LiveGPUState:
        state = self.gpus.get(gpu_index)
        if state is None:
            state = LiveGPUState(index=gpu_index)
            self.gpus[gpu_index] = state
        return state

    def _handle_progress(self, stage: str, message: str, payload: dict[str, object]) -> None:
        if stage == "gpu-burn-output":
            raw_line = str(payload.get("line") or message).strip()
            if raw_line:
                self.gpu_burn_lines.append(raw_line)
                if len(self.gpu_burn_lines) > 160:
                    self.gpu_burn_lines = self.gpu_burn_lines[-160:]
            return
        self.current_stage = stage
        self.status_text = message
        if stage != "gpu-burn-sample":
            self._append_log(message)
        if stage == "suite-start":
            self.host = str(payload.get("host") or self.host)

        if stage == "gpu-inventory":
            for gpu in payload.get("gpus", []):
                if not isinstance(gpu, dict):
                    continue
                gpu_index = int(gpu.get("index", -1))
                state = self._ensure_gpu(gpu_index)
                state.model = str(gpu.get("name") or "未知型号")
                state.serial = str(gpu.get("serial") or "N/A")
                state.pci_bus_id = str(gpu.get("pci_bus_id") or "N/A")
                state.uuid = str(gpu.get("uuid") or "N/A")
            return

        if stage in {"baseline-ecc", "final-ecc"}:
            modes = payload.get("modes", {})
            if isinstance(modes, dict):
                for gpu_index, values in modes.items():
                    if not isinstance(values, dict):
                        continue
                    state = self._ensure_gpu(int(gpu_index))
                    current_raw = str(values.get("current") or "Unknown")
                    pending_raw = values.get("pending")
                    state.ecc = display_ecc_mode(current_raw)
                    if pending_raw and str(pending_raw) != current_raw:
                        state.ecc = f"{state.ecc} -> {display_ecc_mode(str(pending_raw))}"
            counters = payload.get("counters", {})
            if isinstance(counters, dict):
                for gpu_index, values in counters.items():
                    if not isinstance(values, dict):
                        continue
                    state = self._ensure_gpu(int(gpu_index))
                    if stage == "baseline-ecc":
                        self._apply_ecc_snapshot(
                            state,
                            corrected_before=values.get("corrected"),
                            corrected_after=values.get("corrected"),
                            uncorrected_before=values.get("uncorrected"),
                            uncorrected_after=values.get("uncorrected"),
                        )
                    else:
                        self._apply_ecc_snapshot(
                            state,
                            corrected_after=values.get("corrected"),
                            uncorrected_after=values.get("uncorrected"),
                        )
            return

        if stage in {"baseline-nvlink-links", "final-nvlink-links"}:
            counts = payload.get("counts", {})
            if isinstance(counts, dict):
                for gpu_index, value in counts.items():
                    state = self._ensure_gpu(int(gpu_index))
                    if stage == "baseline-nvlink-links":
                        self._apply_nvlink_link_snapshot(state, before=int(value), after=int(value))
                    else:
                        self._apply_nvlink_link_snapshot(state, after=int(value))
                    if int(value) < DEFAULT_NVLINK_EXPECTED_LINKS:
                        state.note = f"NVLink 通道不足: {int(value)}/{DEFAULT_NVLINK_EXPECTED_LINKS}"
            return

        if stage == "gpu-burn-overheat":
            reasons = payload.get("reasons", {})
            if isinstance(reasons, dict):
                for gpu_index, reason in reasons.items():
                    state = self._ensure_gpu(int(gpu_index))
                    state.note = str(reason)
                    state.result_code = "FAIL"
                    state.result = display_result("FAIL")
            return

        if stage in {"baseline-metrics", "gpu-burn-sample"}:
            for sample in payload.get("samples", []):
                if not isinstance(sample, dict):
                    continue
                gpu_index = int(sample.get("gpu_index", -1))
                state = self._ensure_gpu(gpu_index)
                if sample.get("temperature_c") is not None:
                    state.temperature_value = float(sample["temperature_c"])
                    state.temperature = f"{state.temperature_value:.1f} C"
                    self._push_history(state.temperature_history, state.temperature_value)
                if sample.get("power_w") is not None:
                    state.current_power_value = float(sample["power_w"])
                    state.current_power = f"{state.current_power_value:.1f} W"
                if sample.get("utilization_gpu_pct") is not None:
                    state.utilization_value = float(sample["utilization_gpu_pct"])
                    state.utilization = f"{state.utilization_value:.1f} %"
                    self._push_history(state.utilization_history, state.utilization_value)
                power_value = float(sample["power_w"]) if sample.get("power_w") is not None else None
                if power_value is not None and (state.peak_power_value is None or power_value > state.peak_power_value):
                    state.peak_power_value = power_value
                    state.peak_power = f"{power_value:.1f} W"
                self._push_history(state.power_history, power_value)
                used = sample.get("memory_used_mib")
                total = sample.get("memory_total_mib")
                if used is not None and total is not None:
                    state.memory_used_mib = float(used)
                    state.memory_total_mib = float(total)
                    state.memory = f"{state.memory_used_mib:.0f}/{state.memory_total_mib:.0f} MiB"
                if sample.get("sm_clock_mhz") is not None:
                    state.sm_clock_value = float(sample["sm_clock_mhz"])
                    state.sm_clock = f"{state.sm_clock_value:.0f} MHz"
                if sample.get("mem_clock_mhz") is not None:
                    state.mem_clock_value = float(sample["mem_clock_mhz"])
                    state.mem_clock = f"{state.mem_clock_value:.0f} MHz"
            return

        if stage == "gpu-burn-finish":
            burn_ok = payload.get("burn_ok")
            issue_counts = payload.get("issue_counts", {})
            scores = payload.get("scores", {})
            if isinstance(scores, dict):
                for gpu_index, score in scores.items():
                    state = self._ensure_gpu(int(gpu_index))
                    state.burn_value = float(score)
                    state.burn = f"{state.burn_value:.2f} GF/s"
                    self._push_history(state.burn_history, state.burn_value)
            if isinstance(issue_counts, dict):
                for gpu_index, values in issue_counts.items():
                    if not isinstance(values, dict):
                        continue
                    state = self._ensure_gpu(int(gpu_index))
                    warning_count = int(values.get("warning_count") or 0)
                    error_count = int(values.get("error_count") or 0)
                    state.burn_warning_count = str(warning_count)
                    state.burn_error_count = str(error_count)
                    if warning_count > 0 or error_count > 0:
                        state.note = f"gpu-burn WARNING/ERROR {warning_count}/{error_count}"
            for state in self.gpus.values():
                if burn_ok is True and not state.note:
                    state.note = "GPU-Burn 已通过"
                elif burn_ok is False and not state.note:
                    state.note = "GPU-Burn 未通过"
            return

        if stage == "nvlink-finish":
            bandwidth = payload.get("bandwidth", {})
            if isinstance(bandwidth, dict):
                for gpu_index, value in bandwidth.items():
                    state = self._ensure_gpu(int(gpu_index))
                    state.nvlink_value = float(value)
                    state.nvlink = f"{state.nvlink_value:.2f} GB/s"
            return

        if stage in {"suite-finish", "suite-cancelled"}:
            for result in payload.get("results", []):
                if not isinstance(result, dict):
                    continue
                gpu_index = int(result.get("index", -1))
                state = self._ensure_gpu(gpu_index)
                state.model = str(result.get("name") or state.model)
                state.serial = str(result.get("serial") or state.serial)
                state.uuid = str(result.get("uuid") or state.uuid)
                state.pci_bus_id = str(result.get("pci_bus_id") or state.pci_bus_id)
                current_raw = str(result.get("ecc_mode_current") or "Unknown")
                pending_raw = result.get("ecc_mode_pending")
                state.ecc = display_ecc_mode(current_raw)
                if pending_raw and str(pending_raw) != current_raw:
                    state.ecc = f"{state.ecc} -> {display_ecc_mode(str(pending_raw))}"
                if result.get("avg_temp_c") is not None:
                    state.temperature_value = float(result["avg_temp_c"])
                    state.temperature = f"{state.temperature_value:.1f} C"
                if result.get("avg_power_w") is not None:
                    state.current_power_value = float(result["avg_power_w"])
                    state.current_power = f"{state.current_power_value:.1f} W"
                if result.get("max_power_w") is not None:
                    state.peak_power_value = float(result["max_power_w"])
                    state.peak_power = f"{state.peak_power_value:.1f} W"
                if result.get("avg_sm_clock_mhz") is not None:
                    state.sm_clock_value = float(result["avg_sm_clock_mhz"])
                    state.sm_clock = f"{state.sm_clock_value:.0f} MHz"
                if result.get("avg_mem_clock_mhz") is not None:
                    state.mem_clock_value = float(result["avg_mem_clock_mhz"])
                    state.mem_clock = f"{state.mem_clock_value:.0f} MHz"
                if result.get("avg_utilization_gpu_pct") is not None:
                    state.utilization_value = float(result["avg_utilization_gpu_pct"])
                    state.utilization = f"{state.utilization_value:.1f} %"
                if result.get("burn_gflops") is not None:
                    state.burn_value = float(result["burn_gflops"])
                    state.burn = f"{state.burn_value:.2f} GF/s"
                    self._push_history(state.burn_history, state.burn_value)
                if result.get("burn_warning_count") is not None:
                    state.burn_warning_count = str(int(result["burn_warning_count"]))
                if result.get("burn_error_count") is not None:
                    state.burn_error_count = str(int(result["burn_error_count"]))
                if result.get("nvlink_bandwidth_gbps") is not None:
                    state.nvlink_value = float(result["nvlink_bandwidth_gbps"])
                    state.nvlink = f"{state.nvlink_value:.2f} GB/s"
                self._apply_nvlink_link_snapshot(
                    state,
                    before=result.get("nvlink_link_count_before"),
                    after=result.get("nvlink_link_count_after"),
                )
                if result.get("nvlink_crc_error_delta") is not None:
                    state.crc_errors = f"+{int(result['nvlink_crc_error_delta'])}"
                self._apply_ecc_snapshot(
                    state,
                    corrected_before=result.get("ecc_corrected_before"),
                    corrected_after=result.get("ecc_corrected_after"),
                    uncorrected_before=result.get("ecc_uncorrected_before"),
                    uncorrected_after=result.get("ecc_uncorrected_after"),
                )
                notes = result.get("notes") or []
                if isinstance(notes, list) and notes:
                    state.note = str(notes[0])
                result_code = str(result.get("result") or "NOT_RUN")
                assessment_code = str(result.get("assessment") or ("EXCELLENT" if result_code == "PASS" else result_code))
                state.result_code = assessment_code
                state.result = display_result(assessment_code)

    def _build_summary_lines(self, report: RunReport, output_dir: Path, written: dict[str, Path]) -> list[str]:
        failed = sum(1 for item in report.results if item.result in {"FAIL", "ERROR"})
        overall_code = overall_assessment_code(report)
        config_target = str(report.environment.get("target_gpu_index") or "ALL")
        config_label = "全部GPU" if config_target == "ALL" else f"GPU {config_target}"
        lines = [
            f"总体结果: {display_result(overall_code)}",
            f"测试主机: {report.host}",
            f"开始时间: {report.started_at}",
            f"结束时间: {report.finished_at}",
            f"总耗时: {report.duration_sec:.1f} 秒",
            f"GPU 数量: {report.gpu_count} | 异常数量: {failed} | 告警数量: {len(report.warnings) + len(report.errors)}",
            f"运行配置: 温度阈值 {report.environment.get('max_temperature_c', 'N/A')}C | "
            f"Burn {report.environment.get('burn_seconds', 'N/A')}s | "
            f"NVLink {report.environment.get('nvlink_seconds', 'N/A')}s | {config_label}",
            f"输出目录: {output_dir}",
            "",
            "GPU 摘要:",
        ]
        for item in report.results:
            note_text = f" | {item.notes[0]}" if item.notes else ""
            burn_text = format_float(item.burn_gflops, 2)
            nvlink_text = format_float(item.nvlink_bandwidth_gbps, 2)
            crc_text = "N/A" if item.nvlink_crc_error_delta is None else f"+{item.nvlink_crc_error_delta}"
            peak_power_text = format_float(item.max_power_w, 1)
            peak_power_status, _ = self._peak_power_status(item.max_power_w)
            sb_before = self._format_counter(item.ecc_corrected_before)
            sb_after = self._format_counter(item.ecc_corrected_after)
            sb_delta = self._format_delta(item.ecc_corrected_delta)
            db_before = self._format_counter(item.ecc_uncorrected_before)
            db_after = self._format_counter(item.ecc_uncorrected_after)
            db_delta = self._format_delta(item.ecc_uncorrected_delta)
            nvlink_before = "?" if item.nvlink_link_count_before is None else str(item.nvlink_link_count_before)
            nvlink_after = "?" if item.nvlink_link_count_after is None else str(item.nvlink_link_count_after)
            lines.append(
                f"GPU {item.index} {display_result(assessment_code_for_gpu(item))} | ECC {display_ecc_mode(item.ecc_mode_current or 'Unknown')} | "
                f"NVLink通道 {nvlink_before}/{DEFAULT_NVLINK_EXPECTED_LINKS}->{nvlink_after}/{DEFAULT_NVLINK_EXPECTED_LINKS} | "
                f"单比特 {sb_before}->{sb_after} ({sb_delta}) | 双比特 {db_before}->{db_after} ({db_delta}) | "
                f"峰值功耗 {peak_power_text} W({peak_power_status}) | Burn {burn_text} GF/s | "
                f"Burn告警/错误 {item.burn_warning_count if item.burn_warning_count is not None else 'N/A'}/"
                f"{item.burn_error_count if item.burn_error_count is not None else 'N/A'} | "
                f"NVLink {nvlink_text} GB/s | CRC {crc_text}{note_text}"
            )
        lines.extend(
            [
                "",
                "摘要备份:",
            ]
        )
        for key in ("summary_txt", "summary_md", "summary_html", "summary_json", "summary_csv"):
            path = written.get(key)
            if path is not None:
                lines.append(f"- {path.name}")
        if report.warnings:
            lines.extend(["", "告警:"])
            lines.extend(f"- {item}" for item in report.warnings)
        if report.errors:
            lines.extend(["", "错误:"])
            lines.extend(f"- {item}" for item in report.errors)
        return lines

    def _handle_finished(self, event: dict[str, object]) -> None:
        report = event["report"]
        assert isinstance(report, RunReport)
        output_dir = Path(str(event["output_dir"]))
        self.latest_output_dir = output_dir
        self.latest_written = {
            name: Path(path)
            for name, path in (event.get("written") or {}).items()
            if isinstance(name, str) and isinstance(path, str)
        }
        self.cancel_event = None
        self.last_duration_sec = report.duration_sec
        self.result_code = overall_assessment_code(report)
        self.current_stage = "suite-finish"
        self.warning_count = len(report.warnings) + len(report.errors)
        self.fail_count = sum(1 for item in report.results if item.result in {"FAIL", "ERROR"})
        self.summary_lines = self._build_summary_lines(report, output_dir, self.latest_written)
        if report.artifacts.get("run_status") == "CANCELLED":
            self.current_stage = "suite-cancelled"
            self.status_text = f"检测已手动停止，结果已写入 {output_dir}"
            self._append_log("检测已手动停止")
        else:
            self.status_text = f"检测完成，结果已写入 {output_dir}"
        self._append_log(f"检测结束，整体结果: {display_result(self.result_code)}")
        self._append_log(f"结果目录: {output_dir}")

    def _handle_error(self, message: str) -> None:
        self.cancel_event = None
        self.result_code = "ERROR"
        self.current_stage = "suite-error"
        self.warning_count = 1
        self.fail_count = 1
        self.status_text = "检测过程中发生异常"
        self.summary_lines = ["检测异常终止，请查看日志。", "", *message.splitlines()[-12:]]
        self._append_log(message)

    def _drain_queue(self) -> None:
        while True:
            try:
                event = self.queue.get_nowait()
            except queue.Empty:
                return
            event_type = str(event.get("type"))
            if event_type == "progress":
                self._handle_progress(
                    str(event.get("stage")),
                    str(event.get("message")),
                    event.get("payload") if isinstance(event.get("payload"), dict) else {},
                )
            elif event_type == "finished":
                self._handle_finished(event)
            elif event_type == "error":
                self._handle_error(str(event.get("message")))

    def _prompt_text(self, screen: object, prompt: str, current_value: str) -> str | None:
        import curses

        height, width = screen.getmaxyx()
        prompt_y = max(height - 2, 0)
        input_y = height - 1
        prompt_line = shorten(f"{prompt} 当前值: {current_value}。留空表示不修改。", width)
        self._add_line(screen, prompt_y, 0, width, " " * width, 0)
        self._add_line(screen, input_y, 0, width, " " * width, 0)
        self._add_line(screen, prompt_y, 0, width, prompt_line, self._style("warning"))
        self._add_line(screen, input_y, 0, width, "> ", self._style("accent"))
        screen.refresh()

        curses.echo()
        try:
            curses.curs_set(1)
        except Exception:
            pass
        screen.nodelay(False)
        screen.timeout(-1)
        try:
            raw = screen.getstr(input_y, 2, max(width - 3, 1))
        finally:
            curses.noecho()
            try:
                curses.curs_set(0)
            except Exception:
                pass
            screen.nodelay(True)
            screen.timeout(150)
            self._add_line(screen, prompt_y, 0, width, " " * width, 0)
            self._add_line(screen, input_y, 0, width, " " * width, 0)
        value = raw.decode("utf-8", errors="ignore").strip()
        return value or None

    def _set_temperature_threshold(self, screen: object) -> None:
        raw = self._prompt_text(screen, "设置温度阈值(摄氏度)", f"{self.max_temp_c:.1f}")
        if raw is None:
            return
        try:
            value = float(raw)
        except ValueError:
            self._append_log(f"温度阈值输入无效: {raw}")
            return
        if value <= 0:
            self._append_log("温度阈值必须大于 0")
            return
        self.max_temp_c = value
        self._append_log(f"已更新温度阈值为 {self.max_temp_c:.1f}C")

    def _set_burn_seconds(self, screen: object) -> None:
        raw = self._prompt_text(screen, "设置 GPU-Burn 压测时间(秒)", str(self.burn_seconds))
        if raw is None:
            return
        try:
            value = int(raw)
        except ValueError:
            self._append_log(f"GPU-Burn 时长输入无效: {raw}")
            return
        if value <= 0:
            self._append_log("GPU-Burn 时长必须大于 0")
            return
        self.burn_seconds = value
        self._append_log(f"已更新 GPU-Burn 时长为 {self.burn_seconds} 秒")

    def _set_nvlink_seconds(self, screen: object) -> None:
        raw = self._prompt_text(screen, "设置 NVLink 压测时间(秒)", str(self.nvlink_seconds))
        if raw is None:
            return
        try:
            value = int(raw)
        except ValueError:
            self._append_log(f"NVLink 时长输入无效: {raw}")
            return
        if value <= 0:
            self._append_log("NVLink 时长必须大于 0")
            return
        self.nvlink_seconds = value
        self._append_log(f"已更新 NVLink 压测时长为 {self.nvlink_seconds} 秒")

    def _set_target_gpu(self, screen: object) -> None:
        raw = self._prompt_text(screen, "设置目标 GPU 编号，输入 all 表示全部", self._target_gpu_label())
        if raw is None:
            return
        lowered = raw.lower()
        if lowered in {"all", "全部", "a"}:
            self.target_gpu_index = None
            self._append_log("已切换为全部 GPU 压测")
            self._load_idle_snapshot(reset_display=True)
            return
        try:
            value = int(raw)
        except ValueError:
            self._append_log(f"目标 GPU 输入无效: {raw}")
            return
        discovered_gpus, discovery_warnings = discover_gpus()
        for warning in discovery_warnings:
            self._append_log(warning)
        available = sorted(gpu.index for gpu in discovered_gpus)
        if available and value not in available:
            self._append_log(f"当前未发现 GPU {value}，可选 GPU: {', '.join(str(item) for item in available)}")
            return
        self.target_gpu_index = value
        self._append_log(f"已切换为单卡压测模式，仅测试 GPU {value}")
        self._load_idle_snapshot(reset_display=True)

    def _handle_key(self, screen: object, key: int) -> bool:
        if key in (ord("q"), ord("Q")):
            if self._is_running():
                self.exit_after_finish = True
                self._request_stop()
                return False
            return True
        if key in (ord("r"), ord("R")):
            self._start_run()
            return False
        if key in (ord("s"), ord("S")):
            self._request_stop()
            return False
        if key in (ord("o"), ord("O")):
            if self.latest_output_dir is None:
                self._append_log("当前还没有可打开的结果目录")
            else:
                warning = open_path(self.latest_output_dir)
                self._append_log(warning or f"已尝试打开目录: {self.latest_output_dir}")
            return False
        if key in (ord("t"), ord("T"), ord("b"), ord("B"), ord("n"), ord("N"), ord("g"), ord("G")):
            if self._is_running():
                self._append_log("检测运行中，不能修改配置")
                return False
            if key in (ord("t"), ord("T")):
                self._set_temperature_threshold(screen)
            elif key in (ord("b"), ord("B")):
                self._set_burn_seconds(screen)
            elif key in (ord("n"), ord("N")):
                self._set_nvlink_seconds(screen)
            else:
                self._set_target_gpu(screen)
            return False
        return False

    def _style(self, name: str) -> int:
        return getattr(self, "colors", {}).get(name, 0)

    def _result_style(self, result_code: str) -> int:
        if result_code == "EXCELLENT" or result_code == "PASS":
            return self._style("success")
        if result_code == "GOOD":
            return self._style("warning")
        if result_code in {"FAIL", "ERROR"}:
            return self._style("danger")
        if result_code == "NOT_RUN":
            return self._style("warning")
        return self._style("accent")

    def _metric_style(self, value: float | None, warn: float, danger: float, *, inverse: bool = False) -> int:
        if value is None:
            return self._style("muted")
        if inverse:
            if value <= danger:
                return self._style("success")
            if value <= warn:
                return self._style("warning")
            return self._style("danger")
        if value >= danger:
            return self._style("danger")
        if value >= warn:
            return self._style("warning")
        return self._style("success")

    def _peak_power_status(self, value: float | None) -> tuple[str, int]:
        if value is None:
            return "N/A", self._style("muted")
        if value < 250.0:
            return "峰值过低", self._style("danger")
        if value < 280.0:
            return "考虑检查", self._style("warning")
        return "过测", self._style("success")

    def _format_meter_row(
        self,
        label: str,
        value: str,
        raw_value: float | None,
        maximum: float,
        width: int,
        *,
        badge: str | None = None,
        label_width: int = 12,
        value_width: int = 16,
    ) -> str:
        label_cell = pad_to_width(label, min(label_width, max(width // 5, 8)))
        value_cell = pad_to_width(value, min(value_width, max(width // 4, 10)))
        badge_text = f" {badge}" if badge else ""
        reserved = display_width(label_cell) + 1 + display_width(value_cell) + 1 + display_width(badge_text)
        bar_width = max(width - reserved, 8)
        bar = format_bar(raw_value, maximum, bar_width)
        return shorten(f"{label_cell} {value_cell} {bar}{badge_text}", width)

    def _add_line(self, screen: object, y: int, x: int, width: int, text: str, attr: int | None = None) -> None:
        if width <= 0:
            return
        safe = trim_to_width(text, width)
        try:
            screen.addstr(y, x, safe, attr or 0)
        except Exception:
            return

    def _draw_panel(
        self,
        screen: object,
        y: int,
        x: int,
        width: int,
        height: int,
        title: str,
        lines: list[str],
        *,
        border_attr: int | None = None,
        title_attr: int | None = None,
    ) -> None:
        if width < 12 or height < 4:
            return
        border = border_attr if border_attr is not None else self._style("panel")
        title_style = title_attr if title_attr is not None else border
        self._add_line(screen, y, x, width, f"{BOX['tl']}{BOX['h'] * (width - 2)}{BOX['tr']}", border)
        self._add_line(screen, y, x + 1, max(width - 2, 0), shorten(f" {title} ", max(width - 2, 0)), title_style)
        for row in range(1, height - 1):
            self._add_line(screen, y + row, x, 1, BOX["v"], border)
            self._add_line(screen, y + row, x + width - 1, 1, BOX["v"], border)
            self._add_line(screen, y + row, x + 1, width - 2, " " * max(width - 2, 0), 0)
        self._add_line(screen, y + height - 1, x, width, f"{BOX['bl']}{BOX['h'] * (width - 2)}{BOX['br']}", border)
        inner_width = width - 4
        line_cursor = y + 1
        expanded: list[str] = []
        for item in lines:
            wrapped = textwrap.wrap(item, width=max(inner_width, 10), break_long_words=False) or [""]
            expanded.extend(wrapped)
        for index in range(min(len(expanded), height - 2)):
            attr = title_attr if index == 0 and title_attr is not None else self._style("panel")
            self._add_line(
                screen,
                line_cursor + index,
                x + 2,
                inner_width + 1,
                pad_to_width(expanded[index], inner_width),
                attr,
            )

    def _draw_stat_box(
        self,
        screen: object,
        y: int,
        x: int,
        width: int,
        title: str,
        value: str,
        detail: str,
        *,
        value_attr: int | None = None,
    ) -> None:
        self._draw_panel(
            screen,
            y,
            x,
            width,
            5,
            title,
            [value, detail],
            border_attr=self._style("accent"),
            title_attr=value_attr or self._style("title"),
        )

    def _gpu_layout_mode(self, gpu_count: int, usable_width: int) -> str:
        if gpu_count <= 2:
            return "full"
        if gpu_count == 4:
            return "dense-2col" if usable_width >= 108 else "list"
        if gpu_count in {3, 5, 6, 7}:
            return "dense-3col" if usable_width >= 150 else "dense-2col"
        return "list"

    def _build_gpu_card_rows(
        self,
        state: LiveGPUState,
        inner_width: int,
        *,
        dense: bool,
    ) -> tuple[list[str], list[int]]:
        memory_ratio = (
            (state.memory_used_mib or 0.0) / state.memory_total_mib * 100.0
            if state.memory_total_mib
            else None
        )
        peak_power_badge, peak_power_style = self._peak_power_status(state.peak_power_value)
        if dense:
            rows = [
                shorten(f"{state.model} | 序列号 {state.serial}", inner_width),
                shorten(
                    f"ECC {state.ecc} | NVLink通道 {state.nvlink_links_before} -> {state.nvlink_links_after}",
                    inner_width,
                ),
                shorten(
                    f"温 {state.temperature} | 当前功耗 {state.current_power} | 峰值 {state.peak_power} {peak_power_badge} | 利用率 {state.utilization}",
                    inner_width,
                ),
                shorten(
                    f"显存 {state.memory} | SM {state.sm_clock} | MEM {state.mem_clock}",
                    inner_width,
                ),
                shorten(
                    f"GPU-Burn {state.burn} | 告警 {state.burn_warning_count} | 错误 {state.burn_error_count}",
                    inner_width,
                ),
                shorten(
                    f"ECC单 {state.ecc_single_before}->{state.ecc_single_after} ({state.ecc_single_delta}) | 双 {state.ecc_double_before}->{state.ecc_double_after} ({state.ecc_double_delta})",
                    inner_width,
                ),
                shorten(
                    f"NVLink {state.nvlink} | CRC {state.crc_errors} | {state.note or '备注 暂无'}",
                    inner_width,
                ),
            ]
            styles = [
                self._style("panel"),
                self._style("warning")
                if "未开启" in state.ecc or state.nvlink_links_after not in {"N/A", f"{DEFAULT_NVLINK_EXPECTED_LINKS}/{DEFAULT_NVLINK_EXPECTED_LINKS}"}
                else self._style("muted"),
                peak_power_style if state.temperature_value is None else self._metric_style(state.temperature_value, 72.0, self.max_temp_c),
                self._style("info"),
                self._style("danger")
                if state.burn_warning_count not in {"N/A", "0"} or state.burn_error_count not in {"N/A", "0"}
                else self._style("accent"),
                self._style("warning")
                if state.ecc_single_delta not in {"N/A", "+0"} or state.ecc_double_delta not in {"N/A", "+0"}
                else self._style("info"),
                self._style("danger")
                if state.crc_errors not in {"N/A", "+0"}
                else (self._style("warning") if state.note else self._style("muted")),
            ]
            return rows, styles

        rows = [
            shorten(state.model, inner_width),
            shorten(f"序列号 {state.serial} | ECC {state.ecc}", inner_width),
            shorten(f"PCI {state.pci_bus_id} | UUID {state.uuid}", inner_width),
            self._format_meter_row(
                "温度",
                state.temperature,
                state.temperature_value,
                self.max_temp_c,
                inner_width,
            ),
            self._format_meter_row(
                "当前功耗",
                state.current_power,
                state.current_power_value,
                300.0,
                inner_width,
            ),
            self._format_meter_row(
                "GPU峰值功耗",
                state.peak_power,
                state.peak_power_value,
                300.0,
                inner_width,
                badge=peak_power_badge,
            ),
            self._format_meter_row(
                "利用率",
                state.utilization,
                state.utilization_value,
                100.0,
                inner_width,
            ),
            self._format_meter_row(
                "显存",
                state.memory,
                state.memory_used_mib,
                state.memory_total_mib or 1.0,
                inner_width,
            ),
            shorten(
                f"显存占用 {format_float(memory_ratio, 1)} % | SM {state.sm_clock} | MEM {state.mem_clock}",
                inner_width,
            ),
            shorten(f"GPU-Burn GF/s {state.burn}", inner_width),
            shorten(
                f"GPU-Burn告警 {state.burn_warning_count} | 错误 {state.burn_error_count}",
                inner_width,
            ),
            shorten(
                f"ECC单比特 {state.ecc_single_before} -> {state.ecc_single_after} (Δ {state.ecc_single_delta})",
                inner_width,
            ),
            shorten(
                f"ECC双比特 {state.ecc_double_before} -> {state.ecc_double_after} (Δ {state.ecc_double_delta})",
                inner_width,
            ),
            shorten(
                f"NVLink通道 {state.nvlink_links_before} -> {state.nvlink_links_after}",
                inner_width,
            ),
            shorten(f"NVLink {state.nvlink} | CRC {state.crc_errors}", inner_width),
            shorten(
                f"趋势 T {format_sparkline(state.temperature_history, 10)}  U {format_sparkline(state.utilization_history, 10)}  "
                f"P {format_sparkline(state.power_history, 10)}  B {format_sparkline(state.burn_history, 10)}",
                inner_width,
            ),
            shorten(state.note or "备注 暂无", inner_width),
        ]
        styles = [
            self._style("panel"),
            self._style("muted") if "未开启" not in state.ecc else self._style("warning"),
            self._style("muted"),
            self._metric_style(state.temperature_value, 72.0, self.max_temp_c),
            self._metric_style(state.current_power_value, 220.0, 260.0),
            peak_power_style,
            self._metric_style(state.utilization_value, 60.0, 90.0, inverse=True),
            self._style("info"),
            self._style("info"),
            self._style("accent"),
            self._style("danger")
            if state.burn_warning_count not in {"N/A", "0"} or state.burn_error_count not in {"N/A", "0"}
            else self._style("info"),
            self._style("warning") if state.ecc_single_delta not in {"N/A", "+0"} else self._style("info"),
            self._style("warning") if state.ecc_double_delta not in {"N/A", "+0"} else self._style("info"),
            self._style("warning")
            if state.nvlink_links_after not in {"N/A", f"{DEFAULT_NVLINK_EXPECTED_LINKS}/{DEFAULT_NVLINK_EXPECTED_LINKS}"}
            else self._style("info"),
            self._style("info") if state.crc_errors in {"N/A", "+0"} else self._style("danger"),
            self._style("muted"),
            self._style("warning") if state.note else self._style("muted"),
        ]
        return rows, styles

    def _gpu_single_line(self, state: LiveGPUState, width: int) -> str:
        peak_power_badge, _ = self._peak_power_status(state.peak_power_value)
        peak_text = state.peak_power if peak_power_badge == "N/A" else f"{state.peak_power} {peak_power_badge}"
        return shorten(
            (
                f"GPU {state.index} {state.result} | 温 {state.temperature} | 峰值 {peak_text} | 利 {state.utilization} | "
                f"显存 {state.memory} | Burn {state.burn} | NVL {state.nvlink_links_after} | CRC {state.crc_errors} | "
                f"ECCΔ {state.ecc_single_delta}/{state.ecc_double_delta}"
            ),
            width,
        )

    def _draw_gpu_line_panel(
        self,
        screen: object,
        y: int,
        x: int,
        width: int,
        height: int,
        states: list[LiveGPUState],
    ) -> None:
        self._draw_panel(
            screen,
            y,
            x,
            width,
            height,
            "GPU 实时状态",
            [],
            border_attr=self._style("accent"),
            title_attr=self._style("accent"),
        )
        inner_width = width - 4
        visible_rows = max(height - 2, 0)
        if visible_rows <= 0:
            return
        overflow = len(states) - visible_rows
        display_states = states[:visible_rows]
        if overflow > 0 and visible_rows >= 1:
            display_states = states[: visible_rows - 1]
        for row_index, state in enumerate(display_states):
            self._add_line(
                screen,
                y + 1 + row_index,
                x + 2,
                inner_width + 1,
                self._gpu_single_line(state, inner_width),
                self._result_style(state.result_code),
            )
        if overflow > 0 and visible_rows >= 1:
            self._add_line(
                screen,
                y + height - 2,
                x + 2,
                inner_width + 1,
                shorten(f"... 还有 {overflow} 张 GPU 未显示，请放大终端窗口查看。", inner_width),
                self._style("warning"),
            )

    def _draw_gpu_card(
        self,
        screen: object,
        y: int,
        x: int,
        width: int,
        height: int,
        state: LiveGPUState,
        *,
        dense: bool = False,
    ) -> None:
        title = f"GPU {state.index}  {state.result}"
        self._draw_panel(
            screen,
            y,
            x,
            width,
            height,
            title,
            [],
            border_attr=self._result_style(state.result_code),
            title_attr=self._result_style(state.result_code),
        )
        inner_width = width - 4
        rows, styles = self._build_gpu_card_rows(state, inner_width, dense=dense)
        for index, row in enumerate(rows[: max(height - 2, 0)]):
            self._add_line(screen, y + 1 + index, x + 2, inner_width + 1, row, styles[index])

    def _draw_compact(self, screen: object, width: int, height: int, duration: float, stage_label: str) -> None:
        y = 0
        lines = [
            "1Cat-V100-QA 文字检测界面",
            f"结果 {display_result(self.result_code)} | 阶段 {stage_label} | 用时 {duration:.1f}s",
            shorten(
                f"配置 温度 {self.max_temp_c:.1f}C | Burn {self.burn_seconds}s | NVLink {self.nvlink_seconds}s | {self._target_gpu_label()}",
                width,
            ),
            shorten(f"状态 {self.status_text}", width),
            "快捷键 R开始 S停止 T温度 B时长 N链路 G目标GPU O打开结果 Q退出",
        ]
        for line in lines:
            if y >= height:
                break
            self._add_line(screen, y, 0, width, line, self._style("header") if y == 0 else self._style("panel"))
            y += 1
        for gpu_index in sorted(self.gpus):
            if y >= height:
                break
            state = self.gpus[gpu_index]
            self._add_line(
                screen,
                y,
                0,
                width,
                shorten(
                    f"GPU {state.index} {state.result} | NVLink通道 {state.nvlink_links_after} | 单比特 {state.ecc_single_after} ({state.ecc_single_delta}) | 双比特 {state.ecc_double_after} ({state.ecc_double_delta})",
                    width,
                ),
                self._result_style(state.result_code),
            )
            y += 1
        for line in (self.gpu_burn_lines[-2:] or self.logs[-2:] or SUMMARY_PLACEHOLDER):
            if y >= height:
                break
            self._add_line(screen, y, 0, width, shorten(line, width), self._style("muted"))
            y += 1

    def _draw(self, screen: object) -> None:
        screen.erase()
        height, width = screen.getmaxyx()
        duration = self.last_duration_sec
        if self.started_at_monotonic is not None and self._is_running():
            duration = time.monotonic() - self.started_at_monotonic
        spinner = SPINNER_FRAMES[int(time.monotonic() * 8) % len(SPINNER_FRAMES)] if self._is_running() else "■"
        stage_label = STAGE_LABELS.get(self.current_stage, self.current_stage)
        now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if width < 84 or height < 18:
            self._draw_compact(screen, width, height, duration, stage_label)
            screen.refresh()
            return

        left = 1
        usable_width = max(width - 2, 20)

        title = f"{spinner}  1Cat-V100-QA  V100 / CUDA 12 专项质检"
        right = f"{display_result(self.result_code)}  {now_text}"
        right_width = display_width(right)
        self._add_line(
            screen,
            0,
            left,
            max(usable_width - right_width - 2, 0),
            title,
            self._style("header"),
        )
        self._add_line(
            screen,
            0,
            max(width - right_width - 2, left),
            right_width + 1,
            right,
            self._result_style(self.result_code),
        )
        self._add_line(
            screen,
            1,
            left,
            usable_width,
            shorten(
                f"Host {self.host} | 阶段 {stage_label} | 状态 {self.status_text} | 快捷键 R 开始 S 停止 T 温度 B Burn N NVLink G GPU O 结果 Q 退出",
                usable_width,
            ),
            self._style("muted"),
        )

        stats_y = 3
        stat_gap = 2
        stat_columns = 4 if usable_width >= 148 else 2
        stat_width = (usable_width - stat_gap * (stat_columns - 1)) // stat_columns
        stat_items = [
            ("总体结果", display_result(self.result_code), f"异常 {self.fail_count} | 告警 {self.warning_count}", self._result_style(self.result_code)),
            ("当前阶段", stage_label, shorten(self.status_text, stat_width - 4), self._style("accent")),
            ("运行时间", f"{duration:.1f} s", f"GPU {len(self.gpus)} | 输出 {shorten(str(self.output_dir.name), max(stat_width - 14, 8))}", self._style("info")),
            ("运行配置", self._target_gpu_label(), f"温度 {self.max_temp_c:.1f}C | Burn {self.burn_seconds}s | NVLink {self.nvlink_seconds}s", self._style("title")),
        ]
        for index, (title_text, value, detail, attr) in enumerate(stat_items[:stat_columns]):
            x = left + index * (stat_width + stat_gap)
            self._draw_stat_box(screen, stats_y, x, stat_width, title_text, value, detail, value_attr=attr)
        if stat_columns == 2:
            for index, (title_text, value, detail, attr) in enumerate(stat_items[2:], start=0):
                x = left + index * (stat_width + stat_gap)
                self._draw_stat_box(screen, stats_y + 6, x, stat_width, title_text, value, detail, value_attr=attr)
            cards_y = stats_y + 12
        else:
            cards_y = stats_y + 6

        gpu_states = [self.gpus[index] for index in sorted(self.gpus)]
        if not gpu_states:
            gpu_states = [LiveGPUState(index=0, result="等待检测", result_code="IDLE", note="等待开始执行样例")]

        gpu_layout_mode = self._gpu_layout_mode(len(gpu_states), usable_width)
        if gpu_layout_mode == "list":
            gpu_panel_height = min(
                max(len(gpu_states) + 2, 6),
                max(height - cards_y - 8, 6),
            )
            self._draw_gpu_line_panel(screen, cards_y, left, usable_width, gpu_panel_height, gpu_states)
            lower_y = cards_y + gpu_panel_height + 1
        else:
            dense = gpu_layout_mode != "full"
            if gpu_layout_mode == "dense-3col":
                columns = 3
            elif gpu_layout_mode == "dense-2col":
                columns = 2 if len(gpu_states) > 1 and usable_width >= 108 else 1
            else:
                columns = 2 if len(gpu_states) > 1 and usable_width >= 108 else 1
            gutter = 2
            card_width = (usable_width - gutter * (columns - 1)) // columns
            if gpu_layout_mode == "dense-3col":
                card_height = 8
            elif dense:
                card_height = 9
            else:
                card_height = 19 if usable_width >= 120 else 18
            for position, gpu_state in enumerate(gpu_states):
                row = position // columns
                column = position % columns
                x = left + column * (card_width + gutter)
                y = cards_y + row * (card_height + 1)
                self._draw_gpu_card(screen, y, x, card_width, card_height, gpu_state, dense=dense)

            cards_rows = (len(gpu_states) - 1) // columns + 1
            lower_y = cards_y + cards_rows * (card_height + 1)
        remaining_height = height - lower_y - 1
        if remaining_height >= 6:
            summary_lines = self.summary_lines or SUMMARY_PLACEHOLDER
            log_lines = self.logs[-max(remaining_height - 2, 1):] or ["暂无日志"]
            burn_lines = self.gpu_burn_lines[-max(remaining_height - 2, 1):] or ["等待 GPU-Burn 输出"]
            info_lines = [
                f"输出目录: {self.output_dir}",
                f"当前阶段: {stage_label}",
                f"主机: {self.host}",
                f"等待驱动上限: {self.wait_driver_seconds} s",
                f"温度阈值: {self.max_temp_c:.1f} C",
                f"GPU-Burn 时长: {self.burn_seconds} s",
                f"NVLink 时长: {self.nvlink_seconds} s",
                f"目标 GPU: {self._target_gpu_label()}",
                "快捷键: T 温度 | B Burn | N NVLink | G GPU",
            ]
            if self.latest_written:
                info_lines.append("备份文件:")
                info_lines.extend(f"- {path.name}" for path in self.latest_written.values())
            if usable_width >= 180:
                summary_width = max(usable_width * 7 // 24, 38)
                log_width = max(usable_width * 6 // 24, 32)
                burn_width = max(usable_width * 6 // 24, 32)
                info_width = usable_width - summary_width - log_width - burn_width - 6
                self._draw_panel(screen, lower_y, left, summary_width, remaining_height, "检测结论", summary_lines, border_attr=self._style("success"))
                self._draw_panel(
                    screen,
                    lower_y,
                    left + summary_width + 2,
                    log_width,
                    remaining_height,
                    "事件日志",
                    log_lines,
                    border_attr=self._style("accent"),
                )
                self._draw_panel(
                    screen,
                    lower_y,
                    left + summary_width + log_width + 4,
                    burn_width,
                    remaining_height,
                    "GPU-Burn 实时输出",
                    burn_lines,
                    border_attr=self._style("warning"),
                )
                self._draw_panel(
                    screen,
                    lower_y,
                    left + summary_width + log_width + burn_width + 6,
                    info_width,
                    remaining_height,
                    "运行信息",
                    info_lines,
                    border_attr=self._style("info"),
                )
            elif usable_width >= 120 and remaining_height >= 12:
                top_height = max(remaining_height // 2, 6)
                bottom_height = remaining_height - top_height - 1
                left_width = (usable_width - 2) // 2
                right_width = usable_width - left_width - 2
                self._draw_panel(screen, lower_y, left, left_width, top_height, "检测结论", summary_lines, border_attr=self._style("success"))
                self._draw_panel(screen, lower_y, left + left_width + 2, right_width, top_height, "GPU-Burn 实时输出", burn_lines, border_attr=self._style("warning"))
                if bottom_height >= 4:
                    self._draw_panel(screen, lower_y + top_height + 1, left, left_width, bottom_height, "事件日志", log_lines, border_attr=self._style("accent"))
                    self._draw_panel(screen, lower_y + top_height + 1, left + left_width + 2, right_width, bottom_height, "运行信息", info_lines, border_attr=self._style("info"))
            else:
                summary_width = max(usable_width // 2, 28)
                log_width = usable_width - summary_width - 2
                self._draw_panel(screen, lower_y, left, summary_width, remaining_height, "检测结论", summary_lines, border_attr=self._style("success"))
                self._draw_panel(screen, lower_y, left + summary_width + 2, log_width, remaining_height, "GPU-Burn 实时输出", burn_lines, border_attr=self._style("warning"))

        screen.refresh()


def main() -> int:
    return GPUQATextApp().run()
