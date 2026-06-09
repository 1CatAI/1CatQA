from __future__ import annotations

from datetime import datetime
from pathlib import Path
import queue
import threading
import traceback

from gpuqa.cli import build_default_output_dir
from gpuqa.core import run_suite
from gpuqa.desktop import launch_nvtop, open_path
from gpuqa.models import GPUResult, RunReport, assessment_code_for_gpu, display_assessment, overall_assessment_code
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
ECC_MODE_LABELS = {"Enabled": "已开启", "Disabled": "未开启", "Unknown": "未知"}
STATE_STYLES = {
    "IDLE": ("空闲", "#e8ddd0", "#5c4f43"),
    "RUNNING": ("检测中", "#f4e5a5", "#7c5e10"),
    "EXCELLENT": ("优秀", "#d9f3df", "#166534"),
    "GOOD": ("良好", "#f7edc7", "#8a5a13"),
    "PASS": ("通过", "#d9f3df", "#166534"),
    "FAIL": ("不通过", "#f8d9df", "#9f1239"),
    "ERROR": ("错误", "#fee2d5", "#c2410c"),
    "NOT_RUN": ("未运行", "#ece4d8", "#6b5b46"),
}
SUMMARY_FILE_KEYS = ("summary_txt", "summary_md", "summary_html", "summary_json", "summary_csv")
SUMMARY_PLACEHOLDER = "检测完成后，这里会直接显示整体结论、每张 GPU 的结果，以及 summary 备份文件路径。"
WINDOW_SIZE_PRESETS = {
    "紧凑": "1180x760",
    "标准": "1320x860",
    "宽屏": "1480x920",
    "超宽": "1850x1080",
}
DEFAULT_WINDOW_SIZE = "超宽"


def display_result(value: str) -> str:
    return RESULT_LABELS.get(value, display_assessment(value))


def display_ecc_mode(value: str) -> str:
    return ECC_MODE_LABELS.get(value, value)


def format_float(value: float | None, digits: int = 2, fallback: str = "N/A") -> str:
    return fallback if value is None else f"{value:.{digits}f}"


def summarize_counts(report: RunReport) -> dict[str, int]:
    counts = {key: 0 for key in RESULT_LABELS}
    for item in report.results:
        counts[item.result] = counts.get(item.result, 0) + 1
    return counts


def build_backup_label(written: dict[str, Path]) -> str:
    labels = [key.removeprefix("summary_").upper() for key in SUMMARY_FILE_KEYS if key in written]
    return "等待生成" if not labels else " / ".join(labels)


def describe_ecc_mode(result: GPUResult) -> str:
    current_raw = result.ecc_mode_current or "Unknown"
    pending_raw = result.ecc_mode_pending or current_raw
    current = display_ecc_mode(current_raw)
    pending = display_ecc_mode(pending_raw)
    return current if pending == current else f"{current} -> {pending}"


def build_gpu_summary_line(result: GPUResult) -> list[str]:
    assessment_code = assessment_code_for_gpu(result)
    lines = [
        f"GPU {result.index} | {result.name or '未知型号'} | {display_result(assessment_code)}",
        f"  序列号: {result.serial or 'N/A'}",
        f"  ECC: {describe_ecc_mode(result)} | 温度: {format_float(result.avg_temp_c, 1)} / {format_float(result.max_temp_c, 1)} C",
        f"  功耗: {format_float(result.avg_power_w, 1)} / {format_float(result.max_power_w, 1)} W | 利用率: {format_float(result.avg_utilization_gpu_pct, 1)} %",
        f"  NVLink: {format_float(result.nvlink_bandwidth_gbps, 2)} GB/s | 错误增量: {result.nvlink_error_delta if result.nvlink_error_delta is not None else 'N/A'}",
        f"  GPU-Burn 算力: {format_float(result.burn_gflops, 2)} GF/s | ECC 计数: {result.ecc_corrected_delta if result.ecc_corrected_delta is not None else 'N/A'} / {result.ecc_uncorrected_delta if result.ecc_uncorrected_delta is not None else 'N/A'}",
    ]
    if result.notes:
        lines.append(f"  备注: {'；'.join(result.notes)}")
    return lines


def build_gui_summary(report: RunReport, output_dir: Path, written: dict[str, Path]) -> str:
    counts = summarize_counts(report)
    failed = counts.get("FAIL", 0) + counts.get("ERROR", 0)
    overall_code = overall_assessment_code(report)
    lines = [
        f"总体结果: {display_result(overall_code)}",
        f"测试主机: {report.host}",
        f"开始时间: {report.started_at}",
        f"结束时间: {report.finished_at}",
        f"总耗时: {report.duration_sec:.1f} 秒",
        f"GPU 概览: {report.gpu_count} 张 | 通过 {counts.get('PASS', 0)} | 失败 {failed} | 未运行 {counts.get('NOT_RUN', 0)}",
        f"驱动就绪: {'是' if report.driver_ready else '否'}",
        f"输出目录: {output_dir}",
        "",
        "摘要备份:",
    ]
    for key in SUMMARY_FILE_KEYS:
        path = written.get(key)
        if path is not None:
            lines.append(f"  - {path.name}: {path}")
    if report.warnings:
        lines.extend(["", "告警:"])
        lines.extend(f"  - {item}" for item in report.warnings)
    if report.errors:
        lines.extend(["", "错误:"])
        lines.extend(f"  - {item}" for item in report.errors)
    if report.results:
        lines.extend(["", "GPU 结果:"])
        for item in report.results:
            lines.extend(build_gpu_summary_line(item))
            lines.append("")
    return "\n".join(lines).rstrip()


class GPUQAGuiApp:
    def __init__(self, root: "Tk") -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.queue: queue.Queue[dict[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event: threading.Event | None = None
        self.latest_output_dir: Path | None = None
        self.latest_written: dict[str, Path] = {}
        self.row_ids: dict[int, str] = {}
        self.gpu_cards: dict[int, dict[str, object]] = {}

        self.output_dir_var = tk.StringVar(value=str(build_default_output_dir()))
        self.burn_seconds_var = tk.StringVar(value="600")
        self.sample_interval_var = tk.StringVar(value="5.0")
        self.wait_driver_var = tk.StringVar(value="180")
        self.max_temp_var = tk.StringVar(value="80.0")
        self.auto_nvtop_var = tk.BooleanVar(value=True)
        self.window_size_var = tk.StringVar(value=DEFAULT_WINDOW_SIZE)
        self.status_var = tk.StringVar(value="空闲，等待开始检测")
        self.result_var = tk.StringVar(value="尚未开始")
        self.gpu_count_var = tk.StringVar(value="0")
        self.fail_count_var = tk.StringVar(value="0")
        self.warning_count_var = tk.StringVar(value="0")
        self.duration_var = tk.StringVar(value="--")
        self.backup_var = tk.StringVar(value="摘要备份: 等待生成")

        self.start_button = None
        self.stop_button = None
        self.browse_button = None
        self.open_button = None
        self.progress = None
        self.tree = None
        self.gpu_cards_frame = None
        self.log_text = None
        self.summary_text = None
        self.result_badge_label = None

        self._configure_window()
        self._build_layout()
        self._set_state_visual("IDLE")
        self._set_summary_text(SUMMARY_PLACEHOLDER)
        self.root.after(120, self._drain_queue)

    def _configure_window(self) -> None:
        style = self.ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self.root.title("1Cat-V100-QA")
        self.root.geometry(WINDOW_SIZE_PRESETS[DEFAULT_WINDOW_SIZE])
        self.root.minsize(1100, 760)
        self.root.resizable(True, True)
        self.root.configure(bg="#efe6d9")
        style.configure("App.TFrame", background="#efe6d9")
        style.configure("Card.TFrame", background="#fffaf2")
        style.configure("Section.TLabelframe", background="#fffaf2", borderwidth=1)
        style.configure("Section.TLabelframe.Label", background="#fffaf2", foreground="#231f1a", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(16, 10))
        style.configure("Ghost.TButton", font=("Segoe UI", 10), padding=(14, 10))
        style.configure("Treeview", background="#fffaf2", fieldbackground="#fffaf2", foreground="#231f1a", rowheight=30)
        style.configure("Treeview.Heading", background="#efe4d3", foreground="#1f1b16", relief="flat", font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        tk = self.tk
        ttk = self.ttk

        outer = ttk.Frame(self.root, padding=18, style="App.TFrame")
        outer.pack(fill=tk.BOTH, expand=True)

        hero = tk.Frame(outer, bg="#112a24", padx=26, pady=24)
        hero.pack(fill=tk.X)
        hero.grid_columnconfigure(0, weight=1)
        tk.Label(hero, text="1Cat-V100-QA", bg="#112a24", fg="#f6f2eb", font=("Segoe UI", 24, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(
            hero,
            text="面向 V100 的 CUDA 12 专项质检工具，结果会直接显示在界面中，并同步写入 summary 备份。",
            bg="#112a24",
            fg="#c7ddd6",
            font=("Segoe UI", 11),
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.result_badge_label = tk.Label(hero, text="空闲", bg="#e8ddd0", fg="#5c4f43", font=("Segoe UI", 11, "bold"), padx=16, pady=8)
        self.result_badge_label.grid(row=0, column=1, rowspan=2, sticky="e")

        stats = tk.Frame(outer, bg="#efe6d9")
        stats.pack(fill=tk.X, pady=(14, 14))
        stats.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        self._build_stat_card(stats, 0, "总体状态", self.result_var, "#f7efe2", "#3b342d")
        self._build_stat_card(stats, 1, "GPU 数量", self.gpu_count_var, "#edf5ef", "#214b36")
        self._build_stat_card(stats, 2, "异常数量", self.fail_count_var, "#fbe4e9", "#8a2240")
        self._build_stat_card(stats, 3, "告警数量", self.warning_count_var, "#fff2d8", "#7c5e10")
        self._build_stat_card(stats, 4, "测试耗时", self.duration_var, "#e3f0f8", "#15506f")

        controls = ttk.LabelFrame(outer, text="运行设置", padding=16, style="Section.TLabelframe")
        controls.pack(fill=tk.X, pady=(0, 14))
        for column in range(4):
            controls.columnconfigure(column, weight=1 if column in {1, 3} else 0)

        ttk.Label(controls, text="输出目录").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=6)
        ttk.Entry(controls, textvariable=self.output_dir_var).grid(row=0, column=1, sticky=tk.EW, pady=6)
        self.browse_button = ttk.Button(controls, text="浏览", command=self._choose_output_dir, style="Ghost.TButton")
        self.browse_button.grid(row=0, column=2, sticky=tk.E, padx=(10, 0), pady=6)

        ttk.Label(controls, text="压测时长").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=6)
        ttk.Entry(controls, textvariable=self.burn_seconds_var, width=12).grid(row=1, column=1, sticky=tk.W, pady=6)
        ttk.Label(controls, text="采样间隔").grid(row=1, column=2, sticky=tk.W, padx=(16, 10), pady=6)
        ttk.Entry(controls, textvariable=self.sample_interval_var, width=12).grid(row=1, column=3, sticky=tk.W, pady=6)

        ttk.Label(controls, text="驱动等待时长").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=6)
        ttk.Entry(controls, textvariable=self.wait_driver_var, width=12).grid(row=2, column=1, sticky=tk.W, pady=6)
        ttk.Label(controls, text="最高温度阈值").grid(row=2, column=2, sticky=tk.W, padx=(16, 10), pady=6)
        ttk.Entry(controls, textvariable=self.max_temp_var, width=12).grid(row=2, column=3, sticky=tk.W, pady=6)

        ttk.Checkbutton(controls, text="压测期间自动打开 nvtop", variable=self.auto_nvtop_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(10, 4))
        ttk.Label(controls, text="界面大小").grid(row=3, column=2, sticky=tk.W, padx=(16, 10), pady=(10, 4))
        size_picker = ttk.Combobox(controls, textvariable=self.window_size_var, values=list(WINDOW_SIZE_PRESETS), state="readonly", width=10)
        size_picker.grid(row=3, column=3, sticky=tk.W, pady=(10, 4))
        size_picker.bind("<<ComboboxSelected>>", self._apply_window_size)
        actions = ttk.Frame(controls, style="Card.TFrame")
        actions.grid(row=4, column=0, columnspan=4, sticky=tk.EW, pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(actions, text="开始检测", command=self._start_run, style="Accent.TButton")
        self.start_button.grid(row=0, column=1, padx=(10, 0))
        self.stop_button = ttk.Button(actions, text="停止检测", command=self._stop_run, style="Ghost.TButton")
        self.stop_button.grid(row=0, column=2, padx=(10, 0))
        self.stop_button.configure(state=tk.DISABLED)
        self.open_button = ttk.Button(actions, text="打开结果目录", command=self._open_latest_output, style="Ghost.TButton")
        self.open_button.grid(row=0, column=3, padx=(10, 0))
        ttk.Button(actions, text="应用尺寸", command=self._apply_window_size, style="Ghost.TButton").grid(row=0, column=4, padx=(10, 0))

        status_card = ttk.LabelFrame(outer, text="运行状态", padding=16, style="Section.TLabelframe")
        status_card.pack(fill=tk.X, pady=(0, 14))
        status_card.columnconfigure(0, weight=1)
        ttk.Label(status_card, textvariable=self.status_var, font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_card, textvariable=self.backup_var, foreground="#6d6254").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.progress = ttk.Progressbar(status_card, mode="indeterminate")
        self.progress.grid(row=2, column=0, sticky=tk.EW, pady=(12, 0))

        spotlight = ttk.LabelFrame(outer, text="显卡实时参数", padding=12, style="Section.TLabelframe")
        spotlight.pack(fill=tk.X, pady=(0, 14))
        self.gpu_cards_frame = tk.Frame(spotlight, bg="#efe6d9")
        self.gpu_cards_frame.pack(fill=tk.X, expand=True)

        content = ttk.Frame(outer, style="App.TFrame")
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        live_card = ttk.LabelFrame(content, text="实时 GPU 状态", padding=12, style="Section.TLabelframe")
        live_card.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 10))
        live_card.rowconfigure(0, weight=1)
        live_card.columnconfigure(0, weight=1)
        columns = ("gpu", "model", "serial", "ecc", "temp", "power", "util", "memory", "burn", "result")
        self.tree = ttk.Treeview(live_card, columns=columns, show="headings", height=12)
        headings = {
            "gpu": "GPU",
            "model": "型号",
            "serial": "序列号",
            "ecc": "ECC",
            "temp": "温度 C",
            "power": "功耗 W",
            "util": "利用率 %",
            "memory": "显存 MiB",
            "burn": "GPU-Burn GF/s",
            "result": "结果",
        }
        widths = {"gpu": 56, "model": 200, "serial": 170, "ecc": 150, "temp": 88, "power": 88, "util": 84, "memory": 124, "burn": 108, "result": 92}
        for name in columns:
            self.tree.heading(name, text=headings[name])
            self.tree.column(name, width=widths[name], anchor=tk.CENTER if name != "model" else tk.W)
        self.tree.tag_configure("RUNNING", background="#fff7e5")
        self.tree.tag_configure("EXCELLENT", background="#eefaf1", foreground="#166534")
        self.tree.tag_configure("GOOD", background="#fff6dd", foreground="#8a5a13")
        self.tree.tag_configure("PASS", background="#eefaf1", foreground="#166534")
        self.tree.tag_configure("FAIL", background="#fdecef", foreground="#9f1239")
        self.tree.tag_configure("ERROR", background="#fff0e8", foreground="#c2410c")
        self.tree.tag_configure("NOT_RUN", background="#f5f1ea", foreground="#6b5b46")
        tree_scroll = ttk.Scrollbar(live_card, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky=tk.NSEW)
        tree_scroll.grid(row=0, column=1, sticky=tk.NS)

        sidebar = ttk.Frame(content, style="App.TFrame")
        sidebar.grid(row=0, column=1, sticky=tk.NSEW)
        sidebar.rowconfigure(0, weight=2)
        sidebar.rowconfigure(1, weight=1)
        sidebar.columnconfigure(0, weight=1)

        summary_card = ttk.LabelFrame(sidebar, text="检测结论", padding=12, style="Section.TLabelframe")
        summary_card.grid(row=0, column=0, sticky=tk.NSEW, pady=(0, 10))
        summary_card.rowconfigure(0, weight=1)
        summary_card.columnconfigure(0, weight=1)
        self.summary_text = tk.Text(summary_card, wrap="word", bg="#fbf7ef", fg="#231f1a", relief="flat", font=("Consolas", 10), padx=10, pady=10)
        summary_scroll = ttk.Scrollbar(summary_card, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set, state=tk.DISABLED)
        self.summary_text.grid(row=0, column=0, sticky=tk.NSEW)
        summary_scroll.grid(row=0, column=1, sticky=tk.NS)

        log_card = ttk.LabelFrame(sidebar, text="事件日志", padding=12, style="Section.TLabelframe")
        log_card.grid(row=1, column=0, sticky=tk.NSEW)
        log_card.rowconfigure(0, weight=1)
        log_card.columnconfigure(0, weight=1)
        self.log_text = tk.Text(log_card, wrap="word", bg="#fbf8f1", fg="#231f1a", relief="flat", font=("Consolas", 10), padx=10, pady=10)
        log_scroll = ttk.Scrollbar(log_card, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW)
        log_scroll.grid(row=0, column=1, sticky=tk.NS)

    def _build_stat_card(self, parent: "Frame", column: int, title: str, variable: "StringVar", background: str, foreground: str) -> None:
        card = self.tk.Frame(parent, bg=background, padx=16, pady=14)
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
        self.tk.Label(card, text=title, bg=background, fg=foreground, font=("Segoe UI", 9)).pack(anchor="w")
        self.tk.Label(card, textvariable=variable, bg=background, fg=foreground, font=("Segoe UI", 16, "bold")).pack(anchor="w", pady=(8, 0))

    def _build_gpu_metric_tile(
        self,
        parent: "Frame",
        *,
        row: int,
        column: int,
        title: str,
        variable: "StringVar",
        background: str,
        foreground: str,
    ) -> None:
        tile = self.tk.Frame(parent, bg=background, padx=12, pady=10)
        tile.grid(row=row, column=column, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self.tk.Label(tile, text=title, bg=background, fg=foreground, font=("Segoe UI", 9)).pack(anchor="w")
        self.tk.Label(tile, textvariable=variable, bg=background, fg=foreground, font=("Segoe UI", 15, "bold")).pack(anchor="w", pady=(6, 0))

    def _clear_gpu_cards(self) -> None:
        if self.gpu_cards_frame is not None:
            for child in self.gpu_cards_frame.winfo_children():
                child.destroy()
        self.gpu_cards.clear()

    def _ensure_gpu_card(self, gpu_index: int, model: str = "未知", serial: str = "N/A") -> dict[str, object]:
        card = self.gpu_cards.get(gpu_index)
        if card is not None:
            model_var = card["model_var"]
            serial_var = card["serial_var"]
            assert hasattr(model_var, "set")
            assert hasattr(serial_var, "set")
            model_var.set(model or "未知")
            serial_var.set(f"序列号: {serial or 'N/A'}")
            return card

        assert self.gpu_cards_frame is not None
        column_count = 2
        for column_id in range(column_count):
            self.gpu_cards_frame.grid_columnconfigure(column_id, weight=1)

        position = len(self.gpu_cards)
        row = position // column_count
        column = position % column_count
        panel = self.tk.Frame(
            self.gpu_cards_frame,
            bg="#fffaf2",
            padx=16,
            pady=14,
            highlightthickness=1,
            highlightbackground="#d8c9b7",
            highlightcolor="#d8c9b7",
            bd=0,
        )
        panel.grid(row=row, column=column, sticky="nsew", padx=(0 if column == 0 else 10, 0), pady=(0, 10))

        header = self.tk.Frame(panel, bg="#fffaf2")
        header.pack(fill=self.tk.X)
        self.tk.Label(header, text=f"GPU {gpu_index}", bg="#fffaf2", fg="#231f1a", font=("Segoe UI", 16, "bold")).pack(side=self.tk.LEFT)
        badge = self.tk.Label(header, text="检测中", bg="#f4e5a5", fg="#7c5e10", font=("Segoe UI", 10, "bold"), padx=12, pady=6)
        badge.pack(side=self.tk.RIGHT)

        model_var = self.tk.StringVar(value=model or "未知")
        serial_var = self.tk.StringVar(value=f"序列号: {serial or 'N/A'}")
        self.tk.Label(panel, textvariable=model_var, bg="#fffaf2", fg="#2f2a23", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 0))
        self.tk.Label(panel, textvariable=serial_var, bg="#fffaf2", fg="#6b5f53", font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 12))

        metrics = self.tk.Frame(panel, bg="#fffaf2")
        metrics.pack(fill=self.tk.X, expand=True)
        for row_id in range(2):
            metrics.grid_rowconfigure(row_id, weight=1)
        for column_id in range(3):
            metrics.grid_columnconfigure(column_id, weight=1)

        temp_var = self.tk.StringVar(value="N/A")
        power_var = self.tk.StringVar(value="N/A")
        util_var = self.tk.StringVar(value="N/A")
        memory_var = self.tk.StringVar(value="N/A")
        burn_var = self.tk.StringVar(value="N/A")
        ecc_var = self.tk.StringVar(value="未知")

        self._build_gpu_metric_tile(metrics, row=0, column=0, title="温度 C", variable=temp_var, background="#fbe7cf", foreground="#8a5a13")
        self._build_gpu_metric_tile(metrics, row=0, column=1, title="功耗 W", variable=power_var, background="#e7f3ea", foreground="#25613a")
        self._build_gpu_metric_tile(metrics, row=0, column=2, title="利用率 %", variable=util_var, background="#e4eef9", foreground="#1d4f7b")
        self._build_gpu_metric_tile(metrics, row=1, column=0, title="显存 MiB", variable=memory_var, background="#f4e8fb", foreground="#6f2f90")
        self._build_gpu_metric_tile(metrics, row=1, column=1, title="GPU-Burn GF/s", variable=burn_var, background="#fee6dd", foreground="#a3411b")
        self._build_gpu_metric_tile(metrics, row=1, column=2, title="ECC", variable=ecc_var, background="#efe8dd", foreground="#5b4e42")

        card = {
            "panel": panel,
            "badge": badge,
            "model_var": model_var,
            "serial_var": serial_var,
            "temp_var": temp_var,
            "power_var": power_var,
            "util_var": util_var,
            "memory_var": memory_var,
            "burn_var": burn_var,
            "ecc_var": ecc_var,
        }
        self.gpu_cards[gpu_index] = card
        return card

    def _update_gpu_card(self, gpu_index: int, **updates: object) -> None:
        card = self._ensure_gpu_card(gpu_index)
        if "model" in updates:
            model_var = card["model_var"]
            assert hasattr(model_var, "set")
            model_var.set(str(updates["model"] or "未知"))
        if "serial" in updates:
            serial_var = card["serial_var"]
            assert hasattr(serial_var, "set")
            serial_var.set(f"序列号: {updates['serial'] or 'N/A'}")
        value_mapping = {
            "temp": "temp_var",
            "power": "power_var",
            "util": "util_var",
            "memory": "memory_var",
            "burn": "burn_var",
            "ecc": "ecc_var",
        }
        for key, variable_name in value_mapping.items():
            if key not in updates:
                continue
            variable = card[variable_name]
            assert hasattr(variable, "set")
            variable.set(str(updates[key]))
        if "result_tag" in updates:
            badge = card["badge"]
            panel = card["panel"]
            assert hasattr(badge, "configure")
            assert hasattr(panel, "configure")
            state = str(updates["result_tag"])
            label, background, foreground = STATE_STYLES.get(state, STATE_STYLES["RUNNING"])
            badge.configure(text=label, bg=background, fg=foreground)
            border = {
                "EXCELLENT": "#77c091",
                "GOOD": "#d5b35f",
                "PASS": "#77c091",
                "FAIL": "#d98598",
                "ERROR": "#e4a17d",
                "NOT_RUN": "#b9aa95",
                "RUNNING": "#d9be5c",
            }.get(state, "#d8c9b7")
            panel.configure(highlightbackground=border, highlightcolor=border)

    def _set_state_visual(self, state: str) -> None:
        if self.result_badge_label is None:
            return
        label, background, foreground = STATE_STYLES.get(state, STATE_STYLES["IDLE"])
        self.result_badge_label.configure(text=label, bg=background, fg=foreground)

    def _set_summary_text(self, content: str) -> None:
        if self.summary_text is None:
            return
        self.summary_text.configure(state=self.tk.NORMAL)
        self.summary_text.delete("1.0", self.tk.END)
        self.summary_text.insert(self.tk.END, content)
        self.summary_text.see("1.0")
        self.summary_text.configure(state=self.tk.DISABLED)

    def _append_log(self, message: str) -> None:
        if self.log_text is None:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state=self.tk.NORMAL)
        self.log_text.insert(self.tk.END, f"[{stamp}] {message}\n")
        self.log_text.see(self.tk.END)
        self.log_text.configure(state=self.tk.DISABLED)

    def _choose_output_dir(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(initialdir=str(Path(self.output_dir_var.get()).expanduser().parent))
        if selected:
            self.output_dir_var.set(selected)

    def _apply_window_size(self, _event: object | None = None) -> None:
        preset = self.window_size_var.get()
        geometry = WINDOW_SIZE_PRESETS.get(preset)
        if geometry:
            self.root.geometry(geometry)

    def _set_running_state(self, running: bool) -> None:
        state = self.tk.DISABLED if running else self.tk.NORMAL
        if self.start_button is not None:
            self.start_button.configure(state=state)
        if self.stop_button is not None:
            self.stop_button.configure(state=self.tk.NORMAL if running else self.tk.DISABLED)
        if self.browse_button is not None:
            self.browse_button.configure(state=state)
        if self.progress is not None:
            self.progress.start(10) if running else self.progress.stop()

    def _reset_dashboard(self) -> None:
        self.latest_written = {}
        self.row_ids.clear()
        self._clear_gpu_cards()
        self.result_var.set("检测中")
        self.status_var.set("正在准备检测")
        self.gpu_count_var.set("0")
        self.fail_count_var.set("0")
        self.warning_count_var.set("0")
        self.duration_var.set("--")
        self.backup_var.set("摘要备份: 等待生成")
        self._set_state_visual("RUNNING")
        self._set_summary_text("检测正在进行中，完成后会自动在这里展示整体结果和 summary 备份。")
        if self.tree is not None:
            for item_id in self.tree.get_children():
                self.tree.delete(item_id)
        if self.log_text is not None:
            self.log_text.configure(state=self.tk.NORMAL)
            self.log_text.delete("1.0", self.tk.END)
            self.log_text.configure(state=self.tk.DISABLED)

    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            burn_seconds = int(self.burn_seconds_var.get())
            sample_interval = float(self.sample_interval_var.get())
            wait_driver_seconds = int(self.wait_driver_var.get())
            max_temp_c = float(self.max_temp_var.get())
        except ValueError:
            from tkinter import messagebox

            messagebox.showerror("参数错误", "压测时长、采样间隔和温度阈值必须是数字。")
            return

        output_dir = Path(self.output_dir_var.get()).expanduser()
        if output_dir.name == "":
            output_dir = build_default_output_dir()
            self.output_dir_var.set(str(output_dir))
        self.latest_output_dir = output_dir
        self._reset_dashboard()
        self._append_log(f"输出目录: {output_dir}")
        if self.auto_nvtop_var.get():
            _, nvtop_warning = launch_nvtop()
            self._append_log(nvtop_warning or "已在独立终端窗口中启动 nvtop")
        self.cancel_event = threading.Event()
        self._set_running_state(True)
        self.worker = threading.Thread(
            target=self._run_worker,
            kwargs={
                "output_dir": output_dir,
                "burn_seconds": burn_seconds,
                "sample_interval": sample_interval,
                "wait_driver_seconds": wait_driver_seconds,
                "max_temp_c": max_temp_c,
            },
            daemon=True,
        )
        self.worker.start()

    def _stop_run(self) -> None:
        if not self.worker or not self.worker.is_alive() or self.cancel_event is None:
            self._append_log("当前没有正在运行的检测任务")
            return
        if self.cancel_event.is_set():
            return
        self.cancel_event.set()
        self.status_var.set("正在停止检测，请等待当前进程退出")
        self._append_log("已请求停止当前检测")
        if self.stop_button is not None:
            self.stop_button.configure(state=self.tk.DISABLED)

    def _run_worker(self, *, output_dir: Path, burn_seconds: int, sample_interval: float, wait_driver_seconds: int, max_temp_c: float) -> None:
        try:
            report = run_suite(
                burn_seconds=burn_seconds,
                nvlink_seconds=10,
                sample_interval=sample_interval,
                wait_for_driver_seconds=wait_driver_seconds,
                max_temperature_c=max_temp_c,
                gpu_burn_command=None,
                nvlink_bandwidth_command=None,
                target_gpu_index=None,
                progress_callback=self._queue_progress,
                cancel_event=self.cancel_event,
            )
            written = write_outputs(report, output_dir)
        except Exception:
            self.queue.put({"type": "error", "message": traceback.format_exc()})
            return

        self.queue.put({"type": "finished", "report": report, "written": {name: str(path) for name, path in written.items()}, "output_dir": str(output_dir)})

    def _queue_progress(self, stage: str, message: str, payload: dict[str, object] | None) -> None:
        self.queue.put({"type": "progress", "stage": stage, "message": message, "payload": payload or {}})

    def _ensure_gpu_row(self, gpu_index: int, model: str = "未知") -> str:
        assert self.tree is not None
        row_id = self.row_ids.get(gpu_index)
        if row_id:
            return row_id
        row_id = self.tree.insert("", "end", values=(gpu_index, model, "N/A", "未知", "N/A", "N/A", "N/A", "N/A", "N/A", "运行中"), tags=("RUNNING",))
        self.row_ids[gpu_index] = row_id
        return row_id

    def _update_tree_row(self, gpu_index: int, **updates: object) -> None:
        if self.tree is None:
            return
        row_id = self._ensure_gpu_row(gpu_index)
        current = list(self.tree.item(row_id, "values"))
        mapping = {"gpu": 0, "model": 1, "serial": 2, "ecc": 3, "temp": 4, "power": 5, "util": 6, "memory": 7, "burn": 8, "result": 9}
        result_tag = None
        for key, value in updates.items():
            if key == "result_tag":
                result_tag = str(value)
                continue
            if key in mapping:
                current[mapping[key]] = value
        self.tree.item(row_id, values=current)
        if result_tag is not None:
            self.tree.item(row_id, tags=(result_tag,))

    def _handle_progress(self, stage: str, message: str, payload: dict[str, object]) -> None:
        self.status_var.set(message)
        if stage != "gpu-burn-sample":
            self._append_log(message)
        if stage == "gpu-inventory":
            gpus = payload.get("gpus", [])
            self.gpu_count_var.set(str(len(gpus)))
            for gpu in gpus:
                if isinstance(gpu, dict):
                    gpu_index = int(gpu.get("index", -1))
                    model = str(gpu.get("name") or "未知")
                    serial = str(gpu.get("serial") or "N/A")
                    self._ensure_gpu_row(gpu_index, model=model)
                    self._update_tree_row(gpu_index, model=model, serial=serial)
                    self._ensure_gpu_card(gpu_index, model=model, serial=serial)
                    self._update_gpu_card(gpu_index, model=model, serial=serial, result_tag="RUNNING")
            return
        if stage in {"baseline-metrics", "gpu-burn-sample"}:
            for sample in payload.get("samples", []):
                if not isinstance(sample, dict):
                    continue
                gpu_index = int(sample.get("gpu_index", -1))
                used = sample.get("memory_used_mib")
                total = sample.get("memory_total_mib")
                temp = format_float(float(sample["temperature_c"]), 1) if sample.get("temperature_c") is not None else "N/A"
                power = format_float(float(sample["power_w"]), 1) if sample.get("power_w") is not None else "N/A"
                util = format_float(float(sample["utilization_gpu_pct"]), 1) if sample.get("utilization_gpu_pct") is not None else "N/A"
                memory = "N/A" if used is None or total is None else f"{float(used):.0f}/{float(total):.0f}"
                self._update_tree_row(
                    gpu_index,
                    temp=temp,
                    power=power,
                    util=util,
                    memory=memory,
                )
                self._update_gpu_card(
                    gpu_index,
                    temp=temp,
                    power=power,
                    util=util,
                    memory=memory,
                    result_tag="RUNNING",
                )
            return
        if stage == "gpu-burn-finish":
            scores = payload.get("scores", {})
            if isinstance(scores, dict):
                for gpu_index, score in scores.items():
                    self._update_tree_row(int(gpu_index), burn=f"{float(score):.2f}")
                    self._update_gpu_card(int(gpu_index), burn=f"{float(score):.2f}", result_tag="RUNNING")
            return
        if stage in {"suite-finish", "suite-cancelled"}:
            for result in payload.get("results", []):
                if not isinstance(result, dict):
                    continue
                gpu_index = int(result.get("index", -1))
                current_raw = str(result.get("ecc_mode_current") or "Unknown")
                pending_raw = result.get("ecc_mode_pending")
                ecc = display_ecc_mode(current_raw)
                if pending_raw and str(pending_raw) != current_raw:
                    ecc = f"{ecc} -> {display_ecc_mode(str(pending_raw))}"
                result_code = str(result.get("result") or "NOT_RUN")
                assessment_code = str(result.get("assessment") or ("EXCELLENT" if result_code == "PASS" else result_code))
                self._update_tree_row(
                    gpu_index,
                    ecc=ecc,
                    burn=f"{float(result['burn_gflops']):.2f}" if result.get("burn_gflops") is not None else "N/A",
                    result=display_result(assessment_code),
                    result_tag=assessment_code,
                )
                self._update_gpu_card(
                    gpu_index,
                    model=str(result.get("name") or "未知"),
                    serial=str(result.get("serial") or "N/A"),
                    ecc=ecc,
                    burn=f"{float(result['burn_gflops']):.2f}" if result.get("burn_gflops") is not None else "N/A",
                    temp=format_float(float(result["avg_temp_c"]), 1) if result.get("avg_temp_c") is not None else "N/A",
                    power=format_float(float(result["avg_power_w"]), 1) if result.get("avg_power_w") is not None else "N/A",
                    util=format_float(float(result["avg_utilization_gpu_pct"]), 1) if result.get("avg_utilization_gpu_pct") is not None else "N/A",
                    result_tag=assessment_code,
                )

    def _handle_finished(self, event: dict[str, object]) -> None:
        report = event["report"]
        assert isinstance(report, RunReport)
        output_dir = Path(str(event["output_dir"]))
        self.latest_output_dir = output_dir
        self.latest_written = {name: Path(path) for name, path in (event.get("written") or {}).items() if isinstance(name, str) and isinstance(path, str)}
        self.cancel_event = None
        self._set_running_state(False)
        counts = summarize_counts(report)
        failed = counts.get("FAIL", 0) + counts.get("ERROR", 0)
        overall_code = overall_assessment_code(report)
        self.result_var.set(display_result(overall_code))
        if report.artifacts.get("run_status") == "CANCELLED":
            self.status_var.set(f"检测已手动停止，结果已写入 {output_dir}")
        else:
            self.status_var.set(f"检测完成，结果已写入 {output_dir}")
        self.gpu_count_var.set(str(report.gpu_count))
        self.fail_count_var.set(str(failed))
        self.warning_count_var.set(str(len(report.warnings) + len(report.errors)))
        self.duration_var.set(f"{report.duration_sec:.1f} s")
        self.backup_var.set(f"摘要备份: {build_backup_label(self.latest_written)}")
        self._set_state_visual(overall_code)
        self._set_summary_text(build_gui_summary(report, output_dir, self.latest_written))
        if report.artifacts.get("run_status") == "CANCELLED":
            self._append_log("检测已手动停止")
        self._append_log(f"检测结束，整体结果: {display_result(overall_code)}")
        self._append_log(f"结果目录: {output_dir}")

    def _handle_error(self, message: str) -> None:
        from tkinter import messagebox

        self.cancel_event = None
        self._set_running_state(False)
        self.result_var.set("运行失败")
        self.status_var.set("检测过程中发生异常")
        self.fail_count_var.set("1")
        self.warning_count_var.set("1")
        self.backup_var.set("摘要备份: 未生成")
        self._set_state_visual("ERROR")
        self._set_summary_text(f"检测异常终止，请查看日志。\n\n{message}")
        self._append_log(message)
        messagebox.showerror("1Cat-V100-QA", f"程序运行失败，请查看详细错误信息：\n\n{message}")

    def _drain_queue(self) -> None:
        while True:
            try:
                event = self.queue.get_nowait()
            except queue.Empty:
                break
            event_type = str(event.get("type"))
            if event_type == "progress":
                self._handle_progress(str(event.get("stage")), str(event.get("message")), event.get("payload") if isinstance(event.get("payload"), dict) else {})
            elif event_type == "finished":
                self._handle_finished(event)
            elif event_type == "error":
                self._handle_error(str(event.get("message")))
        self.root.after(120, self._drain_queue)

    def _open_latest_output(self) -> None:
        if self.latest_output_dir is None:
            self._append_log("当前还没有可打开的结果目录")
            return
        warning = open_path(self.latest_output_dir)
        if warning:
            self._append_log(warning)


def main() -> int:
    import tkinter as tk

    root = tk.Tk()
    app = GPUQAGuiApp(root)
    root._gpuqa_app = app
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
