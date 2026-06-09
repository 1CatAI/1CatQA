from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ASSESSMENT_LABELS = {
    "EXCELLENT": "优秀",
    "GOOD": "良好",
    "FAIL": "不通过",
    "ERROR": "错误",
    "NOT_RUN": "未运行",
    "RUNNING": "检测中",
    "IDLE": "空闲",
}


@dataclass(slots=True)
class GPUIdentity:
    index: int
    name: str | None = None
    uuid: str | None = None
    serial: str | None = None
    pci_bus_id: str | None = None
    driver_version: str | None = None


@dataclass(slots=True)
class MetricSample:
    timestamp: str
    gpu_index: int
    temperature_c: float | None = None
    power_w: float | None = None
    sm_clock_mhz: float | None = None
    mem_clock_mhz: float | None = None
    utilization_gpu_pct: float | None = None
    memory_used_mib: float | None = None
    memory_total_mib: float | None = None


@dataclass(slots=True)
class GPUResult:
    index: int
    name: str | None = None
    serial: str | None = None
    uuid: str | None = None
    pci_bus_id: str | None = None
    ecc_mode_current: str | None = None
    ecc_mode_pending: str | None = None
    ecc_enabled: bool | None = None
    avg_temp_c: float | None = None
    max_temp_c: float | None = None
    avg_power_w: float | None = None
    max_power_w: float | None = None
    avg_sm_clock_mhz: float | None = None
    avg_mem_clock_mhz: float | None = None
    avg_utilization_gpu_pct: float | None = None
    ecc_corrected_before: int | None = None
    ecc_corrected_after: int | None = None
    ecc_corrected_delta: int | None = None
    ecc_uncorrected_before: int | None = None
    ecc_uncorrected_after: int | None = None
    ecc_uncorrected_delta: int | None = None
    nvlink_link_count_before: int | None = None
    nvlink_link_count_after: int | None = None
    nvlink_error_delta: int | None = None
    nvlink_crc_error_delta: int | None = None
    nvlink_bandwidth_gbps: float | None = None
    burn_gflops: float | None = None
    burn_warning_count: int | None = None
    burn_error_count: int | None = None
    burn_ok: bool | None = None
    result: str = "NOT_RUN"
    assessment: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RunReport:
    host: str
    started_at: str
    finished_at: str
    duration_sec: float
    overall_result: str
    driver_ready: bool
    gpu_count: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    results: list[GPUResult] = field(default_factory=list)
    samples: list[MetricSample] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


def assessment_code_for_gpu(result: GPUResult) -> str:
    if result.assessment:
        return result.assessment
    if result.result == "PASS":
        return "EXCELLENT"
    return result.result


def overall_assessment_code(report: RunReport) -> str:
    if report.overall_result in {"FAIL", "ERROR", "NOT_RUN"}:
        return report.overall_result
    assessments = [assessment_code_for_gpu(item) for item in report.results]
    if not assessments:
        return "NOT_RUN"
    if any(code == "GOOD" for code in assessments):
        return "GOOD"
    if all(code == "EXCELLENT" for code in assessments):
        return "EXCELLENT"
    return report.overall_result


def display_assessment(code: str | None) -> str:
    if code is None:
        return "未知"
    return ASSESSMENT_LABELS.get(code, code)
