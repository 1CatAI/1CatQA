from __future__ import annotations

from dataclasses import asdict
from html import escape
from pathlib import Path
import csv
import json
from typing import Iterable

from gpuqa.models import GPUResult, RunReport, assessment_code_for_gpu, display_assessment, overall_assessment_code


SUMMARY_HEADERS = [
    "GPU",
    "型号",
    "序列号",
    "ECC 模式",
    "NVLink 连接数",
    "平均温度 (C)",
    "最高温度 (C)",
    "ECC 增量",
    "NVLink 错误增量",
    "NVLink CRC 增量",
    "NVLink 带宽 (GB/s)",
    "GPU-Burn 算力 (GF/s)",
    "GPU-Burn 告警计数",
    "GPU-Burn 错误计数",
    "结果",
]


ECC_MODE_LABELS = {
    "Enabled": "已开启",
    "Disabled": "未开启",
    "Unknown": "未知",
}


def _format_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _format_ecc(result: GPUResult) -> str:
    if result.ecc_corrected_delta is None and result.ecc_uncorrected_delta is None:
        return "N/A"
    corrected = result.ecc_corrected_delta if result.ecc_corrected_delta is not None else "?"
    uncorrected = result.ecc_uncorrected_delta if result.ecc_uncorrected_delta is not None else "?"
    return f"{corrected}/{uncorrected}"


def _format_ecc_mode(result: GPUResult) -> str:
    current = ECC_MODE_LABELS.get(result.ecc_mode_current or "Unknown", result.ecc_mode_current or "Unknown")
    pending_raw = result.ecc_mode_pending or result.ecc_mode_current or "Unknown"
    pending = ECC_MODE_LABELS.get(pending_raw, pending_raw)
    if pending == current:
        return current
    return f"{current} (待生效: {pending})"


def _format_nvlink_links(result: GPUResult) -> str:
    before = result.nvlink_link_count_before
    after = result.nvlink_link_count_after
    if before is None and after is None:
        return "N/A"
    before_text = "?" if before is None else str(before)
    after_text = "?" if after is None else str(after)
    return f"{before_text}/6 -> {after_text}/6"


def _display_result(value: str) -> str:
    return display_assessment(value)


def _display_gpu_result(item: GPUResult) -> str:
    return display_assessment(assessment_code_for_gpu(item))


def _display_overall_result(report: RunReport) -> str:
    return display_assessment(overall_assessment_code(report))


def _rows(results: Iterable[GPUResult]) -> list[list[str]]:
    table_rows: list[list[str]] = []
    for item in results:
        table_rows.append(
            [
                str(item.index),
                item.name or "N/A",
                item.serial or "N/A",
                _format_ecc_mode(item),
                _format_nvlink_links(item),
                _format_float(item.avg_temp_c),
                _format_float(item.max_temp_c),
                _format_ecc(item),
                str(item.nvlink_error_delta) if item.nvlink_error_delta is not None else "N/A",
                str(item.nvlink_crc_error_delta) if item.nvlink_crc_error_delta is not None else "N/A",
                _format_float(item.nvlink_bandwidth_gbps),
                _format_float(item.burn_gflops),
                str(item.burn_warning_count) if item.burn_warning_count is not None else "N/A",
                str(item.burn_error_count) if item.burn_error_count is not None else "N/A",
                _display_gpu_result(item),
            ]
        )
    return table_rows


def render_ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [border]
    lines.append(
        "| "
        + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
        + " |"
    )
    lines.append(border)
    for row in rows:
        lines.append(
            "| "
            + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
            + " |"
        )
    lines.append(border)
    return "\n".join(lines)


def render_summary_table(report: RunReport) -> str:
    return render_ascii_table(SUMMARY_HEADERS, _rows(report.results))


def render_markdown_table(report: RunReport) -> str:
    rows = _rows(report.results)
    header = "| " + " | ".join(SUMMARY_HEADERS) + " |"
    separator = "| " + " | ".join("---" for _ in SUMMARY_HEADERS) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def render_run_overview(report: RunReport) -> str:
    lines = [
        f"主机: {report.host}",
        f"开始时间: {report.started_at}",
        f"结束时间: {report.finished_at}",
        f"总体结果: {_display_overall_result(report)}",
        f"驱动就绪: {'是' if report.driver_ready else '否'}",
        f"GPU 数量: {report.gpu_count}",
    ]
    burn_seconds = report.environment.get("burn_seconds")
    nvlink_seconds = report.environment.get("nvlink_seconds")
    max_temperature_c = report.environment.get("max_temperature_c")
    target_gpu_index = report.environment.get("target_gpu_index")
    if any(value is not None for value in (burn_seconds, nvlink_seconds, max_temperature_c, target_gpu_index)):
        target_label = "全部GPU" if target_gpu_index in {None, "ALL"} else f"GPU {target_gpu_index}"
        lines.append(
            f"运行配置: 温度阈值 {max_temperature_c or 'N/A'}C | "
            f"Burn {burn_seconds or 'N/A'}s | NVLink {nvlink_seconds or 'N/A'}s | {target_label}"
        )
    if report.errors:
        lines.append("错误:")
        lines.extend(f"- {item}" for item in report.errors)
    if report.warnings:
        lines.append("告警:")
        lines.extend(f"- {item}" for item in report.warnings)
    return "\n".join(lines)


def render_text_report(report: RunReport) -> str:
    lines = [render_run_overview(report), "", render_summary_table(report)]
    notes = [(item.index, item.notes) for item in report.results if item.notes]
    if notes:
        lines.extend(["", "备注:"])
        for gpu_index, gpu_notes in notes:
            lines.append(f"- GPU {gpu_index}: {'; '.join(gpu_notes)}")
    if report.environment:
        lines.extend(["", "环境信息:"])
        for key, value in report.environment.items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def render_html_report(report: RunReport) -> str:
    rows = _rows(report.results)
    notes = [
        {
            "gpu": str(item.index),
            "notes": item.notes,
        }
        for item in report.results
        if item.notes
    ]
    warnings = "".join(f"<li>{escape(item)}</li>" for item in report.warnings)
    errors = "".join(f"<li>{escape(item)}</li>" for item in report.errors)
    table_head = "".join(f"<th>{escape(header)}</th>" for header in SUMMARY_HEADERS)
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    notes_html = "".join(
        "<tr>"
        f"<td>{escape(item['gpu'])}</td>"
        f"<td>{escape('; '.join(item['notes']))}</td>"
        "</tr>"
        for item in notes
    )
    badge_class = (
        "pass"
        if overall_assessment_code(report) == "EXCELLENT"
        else "warn"
        if overall_assessment_code(report) == "GOOD"
        else "fail"
        if report.overall_result in {"FAIL", "ERROR"}
        else "warn"
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>1Cat-V100-QA 测试报告</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe8;
      --panel: #fffdf9;
      --ink: #231f1a;
      --muted: #6d6254;
      --line: #dacdbd;
      --accent: #0f766e;
      --pass: #166534;
      --fail: #9f1239;
      --warn: #a16207;
      --shadow: 0 16px 48px rgba(35, 31, 26, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 26%),
        linear-gradient(180deg, #f2ebe0 0%, var(--bg) 100%);
      padding: 32px;
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 24px;
      margin-bottom: 24px;
    }}
    h1, h2 {{
      margin: 0 0 16px;
      font-family: Georgia, "Times New Roman", serif;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    .meta-item {{
      background: #faf5ed;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 18px;
      font-weight: 700;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      color: white;
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.04em;
    }}
    .pass {{
      background: var(--pass);
    }}
    .fail {{
      background: var(--fail);
    }}
    .warn {{
      background: var(--warn);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    thead {{
      background: #efe4d3;
    }}
    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      font-size: 14px;
    }}
    tbody tr:nth-child(even) {{
      background: #fcfaf6;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
    }}
    .empty {{
      color: var(--muted);
      font-style: italic;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>1Cat-V100-QA 测试报告</h1>
      <div class="meta">
        <div class="meta-item"><div class="label">主机</div><div class="value">{escape(report.host)}</div></div>
        <div class="meta-item"><div class="label">开始时间</div><div class="value">{escape(report.started_at)}</div></div>
        <div class="meta-item"><div class="label">结束时间</div><div class="value">{escape(report.finished_at)}</div></div>
        <div class="meta-item"><div class="label">总体结果</div><div class="value"><span class="badge {badge_class}">{escape(_display_overall_result(report))}</span></div></div>
        <div class="meta-item"><div class="label">驱动就绪</div><div class="value">{'是' if report.driver_ready else '否'}</div></div>
        <div class="meta-item"><div class="label">GPU 数量</div><div class="value">{report.gpu_count}</div></div>
      </div>
    </section>
    <section class="card">
      <h2>汇总</h2>
      <table>
        <thead>
          <tr>{table_head}</tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </section>
    <section class="card">
      <h2>告警</h2>
      {"<ul>" + warnings + "</ul>" if warnings else '<div class="empty">无告警</div>'}
    </section>
    <section class="card">
      <h2>错误</h2>
      {"<ul>" + errors + "</ul>" if errors else '<div class="empty">无错误</div>'}
    </section>
    <section class="card">
      <h2>备注</h2>
      {
        '<table><thead><tr><th>GPU</th><th>备注</th></tr></thead><tbody>' + notes_html + '</tbody></table>'
        if notes_html
        else '<div class="empty">无备注</div>'
      }
    </section>
  </div>
</body>
</html>
"""


def write_outputs(report: RunReport, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "summary_txt": output_dir / "summary.txt",
        "summary_html": output_dir / "summary.html",
        "summary_json": output_dir / "summary.json",
        "summary_csv": output_dir / "summary.csv",
        "summary_md": output_dir / "summary.md",
        "samples_csv": output_dir / "samples.csv",
    }

    with files["summary_txt"].open("w", encoding="utf-8") as handle:
        handle.write(render_text_report(report))
        handle.write("\n")

    with files["summary_html"].open("w", encoding="utf-8") as handle:
        handle.write(render_html_report(report))

    with files["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2, ensure_ascii=True)

    with files["summary_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gpu_index",
                "name",
                "serial",
                "uuid",
                "pci_bus_id",
                "ecc_mode_current",
                "ecc_mode_pending",
                "ecc_enabled",
                "nvlink_link_count_before",
                "nvlink_link_count_after",
                "avg_temp_c",
                "max_temp_c",
                "avg_power_w",
                "max_power_w",
                "avg_sm_clock_mhz",
                "avg_mem_clock_mhz",
                "avg_utilization_gpu_pct",
                "ecc_corrected_delta",
                "ecc_uncorrected_delta",
                "nvlink_error_delta",
                "nvlink_crc_error_delta",
                "nvlink_bandwidth_gbps",
                "burn_gflops",
                "burn_warning_count",
                "burn_error_count",
                "burn_ok",
                "result",
                "assessment",
                "notes",
            ]
        )
        for item in report.results:
            writer.writerow(
                [
                    item.index,
                    item.name,
                    item.serial,
                    item.uuid,
                    item.pci_bus_id,
                    item.ecc_mode_current,
                    item.ecc_mode_pending,
                    item.ecc_enabled,
                    item.nvlink_link_count_before,
                    item.nvlink_link_count_after,
                    item.avg_temp_c,
                    item.max_temp_c,
                    item.avg_power_w,
                    item.max_power_w,
                    item.avg_sm_clock_mhz,
                    item.avg_mem_clock_mhz,
                    item.avg_utilization_gpu_pct,
                    item.ecc_corrected_delta,
                    item.ecc_uncorrected_delta,
                    item.nvlink_error_delta,
                    item.nvlink_crc_error_delta,
                    item.nvlink_bandwidth_gbps,
                    item.burn_gflops,
                    item.burn_warning_count,
                    item.burn_error_count,
                    item.burn_ok,
                    item.result,
                    _display_gpu_result(item),
                    "; ".join(item.notes),
                ]
            )

    with files["summary_md"].open("w", encoding="utf-8") as handle:
        handle.write("# 1Cat-V100-QA 测试报告\n\n")
        handle.write(render_run_overview(report))
        handle.write("\n\n")
        handle.write(render_markdown_table(report))
        handle.write("\n")

    with files["samples_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "gpu_index",
                "temperature_c",
                "power_w",
                "sm_clock_mhz",
                "mem_clock_mhz",
                "utilization_gpu_pct",
                "memory_used_mib",
                "memory_total_mib",
            ]
        )
        for sample in report.samples:
            writer.writerow(
                [
                    sample.timestamp,
                    sample.gpu_index,
                    sample.temperature_c,
                    sample.power_w,
                    sample.sm_clock_mhz,
                    sample.mem_clock_mhz,
                    sample.utilization_gpu_pct,
                    sample.memory_used_mib,
                    sample.memory_total_mib,
                ]
            )

    return files
