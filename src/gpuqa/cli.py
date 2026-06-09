from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
import sys
import urllib.request
from pathlib import Path

from gpuqa import __version__
from gpuqa.commands import run_command, which_first
from gpuqa.core import run_suite
from gpuqa.desktop import launch_nvtop
from gpuqa.models import assessment_code_for_gpu
from gpuqa.reporting import render_run_overview, render_summary_table, write_outputs
from gpuqa.service import render_systemd_unit


def resolve_desktop_dir() -> Path:
    configured = os.environ.get("GPUQA_OUTPUT_ROOT")
    if configured:
        return Path(configured).expanduser()

    if which_first(["xdg-user-dir"]):
        result = run_command(["xdg-user-dir", "DESKTOP"], timeout=5)
        if result.ok:
            candidate = result.stdout.strip()
            if candidate:
                return Path(candidate).expanduser()

    config = Path.home() / ".config" / "user-dirs.dirs"
    if config.exists():
        for line in config.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.startswith("XDG_DESKTOP_DIR="):
                continue
            value = line.split("=", 1)[1].strip().strip('"')
            value = value.replace("$HOME", str(Path.home()))
            return Path(value).expanduser()

    for fallback in (Path.home() / "Desktop", Path.home() / "桌面"):
        if fallback.exists():
            return fallback
    return Path.home() / "Desktop"


def build_default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return resolve_desktop_dir() / f"1cat-v100-qa-{timestamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="1Cat-V100-QA")
    parser.add_argument("--version", action="version", version=f"1Cat-V100-QA {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="run GPU validation suite")
    run_parser.add_argument("--output-dir")
    run_parser.add_argument("--burn-seconds", type=int, default=600)
    run_parser.add_argument("--sample-interval", type=float, default=5.0)
    run_parser.add_argument("--wait-for-driver-seconds", type=int, default=180)
    run_parser.add_argument("--max-temperature-c", type=float, default=80.0)
    run_parser.add_argument("--nvlink-seconds", type=int, default=10)
    run_parser.add_argument("--gpu-index", type=int)
    run_parser.add_argument("--gpu-burn-command")
    run_parser.add_argument("--nvlink-bandwidth-command")
    run_parser.add_argument("--open-nvtop", action="store_true")
    run_parser.add_argument("--json", action="store_true", help="output machine-readable JSON to stdout")
    run_parser.add_argument("--quiet", action="store_true", help="suppress all output except JSON (implies --json)")
    run_parser.add_argument("--webhook-url", help="POST full result JSON to this URL on completion")

    serve_parser = subparsers.add_parser("serve", help="start HTTP API server for agent-driven testing")
    serve_parser.add_argument("--host", default="0.0.0.0", help="listen host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8765, help="listen port (default: 8765)")

    service_parser = subparsers.add_parser("print-service", help="print systemd unit template")
    service_parser.add_argument("--working-directory", default="/opt/gpuqa")
    service_parser.add_argument("--output-dir", default="/var/lib/gpuqa/latest")
    service_parser.add_argument("--user", default="root")
    service_parser.add_argument("--burn-seconds", type=int, default=600)
    service_parser.add_argument("--nvlink-seconds", type=int, default=10)
    service_parser.add_argument("--sample-interval", type=float, default=5.0)
    service_parser.add_argument("--wait-for-driver-seconds", type=int, default=180)
    service_parser.add_argument("--max-temperature-c", type=float, default=80.0)
    service_parser.add_argument("--gpu-index", type=int)

    return parser


def _results_as_dict(report, written: dict[str, Path] | None = None):
    from gpuqa.models import overall_assessment_code
    results = []
    for item in report.results:
        d = asdict(item)
        d["assessment_label"] = assessment_code_for_gpu(item)
        results.append(d)
    out = {
        "host": report.host,
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "duration_sec": report.duration_sec,
        "overall_result": report.overall_result,
        "overall_assessment": overall_assessment_code(report),
        "driver_ready": report.driver_ready,
        "gpu_count": report.gpu_count,
        "errors": report.errors,
        "warnings": report.warnings,
        "environment": report.environment,
        "results": results,
        "artifacts": report.artifacts,
    }
    if written:
        out["output_files"] = {name: str(path) for name, path in written.items()}
    return out


def _send_webhook(url: str, payload: dict) -> bool:
    try:
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=30)
        return True
    except Exception as exc:
        print(f"webhook delivery failed: {exc}", file=sys.stderr)
        return False


def command_run(args: argparse.Namespace) -> int:
    use_json = args.json or args.quiet
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else build_default_output_dir()

    nvtop_warning: str | None = None
    if args.open_nvtop:
        _, nvtop_warning = launch_nvtop()

    report = run_suite(
        burn_seconds=args.burn_seconds,
        nvlink_seconds=args.nvlink_seconds,
        sample_interval=args.sample_interval,
        wait_for_driver_seconds=args.wait_for_driver_seconds,
        max_temperature_c=args.max_temperature_c,
        gpu_burn_command=args.gpu_burn_command,
        nvlink_bandwidth_command=args.nvlink_bandwidth_command,
        target_gpu_index=args.gpu_index,
    )
    if nvtop_warning:
        report.warnings.append(nvtop_warning)

    written = write_outputs(report, output_dir)

    if use_json:
        payload = _results_as_dict(report, written)
        if not args.quiet:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        if args.webhook_url:
            _send_webhook(args.webhook_url, payload)
    else:
        print(render_run_overview(report))
        print()
        print(render_summary_table(report))
        print()
        print("输出文件:")
        for name, path in written.items():
            print(f"- {name}: {path}")
        if args.webhook_url:
            payload = _results_as_dict(report, written)
            if _send_webhook(args.webhook_url, payload):
                print()
                print(f"结果已推送至: {args.webhook_url}")

    if report.overall_result == "PASS":
        return 0
    if report.overall_result == "NOT_RUN":
        return 2
    return 1


def command_serve(args: argparse.Namespace) -> int:
    from gpuqa.httpd import serve
    serve(host=args.host, port=args.port)
    return 0


def command_print_service(args: argparse.Namespace) -> int:
    print(
        render_systemd_unit(
            working_directory=args.working_directory,
            output_directory=args.output_dir,
            user=args.user,
            burn_seconds=args.burn_seconds,
            nvlink_seconds=args.nvlink_seconds,
            sample_interval=args.sample_interval,
            wait_for_driver_seconds=args.wait_for_driver_seconds,
            max_temperature_c=args.max_temperature_c,
            gpu_index=args.gpu_index,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return command_run(args)
    if args.command == "serve":
        return command_serve(args)
    if args.command == "print-service":
        return command_print_service(args)

    parser.print_help(sys.stderr)
    return 2
