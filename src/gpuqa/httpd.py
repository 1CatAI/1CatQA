"""Lightweight HTTP API server for agent-driven GPU QA.

Uses stdlib only — zero extra dependencies. Agents (OpenClaw, etc.) can
POST /api/run to trigger tests and poll /api/status or /api/results/<id>
for outcomes.
"""

from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any

from gpuqa import __version__
from gpuqa.cli import build_default_output_dir
from gpuqa.core import run_suite, utc_now
from gpuqa.models import RunReport


_MAX_RESULTS = 50

_status_lock = threading.Lock()
_current_run_id: str | None = None
_current_progress: str = "idle"
_current_started: str | None = None
_last_report: RunReport | None = None
_active_error: str | None = None
_results_store: dict[str, dict[str, Any]] = {}


def _now_iso() -> str:
    return utc_now()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        if os.environ.get("1CATQA_HTTP_VERBOSE"):
            super().log_message(fmt, *args)

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._handle_health()
        elif self.path == "/api/status":
            self._handle_status()
        elif self.path == "/api/results":
            self._handle_list_results()
        elif self.path.startswith("/api/results/"):
            run_id = self.path[len("/api/results/"):].strip("/")
            self._handle_get_result(run_id)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/api/run":
            self._handle_run()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_health(self) -> None:
        gpu_count = 0
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                gpu_count = len([l for l in result.stdout.strip().splitlines() if l.strip()])
        except Exception:
            pass

        self._send_json(200, {
            "status": "ok",
            "version": __version__,
            "gpu_count": gpu_count,
            "hostname": socket.gethostname(),
            "server_time": _now_iso(),
        })

    def _handle_status(self) -> None:
        with _status_lock:
            self._send_json(200, {
                "running": _current_run_id is not None,
                "run_id": _current_run_id,
                "progress": _current_progress,
                "started_at": _current_started,
                "last_error": _active_error,
            })

    def _handle_list_results(self) -> None:
        runs = []
        for run_id, entry in sorted(
            _results_store.items(),
            key=lambda kv: kv[1].get("started_at", ""),
            reverse=True,
        ):
            runs.append({
                "run_id": entry["run_id"],
                "started_at": entry.get("started_at"),
                "finished_at": entry.get("finished_at"),
                "overall_result": entry.get("overall_result"),
                "gpu_count": entry.get("gpu_count", 0),
                "output_dir": str(entry.get("output_dir", "")),
            })
        self._send_json(200, {"runs": runs})

    def _handle_get_result(self, run_id: str) -> None:
        entry = _results_store.get(run_id)
        if entry is None:
            self._send_json(404, {"error": f"run {run_id} not found"})
            return
        self._send_json(200, entry)

    def _handle_run(self) -> None:
        global _current_run_id, _current_progress, _current_started, _active_error

        with _status_lock:
            if _current_run_id is not None:
                self._send_json(409, {
                    "error": "A test run is already in progress",
                    "run_id": _current_run_id,
                })
                return

        body = self._read_body()
        run_id = f"run-{int(time.time())}"
        output_dir = body.get("output_dir")

        with _status_lock:
            _current_run_id = run_id
            _current_progress = "starting"
            _current_started = _now_iso()
            _active_error = None

        self._send_json(202, {
            "run_id": run_id,
            "status": "accepted",
            "message": "Test started. Poll /api/status or /api/results/{run_id} for results.",
        })

        thread = threading.Thread(
            target=_execute_run,
            args=(
                run_id,
                output_dir,
                body.get("burn_seconds", 600),
                body.get("nvlink_seconds", 10),
                body.get("sample_interval", 5.0),
                body.get("wait_for_driver_seconds", 180),
                body.get("max_temperature_c", 80.0),
                body.get("gpu_index"),
                body.get("webhook_url"),
            ),
            daemon=True,
        )
        thread.start()


def _execute_run(
    run_id: str,
    output_dir: str | None,
    burn_seconds: int,
    nvlink_seconds: int,
    sample_interval: float,
    wait_for_driver_seconds: int,
    max_temperature_c: float,
    gpu_index: int | None,
    webhook_url: str | None,
) -> None:
    global _current_run_id, _current_progress, _last_report, _active_error

    try:
        _set_progress("collecting_gpu_info")
        report = run_suite(
            burn_seconds=burn_seconds,
            nvlink_seconds=nvlink_seconds,
            sample_interval=sample_interval,
            wait_for_driver_seconds=wait_for_driver_seconds,
            max_temperature_c=max_temperature_c,
            gpu_burn_command=None,
            nvlink_bandwidth_command=None,
            target_gpu_index=gpu_index,
            progress_callback=_api_progress,
        )
        _last_report = report

        resolved_dir = Path(output_dir) if output_dir else build_default_output_dir()
        from gpuqa.reporting import write_outputs
        written = write_outputs(report, resolved_dir)

        entry = _build_result_entry(run_id, report, str(resolved_dir), written)
        _results_store[run_id] = entry
        if len(_results_store) > _MAX_RESULTS:
            oldest = sorted(_results_store.keys())[0]
            del _results_store[oldest]

        if webhook_url:
            _fire_webhook(webhook_url, entry)
    except Exception as exc:
        _active_error = str(exc)
        _results_store[run_id] = {
            "run_id": run_id,
            "error": str(exc),
            "started_at": _current_started,
            "finished_at": _now_iso(),
            "overall_result": "ERROR",
        }
    finally:
        with _status_lock:
            _current_run_id = None
            _current_progress = "idle"


def _set_progress(stage: str) -> None:
    global _current_progress
    with _status_lock:
        _current_progress = stage


def _api_progress(stage: str, detail: str, meta: dict[str, object] | None) -> None:
    _set_progress(f"{stage}:{detail}")


def _build_result_entry(
    run_id: str,
    report: RunReport,
    output_dir: str,
    written: dict[str, str],
) -> dict[str, Any]:
    entry: dict[str, Any] = asdict(report)
    entry["run_id"] = run_id
    entry["output_dir"] = output_dir
    entry["output_files"] = {
        name: str(path) for name, path in written.items()
    }
    return entry


def _fire_webhook(url: str, entry: dict[str, Any]) -> None:
    try:
        import urllib.request
        payload = json.dumps(entry, indent=2, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception:
        pass


def serve(
    host: str = "0.0.0.0",
    port: int = 8765,
) -> None:
    server = HTTPServer((host, port), _Handler)
    print(f"1CatQA {__version__} HTTP API listening on http://{host}:{port}")
    print(f"Health check: http://{host}:{port}/api/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
