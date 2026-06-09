from __future__ import annotations


def render_systemd_unit(
    *,
    working_directory: str,
    output_directory: str,
    user: str = "root",
    burn_seconds: int = 600,
    nvlink_seconds: int = 10,
    sample_interval: float = 5.0,
    wait_for_driver_seconds: int = 180,
    max_temperature_c: float = 80.0,
    gpu_index: int | None = None,
) -> str:
    gpu_index_arg = f" --gpu-index {gpu_index}" if gpu_index is not None else ""
    return f"""[Unit]
Description=1Cat-V100-QA automatic validation for V100 (CUDA 12 optimized)
After=multi-user.target nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
User={user}
WorkingDirectory={working_directory}
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/env python3 -m gpuqa run --output-dir {output_directory} --burn-seconds {burn_seconds} --nvlink-seconds {nvlink_seconds} --sample-interval {sample_interval} --wait-for-driver-seconds {wait_for_driver_seconds} --max-temperature-c {max_temperature_c}{gpu_index_arg}
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
