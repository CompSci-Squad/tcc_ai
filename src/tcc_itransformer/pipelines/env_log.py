"""Capture the runtime environment (Python, packages, GPU, git SHA) for reproducibility."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_package_versions() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["uv", "pip", "list", "--format=json"],
            capture_output=True, text=True, check=True,
        )
        packages = json.loads(result.stdout)
        return {pkg["name"]: pkg["version"] for pkg in packages}
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return {}


def get_gpu_info() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "devices": [
            {
                "name": torch.cuda.get_device_name(i),
                "memory_total_mb": round(
                    torch.cuda.get_device_properties(i).total_mem / (1024**2)
                ),
            }
            for i in range(torch.cuda.device_count())
        ],
        "cuda_version": torch.version.cuda,
        "cudnn_version": str(torch.backends.cudnn.version()),
    }


def collect_environment() -> dict[str, object]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "git_commit": get_git_commit(),
        "torch_version": torch.__version__,
        "gpu": get_gpu_info(),
        "packages": get_package_versions(),
    }


def write_environment(output_path: Path = Path("docs/environment.json")) -> Path:
    env_info = collect_environment()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(env_info, indent=2, default=str))
    return output_path
