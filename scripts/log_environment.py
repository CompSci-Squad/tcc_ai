"""Log complete environment information for reproducibility."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def get_git_commit() -> str:
    """Return current git commit hash or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_package_versions() -> dict[str, str]:
    """Return installed package versions via uv pip list."""
    try:
        result = subprocess.run(
            ["uv", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        packages = json.loads(result.stdout)
        return {pkg["name"]: pkg["version"] for pkg in packages}
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return {}


def get_gpu_info() -> dict[str, object]:
    """Return GPU information from PyTorch CUDA runtime."""
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
    """Collect all environment metadata into a single dictionary."""
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


def main() -> None:
    """Collect and save environment information to docs/environment.json."""
    env_info = collect_environment()

    output_path = Path("docs/environment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(env_info, f, indent=2, default=str)

    print(f"Environment logged to {output_path}")
    print(f"  Python: {env_info['python_version'].split()[0]}")
    print(f"  PyTorch: {env_info['torch_version']}")
    gpu_status = "available" if env_info["gpu"]["available"] else "not available"
    print(f"  GPU: {gpu_status}")
    print(f"  Git: {env_info['git_commit'][:8]}")


if __name__ == "__main__":
    main()
