#!/usr/bin/env python3
"""Collect non-secret runtime and hardware provenance for study artifacts."""
import csv
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


PACKAGE_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "safetensors",
    "numpy",
    "scipy",
    "scikit-learn",
    "peft",
)


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_versions():
    versions = {}
    for name in PACKAGE_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _torch_cuda():
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "torch_importable": False,
            "pytorch_cuda_available": None,
            "pytorch_cuda_version": None,
            "pytorch_device_count": None,
            "pytorch_cudnn_version": None,
        }
    cuda_available = bool(torch.cuda.is_available())
    return {
        "torch_importable": True,
        "pytorch_cuda_available": cuda_available,
        "pytorch_cuda_version": torch.version.cuda,
        "pytorch_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "pytorch_cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
    }


def _parse_memory_mib(text):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _nvidia_smi(gpu_id=None):
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    if gpu_id is not None and str(gpu_id).strip():
        cmd.insert(1, "-i")
        cmd.insert(2, str(gpu_id))
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return {
            "available": False,
            "error": "nvidia-smi not found",
            "gpus": [],
        }
    if proc.returncode != 0:
        return {
            "available": False,
            "error": proc.stderr.strip()[:240],
            "gpus": [],
        }
    rows = []
    for fields in csv.reader(proc.stdout.splitlines()):
        if len(fields) != 5:
            continue
        index, name, uuid, driver, memory_total = [field.strip() for field in fields]
        rows.append({
            "index": index,
            "name": name,
            "uuid_sha256": _sha256_text(uuid) if uuid else None,
            "driver_version": driver,
            "memory_total_mib": _parse_memory_mib(memory_total),
        })
    return {
        "available": True,
        "error": "",
        "gpus": rows,
    }


def collect_run_environment(gpu_id=None):
    """Return a bounded, non-secret environment receipt for final study provenance."""
    hostname = platform.node()
    nvidia = _nvidia_smi(gpu_id)
    return {
        "schema": "run_environment_v1",
        "collected_at": _utc_now(),
        "gpu_id_requested": str(gpu_id) if gpu_id is not None and str(gpu_id).strip() else None,
        "hostname_sha256": _sha256_text(hostname) if hostname else None,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_basename": os.path.basename(sys.executable),
        },
        "packages": _package_versions(),
        "cuda": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            **_torch_cuda(),
        },
        "nvidia_smi": {
            "available": nvidia["available"],
            "error": nvidia["error"],
        },
        "gpus": nvidia["gpus"],
    }


if __name__ == "__main__":
    import json

    json.dump(collect_run_environment(os.environ.get("GPU_ID")), sys.stdout, indent=2)
    sys.stdout.write("\n")
