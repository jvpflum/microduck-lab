#!/usr/bin/env python3
"""Emit a machine-readable MicroDuck training readiness report."""

import json
import platform
import sys

import mujoco
import torch
import warp as wp


cuda_available = torch.cuda.is_available()
report = {
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": cuda_available,
    "cuda_device_count": torch.cuda.device_count(),
    "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
    "cuda_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
    "warp": wp.__version__,
    "mujoco": mujoco.__version__,
}
print(json.dumps(report, indent=2, sort_keys=True))

if platform.machine() != "aarch64":
    print("Expected aarch64 on DGX Spark.", file=sys.stderr)
    raise SystemExit(1)
if not cuda_available:
    print("CUDA is not available to PyTorch.", file=sys.stderr)
    raise SystemExit(1)
if torch.version.cuda is None:
    print("The installed ARM64 PyTorch wheel is CPU-only.", file=sys.stderr)
    raise SystemExit(1)
