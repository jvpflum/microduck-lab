#!/usr/bin/env python3
"""Write lightweight NVIDIA utilization telemetry to JSONL during training."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path


FULL_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)
FALLBACK_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
)


def parent_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def sample() -> list[dict[str, object]]:
    fields = FULL_FIELDS
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        # Grace Hopper exposes unified memory through Linux rather than the
        # conventional NVML VRAM counters. RTX cards use the full query.
        fields = FALLBACK_FIELDS
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
    mem_available_mib = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available_mib = float(line.split()[1]) / 1024.0
                break
    except (OSError, ValueError, IndexError):
        pass
    rows = []
    def optional_float(value: str) -> float | None:
        return None if value in {"[N/A]", "N/A", ""} else float(value)

    for line in result.stdout.splitlines():
        values = [field.strip() for field in line.split(",")]
        if len(values) != len(fields):
            continue
        values_by_name = dict(zip(fields, values, strict=True))
        power = values_by_name["power.draw"]
        rows.append(
            {
                "gpu_index": int(values_by_name["index"]),
                "gpu_name": values_by_name["name"],
                "gpu_utilization_percent": float(values_by_name["utilization.gpu"]),
                "vram_used_mib": optional_float(values_by_name.get("memory.used", "")),
                "vram_total_mib": optional_float(values_by_name.get("memory.total", "")),
                "system_memory_available_mib": mem_available_mib,
                "temperature_c": float(values_by_name["temperature.gpu"]),
                "power_w": optional_float(power),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--parent-pid", type=int, default=os.getppid())
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    while parent_alive(args.parent_pid):
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            rows = sample()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            rows = [{"error": str(exc)}]
        with args.output.open("a", encoding="utf-8") as handle:
            for row in rows:
                record = {"timestamp": timestamp, **row}
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                if "error" not in row:
                    memory = (
                        f"vram={row['vram_used_mib']:.0f}/{row['vram_total_mib']:.0f} MiB "
                        if row["vram_used_mib"] is not None else
                        f"available-unified={row['system_memory_available_mib']:.0f} MiB "
                    )
                    print(
                        "[gpu] "
                        f"util={row['gpu_utilization_percent']:.0f}% "
                        f"{memory}"
                        f"temp={row['temperature_c']:.0f} C",
                        flush=True,
                    )
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
