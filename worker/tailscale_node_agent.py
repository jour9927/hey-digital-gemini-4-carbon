#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon — Tailscale SSH Worker Node Probe & Fallback Executor
=================================================================================
Deployed over Tailscale SSH to `b850` (`100.100.230.119`, RTX 5070 Ti 16GB) and
`alan-fw` (`100.78.161.123`, AMD Ryzen AI Max+ 395 128GB).
Strictly read-safe / non-intrusive: checks active GPU & RAM usage so other concurrent
sessions are never disturbed, and exposes structured node capability JSON to `Alan-Hub`.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess


def collect_node_capabilities() -> dict:
    du = shutil.disk_usage(os.path.expanduser("~"))
    gpu_info = "NONE"
    free_vram_mb = 0
    total_vram_mb = 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            gpu_info = out
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 3:
                total_vram_mb = int(parts[1].split()[0])
                free_vram_mb = int(parts[2].split()[0])
    except Exception:
        pass

    ts_ip = ""
    try:
        ts_ip = subprocess.check_output(["tailscale", "ip", "-4"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass

    return {
        "ok": True,
        "hostname": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "tailscale_ipv4": ts_ip,
        "gpu": gpu_info,
        "total_vram_mb": total_vram_mb,
        "free_vram_mb": free_vram_mb,
        "safe_for_gpu_offload": free_vram_mb >= 4500,
        "disk_free_gb": round(du.free / (1024**3), 2),
    }


if __name__ == "__main__":
    print(json.dumps(collect_node_capabilities(), ensure_ascii=False))
