#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon — Distributed Compute Dispatcher (Local Master)
===========================================================================
Routes heavy PyTorch workloads (Voice Cloning, Demucs/Whisper, FLUX/SDXL Image Gen, LLM)
across available nodes in priority order while keeping ALL master scripts and outputs safe
on the local machine (`Alan-Hub`):

Priority 1: Ephemeral High-Speed Node — Colab RTX Pro 6000 Blackwell (96GB VRAM) via HTTPS Bridge
Priority 2: Tailscale SSH CUDA Node — `b850` (`alan-eagle@100.100.230.119`, RTX 5070 Ti 16GB, auto-checks free VRAM so concurrent sessions are never disturbed)
Priority 3: Tailscale SSH High-Memory Node — `alan-fw` (`claw@100.78.161.123`, AMD Ryzen AI Max+ 395 128GB)
Priority 4: Local Apple Silicon Node — `Alan-Hub` (M3 Pro 36GB MPS) / `alan-mba` (M4 Air 24GB)
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ColabBridgeConfig:
    url: str
    token: str = "hd_g4c_rtx6000_9f8a7b6c5d4e3f2a1b0c9d8e"
    timeout: int = 60


class ColabRTX6000Client:
    """HTTPS client for our isolated `/content/heydigital_g4c` daemon (:19464) on RTX Pro 6000."""

    def __init__(self, cfg: ColabBridgeConfig):
        self.base_url = cfg.url.rstrip("/")
        self.token = cfg.token
        self.timeout = cfg.timeout

    def _headers(self, extra: Optional[dict] = None) -> dict:
        h = {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "HeyDigital-Gemini4Carbon/1.0",
        }
        if extra:
            h.update(extra)
        return h

    def health(self) -> dict:
        req = urllib.request.Request(f"{self.base_url}/health", headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def put_file(self, local_path: Path, remote_rel: str) -> dict:
        q = urllib.parse.urlencode({"path": remote_rel})
        data = local_path.read_bytes()
        req = urllib.request.Request(
            f"{self.base_url}/put?{q}",
            data=data,
            headers=self._headers({"Content-Type": "application/octet-stream", "Content-Length": str(len(data))}),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=max(self.timeout, 120)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def put_bytes(self, data: bytes, remote_rel: str) -> dict:
        q = urllib.parse.urlencode({"path": remote_rel})
        req = urllib.request.Request(
            f"{self.base_url}/put?{q}",
            data=data,
            headers=self._headers({"Content-Type": "application/octet-stream", "Content-Length": str(len(data))}),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=max(self.timeout, 120)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_file(self, remote_rel: str, local_path: Path) -> Path:
        q = urllib.parse.urlencode({"path": remote_rel})
        req = urllib.request.Request(f"{self.base_url}/get?{q}", headers=self._headers(), method="GET")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(req, timeout=max(self.timeout, 180)) as resp:
            local_path.write_bytes(resp.read())
        return local_path

    def get_tar_dir(self, remote_rel: str, local_dir: Path) -> Path:
        q = urllib.parse.urlencode({"path": remote_rel, "tar": "1"})
        req = urllib.request.Request(f"{self.base_url}/get?{q}", headers=self._headers(), method="GET")
        local_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(req, timeout=max(self.timeout, 300)) as resp:
            raw = resp.read()
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            tf.extractall(path=str(local_dir), filter="data")
        return local_dir

    def exec_async(self, cmd: str, cwd: str = "repo", env: Optional[dict] = None, job_id: Optional[str] = None) -> str:
        payload = {"cmd": cmd, "cwd": cwd}
        if env:
            payload["env"] = env
        if job_id:
            payload["id"] = job_id
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/exec",
            data=body,
            headers=self._headers({"Content-Type": "application/json", "Content-Length": str(len(body))}),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        return res["id"]

    def wait_job(self, job_id: str, poll_sec: float = 3.0, max_wait_sec: float = 1800.0) -> dict:
        t0 = time.time()
        last_line = ""
        while (time.time() - t0) < max_wait_sec:
            q = urllib.parse.urlencode({"id": job_id, "tail": "12000"})
            req = urllib.request.Request(f"{self.base_url}/job?{q}", headers=self._headers(), method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                info = json.loads(resp.read().decode("utf-8"))
            out = (info.get("output") or "").strip()
            if out:
                cur_tail = out.splitlines()[-1]
                if cur_tail != last_line:
                    print(f"  [RTX6000:{job_id}] {cur_tail}")
                    last_line = cur_tail
            if info.get("status") in ("done", "failed"):
                if info.get("status") == "failed":
                    raise RuntimeError(f"Remote job {job_id} failed (rc={info.get('rc')}):\n{out[-3000:]}")
                return info
            time.sleep(poll_sec)
        raise TimeoutError(f"Remote job {job_id} timed out after {max_wait_sec}s")


class TailscaleClusterInspector:
    """Non-intrusive health & free-resource probe across Tailscale SSH nodes (`b850`, `alan-fw`, `alan-mba`)."""

    def __init__(self, repo_root: Path):
        self.ssh_cfg = repo_root / "config" / "ssh_config"

    def probe_node(self, host: str) -> dict:
        cmd = [
            "ssh", "-F", str(self.ssh_cfg), "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            host,
            "hostname && uname -s -m && (nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo NO_NVIDIA)"
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            if p.returncode != 0:
                return {"host": host, "reachable": False, "error": p.stderr.strip()}
            lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
            gpu_line = lines[2] if len(lines) >= 3 else "NO_NVIDIA"
            free_vram_mb = 0
            if gpu_line != "NO_NVIDIA" and "," in gpu_line:
                parts = [x.strip() for x in gpu_line.split(",")]
                if len(parts) >= 3:
                    free_vram_mb = int(parts[2].split()[0])
            return {
                "host": host,
                "reachable": True,
                "hostname": lines[0] if lines else host,
                "arch": lines[1] if len(lines) > 1 else "",
                "gpu": gpu_line,
                "free_vram_mb": free_vram_mb,
            }
        except Exception as exc:
            return {"host": host, "reachable": False, "error": str(exc)}

    def probe_all(self) -> list[dict]:
        return [self.probe_node(h) for h in ("b850", "alan-fw", "alan-mba")]
