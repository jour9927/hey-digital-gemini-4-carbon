#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon — Master Colab RTX Pro 6000 Bridge Daemon (Local Master Copy)
Runs in an isolated workspace (/local-scratch/heydigital_g4c -> /content/heydigital_g4c) on port 19464.
Never touches any other session's processes, ports, or directories.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import secrets
DAEMON_PORT = 19464
DEFAULT_TOKEN = os.environ.get("HEYDIGITAL_WORKER_TOKEN") or secrets.token_urlsafe(24)


def build_daemon_source() -> str:
    return '''#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, os, shutil, subprocess, tarfile, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WORK = Path('/content/heydigital_g4c').resolve()
TOKEN = ''
JOBS = {}
LOCK = threading.Lock()

def _safe(rel: str) -> Path:
    p = (WORK / rel.lstrip('/')).resolve()
    if p != WORK and WORK not in p.parents:
        raise ValueError('outside work root')
    return p

def _env(extra=None):
    e = dict(os.environ)
    vb = WORK / '.venv/bin'
    e['PATH'] = ':'.join(p for p in [str(vb) if vb.exists() else '', '/usr/local/cuda/bin', '/usr/local/bin', e.get('PATH', '')] if p)
    e['PYTHONUNBUFFERED'] = '1'
    e['HF_HOME'] = str(WORK / 'hf_cache')
    e['TORCH_HOME'] = str(WORK / 'torch_cache')
    if extra:
        e.update({str(k): str(v) for k, v in extra.items()})
    return e

def _run(jid: str, cmd: str, cwd: Path, lp: Path, extra):
    t0 = time.time()
    with lp.open('wb') as lf:
        pr = subprocess.Popen(['bash', '-lc', cmd], cwd=str(cwd), stdout=lf, stderr=subprocess.STDOUT, env=_env(extra), start_new_session=True)
        with LOCK:
            JOBS[jid]['pid'] = pr.pid
        rc = pr.wait()
    with LOCK:
        JOBS[jid].update(status='done' if rc == 0 else 'failed', rc=rc, elapsed_sec=round(time.time() - t0, 2))

class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *a):
        pass
    def _ok(self) -> bool:
        if self.headers.get('Authorization') == f'Bearer {TOKEN}':
            return True
        self._j(401, {'ok': False, 'error': 'unauthorized'})
        return False
    def _j(self, code: int, obj: dict):
        b = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if not self._ok():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == '/health':
            with LOCK:
                run = [k for k, v in JOBS.items() if v.get('status') == 'running']
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader'], text=True).strip()
            du = shutil.disk_usage(str(WORK))
            return self._j(200, {'ok': True, 'work': str(WORK), 'cpus': os.cpu_count(), 'disk_free_gb': round(du.free / (1024**3), 2), 'gpu': gpu, 'running': run})
        if u.path == '/job':
            jid = (q.get('id') or [''])[0]
            tail = int((q.get('tail') or ['12000'])[0])
            with LOCK:
                info = dict(JOBS.get(jid) or {})
            if not info:
                return self._j(404, {'ok': False, 'error': 'unknown'})
            lp = Path(info['log'])
            txt = lp.read_text(errors='replace')[-tail:] if lp.exists() else ''
            return self._j(200, {'ok': True, **info, 'output': txt})
        if u.path == '/get':
            t = _safe((q.get('path') or [''])[0])
            if not t.exists():
                return self._j(404, {'ok': False, 'error': 'not found'})
            if (q.get('tar') or ['0'])[0] == '1':
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode='w:gz') as tf:
                    for ch in (sorted(t.iterdir()) if t.is_dir() else [t]):
                        tf.add(str(ch), arcname=ch.name)
                d = buf.getvalue()
                self.send_response(200)
                self.send_header('Content-Type', 'application/gzip')
                self.send_header('Content-Length', str(len(d)))
                self.end_headers()
                self.wfile.write(d)
                return
            sz = t.stat().st_size
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(sz))
            self.end_headers()
            with t.open('rb') as fh:
                shutil.copyfileobj(fh, self.wfile, length=1048576)
            return
        self._j(404, {'ok': False, 'error': 'not found'})
    def do_POST(self):
        if not self._ok():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        n = int(self.headers.get('Content-Length') or '0')
        body = self.rfile.read(n) if n > 0 else b''
        if u.path == '/put':
            t = _safe((q.get('path') or [''])[0])
            if (q.get('untar') or ['0'])[0] == '1':
                t.mkdir(parents=True, exist_ok=True)
                with tarfile.open(fileobj=io.BytesIO(body), mode='r:gz') as tf:
                    tf.extractall(path=str(t), filter='data')
                return self._j(200, {'ok': True, 'extracted_to': str(t), 'bytes': len(body)})
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(body)
            return self._j(200, {'ok': True, 'saved': str(t), 'bytes': len(body)})
        if u.path == '/exec':
            req = json.loads(body.decode('utf-8') or '{}')
            jid = req.get('id') or ('job_' + uuid.uuid4().hex[:10])
            cwd = _safe(req.get('cwd') or 'repo')
            cwd.mkdir(parents=True, exist_ok=True)
            lp = WORK / 'logs' / f'{jid}.log'
            lp.parent.mkdir(parents=True, exist_ok=True)
            with LOCK:
                JOBS[jid] = {'id': jid, 'cmd': req['cmd'], 'status': 'running', 'rc': None, 'started_at': time.time(), 'log': str(lp)}
            threading.Thread(target=_run, args=(jid, req['cmd'], cwd, lp, req.get('env')), daemon=True).start()
            return self._j(200, {'ok': True, 'id': jid})
        self._j(404, {'ok': False, 'error': 'not found'})

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=19464)
    ap.add_argument('--token', required=True)
    ap.add_argument('--work', required=True)
    a = ap.parse_args()
    TOKEN = a.token
    WORK = Path(a.work).resolve()
    WORK.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(('127.0.0.1', a.port), H).serve_forever()
'''


def bootstrap_remote():
    candidates = [Path("/local-scratch"), Path("/content")]
    best_mount = max(
        [p for p in candidates if p.exists() and os.access(p, os.W_OK)],
        key=lambda p: shutil.disk_usage(str(p)).free,
    )
    real_work = best_mount / "heydigital_g4c"
    real_work.mkdir(parents=True, exist_ok=True)
    link = Path("/content/heydigital_g4c")
    if link != real_work and not link.exists():
        link.symlink_to(real_work)
    for sub in ("bin", "logs", "repo", "hf_cache", "torch_cache", "jobs", "voices", "outputs"):
        (real_work / sub).mkdir(parents=True, exist_ok=True)

    daemon_py = real_work / "heydigital_daemon_19464.py"
    daemon_py.write_text(build_daemon_source(), encoding="utf-8")
    daemon_py.chmod(0o755)

    subprocess.run("pkill -f '[h]eydigital_daemon_19464.py' || true", shell=True)
    subprocess.run("pkill -f '[h]eydigital_cf_19464' || true", shell=True)
    time.sleep(0.4)

    cf = real_work / "bin" / "heydigital_cf_19464"
    if not cf.exists():
        for cand in [Path("/usr/local/bin/cloudflared"), Path("/content/tradeoff_layer_carbon/bin/tradeoff_cf_19531"), Path("/content/mandushuqian_a100/mandushuqian_cf_19917")]:
            if cand.exists():
                shutil.copy2(cand, cf)
                break
        if not cf.exists():
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                str(cf),
            )
        cf.chmod(0o755)

    subprocess.Popen(
        [sys.executable, "-u", str(daemon_py), "--port", str(DAEMON_PORT), "--token", DEFAULT_TOKEN, "--work", str(real_work)],
        stdin=subprocess.DEVNULL,
        stdout=open(real_work / "daemon.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(0.5)

    cf_log = real_work / "cf_19464.log"
    cf_log.write_text("")
    subprocess.Popen(
        [str(cf), "tunnel", "--protocol", "http2", "--no-autoupdate", "--url", f"http://127.0.0.1:{DAEMON_PORT}"],
        stdin=subprocess.DEVNULL,
        stdout=open(cf_log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    url = ""
    for _ in range(50):
        time.sleep(0.5)
        if cf_log.exists():
            txt = cf_log.read_text(errors="ignore")
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
            if m and "Registered tunnel connection" in txt:
                url = m.group(0)
                break
            elif m:
                url = m.group(0)

    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
        text=True,
    ).strip()
    du = shutil.disk_usage(str(real_work))
    info = {
        "url": url,
        "token": DEFAULT_TOKEN,
        "port": DAEMON_PORT,
        "work": str(real_work),
        "gpu": gpu,
        "cpus": os.cpu_count(),
        "disk_free_gb": round(du.free / (1024**3), 2),
    }
    (real_work / "connection_info.json").write_text(json.dumps(info, indent=2))
    print("HEYDIGITAL_RTX6000_READY=" + json.dumps(info))


if __name__ == "__main__":
    bootstrap_remote()
