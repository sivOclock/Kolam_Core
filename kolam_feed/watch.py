"""Live view of a training run: serves watch.html, the run folder (log.jsonl, samples.png) and
/stats.json (GPU, CPU, RAM right now), so the page needs nothing else.

    python -m kolam_feed.watch runs/tiny --port 8800     # then open http://localhost:8800
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("watch.html")
GPU_Q = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
_cpu_prev: tuple[int, int] | None = None
RUN = ""


def _gpu() -> dict | None:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_Q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
    name, *vals = [v.strip() for v in out.split(",")]
    num = [float(v) if v.replace(".", "", 1).isdigit() else None for v in vals]
    return dict(zip(["name", "util", "mem_used", "mem_total", "temp", "power", "power_cap"], [name, *num]))


def _cpu() -> float | None:
    """Busy share of all cores since the last call, from /proc/stat."""
    global _cpu_prev
    try:
        f = [int(v) for v in open("/proc/stat").readline().split()[1:]]
    except OSError:
        return None
    idle, total = f[3] + f[4], sum(f)
    prev, _cpu_prev = _cpu_prev, (idle, total)
    if prev is None or total == prev[1]:
        return None
    return 100 * (1 - (idle - prev[0]) / (total - prev[1]))


def _ram() -> dict:
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        m[k] = int(v.split()[0]) / 2**20                    # kB -> GB
    return {"used": m["MemTotal"] - m["MemAvailable"], "total": m["MemTotal"]}


def stats() -> dict:
    return {"t": time.time(), "run": RUN, "gpu": _gpu(), "cpu": _cpu(), "cores": os.cpu_count(),
            "load": os.getloadavg()[0], "ram": _ram()}


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/watch.html"):
            return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/stats.json":
            return self._send(json.dumps(stats()).encode(), "application/json")
        return super().do_GET()

    def _send(self, body: bytes, kind: str):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):                               # quiet: the page polls every few seconds
        pass


def main() -> None:
    p = argparse.ArgumentParser(prog="kolam_feed.watch")
    p.add_argument("run", help="the training run folder, e.g. runs/tiny")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--host", default="0.0.0.0")
    a = p.parse_args()
    global RUN
    RUN = a.run
    _cpu()
    print(f"watching {a.run}: http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=a.run)).serve_forever()


if __name__ == "__main__":
    main()
