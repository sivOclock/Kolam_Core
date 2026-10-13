"""CLI.

    python -m kolam_feed verify data/enc_1m --n 5000    # re-check kolam_gen output; exit 1 on any failure
    python -m kolam_feed preview data/enc_1m -o sheet.png --n 36
    python -m kolam_feed bench data/enc_1m --size 448   # images per second, one process
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from PIL import Image

from .data import KolamDataset
from .verify import verify


def _verify(a):
    r = verify(a.src, n=a.n, seed=a.seed)
    print(f"{r['kolams']} kolams, {r['checked']} checked in full")
    print("failures:", json.dumps(r["failures"]) if r["failures"] else "none")
    for k, v in r["examples"].items():
        print(f"  {k}: {v}")
    for k, v in r["dist"].items():
        top = ", ".join(f"{x}: {c}" for x, c in list(v.items())[:12])
        print(f"{k}: {top}")
    sys.exit(0 if r["ok"] else 1)


def _preview(a):
    ds = KolamDataset(a.src, size=a.tile)
    cols = 6
    rows = (min(a.n, len(ds)) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * a.tile, rows * a.tile), (10, 10, 12))
    for i in range(min(a.n, len(ds))):
        sheet.paste(ds.render(i)[0].convert("RGB"), ((i % cols) * a.tile, (i // cols) * a.tile))
    sheet.save(a.out)
    print(f"saved {a.out}")


def _bench(a):
    ds = KolamDataset(a.src, size=a.size)
    n = min(a.n, len(ds))
    t = time.time()
    for i in range(n):
        ds[i]
    dt = time.time() - t
    print(f"{n} images at {a.size}px in {dt:.1f}s: {n / dt:.1f} images/s per process")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kolam_feed")
    sub = p.add_subparsers(required=True)
    v = sub.add_parser("verify", help="re-check kolam_gen output with the Python rules")
    v.add_argument("src")
    v.add_argument("--n", type=int, default=2000, help="kolams checked in full (all lines get the cheap checks)")
    v.add_argument("--seed", type=int, default=0)
    v.set_defaults(fn=_verify)
    s = sub.add_parser("preview", help="contact sheet of rendered training images")
    s.add_argument("src")
    s.add_argument("-o", "--out", default="sheet.png")
    s.add_argument("--n", type=int, default=36)
    s.add_argument("--tile", type=int, default=256)
    s.set_defaults(fn=_preview)
    b = sub.add_parser("bench", help="render speed")
    b.add_argument("src")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--size", type=int, default=448)
    b.set_defaults(fn=_bench)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
