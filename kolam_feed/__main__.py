"""CLI.

    python -m kolam_feed verify data/enc_1m --n 5000    # re-check kolam_gen output; exit 1 on any failure
    python -m kolam_feed preview data/enc_1m -o sheet.png --n 36 --aug photo   # presets: none, light, photo, real
    python -m kolam_feed augs data/enc_1m -o augs.png --n 4 --tile 224    # each augmentation on its own
    python -m kolam_feed bench data/enc_1m --size 448   # images per second, one process
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from PIL import Image

import numpy as np
from PIL import ImageDraw

from .augment import Augment, camera, effects, turn
from .compiler import compile_hypergraph
from .data import KolamDataset
from .encoding import from_encoding
from .render import Style, random_style, render_kolam
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
    ds = KolamDataset(a.src, size=a.tile, augment=a.aug, seed=a.seed)
    cols = 6
    rows = (min(a.n, len(ds)) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * a.tile, rows * a.tile), (10, 10, 12))
    for i in range(min(a.n, len(ds))):
        sheet.paste(ds.render(i)["image"].convert("RGB"), ((i % cols) * a.tile, (i // cols) * a.tile))
    sheet.save(a.out)
    print(f"saved {a.out}")


def _augs(a):
    """Rows: kolams. Columns: the clean picture, the turns, random styles (colours) each with a new
    camera angle and glare, then each image change alone, then the full presets. Same seed per cell, so the sheet is reproducible."""
    ds = KolamDataset(a.src, augment="none")
    pick = np.random.default_rng(a.seed).choice(len(ds), min(a.n, len(ds)), replace=False)
    fx = effects()
    looks = ["clean", "rot90", "mirror_lr", *(f"style {k} + angle + glare" for k in range(1, 6)), *fx]
    T, top, per = a.tile, 22, a.per_row
    band = top + len(pick) * T
    bands = (len(looks) + per - 1) // per
    sheet = Image.new("RGB", (per * T, bands * band), (10, 10, 12))
    d = ImageDraw.Draw(sheet)
    for j, name in enumerate(looks):
        x0, y0 = (j % per) * T, (j // per) * band
        d.text((x0 + 6, y0 + 5), name, fill=(230, 230, 230))
        for r, i in enumerate(pick):
            h = from_encoding(ds.encodings[i].split("|"))
            rng = np.random.default_rng([a.seed, int(i), j])
            style, aug = Style(size=T), fx.get(name, camera() if name.startswith("style") else Augment())
            if name in ("rot90", "mirror_lr"):
                h, _ = turn(h, rng, ops=(name,))
            elif name.startswith("style") or name.endswith("preset"):
                style = random_style(rng, T)
            img = render_kolam(compile_hypergraph(h, check=False), style)
            img, _ = aug.image(img, rng, style.bg, style.ink)
            sheet.paste(img.convert("RGB"), (x0, y0 + top + r * T))
    sheet.save(a.out)
    print(f"saved {a.out}: {len(pick)} kolams x {len(looks)} looks")


def _bench(a):
    ds = KolamDataset(a.src, size=a.size, augment=a.aug)
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
    s.add_argument("--aug", default="light", help="augmentation preset: none, light, photo, real")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_preview)
    g = sub.add_parser("augs", help="sheet of every augmentation on its own")
    g.add_argument("src")
    g.add_argument("-o", "--out", default="augs.png")
    g.add_argument("--n", type=int, default=4)
    g.add_argument("--tile", type=int, default=224)
    g.add_argument("--per-row", type=int, default=10, help="looks per band")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(fn=_augs)
    b = sub.add_parser("bench", help="render speed")
    b.add_argument("src")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--size", type=int, default=448)
    b.add_argument("--aug", default="light", help="augmentation preset: none, light, photo, real")
    b.set_defaults(fn=_bench)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
