"""CLI.

    python -m kolam_press draw "c([0,0],[0,3]); hl([1,1],[2,1])" -o kolam.png
    python -m kolam_press draw -f examples/showcase.kolam -o showcase.png --view panel --seq
    python -m kolam_press generate -n 2000 -o data/kolams --preview 24
    python -m kolam_press evaluate data/kolams
    python -m kolam_press gallery data/kolams -o gallery.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from .compiler import compile_hypergraph
from .dsl import format_program, parse
from .evaluate import evaluate, load
from .generator import GenConfig, generate_dataset
from .geometry import check_geometry
from .hypergraph import KolamError
from .render import Style, render_brush, render_kolam, render_panel


def _draw(a):
    text = Path(a.file).read_text() if a.file else a.program
    try:
        h = parse(text)
    except KolamError as e:
        sys.exit(f"error: {e}")
    c = compile_hypergraph(h)
    geo = check_geometry(c)
    prog = format_program(h)
    print(prog)
    print(f"strokes={geo['n_strokes']} crossings={geo['n_crossings']} proper={geo['ok']} {geo['issues'] or ''}")
    if a.seq:
        print(c.sequence_text())
    if a.json:
        print(json.dumps(h.to_json()))
    st = Style(size=a.size)
    img = {"panel": lambda: render_panel(c, prog, size=a.size),
           "kolam": lambda: render_kolam(c, st),
           "colour": lambda: render_kolam(c, st, coloured=True),
           "brush": lambda: render_brush(h, st)}[a.view]()
    img.save(a.out)
    print(f"saved {a.out}")


def _generate(a):
    cfg = GenConfig(min_size=a.min_size, max_size=a.max_size, image_size=a.size, augment=not a.no_augment)
    out = generate_dataset(a.out, a.n, seed=a.seed, cfg=cfg, workers=a.workers, preview=a.preview)
    print(f"wrote {a.n} samples to {out}")


def _evaluate(a):
    r = evaluate(a.dataset, probe=not a.no_probe, probe_n=a.probe_n)
    print((Path(a.dataset) / "report.md").read_text())
    bad = sum(v["failures"] for v in r["checks"].values()) + r["label_to_image_collisions"]
    sys.exit(1 if bad else 0)


def _gallery(a):
    recs = load(a.dataset)[: a.k]
    cols = a.cols
    t = a.tile
    rows = (len(recs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * t, rows * t), (10, 10, 12))
    for i, r in enumerate(recs):
        im = Image.open(Path(a.dataset) / r["image"]).convert("RGB").resize((t - 4, t - 4), Image.LANCZOS)
        sheet.paste(im, ((i % cols) * t + 2, (i // cols) * t + 2))
    sheet.save(a.out)
    print(f"saved {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="kolam_press", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw", help="render a brush program")
    d.add_argument("program", nargs="?", default="")
    d.add_argument("-f", "--file")
    d.add_argument("-o", "--out", default="kolam.png")
    d.add_argument("--view", choices=["panel", "kolam", "colour", "brush"], default="panel")
    d.add_argument("--size", type=int, default=384)
    d.add_argument("--seq", action="store_true", help="print the compiled lattice sequence")
    d.add_argument("--json", action="store_true", help="print the hypergraph label")
    d.set_defaults(fn=_draw)
    g = sub.add_parser("generate", help="generate a dataset")
    g.add_argument("-n", type=int, default=1000)
    g.add_argument("-o", "--out", default="data/kolams")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--size", type=int, default=384)
    g.add_argument("--min-size", type=int, default=3)
    g.add_argument("--max-size", type=int, default=8)
    g.add_argument("--workers", type=int, default=0)
    g.add_argument("--preview", type=int, default=0, help="also write brush|colour|image panels for the first N")
    g.add_argument("--no-augment", action="store_true")
    g.set_defaults(fn=_generate)
    e = sub.add_parser("evaluate", help="check a dataset and write report.md")
    e.add_argument("dataset")
    e.add_argument("--no-probe", action="store_true")
    e.add_argument("--probe-n", type=int, default=1500)
    e.set_defaults(fn=_evaluate)
    s = sub.add_parser("gallery", help="contact sheet of training images")
    s.add_argument("dataset")
    s.add_argument("-o", "--out", default="gallery.png")
    s.add_argument("-k", type=int, default=24)
    s.add_argument("--cols", type=int, default=6)
    s.add_argument("--tile", type=int, default=192)
    s.set_defaults(fn=_gallery)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
