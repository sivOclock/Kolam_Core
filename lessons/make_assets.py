"""Build the assets the lesson pages use (gitignored, local only).

Run from the repo root:  .venv/bin/python lessons/make_assets.py

Lesson 1: a real 2x2 kolam (crossing-loop square + one crescent horn), rendered by
kolam_feed.render, plus every gap's pixel position so the page can point at gaps while
it steps through the encoding token by token. The kolam passes compile_hypergraph()
strict, i.e. it is exactly the kind of label kolam_gen emits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kolam_feed.compiler import compile_hypergraph
from kolam_feed.encoding import encoding_grid, to_encoding, to_tokens
from kolam_feed.hypergraph import Relation, from_gates
from kolam_feed.render import Style, render_kolam
from kolam_feed.spec import CROSS, HORN, gate_between

OUT = Path(__file__).resolve().parent / "assets"


def lesson1():
    rows = cols = 2
    size = 512

    gates = {}
    loop = [(0, 0), (0, 1), (1, 1), (1, 0)]
    for a, b in zip(loop, loop[1:] + loop[:1]):
        gates[gate_between(a, b)] = CROSS
    horns = [Relation(HORN, ((0, 0),), "PLAIN_HORN", False, "W")]
    h = from_gates(rows, cols, gates, horns)

    enc = to_encoding(h)
    tokens = to_tokens(enc)
    c = compile_hypergraph(h)          # strict=True: must be a valid dataset label
    img, M = render_kolam(c, Style(size=size, bg=(30, 34, 44)), frame=True)

    g = encoding_grid(enc)
    gaps = []
    for li in range(2 * rows + 1):
        for lj in range(1 - li % 2, 2 * cols + 1, 2):
            p = M @ np.array([lj, li, 1.0])
            gaps.append({"li": li, "lj": lj, "d": int(g[li, lj]),
                         "x": round(float(p[0]), 1), "y": round(float(p[1]), 1)})

    tokmap, gi = [], 0
    for t in tokens:
        if t == "|":
            tokmap.append(-1)
        elif t == "<end>":
            tokmap.append(-2)
        else:
            tokmap.append(gi)
            gi += 1
    assert gi == len(gaps), "token order and gap order disagree"

    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / "lesson-1.png")
    data = {"size": size, "rows": rows, "cols": cols, "enc": enc,
            "tokens": tokens, "tokmap": tokmap, "gaps": gaps}
    (OUT / "lesson-1.js").write_text("window.LESSON1 = " + json.dumps(data, indent=1) + ";\n")
    print("encoding:", enc)
    print("tokens:   ", tokens)
    print("wrote assets/lesson-1.png and assets/lesson-1.js")


if __name__ == "__main__":
    lesson1()
