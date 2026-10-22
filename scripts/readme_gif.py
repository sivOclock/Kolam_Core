"""A kolam drawing itself, as a GIF for the README: the dots appear, then each line is traced at a steady
pen speed, on the chalkboard look of the site. Drawn with the dataset's own geometry (geometry.drawn).

    python scripts/readme_gif.py data/enc_demo_100k 66 docs/readme/drawing.gif
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from kolam_feed.compiler import compile_hypergraph
from kolam_feed.data import load_encodings
from kolam_feed.encoding import from_encoding
from kolam_feed.geometry import drawn, per_segment, sample_stroke
from kolam_feed.spec import to_lattice

S, SS = 560, 2                       # picture side, supersampling
BG0, BG1 = (46, 107, 82), (26, 53, 48)
CHALK = (249, 250, 251)


def board() -> Image.Image:
    """The chalkboard: a soft radial gradient, like the site's boards."""
    y, x = np.mgrid[0:S * SS, 0:S * SS] / (S * SS)
    t = np.clip(np.hypot(x - 0.55, y - 0.45) / 0.75, 0, 1)[..., None]
    a = (np.array(BG0) * (1 - t) + np.array(BG1) * t).astype(np.uint8)
    return Image.fromarray(a)


def main(src: str, index: int, out: str, frames_dots: int = 14, frames_lines: int = 70, hold: int = 30) -> None:
    enc = load_encodings(src)[0][index].split("|")
    h = from_encoding(enc)
    c = compile_hypergraph(h, check=False)
    shown, _ = drawn(c)
    lines = [sample_stroke(s, per_segment(s, 16)) for s in shown]
    lines = [np.vstack([P, P[:1]]) for P in lines]
    pts = np.vstack(lines)
    lo, hi = np.minimum(pts.min(0), 0), np.maximum(pts.max(0), [2 * h.rows, 2 * h.cols])
    side = (hi - lo).max() + 2.4
    scale = S * SS / side
    off = (S * SS - (hi - lo)[::-1] * scale) / 2 - lo[::-1] * scale
    px = lambda P: [(off[0] + q[1] * scale, off[1] + q[0] * scale) for q in P]
    dots = sorted(c.config.ring_dots or {(r, k) for r in range(h.rows) for k in range(h.cols)})
    width, r = max(2, int(0.11 * scale)), 0.13 * scale
    base = board()
    seg_len = [np.hypot(*np.diff(P, axis=0).T) for P in lines]
    total = sum(s.sum() for s in seg_len)
    out_frames: list[Image.Image] = []

    def snapshot(im):
        out_frames.append(im.resize((S, S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.35)))

    im = base.copy()
    d = ImageDraw.Draw(im)
    snapshot(im)
    for f in range(1, frames_dots + 1):                           # the dots, row by row
        for (i, j) in dots[: round(len(dots) * f / frames_dots)]:
            (x, y), = px([to_lattice((i, j))])
            d.ellipse([x - r, y - r, x + r, y + r], fill=CHALK)
        snapshot(im)
    budget = total / frames_lines                                 # the lines, at one pen speed
    li, k, left = 0, 0, budget
    while li < len(lines):
        P, L = lines[li], seg_len[li]
        while li < len(lines) and left > 0:
            P, L = lines[li], seg_len[li]
            d.line(px(P[k:k + 2]), fill=CHALK, width=width, joint="curve")
            left -= L[k]
            k += 1
            if k >= len(L):
                li, k = li + 1, 0
        snapshot(im)
        left = budget
    pal = out_frames[-1].quantize(colors=64, method=Image.Quantize.MEDIANCUT)   # one palette: the chalk stays white
    out_frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in out_frames]
    out_frames += [out_frames[-1]] * hold                         # a rest on the finished kolam
    dur = [70] + [60] * frames_dots + [40] * (len(out_frames) - frames_dots - 1 - hold) + [80] * hold
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:], duration=dur, loop=0, optimize=True)
    print(f"{out}: {h.rows}x{h.cols} dots, {len(shown)} lines, {len(out_frames)} frames")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
