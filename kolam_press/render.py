"""Rasterise compiled kolams (training images) and brush programs (skeleton previews)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .compiler import Compiled
from .geometry import sample_stroke
from .hypergraph import Hypergraph
from .spec import CROSSING_LOOP, HORN, LEFT_HORN, PARALLEL_LOOP, RIGHT_HORN, left_of, to_lattice

BRUSH_COLORS = {CROSSING_LOOP: (236, 98, 78), PARALLEL_LOOP: (74, 158, 232), HORN: (86, 196, 120)}


@dataclass
class Style:
    size: int = 384
    bg: tuple = (22, 26, 36)
    ink: tuple = (238, 234, 222)
    dot: tuple | None = None
    line_width: float = 0.10     # lattice units (dots are 2 apart)
    dot_radius: float = 0.12
    margin: float = 0.8
    show_dots: bool = True
    supersample: int = 3
    blur: float = 0.0            # px, after downsampling
    noise: float = 0.0           # gaussian std, 0..255
    jitter: float = 0.0          # lattice units, per control point
    seed: int = 0

    def to_json(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def random_style(rng: np.random.Generator, size: int = 384) -> Style:
    """Label-preserving appearance augmentation: palette, widths, dots, blur, noise, hand jitter."""
    kind = rng.choice(["chalk", "ink", "colour"], p=[0.5, 0.3, 0.2])
    if kind == "chalk":
        bg = tuple(int(x) for x in rng.integers([10, 12, 16], [70, 60, 60]))
        ink = tuple(int(x) for x in rng.integers([215, 210, 200], [256, 256, 246]))
    elif kind == "ink":
        bg = tuple(int(x) for x in rng.integers([225, 218, 205], [256, 252, 245]))
        ink = tuple(int(x) for x in rng.integers([0, 0, 0], [70, 60, 90]))
    else:
        bg = tuple(int(x) for x in rng.integers([90, 20, 20], [200, 90, 80]))
        ink = tuple(int(x) for x in rng.integers([225, 205, 150], [256, 256, 256]))
    return Style(size=size, bg=bg, ink=ink,
                 line_width=float(rng.uniform(0.07, 0.15)),
                 dot_radius=float(rng.uniform(0.08, 0.16)),
                 margin=float(rng.uniform(0.5, 1.4)),
                 show_dots=bool(rng.random() < 0.9),
                 blur=float(rng.choice([0.0, 0.0, rng.uniform(0.3, 1.0)])),
                 noise=float(rng.choice([0.0, rng.uniform(2, 10)])),
                 jitter=float(rng.choice([0.0, rng.uniform(0.01, 0.05)])),
                 seed=int(rng.integers(1 << 31)))


class _Canvas:
    def __init__(self, rows: int, cols: int, style: Style):
        self.s = style
        ss = style.supersample
        self.W = style.size * ss
        H, W = 2 * rows, 2 * cols
        self.scale = self.W / (max(H, W) + 2 * style.margin)
        self.ox = (self.W - W * self.scale) / 2
        self.oy = (self.W - H * self.scale) / 2
        self.img = Image.new("RGB", (self.W, self.W), style.bg)
        self.draw = ImageDraw.Draw(self.img)
        self.rows, self.cols = rows, cols

    def px(self, P: np.ndarray) -> list[tuple[float, float]]:
        P = np.asarray(P, float)
        return list(zip(self.ox + P[:, 1] * self.scale, self.oy + P[:, 0] * self.scale))

    def lw(self, w: float) -> int:
        return max(1, int(round(w * self.scale)))

    def dots(self, colour=None, radius=None):
        r = (radius or self.s.dot_radius) * self.scale
        col = colour or self.s.dot or self.s.ink
        for i in range(self.rows):
            for j in range(self.cols):
                (x, y), = self.px(np.array([to_lattice((i, j))]))
                self.draw.ellipse([x - r, y - r, x + r, y + r], fill=col)

    def finish(self) -> Image.Image:
        img = self.img.resize((self.s.size, self.s.size), Image.LANCZOS)
        if self.s.blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(self.s.blur))
        if self.s.noise > 0:
            rng = np.random.default_rng(self.s.seed)
            a = np.asarray(img, dtype=np.float32) + rng.normal(0, self.s.noise, (self.s.size, self.s.size, 1))
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return img


def _jittered(strokes, jitter: float, seed: int):
    if jitter <= 0:
        return strokes
    rng = np.random.default_rng(seed)
    pts = sorted({p for s in strokes for p in s})
    off = dict(zip(pts, rng.normal(0, jitter, (len(pts), 2))))
    return [[(p[0] + off[p][0], p[1] + off[p][1]) for p in s] for s in strokes]


def render_kolam(c: Compiled, style: Style | None = None, coloured: bool = False) -> Image.Image:
    """The kolam image. coloured=True paints each part with the brush that made it."""
    style = style or Style()
    cv = _Canvas(c.config.rows, c.config.cols, style)
    if style.show_dots:
        cv.dots()
    n = 16
    w = cv.lw(style.line_width)
    rels = c.hypergraph.relations
    for s, own in zip(_jittered(c.strokes, style.jitter, style.seed), c.owners):
        P = sample_stroke(s, n)
        if not coloured:
            xy = cv.px(np.vstack([P, P[:1]]))
            cv.draw.line(xy, fill=style.ink, width=w, joint="curve")
            continue
        k = len(s)
        for i in range(k):
            seg = np.vstack([P[i * n:(i + 1) * n], P[((i + 1) % k) * n][None]])
            colr = BRUSH_COLORS[rels[own[i]].type] if own[i] >= 0 else style.ink
            cv.draw.line(cv.px(seg), fill=colr, width=w, joint="curve")
            r = w / 2
            for x, y in cv.px(seg[[0, -1]]):
                cv.draw.ellipse([x - r, y - r, x + r, y + r], fill=colr)
    return cv.finish()


def render_brush(h: Hypergraph, style: Style | None = None) -> Image.Image:
    """The brush program itself: coloured strokes through the dots (what you 'draw')."""
    style = style or Style()
    cv = _Canvas(h.rows, h.cols, style)
    cv.dots(colour=(120, 124, 138), radius=0.16)
    w = cv.lw(0.22)
    for r in h.relations:
        col = BRUSH_COLORS[r.type]
        L = np.array([to_lattice(v) for v in r.vertices], float)
        if r.type == HORN:
            a, b = L
            f = (b - a) / 2
            lft = np.array(left_of((int(f[0]), int(f[1]))), float)
            tip = b + 0.55 * f
            cv.draw.line(cv.px(np.array([a, tip])), fill=col, width=w)
            barbs = {LEFT_HORN: [1], RIGHT_HORN: [-1]}.get(r.variant, [1, -1])
            for sgn in barbs:
                cv.draw.line(cv.px(np.array([tip, tip - 0.6 * f + sgn * 0.5 * lft])), fill=col, width=w)
            continue
        if r.closed:
            L = np.vstack([L, L[:1]])
        cv.draw.line(cv.px(L), fill=col, width=w, joint="curve")
        rad = 0.3 * cv.scale
        for x, y in cv.px(L):
            cv.draw.ellipse([x - rad, y - rad, x + rad, y + rad], outline=col, width=max(1, w // 2))
    return cv.finish()


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def render_panel(c: Compiled, program: str = "", size: int = 320, train_style: Style | None = None,
                 title: str = "") -> Image.Image:
    """brush program | kolam coloured by brush | training image, with the program underneath."""
    base = Style(size=size)
    views = [render_brush(c.hypergraph, base), render_kolam(c, base, coloured=True),
             render_kolam(c, replace(train_style, size=size) if train_style else base)]
    gap, pad = 8, 6
    lines = ([title] if title else []) + program.splitlines()
    font = _font(13)
    text_h = 17 * len(lines) + 2 * pad
    W = 3 * size + 4 * gap
    out = Image.new("RGB", (W, size + 2 * gap + text_h), (14, 16, 22))
    for i, v in enumerate(views):
        out.paste(v, (gap + i * (size + gap), gap))
    d = ImageDraw.Draw(out)
    for k, line in enumerate(lines):
        d.text((gap + pad, size + 2 * gap + pad + 17 * k), line, fill=(200, 204, 214), font=font)
    return out
