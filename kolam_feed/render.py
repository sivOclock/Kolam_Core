"""Rasterise compiled kolams: the training images (random styles) and the coloured-by-brush view."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .compiler import Compiled
from .geometry import drawn, per_segment, sample_stroke
from .spec import CROSSING_LOOP, HORN, PARALLEL_LOOP, to_lattice

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
    def __init__(self, rows: int, cols: int, style: Style, extent=None):
        """extent (top, bottom, left, right) in lattice units; default the dot grid's box."""
        self.s = style
        ss = style.supersample
        self.W = style.size * ss
        t, b, lf, rt = extent or (0, 2 * rows, 0, 2 * cols)
        H, W = b - t, rt - lf
        self.scale = self.W / (max(H, W) + 2 * style.margin)
        self.ox = (self.W - W * self.scale) / 2 - lf * self.scale
        self.oy = (self.W - H * self.scale) / 2 - t * self.scale
        self.img = Image.new("RGB", (self.W, self.W), style.bg)
        self.draw = ImageDraw.Draw(self.img)
        self.rows, self.cols = rows, cols

    def px(self, P: np.ndarray) -> list[tuple[float, float]]:
        P = np.asarray(P, float)
        return list(zip(self.ox + P[:, 1] * self.scale, self.oy + P[:, 0] * self.scale))

    def lw(self, w: float) -> int:
        return max(1, int(round(w * self.scale)))

    def dots(self, colour=None, radius=None, only=None):
        r = (radius or self.s.dot_radius) * self.scale
        col = colour or self.s.dot or self.s.ink
        for i in range(self.rows):
            for j in range(self.cols):
                if only is not None and (i, j) not in only:
                    continue
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
    """Hand wobble: a smooth noise field (random offsets at lattice points, bilinear in between), so
    shared points move together and a densely drawn horn stroke stays smooth."""
    if jitter <= 0:
        return strokes
    P = np.vstack([np.asarray(s, float) for s in strokes])
    lo = np.floor(P.min(0)).astype(int) - 1
    hi = np.ceil(P.max(0)).astype(int) + 2
    rng = np.random.default_rng(seed)
    N = rng.normal(0, jitter, (hi[0] - lo[0], hi[1] - lo[1], 2))

    def move(p):
        x = np.asarray(p, float) - lo
        i, j = np.floor(x).astype(int)
        a, b = x - (i, j)
        o = ((1 - a) * (1 - b) * N[i, j] + a * (1 - b) * N[i + 1, j]
             + (1 - a) * b * N[i, j + 1] + a * b * N[i + 1, j + 1])
        return (p[0] + o[0], p[1] + o[1])
    return [[move(p) for p in s] for s in strokes]


def render_kolam(c: Compiled, style: Style | None = None, coloured: bool = False) -> Image.Image:
    """The kolam image. coloured=True paints each part with the brush that made it. Only dots some
    relation uses are drawn, and the frame grows to fit a crescent that points off the grid."""
    style = style or Style()
    rows, cols = c.config.rows, c.config.cols
    shown, owners = drawn(c)
    P = np.vstack([np.asarray(s, float) for s in shown]) if shown else np.zeros((1, 2))
    ext = (min(0, P[:, 0].min()), max(2 * rows, P[:, 0].max()), min(0, P[:, 1].min()), max(2 * cols, P[:, 1].max()))
    cv = _Canvas(rows, cols, style, ext)
    if style.show_dots:
        cv.dots(only=c.config.ring_dots)
    w = cv.lw(style.line_width)
    rels = c.hypergraph.relations
    for s, own in zip(_jittered(shown, style.jitter, style.seed), owners):
        n = per_segment(s, 16)
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
