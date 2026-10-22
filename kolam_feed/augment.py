"""Augmentation: every training image a new look, the label always exact.

Three layers, all seeded by one numpy Generator (so any image can be made again):

  1. style      render.random_style: palette (chalk / ink / colour), line width, dots, margin, blur,
                noise, hand jitter. Drawn at render time.
  2. turn       one of the 8 turns/mirrors of the dot grid (dihedral group D4). It changes the label,
                so the label is turned with it (mirrors swap LEFT_HORN and RIGHT_HORN, spec §8.5).
  3. image      label-preserving changes to the rendered picture: rotation (up to +-30 degrees, shrunk to
                fit), camera tilt (foreshortening), perspective, scale and shift; glare (flash, sun sheen, wash); a real-world floor behind the kolam (cement, red oxide,
                mud wash, sand, tiles, granite, wood, paper, chalkboard, asphalt); real ink colours (rice
                flour, chalk, turmeric, kumkum, rangoli powders, several at once) and thicker lines;
                petals and dirt; lighting gradient, shadow, vignette, colour, contrast, gamma; broken chalk
                lines; motion blur, defocus, sensor noise, low resolution, JPEG. Each returns the 3x3 pixel transform it applied, so dense
                targets (dot positions for a CNN) can be moved the same way.

    aug = Augment.preset("photo")
    h2, op = turn(h, rng)                   # label + its turned picture
    img, H = aug.image(img, rng, bg, ink)   # H maps old pixel (x, y, 1) -> new pixel
"""
from __future__ import annotations

import io
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .hypergraph import Hypergraph, Relation
from .spec import HORN, HORN_DIRS, LEFT_HORN, RIGHT_HORN, horn_dir

# ------------------------------------------------------------------ 2. turns and mirrors (label-changing)
OPS = ("id", "rot90", "rot180", "rot270", "mirror_lr", "mirror_ud", "transpose", "antitranspose")
_REFLECTIONS = {"mirror_lr", "mirror_ud", "transpose", "antitranspose"}


def _vertex(v, op: str, rows: int, cols: int):
    r, c = v
    return {"id": (r, c), "rot90": (c, rows - 1 - r), "rot180": (rows - 1 - r, cols - 1 - c),
            "rot270": (cols - 1 - c, r), "mirror_lr": (r, cols - 1 - c), "mirror_ud": (rows - 1 - r, c),
            "transpose": (c, r), "antitranspose": (cols - 1 - c, rows - 1 - r)}[op]


def transform(h: Hypergraph, op: str) -> Hypergraph:
    """The label of the picture turned/mirrored by `op` (rot90 = a quarter turn clockwise on screen)."""
    rows, cols = (h.cols, h.rows) if op in ("rot90", "rot270", "transpose", "antitranspose") else (h.rows, h.cols)
    rels = []
    for r in h.relations:
        vs = tuple(_vertex(v, op, h.rows, h.cols) for v in r.vertices)
        variant, d = r.variant, r.dir
        if r.type == HORN:
            if op in _REFLECTIONS and variant in (LEFT_HORN, RIGHT_HORN):
                variant = RIGHT_HORN if variant == LEFT_HORN else LEFT_HORN
            f = horn_dir(d)
            b = _vertex((r.vertices[0][0] + f[0], r.vertices[0][1] + f[1]), op, h.rows, h.cols)
            d = next(k for k in HORN_DIRS if horn_dir(k) == (b[0] - vs[0][0], b[1] - vs[0][1]))
        rels.append(Relation(r.type, vs, variant, r.closed, d))
    return Hypergraph(rows, cols, tuple(rels)).canonical()


def turn(h: Hypergraph, rng: np.random.Generator, ops=OPS) -> tuple[Hypergraph, str]:
    op = ops[rng.integers(len(ops))]
    return transform(h, op), op


# ------------------------------------------------------------------ 3. image changes (label-preserving)
@dataclass
class Augment:
    """Probabilities (p_*) and ranges of the label-preserving image changes."""
    p_turn: float = 0.0             # chance of a D4 turn/mirror (label turned with it)
    p_geom: float = 0.0             # small rotation / perspective / scale / shift
    rotate: float = 8.0             # degrees, +-
    perspective: float = 0.06       # corner jitter, fraction of the image side
    scale: tuple = (0.85, 1.1)
    shift: float = 0.05             # fraction of the image side
    fit: bool = False               # shrink a rotated picture so its corners stay in the frame
    tilt: float = 0.0               # camera tilt: the far side of the picture shrinks by up to this fraction
    p_glare: float = 0.0            # flash hotspot, sun sheen on a shiny floor, or a soft wash
    p_surface: float = 0.0          # a real-world floor behind the kolam
    surfaces: tuple = ()            # floors to pick from (default: all of SURFACES)
    p_photo_floor: float = 0.0      # a blurred, dulled crop of a real photo as the floor (from `photos`), so
    photos: tuple = ()              #   "looks like a photo" never tells a kolam from a non-kolam
    p_ink: float = 0.0              # a real ink colour: rice flour, chalk, turmeric, kumkum, rangoli powders
    p_multi: float = 0.0            # several ink colours across the kolam
    p_thick: float = 0.0            # a thicker line
    p_debris: float = 0.0           # petals, leaves and dirt on the floor
    p_light: float = 0.0            # brightness gradient across the image
    p_shadow: float = 0.0           # a soft dark band
    p_texture: float = 0.0          # floor / paper texture (low-frequency noise)
    p_colour: float = 0.0           # brightness, contrast, gamma, channel gains
    p_chalk: float = 0.0            # broken, powdery lines
    p_lowres: float = 0.0           # down- then up-sample
    p_jpeg: float = 0.0
    p_gray: float = 0.0
    p_vignette: float = 0.0
    p_motion: float = 0.0           # camera shake
    p_defocus: float = 0.0
    p_noise: float = 0.0            # sensor noise

    @staticmethod
    def preset(name: str) -> "Augment":
        """none: style only. light: synthetic-looking variety. photo: towards phone photos of real kolams.
        real: photo plus bigger angles, real floors and inks, petals and dirt, camera effects."""
        if name == "none":
            return Augment()
        light = Augment(p_turn=0.5, p_geom=0.3, rotate=4.0, perspective=0.03, scale=(0.9, 1.05), shift=0.03,
                        p_light=0.2, p_texture=0.2, p_colour=0.3, p_chalk=0.2, p_lowres=0.1, p_jpeg=0.2)
        if name == "light":
            return light
        if name == "photo":
            return replace(light, p_turn=0.5, p_geom=0.8, rotate=10.0, perspective=0.07, scale=(0.8, 1.1),
                           shift=0.06, p_light=0.6, p_shadow=0.3, p_texture=0.6, p_colour=0.7, p_chalk=0.5,
                           p_lowres=0.3, p_jpeg=0.6, p_gray=0.1)
        if name == "real":
            return replace(Augment.preset("photo"), p_geom=1.0, rotate=30.0, fit=True, tilt=0.35, perspective=0.05,
                           scale=(0.92, 1.1), p_glare=0.7,
                           p_surface=0.85, p_chalk=0.3,
                           p_ink=0.7, p_multi=0.15, p_thick=0.4, p_debris=0.3, p_texture=0.3, p_vignette=0.4,
                           p_motion=0.15, p_defocus=0.2, p_noise=0.4)
        raise ValueError(f"unknown augmentation preset {name!r}: none, light, photo, real")

    def image(self, img: Image.Image, rng: np.random.Generator, bg, ink=None) -> tuple[Image.Image, np.ndarray]:
        """Apply the image changes; returns the new image and the 3x3 pixel transform (old -> new).
        bg / ink: the render's background and line colours (ink is guessed when not given)."""
        H = np.eye(3)
        if rng.random() < self.p_geom:
            img, H = self._geom(img, rng, bg)
        a = np.asarray(img, dtype=np.float32)
        bg = np.asarray(bg, np.float32)
        if rng.random() < self.p_chalk:
            a = _chalk(a, rng, bg)
        surface, new_ink, multi, thick = (rng.random() < p for p in (self.p_surface, self.p_ink, self.p_multi,
                                                                       self.p_thick))
        if self.photos and rng.random() < self.p_photo_floor:
            surface = _photo_floor(a.shape[:2], rng, self.photos)
        if surface is not False or new_ink or multi or thick:
            a = self._repaint(a, rng, bg, ink, surface, new_ink, multi, thick)
        if rng.random() < self.p_debris:
            a = _debris(a, rng)
        if rng.random() < self.p_texture:
            a = a * (1 + 0.12 * _smooth_noise(a.shape[:2], rng, cells=int(rng.integers(4, 24))))[..., None]
        if rng.random() < self.p_light:
            a = a * _gradient(a.shape[:2], rng, 0.35)[..., None]
        if rng.random() < self.p_shadow:
            a = a * _shadow(a.shape[:2], rng)[..., None]
        if rng.random() < self.p_glare:
            w = _glare(a.shape[:2], rng)
            a = a + (np.array([255, 252, 240], np.float32) - a) * w[..., None]
        if rng.random() < self.p_vignette:
            a = a * _vignette(a.shape[:2], rng)[..., None]
        if rng.random() < self.p_colour:
            a = _colour(a, rng)
        if rng.random() < self.p_gray:
            a = a.mean(2, keepdims=True).repeat(3, 2)
        blurred = rng.random() < self.p_motion          # at most one blur, so dense kolams stay readable
        if blurred:
            a = _motion(a, rng)
        if rng.random() < self.p_noise:
            a = a + rng.normal(0, rng.uniform(3, 12), a.shape[:2] + (1 if rng.random() < 0.5 else 3,))
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        if not blurred and rng.random() < self.p_defocus:
            blurred = True
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.6) * img.size[0] / 384))
        if not blurred and rng.random() < self.p_lowres:
            s = img.size[0]
            k = float(rng.uniform(0.35, 0.7))
            img = img.resize((max(16, int(s * k)),) * 2, Image.BILINEAR).resize((s, s), Image.BILINEAR)
        if rng.random() < self.p_jpeg:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=int(rng.integers(25, 85)))
            img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        return img, H

    def _repaint(self, a, rng, bg, ink, surface, new_ink, multi, thick):
        """Split the picture into line and background (alpha from the distance to bg), then paint the
        line in new colours over a new floor. The line's shape, and so the label, is untouched."""
        d = np.abs(a - bg).sum(2)
        ink = np.asarray(ink, np.float32) if ink is not None else a.reshape(-1, 3)[d.argmax()]
        alpha = np.clip(d / max(np.abs(ink - bg).sum(), 1.0), 0, 1)
        if thick and alpha.mean() < 0.06:                           # only where the lines have room
            k = 3 if a.shape[0] < 300 else 5
            m = Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(k))
            alpha = np.asarray(m.filter(ImageFilter.GaussianBlur(0.6)), np.float32) / 255
        if isinstance(surface, np.ndarray):                         # a photo floor, already made
            floor = surface
        elif surface:
            floor = _surface(a.shape[:2], rng, rng.choice(self.surfaces or SURFACES))
        else:
            floor = np.broadcast_to(bg, a.shape).astype(np.float32)
        lum = float((floor @ [0.3, 0.59, 0.11]).mean())
        if multi:
            cols = [_ink_colour(rng, lum, coloured=True) for _ in range(int(rng.integers(2, 5)))]
            fields = np.stack([_smooth_noise(a.shape[:2], rng, int(rng.integers(2, 6))) for _ in cols])
            w = np.exp(fields * 6)
            colour = np.einsum("khw,kc->hwc", w / w.sum(0), np.array(cols, np.float32))
        else:
            c = _ink_colour(rng, lum) if new_ink else ink
            if abs(float(np.dot(c, [0.3, 0.59, 0.11])) - lum) < 80:     # keep the line visible on its floor
                c = np.array([245, 240, 228] if lum < 128 else [25, 22, 30], np.float32)
            colour = np.broadcast_to(np.asarray(c, np.float32), a.shape)
        if new_ink or multi:                                            # powder: grainy, not quite solid
            alpha = alpha * (1 - rng.uniform(0, 0.35) * rng.random(a.shape[:2])).astype(np.float32)
        return floor * (1 - alpha[..., None]) + colour * alpha[..., None]

    def _geom(self, img, rng, bg):
        s = img.size[0]
        c = (s - 1) / 2
        t = np.radians(rng.uniform(-self.rotate, self.rotate))
        k = rng.uniform(*self.scale)
        if self.fit:
            k /= abs(np.cos(t)) + abs(np.sin(t))
        dx, dy = rng.uniform(-self.shift, self.shift, 2) * s
        A = np.array([[k * np.cos(t), -k * np.sin(t), 0], [k * np.sin(t), k * np.cos(t), 0], [0, 0, 1]])
        T = np.array([[1, 0, c + dx], [0, 1, c + dy], [0, 0, 1]]) @ A @ np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]])
        src = np.array([[0, 0], [s, 0], [s, s], [0, s]], float)          # then a perspective nudge of the corners
        dst = _apply(T, _tilt(src, s, rng, self.tilt)) + rng.uniform(-self.perspective, self.perspective, (4, 2)) * s
        H = _homography(src, dst)
        Hi = np.linalg.inv(H)                                             # PIL wants new pixel -> old pixel
        coeffs = (Hi / Hi[2, 2]).ravel()[:8]
        out = img.transform((s, s), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC, fillcolor=tuple(bg))
        return out, H


def _tilt(P, s, rng, amount: float):
    """A camera looking at the floor from an angle: one side (the far one) shrinks and the picture
    foreshortens towards it, keeping the frame's centre."""
    if amount <= 0:
        return P
    f = rng.uniform(0, amount)
    side = rng.integers(4)                                   # far side: top, right, bottom, left
    c = s / 2
    Q = P - c
    turn = np.array([[0, -1], [1, 0]]) if side % 2 else np.eye(2)
    turn = turn if side < 2 else -turn
    Q = Q @ turn                                              # far side now on top (y = -c)
    t = (c - Q[:, 1]) / (2 * c)                               # 0 at the near side, 1 at the far side
    Q = np.c_[Q[:, 0] * (1 - f * t), Q[:, 1] * (1 - 0.5 * f) - 0.25 * f * c]
    return Q @ turn.T + c


def _apply(H, P):
    q = np.c_[P, np.ones(len(P))] @ H.T
    return q[:, :2] / q[:, 2:]


def _homography(src, dst) -> np.ndarray:
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
    h = np.linalg.solve(np.array(A, float), dst.ravel())
    return np.append(h, 1).reshape(3, 3)


def _smooth_noise(shape, rng, cells) -> np.ndarray:
    """Smooth noise in [-1, 1]: a coarse random grid (cells, or (rows, cols) of them), bicubically enlarged."""
    g = rng.uniform(-1, 1, (cells, cells) if np.isscalar(cells) else cells).astype(np.float32)
    im = Image.fromarray(((g + 1) * 127.5).astype(np.uint8)).resize(shape[::-1], Image.BICUBIC)
    return np.asarray(im, np.float32) / 127.5 - 1


def _gradient(shape, rng, amount: float) -> np.ndarray:
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    t = rng.uniform(0, 2 * np.pi)
    g = (np.cos(t) * (x / w - 0.5) + np.sin(t) * (y / h - 0.5)) * 2
    return 1 + amount * rng.uniform(0.3, 1) * g


def _shadow(shape, rng) -> np.ndarray:
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    t = rng.uniform(0, np.pi)
    d = np.cos(t) * (x - rng.uniform(0, w)) + np.sin(t) * (y - rng.uniform(0, h))
    soft = rng.uniform(0.03, 0.15) * w
    return 1 - rng.uniform(0.2, 0.5) / (1 + np.exp(-d / soft))


def _colour(a, rng) -> np.ndarray:
    m = a.mean()
    a = (a - m) * rng.uniform(0.7, 1.3) + m * rng.uniform(0.75, 1.25)
    a = a * rng.uniform(0.85, 1.15, 3)
    g = rng.uniform(0.75, 1.35)
    return 255 * (np.clip(a, 0, 255) / 255) ** g


def _chalk(a, rng, bg) -> np.ndarray:
    """Powdery, broken lines: some ink pixels fade towards the background, in clumps."""
    ink = np.abs(a - bg).sum(2) > 60
    keep = _smooth_noise(a.shape[:2], rng, cells=int(rng.integers(30, 90)))
    fade = np.clip((rng.uniform(-0.2, 0.3) - keep) * 2.5, 0, 1) * ink
    fade = np.maximum(fade, (rng.random(a.shape[:2]) < 0.15) * ink * 0.6)
    out = a * (1 - fade[..., None]) + bg * fade[..., None]
    return np.asarray(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.4)),
                      np.float32)


def _vignette(shape, rng) -> np.ndarray:
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((x / w - rng.uniform(0.4, 0.6)) ** 2 + (y / h - rng.uniform(0.4, 0.6)) ** 2) * 2
    return 1 - rng.uniform(0.2, 0.55) * r2


def _glare(shape, rng) -> np.ndarray:
    """How far each pixel is washed towards white, 0..~0.6: a flash hotspot, a sun sheen across a shiny
    floor, or a soft wash from one side."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = x / w, y / h
    kind = rng.integers(3)
    cx, cy = rng.uniform(0.1, 0.9, 2)
    if kind == 0:                                                     # flash: a round hotspot
        r = rng.uniform(0.08, 0.3)
        g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r))
    elif kind == 1:                                                   # sun sheen: a long soft band
        t = rng.uniform(0, np.pi)
        d = np.cos(t) * (x - cx) + np.sin(t) * (y - cy)
        along = -np.sin(t) * (x - cx) + np.cos(t) * (y - cy)
        g = np.exp(-d * d / (2 * rng.uniform(0.03, 0.12) ** 2)) * np.exp(-along * along / (2 * rng.uniform(0.3, 0.8) ** 2))
    else:                                                             # wash from a window: one side brighter
        t = rng.uniform(0, 2 * np.pi)
        g = np.clip(0.5 + np.cos(t) * (x - 0.5) + np.sin(t) * (y - 0.5), 0, 1) ** 2
    return (g * rng.uniform(0.2, 0.5)).astype(np.float32)


def _motion(a, rng) -> np.ndarray:
    """Camera shake: the mean of copies shifted along one direction."""
    n = int(rng.integers(3, 8)) * a.shape[0] // 384 + 2
    t = rng.uniform(0, np.pi)
    out = np.zeros_like(a)
    for s in np.linspace(-n / 2, n / 2, n):
        out += np.roll(a, (int(round(s * np.sin(t))), int(round(s * np.cos(t)))), (0, 1))
    return out / n


# ------------------------------------------------------------------ real-world floors and inks
SURFACES = ("cement", "red_oxide", "mud", "sand", "tiles", "granite", "wood", "paper", "chalkboard", "asphalt")
INKS = {"flour": (240, 236, 222), "chalk": (232, 232, 228), "turmeric": (238, 186, 30), "kumkum": (200, 30, 45),
        "green": (40, 160, 70), "blue": (40, 90, 200), "pink": (230, 80, 150), "orange": (240, 120, 30),
        "purple": (120, 50, 160), "black": (25, 22, 30)}


def _ink_colour(rng, floor_lum: float, coloured: bool = False) -> np.ndarray:
    """A real kolam ink that stands out from the floor: often rice flour / chalk, else a powder colour."""
    names = list(INKS)
    p = np.array([0 if coloured else 3, 0 if coloured else 1, 1, 1, 1, 1, 1, 1, 0.5, 0 if coloured else 1.0])
    for _ in range(8):
        c = np.array(INKS[names[rng.choice(len(names), p=p / p.sum())]], np.float32) * rng.uniform(0.9, 1.05, 3)
        if abs(float(c @ [0.3, 0.59, 0.11]) - floor_lum) > 80:
            return np.clip(c, 0, 255)
    return np.array([245, 240, 228] if floor_lum < 128 else [25, 22, 30], np.float32)


def _grain(shape, rng, amount: float) -> np.ndarray:
    return rng.normal(0, amount, shape).astype(np.float32)


def _photo_floor(shape, rng, photos) -> np.ndarray:
    """A random photo crop as a floor: blurred and pulled towards its mean colour, so a kolam drawn on
    it stays readable while the floor still looks photographic."""
    im = Image.open(photos[int(rng.integers(len(photos)))]).convert("RGB")
    w, h = im.size
    c = int(min(w, h) * rng.uniform(0.5, 1.0))
    x, y = int(rng.integers(0, w - c + 1)), int(rng.integers(0, h - c + 1))
    im = im.crop((x, y, x + c, y + c)).resize((shape[1], shape[0]), Image.BILINEAR)
    im = im.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 5.0) * shape[0] / 384))
    a = np.asarray(im, np.float32)
    k = rng.uniform(0.45, 0.8)
    return a * (1 - k) + a.mean((0, 1)) * k


def _surface(shape, rng, kind: str) -> np.ndarray:
    """A procedural floor, (h, w, 3) in 0..255."""
    h, w = shape
    n = lambda c: _smooth_noise(shape, rng, c)
    def tint(base, spread):
        return np.asarray(base, np.float32) + rng.uniform(-spread, spread, 3)
    if kind == "cement":
        v = tint([150, 148, 142], 25)
        g = 18 * n(6) + 10 * n(40) + _grain(shape, rng, 9) - 40 * (rng.random(shape) < 0.004)
        return v + g[..., None]
    if kind == "red_oxide":
        v = tint([150, 52, 42], 20)
        return v + (14 * n(5) + 4 * _grain(shape, rng, 1))[..., None] + 35 * np.clip(n(3), 0, 1)[..., None]
    if kind == "mud":
        v = tint([115, 98, 62], 18)
        streak = n((int(rng.integers(30, 70)), 4)) if rng.random() < 0.5 else n((4, int(rng.integers(30, 70))))
        return v + (16 * streak + 10 * n(12) + _grain(shape, rng, 6))[..., None]
    if kind == "sand":
        v = tint([196, 166, 122], 18)
        return v + (10 * n(8) + _grain(shape, rng, 14))[..., None]
    if kind == "tiles":
        base = tint([[226, 220, 205], [180, 90, 60], [150, 150, 150], [235, 235, 235]][rng.integers(4)], 15)
        s = int(rng.uniform(0.12, 0.35) * w)
        y, x = np.mgrid[0:h, 0:w]
        ox, oy = rng.integers(0, s, 2)
        ti, tj = (y + oy) // s, (x + ox) // s
        jitter = rng.uniform(-12, 12, (ti.max() + 1, tj.max() + 1, 3)).astype(np.float32)
        out = base + jitter[ti, tj] + (6 * n(20) + _grain(shape, rng, 3))[..., None]
        vein = np.abs(np.sin(n(5) * 9 + n(17) * 3))
        out -= (rng.uniform(0, 30) * (vein < 0.08))[..., None]
        grout = ((y + oy) % s < max(2, s // 40)) | ((x + ox) % s < max(2, s // 40))
        out[grout] = base * 0.55
        return out
    if kind == "granite":
        v = tint([60, 58, 62], 20)
        g = _grain(shape, rng, 10) + 50 * (rng.random(shape) < 0.05) * rng.uniform(-1, 1, shape)
        return v + (np.asarray(Image.fromarray(np.clip(g + 128, 0, 255).astype(np.uint8))
                               .filter(ImageFilter.GaussianBlur(0.8)), np.float32) - 128)[..., None]
    if kind == "wood":
        a, b = tint([150, 105, 65], 20), tint([110, 72, 42], 15)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        along = x if rng.random() < 0.5 else y
        t = along / w * rng.uniform(5, 12) + 0.6 * n((3, 8))
        m = (0.5 + 0.5 * np.sin(t * 2 * np.pi)) ** 3
        fine = n((4, 90)) if along is x else n((90, 4))
        return (a * (1 - m[..., None]) + b * m[..., None]) + (6 * fine + _grain(shape, rng, 4))[..., None]
    if kind == "paper":
        v = tint([236, 230, 214], 10)
        return v + (5 * n(10) + 4 * n(80) + _grain(shape, rng, 3))[..., None]
    if kind == "chalkboard":
        v = tint([[34, 52, 40], [30, 32, 36]][rng.integers(2)], 8)
        return v + (12 * np.clip(n(7), 0, 1) + 4 * n(30) + _grain(shape, rng, 3))[..., None]
    if kind == "asphalt":
        v = tint([62, 62, 64], 12)
        return v + (8 * n(10) + _grain(shape, rng, 16))[..., None]
    raise ValueError(f"unknown surface {kind!r}")


def _debris(a, rng) -> np.ndarray:
    """Petals, leaves and specks of dirt: small, so the kolam stays readable."""
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    s = a.shape[0] / 384
    cols = [(240, 140, 20), (250, 200, 40), (200, 30, 50), (240, 240, 230), (60, 130, 50), (70, 50, 35)]
    for _ in range(int(rng.integers(4, 30))):
        x, y = rng.uniform(0, a.shape[1]), rng.uniform(0, a.shape[0])
        r = rng.uniform(1, 5) * s
        c = cols[rng.integers(len(cols))]
        d.ellipse([x - r, y - r * rng.uniform(0.4, 1), x + r, y + r * rng.uniform(0.4, 1)], fill=c)
    return np.asarray(img, np.float32)


# ------------------------------------------------------------------ one effect at a time (for the sheet)
def camera() -> Augment:
    """A new angle and glare on every picture (the sheet's style columns)."""
    return Augment(p_geom=1.0, rotate=30.0, fit=True, tilt=0.35, perspective=0.03, scale=(0.95, 1.05), p_glare=0.8)


def effects() -> dict[str, Augment]:
    """Each image change on its own at full strength, for `python -m kolam_feed augs`."""
    g = dict(p_geom=1.0, rotate=0.0, perspective=0.0, scale=(1.0, 1.0), shift=0.0)
    fx = {
        "rotate 30": Augment(**g | {"rotate": 30.0, "fit": True}),
        "perspective": Augment(**g | {"perspective": 0.1}),
        "camera tilt": Augment(**g | {"tilt": 0.35, "fit": True}),
        "glare": Augment(p_glare=1.0),
        "scale+shift": Augment(**g | {"scale": (0.8, 1.1), "shift": 0.06}),
    }
    fx |= {k: Augment(p_surface=1.0, surfaces=(k,)) for k in SURFACES}
    fx |= {
        "ink colour": Augment(p_ink=1.0), "multi-colour": Augment(p_multi=1.0), "thick line": Augment(p_thick=1.0),
        "petals+dirt": Augment(p_debris=1.0),
        "light": Augment(p_light=1.0), "shadow": Augment(p_shadow=1.0), "vignette": Augment(p_vignette=1.0),
        "texture": Augment(p_texture=1.0), "colour": Augment(p_colour=1.0), "chalk": Augment(p_chalk=1.0),
        "motion blur": Augment(p_motion=1.0), "defocus": Augment(p_defocus=1.0), "noise": Augment(p_noise=1.0),
        "low-res": Augment(p_lowres=1.0), "jpeg": Augment(p_jpeg=1.0), "gray": Augment(p_gray=1.0),
    }
    return fx | {f"{k} preset": Augment.preset(k) for k in ("light", "photo", "real")}
