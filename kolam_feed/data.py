"""Encoding shards -> (image, target) on the fly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from PIL import Image, ImageDraw

from .augment import SURFACES, Augment, turn
from .compiler import compile_hypergraph
from .encoding import VOCAB, from_encoding, to_encoding, to_tokens
from .render import Style, _Canvas, random_style, render_kolam


def shard_files(src: str | Path) -> list[Path]:
    src = Path(src)
    return sorted(src.glob("shard_*.jsonl")) if src.is_dir() else [src]


class Packed:
    """A read-only list of strings in one byte buffer + offsets. DataLoader workers are forked copies of
    the dataset; a list of millions of str objects gets copied into every worker bit by bit (reading an
    object writes its refcount), a numpy buffer stays shared."""

    def __init__(self, items: list[str]):
        b = [x.encode() for x in items]
        self.off = np.zeros(len(b) + 1, np.int64)
        np.cumsum([len(x) for x in b], out=self.off[1:])
        self.buf = np.frombuffer(b"".join(b), np.uint8)

    def __len__(self) -> int:
        return len(self.off) - 1

    def __getitem__(self, i: int) -> str:
        return self.buf[self.off[i]:self.off[i + 1]].tobytes().decode()

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def load_encodings(src: str | Path) -> tuple[Packed, Packed]:
    """Every kolam in a kolam_gen output dir (or one .jsonl file): its encoding as one "|"-joined
    string and its system, each packed (Packed) so millions of them stay shared with the workers."""
    encs, systems = [], []
    for f in shard_files(src):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                encs.append("|".join(r["encoding"]))
                systems.append(r.get("system", ""))
    return Packed(encs), Packed(systems)


def target_text(enc: list[str]) -> str:
    """The model's answer: the encoding rows, one per line (spec §12)."""
    return "\n".join(enc)


def collate(items: list[dict]) -> dict:
    """Batch items as lists per key (PIL images stay PIL, ready for a VLM processor). Use as
    DataLoader(ds, batch_size=..., collate_fn=collate), or give the dataset a transform that turns
    each item into tensors and keep PyTorch's default collate."""
    return {k: [it[k] for it in items] for k in items[0]}


class KolamDataset:
    """Map-style dataset for torch.utils.data.DataLoader: item i is kolam i, augmented from
    (seed, epoch, i) (kolam_feed/augment.py): a random style when drawn (style=True), then maybe a
    turn/mirror (the label turned with it) and image changes from `augment` (a preset name: "none",
    "light", "photo", or an Augment). Kolams are only ever drawn on floors (augment.SURFACES), never
    on the non-kolam photos (those are NotKolamDataset's alone). Call set_epoch(e) before each epoch for new looks (with
    persistent_workers=True the workers keep their own copy, so leave that off). style=False and
    augment="none" give the plain default picture. `transform(item) -> anything` runs last, e.g. to
    apply an image processor and tokenizer."""

    vocab = VOCAB

    def __init__(self, src: str | Path, size: int = 448, seed: int = 0, style: bool = True,
                 augment: str | Augment = "light", transform=None):
        self.encodings, self.systems = load_encodings(src)
        self.size, self.seed, self.style, self.epoch = size, seed, style, 0
        self.aug = Augment.preset(augment) if isinstance(augment, str) else augment
        self.transform = transform

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.encodings)

    def render(self, i: int) -> dict:
        """Kolam i as {image, encoding, turn, H, M}: H maps the clean render's pixels to the image's,
        M maps a lattice point (lj, li, 1) to its pixel in the image (x, y, 1)."""
        enc = self.encodings[i].split("|")
        h = from_encoding(enc)
        rng = np.random.default_rng([self.seed, self.epoch, i])
        op = "id"
        if rng.random() < self.aug.p_turn:
            h, op = turn(h, rng)
            enc = to_encoding(h)
        style = random_style(rng, self.size) if self.style else Style(size=self.size)
        img, M = render_kolam(compile_hypergraph(h, check=False), style, frame=True)
        img, H = self.aug.image(img, rng, style.bg, style.ink)
        return {"image": img, "encoding": enc, "turn": op, "H": H, "M": H @ M}

    def __getitem__(self, i: int) -> dict:
        r = self.render(i)
        item = {"image": r["image"], "encoding": r["encoding"], "text": target_text(r["encoding"]),
                "tokens": to_tokens(r["encoding"]), "system": self.systems[i], "turn": r["turn"], "H": r["H"], "M": r["M"]}
        return self.transform(item) if self.transform else item


PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NOT_KOLAM_KINDS = ("photo", "floor", "dots", "scribble")


class NotKolamDataset:
    """Pictures that are not kolams, for the model's "is this a kolam?" head: item i is
    {"image", "kind"}, drawn from (seed, epoch, i) like KolamDataset and put through the same
    `augment`, so camera, floor and JPEG effects can't tell a kolam from a non-kolam. Kinds:
      photo     a random square crop of a photo under `photos` (e.g. Imagenette: cats, dogs, trucks)
      floor     an empty floor or paper, no kolam on it
      dots      a dot grid with no lines (looks most like a kolam)
      scribble  random smooth loops in kolam ink (lines that make no kolam)
    split "train" / "val": every 10th photo is held out for val."""

    def __init__(self, photos: str | Path | None, size: int = 448, seed: int = 0, augment: str | Augment = "light",
                 split: str = "train", n: int | None = None):
        files = sorted(f for f in Path(photos).rglob("*") if f.suffix.lower() in PHOTO_EXT) if photos else []
        self.files = [f for k, f in enumerate(files) if (k % 10 == 0) == (split == "val")]
        self.size, self.seed, self.epoch = size, seed, 0
        self.aug = Augment.preset(augment) if isinstance(augment, str) else augment
        self.n = n or max(2 * len(self.files), 2000)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def render(self, i: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch, i, 7])
        kinds, p = (NOT_KOLAM_KINDS, [0.5, 0.15, 0.2, 0.15]) if self.files else (NOT_KOLAM_KINDS[1:], [0.3, 0.4, 0.3])
        kind = str(rng.choice(kinds, p=p))
        style = random_style(rng, self.size)
        S = self.size
        if kind == "photo":
            im = Image.open(self.files[int(rng.integers(len(self.files)))]).convert("RGB")
            w, h = im.size
            c = int(min(w, h) * rng.uniform(0.6, 1.0))
            x, y = int(rng.integers(0, w - c + 1)), int(rng.integers(0, h - c + 1))
            img = im.crop((x, y, x + c, y + c)).resize((S, S), Image.BILINEAR)
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif kind == "floor":
            img = Image.new("RGB", (S, S), style.bg)
            if rng.random() < 0.8:                     # a real floor, else the plain board / paper
                img, _ = Augment(p_surface=1.0, surfaces=(str(rng.choice(SURFACES)),)).image(img, rng, style.bg)
        elif kind == "dots":
            rows, cols = (int(v) for v in rng.integers(2, 13, 2))
            cv = _Canvas(rows, cols, style)
            cv.dots()
            img = cv.finish()
        else:
            img = Image.new("RGB", (S * 2, S * 2), style.bg)
            d = ImageDraw.Draw(img)
            w = max(2, int(S * 2 * rng.uniform(0.006, 0.02)))
            for _ in range(int(rng.integers(1, 6))):
                P = rng.uniform(0.1, 0.9, (int(rng.integers(4, 12)), 2)) * S * 2
                d.line(_smooth(P, closed=rng.random() < 0.6), fill=style.ink, width=w, joint="curve")
            img = img.resize((S, S), Image.LANCZOS)
        img, _ = self.aug.image(img, rng, style.bg, style.ink)
        return {"image": img, "kind": kind}

    def __getitem__(self, i: int) -> dict:
        return self.render(i)


def _smooth(P: np.ndarray, closed: bool, n: int = 12) -> list[tuple[float, float]]:
    """Points along the chained quadratic Bezier through midpoints of P (the kolam line's own curve)."""
    if closed:
        P = np.vstack([P, P[:2]])
    M = (P[:-1] + P[1:]) / 2
    out = [tuple(M[0])]
    t = np.linspace(0, 1, n)[1:, None]
    for k in range(1, len(M)):
        a, b, c = M[k - 1], P[k], M[k]
        out += [tuple(q) for q in (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t ** 2 * c]
    return out
