"""Encoding shards -> (image, target) on the fly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .augment import Augment, turn
from .compiler import compile_hypergraph
from .encoding import VOCAB, from_encoding, to_encoding, to_tokens
from .render import Style, random_style, render_kolam


def shard_files(src: str | Path) -> list[Path]:
    src = Path(src)
    return sorted(src.glob("shard_*.jsonl")) if src.is_dir() else [src]


def load_encodings(src: str | Path) -> tuple[list[str], list[str]]:
    """Every kolam in a kolam_gen output dir (or one .jsonl file): its encoding as one "|"-joined
    string (compact in memory) and its system."""
    encs, systems = [], []
    for f in shard_files(src):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                encs.append("|".join(r["encoding"]))
                systems.append(r.get("system", ""))
    return encs, systems


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
    "light", "photo", or an Augment). Call set_epoch(e) before each epoch for new looks (with
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
