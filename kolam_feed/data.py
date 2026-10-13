"""Encoding shards -> (image, target) on the fly."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kolam_press.compiler import compile_hypergraph
from kolam_press.encoding import VOCAB, from_encoding, to_tokens
from kolam_press.render import Style, random_style, render_kolam


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
    """Map-style dataset for torch.utils.data.DataLoader: item i is kolam i rendered in a random style
    drawn from (seed, epoch, i). Call set_epoch(e) before each epoch for new looks (with
    persistent_workers=True the workers keep their own copy, so leave that off). With augment=False
    every item is drawn in the plain default style. `transform(item) -> anything` runs last, e.g. to
    apply an image processor and tokenizer."""

    vocab = VOCAB

    def __init__(self, src: str | Path, size: int = 448, seed: int = 0, augment: bool = True, transform=None):
        self.encodings, self.systems = load_encodings(src)
        self.size, self.seed, self.augment, self.epoch = size, seed, augment, 0
        self.transform = transform

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.encodings)

    def render(self, i: int):
        enc = self.encodings[i].split("|")
        c = compile_hypergraph(from_encoding(enc), check=False)
        rng = np.random.default_rng([self.seed, self.epoch, i])
        style = random_style(rng, self.size) if self.augment else Style(size=self.size)
        return render_kolam(c, style), enc

    def __getitem__(self, i: int) -> dict:
        img, enc = self.render(i)
        item = {"image": img, "encoding": enc, "text": target_text(enc), "tokens": to_tokens(enc),
                "system": self.systems[i]}
        return self.transform(item) if self.transform else item
