"""kolam_feed: kolam encodings (from kolam_gen) -> images rendered on the fly -> the model.

    from kolam_feed import KolamDataset
    ds = KolamDataset("data/enc_1m", size=448)
    item = ds[0]          # {"image": PIL.Image, "encoding": [...], "text": "...", "tokens": [...]}

Rendering uses kolam_press (the source of truth): from_encoding -> compile -> render_kolam with a
fresh random style per (seed, epoch, index), so every epoch shows each kolam in a new look.
"""
from .data import KolamDataset, collate, load_encodings, target_text
from .verify import verify

__all__ = ["KolamDataset", "collate", "load_encodings", "target_text", "verify"]
