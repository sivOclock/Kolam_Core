"""Re-check kolam_gen output with the Python rules (kolam_press is the source of truth).

Every line: the encoding parses, rows and cols agree, no two kolams are the same. A random sample of
n kolams is then checked in full: a valid canonical dataset label, compiles, passes the geometry
check, the drawing decodes back to the exact label, and the label gives back the same encoding.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

from kolam_press.compiler import compile_hypergraph, decode, observe
from kolam_press.encoding import from_encoding, to_encoding
from kolam_press.geometry import check_geometry
from kolam_press.hypergraph import KolamError

from .data import shard_files


def _lines(src):
    for f in shard_files(src):
        with open(f) as fh:
            yield from fh


def verify(src, n: int = 2000, seed: int = 0, log=print) -> dict:
    """Streams the shards twice (cheap checks, then the n sampled lines), so memory stays small."""
    fails: Counter = Counter()
    examples: dict = {}

    def fail(k, r, msg=""):
        fails[k] += 1
        examples.setdefault(k, f"id {r.get('id')}: {msg}"[:300])

    seen = set()                              # 16-byte fingerprints, not the encodings
    dist = {k: Counter() for k in ("system", "symmetry", "grid", "strokes", "horns")}
    total = 0
    for line in _lines(src):
        r = json.loads(line)
        total += 1
        key = hashlib.blake2b("|".join(r["encoding"]).encode(), digest_size=16).digest()
        if key in seen:
            fail("duplicate", r)
        seen.add(key)
        if len(r["encoding"]) != 2 * r["rows"] + 1 or len(r["encoding"][0]) != r["cols"]:
            fail("size_mismatch", r)
        dist["system"][r.get("system")] += 1
        dist["symmetry"][r.get("symmetry")] += 1
        dist["grid"][f"{r['rows']}x{r['cols']}"] += 1
        dist["strokes"][min(r.get("strokes", 0), 10)] += 1
        dist["horns"][min(r.get("horns", 0), 10)] += 1
    pick = set(np.random.default_rng(seed).permutation(total)[:n].tolist())
    sample = [json.loads(line) for i, line in enumerate(_lines(src)) if i in pick]
    for k, r in enumerate(sample, 1):
        try:
            h = from_encoding(r["encoding"])
        except KolamError as e:
            fail("unreadable", r, str(e))
            continue
        errs = h.validate()
        if errs:
            fail("invalid_label", r, "; ".join(errs))
            continue
        if to_encoding(h) != r["encoding"]:
            fail("encoding_round_trip", r)
        c = compile_hypergraph(h)
        geo = check_geometry(c)
        if not geo["ok"]:
            fail("geometry", r, "; ".join(geo["issues"]))
        if geo["n_strokes"] != r.get("strokes", geo["n_strokes"]) or geo["n_crossings"] != r.get("crossings", geo["n_crossings"]):
            fail("stats_mismatch", r, f"python {geo['n_strokes']}/{geo['n_crossings']}")
        try:
            if decode(h.rows, h.cols, *observe(c.config)) != h:
                fail("decode_mismatch", r)
        except KolamError as e:
            fail("decode_mismatch", r, str(e))
        if k % 500 == 0:
            log(f"  checked {k}/{len(sample)}")
    return {"kolams": total, "checked": len(sample), "failures": dict(fails), "examples": examples,
            "ok": not fails, "dist": {k: dict(v.most_common()) for k, v in dist.items()}}
