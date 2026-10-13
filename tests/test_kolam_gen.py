"""kolam_gen (Rust) against kolam_press (Python, the source of truth), and kolam_feed.

The Rust tests run when the binary is built: cd kolam_gen && cargo build --release
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from kolam_feed import KolamDataset, collate, verify
from kolam_press import parse
from kolam_press.compiler import compile_hypergraph
from kolam_press.encoding import from_encoding, to_encoding
from kolam_press.generator import GenConfig, sample_hypergraph
from kolam_press.geometry import check_geometry
from kolam_press.hypergraph import Hypergraph
from kolam_press.systems import WFC_SOURCE_KOLAM, _crop, photo_tiles, substitution, switching, wfc

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "kolam_gen" / "target" / "release" / "kolam_gen"
needs_bin = pytest.mark.skipif(not BIN.exists(), reason="kolam_gen not built (cargo build --release)")


def test_wfc_source_matches_python():
    src = (ROOT / "kolam_gen" / "src" / "systems.rs").read_text()
    block = src.split("WFC_SOURCE_ENCODING")[1].split("];")[0]
    assert re.findall(r'"([0-8]+)"', block) == to_encoding(parse(WFC_SOURCE_KOLAM))


@needs_bin
def test_geometry_check_matches_python(tmp_path):
    """Same kolams (passing and failing) -> same verdict and stats in Rust and Python."""
    rng = np.random.default_rng(11)
    tiles = photo_tiles()
    hs = [sample_hypergraph(rng, GenConfig())[0] for _ in range(40)]
    for make in (substitution, switching, lambda r: wfc(r, tiles)):
        for _ in range(4):
            h, _ = make(rng)
            if h.relations and not h.validate():
                hs.append(Hypergraph(*_crop(h)))
    hs = [h for h in hs if h.relations]
    f = tmp_path / "in.jsonl"
    f.write_text("".join(json.dumps({"encoding": to_encoding(h)}) + "\n" for h in hs))
    out = subprocess.run([BIN, "check", f], capture_output=True, text=True, check=True).stdout.splitlines()
    for h, line in zip(hs, out, strict=True):
        r, g = json.loads(line), check_geometry(compile_hypergraph(h))
        assert (r["ok"], r["n_strokes"], r["n_crossings"]) == (g["ok"], g["n_strokes"], g["n_crossings"])
        assert r["min_dot"] == pytest.approx(g["min_dot_clearance"], abs=1e-3)


@needs_bin
def test_generated_encodings_pass_python_checks(tmp_path):
    subprocess.run([BIN, "generate", "-n", "200", "-o", tmp_path, "--shard", "120"], check=True, capture_output=True)
    r = verify(tmp_path, n=60, log=lambda *_: None)
    assert r["ok"], r["examples"]
    assert r["kolams"] == 200 and set(r["dist"]["system"]) == {"random", "substitution", "switching", "wfc"}
    # resume: finished shards are kept, the result is the same
    before = (tmp_path / "shard_00000.jsonl").read_text()
    (tmp_path / "shard_00001.jsonl").unlink()
    subprocess.run([BIN, "generate", "-n", "200", "-o", tmp_path, "--shard", "120"], check=True, capture_output=True)
    assert (tmp_path / "shard_00000.jsonl").read_text() == before
    assert verify(tmp_path, n=0)["kolams"] == 200


def test_dataset_renders_on_the_fly(tmp_path):
    rng = np.random.default_rng(3)
    encs = [to_encoding(sample_hypergraph(rng, GenConfig())[0]) for _ in range(3)]
    (tmp_path / "shard_00000.jsonl").write_text("".join(json.dumps({"encoding": e, "system": "random"}) + "\n" for e in encs))
    ds = KolamDataset(tmp_path, size=128)
    assert len(ds) == 3
    item = ds[1]
    assert item["image"].size == (128, 128) and item["encoding"] == encs[1]
    assert from_encoding(item["text"].split("\n")) == from_encoding(encs[1])
    a = np.asarray(ds[1]["image"])
    ds.set_epoch(1)
    assert not np.array_equal(a, np.asarray(ds[1]["image"]))     # a new look each epoch
    ds.set_epoch(0)
    assert np.array_equal(a, np.asarray(ds[1]["image"]))         # reproducible
    batch = collate([ds[0], ds[2]])
    assert len(batch["image"]) == 2 and batch["encoding"] == [encs[0], encs[2]]
    ds.transform = lambda it: (np.asarray(it["image"]).shape, it["text"])
    assert ds[0][0] == (128, 128, 3)
