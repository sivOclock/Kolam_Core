"""The Python kolam model in kolam_feed: compiler, drawing, geometry check, decoding, rendering, dataset."""
import json

import numpy as np
import pytest
from conftest import kolam

from kolam_feed import KolamDataset, collate
from kolam_feed.compiler import compile_hypergraph, decode, observe
from kolam_feed.encoding import from_encoding, to_encoding
from kolam_feed.geometry import check_geometry
from kolam_feed.render import Style, render_kolam

C, P, H = "CROSSING_LOOP", "PARALLEL_LOOP", "HORN"


def build_crossing_loop(n_dots: int):
    """Reference CROSSING_LOOP from the original research notebook
    (archive/kolam_engine_demo.ipynb): n dots in row 0, closed figure-8 chain."""
    stroke = [(0, 1)]
    for k in range(n_dots - 1):                         # forward zigzag
        stroke.append((1, 2 * k + 2))
        stroke.append((2 if k % 2 == 0 else 0, 2 * k + 3))
    last_lj = 2 * (n_dots - 1) + 1                      # arc around the last dot
    stroke.append((1, last_lj + 1))
    stroke.append((2 - stroke[-2][0], last_lj))
    for k in range(n_dots - 2, -1, -1):                 # backward zigzag: 2nd pass makes the X
        stroke.append((1, 2 * k + 2))
        stroke.append((2 if k % 2 == 0 else 0, 2 * k + 1))
    stroke.append((1, 0))                               # arc around the first dot, close
    stroke.append((0, 1))
    return stroke


def test_crossing_loop_reproduces_notebook_build_crossing_loop():
    for n in range(2, 10):
        c = compile_hypergraph(kolam(1, n, (C, [(0, k) for k in range(n)])))
        assert len(c.strokes) == 1
        s = c.strokes[0]
        assert s + s[:1] == build_crossing_loop(n)


def test_geometry_and_decoding_match_the_python_reference(kolams):
    """Same verdict and stats as the retired Python generator gave, failing kolams included; every
    kolam decodes back from its drawing."""
    for k in kolams:
        h = from_encoding(k["encoding"])
        c = compile_hypergraph(h, check=False)
        g = check_geometry(c)
        assert (g["ok"], g["n_strokes"], g.get("n_crossings")) == (k["ok"], k["n_strokes"], k["n_crossings"])
        assert g.get("min_dot_clearance") == pytest.approx(k["min_dot"], abs=1e-6)
        assert decode(h.rows, h.cols, *observe(c.config)) == h


def test_parallel_loop_has_no_crossings_and_one_stroke():
    g = check_geometry(compile_hypergraph(kolam(3, 4, (P, [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]))))
    assert g["ok"] and g["n_strokes"] == 1 and g["n_crossings_found"] == 0


def test_horn_variants_are_distinct_pictures():
    imgs = [np.asarray(render_kolam(compile_hypergraph(kolam(1, 2, (C, [(0, 0), (0, 1)]), (H, (0, 1), v, "E"))),
                                    Style(size=128)))
            for v in ("PLAIN_HORN", "LEFT_HORN", "RIGHT_HORN")]
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.abs(imgs[i].astype(int) - imgs[j]).sum() > 1000


def test_dataset_renders_on_the_fly(tmp_path, kolams):
    encs = [k["encoding"] for k in kolams[:3]]
    (tmp_path / "shard_00000.jsonl").write_text("".join(json.dumps({"encoding": e, "system": "random"}) + "\n" for e in encs))
    ds = KolamDataset(tmp_path, size=128)
    assert len(ds) == 3
    item = ds[1]
    assert item["image"].size == (128, 128) and item["encoding"] == encs[1]
    assert to_encoding(from_encoding(item["text"].split("\n"))) == encs[1]
    a = np.asarray(ds[1]["image"])
    ds.set_epoch(1)
    assert not np.array_equal(a, np.asarray(ds[1]["image"]))     # a new look each epoch
    ds.set_epoch(0)
    assert np.array_equal(a, np.asarray(ds[1]["image"]))         # reproducible
    batch = collate([ds[0], ds[2]])
    assert len(batch["image"]) == 2 and batch["encoding"] == [encs[0], encs[2]]
    ds.transform = lambda it: (np.asarray(it["image"]).shape, it["text"])
    assert ds[0][0] == (128, 128, 3)
