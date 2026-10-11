import json
from pathlib import Path

import numpy as np
import pytest

from kolam_press import (Hypergraph, KolamError, Relation, check_geometry, compile_hypergraph, decode,
                         format_program, is_isomorphic, observe, parse, render_kolam)
from kolam_press.generator import GenConfig, brush_line_components, generate_sample, sample_hypergraph
from kolam_press.hypergraph import transform_relation
from kolam_press.render import Style


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


# ------------------------------------------------------------------ compiler
def test_crossing_brush_reproduces_notebook_build_crossing_loop():
    for n in range(2, 10):
        c = compile_hypergraph(parse(f"c([0,0],[0,{n - 1}])"))
        assert len(c.strokes) == 1
        s = c.strokes[0]
        assert s + s[:1] == build_crossing_loop(n)


def test_three_brushes_compile_to_proper_kolam():
    h = parse(Path(__file__).resolve().parents[1].joinpath("examples/showcase.kolam").read_text())
    c = compile_hypergraph(h)
    g = check_geometry(c)
    assert g["ok"], g["issues"]
    assert g["n_crossings"] == 8 and g["n_crossings_found"] == 8
    assert all(p[0] % 2 == 0 or p[1] % 2 == 0 for s in c.strokes for p in s), "path visits an anchor"


def test_parallel_loop_has_no_crossings_and_one_stroke():
    c = compile_hypergraph(parse("p([0,0],[0,3],[2,3])"))
    g = check_geometry(c)
    assert g["ok"] and g["n_strokes"] == 1 and g["n_crossings_found"] == 0


def test_horn_variants_are_distinct_pictures():
    imgs = [np.asarray(render_kolam(compile_hypergraph(parse(f"{b}([1,0],[1,1])")), Style(size=128)))
            for b in ("h", "hl", "hr")]
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.abs(imgs[i].astype(int) - imgs[j]).sum() > 1000


def test_mirror_swaps_left_and_right_horns():
    """Mirroring the picture == mirroring the label (LEFT <-> RIGHT), checked on pixels."""
    h = parse("grid(3,4); c([0,0],[0,2]); hl([1,0],[2,0]); hr([1,3],[1,2]); h([2,3],[2,2])")
    mh = Hypergraph(h.rows, h.cols,
                    tuple(transform_relation(r, "mirror_lr", h.rows, h.cols) for r in h.relations)).canonical()
    assert {r.variant for r in mh.relations if r.type == "HORN"} == {"LEFT_HORN", "RIGHT_HORN", "PLAIN_HORN"}
    a = np.asarray(render_kolam(compile_hypergraph(h), Style(size=200)), dtype=int)
    b = np.asarray(render_kolam(compile_hypergraph(mh), Style(size=200)), dtype=int)
    assert np.abs(a[:, ::-1] - b).mean() < 1.0


def test_sequence_text():
    c = compile_hypergraph(parse("c([0,0],[0,1])"))
    assert c.sequence_text().startswith("[0,1] → [1,2] → [2,3]")


# ------------------------------------------------------------------ DSL + labels
def test_dsl_aliases_endpoints_and_rings():
    assert parse("h1([0,0],[0,1])") == parse("hl([0,0],[0,1])") == parse("LEFT_HORN([0,0],[0,1])")
    assert parse("c([0,0],[0,3])").relations[0].vertices == ((0, 0), (0, 1), (0, 2), (0, 3))
    ring = parse("p([0,0],[0,1],[1,1],[1,0],[0,0])").relations[0]
    assert ring.closed and len(ring.vertices) == 4


@pytest.mark.parametrize("prog,msg", [
    ("c([0,0],[2,2])", "diagonal"),
    ("h([0,0],[1,1])", "neighbours"),
    ("x([0,0],[0,1])", "unknown brush"),
    ("c([0,0],[0,1]); c([0,1],[1,1])", "same-brush"),
    ("c([0,0],[0,1]); h([1,1],[0,1])", "tip dot"),
    ("c([0,0],[0,2]); p([0,1],[0,2])", "can.t be both"),
    ("grid(2,2); c([0,0],[0,3])", "outside"),
])
def test_dsl_rejects_bad_programs(prog, msg):
    with pytest.raises(KolamError, match=msg):
        parse(prog)


def test_crossing_and_parallel_may_share_dots():
    h = parse("c([0,0],[0,2]); p([0,1],[2,1])")
    assert check_geometry(compile_hypergraph(h))["ok"]


def test_canonical_labels_ignore_start_and_direction():
    a = parse("c([0,0],[0,2])")
    b = parse("c([0,2],[0,0])")
    assert a == b
    r1 = parse("c([0,0],[0,1],[1,1],[1,0],[0,0])")
    r2 = parse("c([1,1],[0,1],[0,0],[1,0],[1,1])")
    assert r1 == r2
    assert parse("h([0,0],[0,1])") != parse("h([0,1],[0,0])")      # horn order is visible


def test_program_and_json_roundtrip():
    h = parse(Path(__file__).resolve().parents[1].joinpath("examples/showcase.kolam").read_text())
    assert parse(format_program(h)) == h
    assert Hypergraph.from_json(json.loads(json.dumps(h.to_json()))) == h


def test_isomorphism():
    a = parse("grid(4,4); c([0,0],[0,2]); hl([0,1],[1,1])")
    b = parse("grid(4,4); c([2,1],[2,3]); hl([2,2],[3,2])")     # translated
    c = parse("grid(4,4); c([0,0],[0,2]); hr([0,1],[1,1])")     # other variant
    d = parse("grid(4,4); c([0,0],[0,2]); hl([0,0],[1,0])")     # horn on the chain's end, not middle
    assert is_isomorphic(a, b)
    assert not is_isomorphic(a, c)
    assert not is_isomorphic(a, d)


# ------------------------------------------------------------------ generator
@pytest.fixture(scope="module")
def samples():
    cfg = GenConfig(augment=False, image_size=160)
    return [generate_sample(i, 123, cfg) for i in range(60)]


def test_generated_samples_are_proper_and_decodable(samples):
    for rec, c, _ in samples:
        h = Hypergraph.from_json(rec["hypergraph"])
        assert not h.validate()
        assert parse(rec["program"]) == h
        g = check_geometry(c)
        assert g["ok"], (rec["id"], g["issues"])
        assert decode(h.rows, h.cols, *observe(c.config)) == h


def test_forest_brush_lines_give_one_stroke_per_piece(samples):
    seen = 0
    for rec, c, _ in samples:
        forest, pieces = brush_line_components(c.hypergraph.relations)
        if forest:
            seen += 1
            assert len(c.strokes) == pieces, rec["id"]
    assert seen > 10


def test_generation_is_deterministic():
    cfg = GenConfig(image_size=128)
    a, ca, sa = generate_sample(5, 99, cfg)
    b, cb, sb = generate_sample(5, 99, cfg)
    assert a == b
    assert render_kolam(ca, sa).tobytes() == render_kolam(cb, sb).tobytes()


def test_sampler_covers_all_brushes_and_symmetries():
    cfg = GenConfig()
    types, syms = set(), set()
    for i in range(80):
        h, meta = sample_hypergraph(np.random.default_rng(i), cfg)
        types |= {(r.type, r.variant) for r in h.relations}
        syms.add(meta["symmetry"])
    assert {t for t, _ in types} == {"CROSSING_LOOP", "PARALLEL_LOOP", "HORN"}
    assert {v for _, v in types if v} == {"PLAIN_HORN", "LEFT_HORN", "RIGHT_HORN"}
    assert len(syms) >= 5


def test_relation_rejects_bad_json():
    h = Hypergraph(2, 2, (Relation("HORN", ((0, 0), (0, 1)), "UP_HORN"),))
    assert h.validate()


# ------------------------------------------------------------------ free drawing (strict=False)
@pytest.mark.parametrize("prog", [
    "grid(3,3); p([0,0],[0,2]); p([0,1],[2,1])",           # same-brush loops sharing a dot
    "grid(3,3); c([0,0],[0,2]); c([0,1],[2,1])",
    "grid(2,3); c([0,0],[0,2]); c([0,1],[0,2])",           # two strokes on one gap, same state
])
def test_free_drawings_draw_cleanly(prog):
    h = parse(prog, strict=False)
    with pytest.raises(KolamError):
        parse(prog)                                        # not a dataset label
    g = check_geometry(compile_hypergraph(h, strict=False))
    assert g["ok"], g["issues"]


@pytest.mark.parametrize("prog,msg", [
    ("c([0,0],[1,1])", "diagonal"),                        # diagonal dots share no gate
    ("c([0,0],[0,2]); p([0,1],[0,2])", "can.t be both"),   # one gap, two states
])
def test_free_drawing_still_refuses_the_impossible(prog, msg):
    with pytest.raises(KolamError, match=msg):
        parse(prog, strict=False)
