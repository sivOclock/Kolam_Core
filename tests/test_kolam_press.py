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
    imgs = [np.asarray(render_kolam(compile_hypergraph(parse(f"grid(1,2); c([0,0],[0,1]); {b}([0,1],E)")),
                                    Style(size=128)))
            for b in ("h", "hl", "hr")]
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.abs(imgs[i].astype(int) - imgs[j]).sum() > 1000


def test_mirror_swaps_left_and_right_horns():
    """Mirroring the picture == mirroring the label (LEFT <-> RIGHT), checked on pixels."""
    h = parse("grid(3,4); c([0,0],[0,2]); hl([0,0],W); hr([1,3],S); h([2,1],W); p([2,1],[2,2])")
    mh = Hypergraph(h.rows, h.cols,
                    tuple(transform_relation(r, "mirror_lr", h.rows, h.cols) for r in h.relations)).canonical()
    assert {(r.variant, r.dir) for r in mh.relations if r.type == "HORN"} == {
        ("RIGHT_HORN", "E"), ("LEFT_HORN", "S"), ("PLAIN_HORN", "E")}
    a = np.asarray(render_kolam(compile_hypergraph(h), Style(size=200)), dtype=int)
    b = np.asarray(render_kolam(compile_hypergraph(mh), Style(size=200)), dtype=int)
    assert np.abs(a[:, ::-1] - b).mean() < 1.0


def test_sequence_text():
    c = compile_hypergraph(parse("c([0,0],[0,1])"))
    assert c.sequence_text().startswith("[0,1] → [1,2] → [2,3]")


# ------------------------------------------------------------------ DSL + labels
def test_dsl_aliases_endpoints_and_rings():
    assert parse("h1([0,0],E)") == parse("hl([0,0],E)") == parse("LEFT_HORN([0,0],E)")
    assert parse("c([0,0],[0,3])").relations[0].vertices == ((0, 0), (0, 1), (0, 2), (0, 3))
    ring = parse("p([0,0],[0,1],[1,1],[1,0],[0,0])").relations[0]
    assert ring.closed and len(ring.vertices) == 4


@pytest.mark.parametrize("prog,msg", [
    ("c([0,0],[2,2])", "diagonal"),
    ("h([0,0],[0,1])", "one dot and a direction"),         # the retired two-dot horn
    ("h([0,0],NE)", "one dot and a direction"),
    ("x([0,0],[0,1])", "unknown brush"),
    ("c([0,0],[0,1]); h([0,0],E)", "along the brush line"),
    ("grid(1,2); h([0,0],E); h([0,1],W)", "too close"),     # two short stubs nose to nose
    ("grid(3,3); c([0,0],[0,2]); h([1,1],N)", "in use"),     # pointing at a used dot: only a stub
    ("grid(1,3); hl([0,0],E); hr([0,2],W)", "curls another way"),       # meeting horns curl alike
    ("grid(2,2); p([0,0],[0,1],[1,1]); h([0,0],S); h([1,1],W)", "inner corner"),
    ("h([0,0],E); hl([0,0],E)", "already has a horn"),
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
    assert parse("h([0,0],E)") != parse("h([0,0],W)")              # a horn's direction is visible


def test_program_and_json_roundtrip():
    h = parse(Path(__file__).resolve().parents[1].joinpath("examples/showcase.kolam").read_text())
    assert parse(format_program(h)) == h
    assert Hypergraph.from_json(json.loads(json.dumps(h.to_json()))) == h


def test_isomorphism():
    a = parse("grid(4,4); c([0,0],[0,2]); hl([0,1],S)")
    b = parse("grid(4,4); c([2,1],[2,3]); hl([2,2],S)")         # translated
    c = parse("grid(4,4); c([0,0],[0,2]); hr([0,1],S)")         # other variant
    d = parse("grid(4,4); c([0,0],[0,2]); hl([0,0],S)")         # horn on the chain's end, not middle
    e = parse("grid(4,4); c([0,0],[0,2]); hl([0,1],N)")         # other direction
    assert is_isomorphic(a, b)
    assert not any(is_isomorphic(a, x) for x in (c, d, e))


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


def test_generator_makes_branching_dots(samples):
    """Lines of one brush meeting at a dot ('+' or 'T', like sikku kolams) appear in the data."""
    from kolam_press.spec import to_lattice
    branching = 0
    for _, c, _ in samples:
        g = c.config.gates
        for v in c.hypergraph.used_vertices():
            li, lj = to_lattice(v)
            st = [g.get(x, ("",))[0] for x in ((li - 1, lj), (li + 1, lj), (li, lj - 1), (li, lj + 1))]
            branching += max(st.count("CROSS"), st.count("JOIN")) >= 3
    assert branching > 20


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
    assert Hypergraph(2, 2, (Relation("HORN", ((0, 0),), "UP_HORN", dir="N"),)).validate()
    assert Hypergraph(2, 2, (Relation("HORN", ((0, 0),), "LEFT_HORN", dir="NE"),)).validate()
    assert Hypergraph(2, 2, (Relation("HORN", ((0, 0), (0, 1)), "LEFT_HORN", dir="N"),)).validate()


# ------------------------------------------------------------------ one picture, one label (gaps)
@pytest.mark.parametrize("prog", [
    "grid(3,3); p([0,0],[0,2]); p([0,1],[2,1])",           # same-brush loops sharing a dot ('T')
    "grid(3,3); c([0,0],[0,2]); c([0,1],[2,1])",
    "grid(2,3); c([0,0],[0,2]); c([0,1],[0,2])",           # two strokes on one gap, same state
])
def test_any_drawing_is_a_label_once_canonical(prog):
    h = parse(prog)                                        # parse returns the canonical label
    assert h == h.canonical() and not h.validate()
    g = check_geometry(compile_hypergraph(h))
    assert g["ok"], g["issues"]
    c = compile_hypergraph(h)
    assert decode(h.rows, h.cols, *observe(c.config)) == h


def test_same_picture_same_label():
    """A '+' of crossings drawn as straight+straight or as two L's: one picture, so one label."""
    a = parse("grid(3,3); c([1,0],[1,2]); c([0,1],[2,1])")
    b = parse("grid(3,3); c([1,0],[1,1],[0,1]); c([2,1],[1,1],[1,2])")
    c = parse("grid(3,3); c([1,0],[1,1],[2,1]); c([0,1],[1,1],[1,2])")
    assert a == b == c
    assert format_program(a) == "grid(3, 3)\nc([0,1],[2,1])\nc([1,0],[1,2])"
    assert parse("c([0,0],[0,1]); c([0,1],[1,1])") == parse("c([0,0],[0,1],[1,1])")   # corners stay one line
    assert parse("c([1,1],[1,3],[3,3],[3,1],[1,1])").relations[0].closed            # rings stay rings


def test_gate_codes_are_the_answer_key():
    from kolam_press.hypergraph import from_gate_codes, gate_codes
    h = parse("grid(3,3); c([1,0],[1,2]); c([0,1],[2,1]); p([2,0],[2,1]); hl([0,0],W)")
    codes = gate_codes(h)
    assert codes == {"h": ["00", "11", "20"], "v": ["010", "010"]}
    assert from_gate_codes(3, 3, codes, [r for r in h.relations if r.type == "HORN"]) == h


def test_photo_kolam_is_a_dataset_label():
    """The sikku kolam read off a photo: crossing lines branch at 29 dots, still one exact label."""
    h = parse(Path(__file__).resolve().parents[1].joinpath("examples/sikku_diamond.kolam").read_text())
    c = compile_hypergraph(h)
    assert check_geometry(c)["ok"]
    assert decode(h.rows, h.cols, *observe(c.config)) == h


@pytest.mark.parametrize("prog,msg", [
    ("c([0,0],[1,1])", "diagonal"),                        # diagonal dots share no gate
    ("c([0,0],[0,2]); p([0,1],[0,2])", "can.t be both"),   # one gap, two states
])
def test_free_drawing_still_refuses_the_impossible(prog, msg):
    with pytest.raises(KolamError, match=msg):
        parse(prog, strict=False)


# ------------------------------------------------------------------ drawing primitives (docs §10)
def test_parallel_capsule_has_straight_walls_and_round_ends():
    """A parallel run is a capsule: walls 0.75 from the dots all along, ends ~0.7 (like crossing)."""
    from kolam_press.compiler import build_config, trace
    from kolam_press.geometry import capsule_points, sample_stroke
    h = parse("p([0,0],[0,3])")
    cfg = build_config(h)
    P = np.vstack([sample_stroke(capsule_points(cfg, s), 80) for s in trace(cfg)])
    top = P[P[:, 0] < 1]
    walls = [1 - top[np.abs(top[:, 1] - lj) < 0.04][:, 0].min() for lj in (1.5, 2, 3, 4, 5, 6)]
    assert max(walls) - min(walls) < 0.01 and abs(walls[0] - 0.75) < 0.01          # no pinch
    assert 0.65 < P[:, 1].max() - 7 < 0.76                                          # round end


def test_box_of_four_parallel_gaps_drops_the_inner_loop():
    from kolam_press.compiler import build_config, trace
    from kolam_press.geometry import is_box_hole
    cfg = build_config(parse("p([0,0],[0,1],[1,1],[1,0],[0,0])"))
    strokes = trace(cfg)
    assert len(strokes) == 2 and sum(is_box_hole(cfg, s) for s in strokes) == 1
    cfg = build_config(parse("p([0,0],[0,1],[1,1],[1,0])"))                       # open U: no box
    assert not any(is_box_hole(cfg, s) for s in trace(cfg))


def test_crossing_between_two_tubes_has_room():
    """A 'ladder': two parallel tubes side by side, linked by a crossing. The walls lean in at the X."""
    for prog in ["grid(3,2); p([0,0],[2,0]); p([0,1],[2,1]); c([1,0],[1,1])",
                 "grid(3,2); p([0,0],[2,0]); p([0,1],[2,1]); c([0,0],[0,1])"]:
        g = check_geometry(compile_hypergraph(parse(prog, strict=False), strict=False))
        assert g["ok"], (prog, g["issues"])


# ------------------------------------------------------------------ horns (docs §10.6)
HORN_CASES = [
    "grid(3,3); h([1,1],N)",                                   # lone dot
    "grid(3,3); c([1,0],[1,1]); hl([1,1],E)",                  # chain end
    "grid(3,3); p([1,0],[1,1]); hl([1,1],E)",                  # tube end
    "grid(3,3); p([1,0],[1,2]); h([1,1],N)",                   # tube side
    "grid(3,3); hl([1,1],N); hl([1,1],S)",                     # two horns on one dot
    "grid(3,3); hl([1,1],N); hr([1,1],E)",                     # side by side: each stops at 45 degrees
    "grid(3,3); h([1,1],N); h([1,1],E); h([1,1],S); h([1,1],W)",
    "grid(2,3); c([0,0],[1,0]); c([0,2],[1,2]); h([0,0],E); h([0,2],W)",      # facing: tips meet
    "grid(2,3); p([0,0],[1,0]); p([0,2],[1,2]); hl([0,0],E); hl([0,2],W)",
    "grid(2,2); c([0,0],[0,1],[1,1]); hr([0,0],S); hr([1,1],W)",              # meeting at 90 degrees
    "grid(5,5); c([1,1],[1,3],[3,3],[3,1],[1,1]); hl([1,2],S); hl([3,2],N); hl([2,1],E); hl([2,3],W)",
    "grid(1,3); c([0,0],[0,2]); hl([0,0],W); hr([0,2],E)",     # off the grid: crescents past the border
]


@pytest.mark.parametrize("prog", HORN_CASES)
def test_horns_draw_cleanly_and_decode(prog):
    h = parse(prog)
    c = compile_hypergraph(h)
    g = check_geometry(c)
    assert g["ok"], g["issues"]
    assert decode(h.rows, h.cols, *observe(c.config)) == h


def test_horn_size_follows_the_spot_it_points_at():
    from kolam_press.spec import to_lattice
    def reach(prog):                    # short horns are free drawings only, not dataset labels
        c = compile_hypergraph(parse(prog, strict=False), strict=False)
        t = next(t for t in c.config.tips if t.dot == (1, 1))
        o = to_lattice(t.dot)
        return round(((t.point[0] - o[0]) ** 2 + (t.point[1] - o[1]) ** 2) ** 0.5, 3)
    assert reach("grid(3,3); h([1,1],N)") == 2.0                            # empty spot: crescent
    assert reach("grid(3,3); c([0,0],[0,2]); h([1,1],N)") == 1.05          # used dot: short
    assert reach("grid(3,3); h([1,1],N); h([0,1],E)") == 1.05              # a horn's own dot counts as used


def test_horn_grows_smoothly_out_of_its_line():
    """No lump where the horn starts: away from the tip the line turns gently everywhere."""
    from kolam_press.geometry import drawn, sample_stroke
    c = compile_hypergraph(parse("grid(3,3); c([1,0],[1,1]); hl([1,1],E)"))
    (s,), _ = drawn(c)
    P = sample_stroke(s, 8)
    v0, v1 = P - np.roll(P, 1, axis=0), np.roll(P, -1, axis=0) - P
    cos = (v0 * v1).sum(1) / (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1))
    tip = np.array(c.config.tips[0].point)
    far = np.linalg.norm(P - tip, axis=1) > 0.05
    assert cos[far].min() > np.cos(np.radians(12))


def test_unused_dots_are_not_drawn():
    a = np.asarray(render_kolam(compile_hypergraph(parse("grid(3,3); c([1,0],[1,1])")), Style(size=96)), int)
    b = np.asarray(render_kolam(compile_hypergraph(parse("grid(3,3); c([1,0],[1,1])")),
                                Style(size=96, show_dots=False)), int)
    corner = (slice(0, 30), slice(0, 30))                    # dot [0,0] is unused
    assert np.abs(a[corner] - b[corner]).sum() == 0
