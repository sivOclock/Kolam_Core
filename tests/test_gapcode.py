import json

import numpy as np
import pytest

from kolam_press import KolamError, parse
from kolam_press.gapcode import (allowed_next, from_grid, from_sparse, from_tokens, sparse_grid, to_grid, to_sparse,
                                 to_tokens)
from kolam_press.generator import GenConfig, generate_sample
from kolam_press.systems import substitution, switching

SWITCHING_0010 = ("grid(5,5); c([0,2],[4,2]); c([1,1],[1,3]); c([2,0],[2,4]); c([3,1],[3,3]); "
                  "hl([0,2],W); hl([4,2],E); hr([0,2],E); hr([4,2],W)")


def same(a, b):
    return json.dumps(a.canonical().to_json()) == json.dumps(b.canonical().to_json())


def test_example_code():
    h = parse(SWITCHING_0010)
    assert to_sparse(h) == [(1, 4, 7), (1, 6, 5), (2, 5, 1), (3, 4, 1), (3, 6, 1), (4, 5, 1), (5, 2, 1), (5, 4, 1),
                            (5, 6, 1), (5, 8, 1), (6, 5, 1), (7, 4, 1), (7, 6, 1), (8, 5, 1), (9, 4, 8), (9, 6, 4)]
    assert to_tokens(to_sparse(h))[:6] == ["<r1>", "<c4>", "<7>", "<c6>", "<5>", "<r2>"]
    assert same(from_sparse(to_sparse(h)), h)


@pytest.mark.parametrize("variant,dir_,digit", [
    ("h", "E", 3), ("hl", "E", 4), ("hr", "E", 5), ("h", "S", 3), ("hl", "S", 4), ("hr", "S", 5),
    ("h", "W", 6), ("hl", "W", 7), ("hr", "W", 8), ("h", "N", 6), ("hl", "N", 7), ("hr", "N", 8)])
def test_horn_digits(variant, dir_, digit):
    line = "c([0,1],[2,1])" if dir_ in "EW" else "c([1,0],[1,2])"      # the horn points away from its line
    h = parse(f"grid(3,3); {line}; {variant}([1,1],{dir_})")
    g = to_grid(h)
    assert sorted(set(g.ravel()) - {0, 1}) == [digit]
    assert same(from_grid(g), h)


def test_grids_of_generated_kolams_round_trip():
    hs = [parse(generate_sample(i, 3, GenConfig())[0]["program"]) for i in range(40)]
    for make, seed in ((substitution, 1), (switching, 2)):
        rng = np.random.default_rng(seed)
        hs += [make(rng)[0] for _ in range(4)]
    for h in hs:
        if not h.relations:
            continue
        g = to_grid(h)
        assert same(from_grid(g), h)
        items = to_sparse(h)
        assert all(x != 0 for *_, x in items)
        assert same(from_sparse(items, h.rows, h.cols), h)
        assert from_tokens(to_tokens(items)) == items


def test_sparse_reads_the_size_of_a_cropped_kolam():
    rec = generate_sample(5, 0, GenConfig())[0]
    items = [tuple(t) for t in rec["sparse"]]
    assert sparse_grid(items).shape == (2 * rec["grid"]["rows"] + 1, 2 * rec["grid"]["cols"] + 1)


def test_constrained_decoding_accepts_every_valid_sequence():
    toks = to_tokens(to_sparse(parse(SWITCHING_0010)))
    for k, t in enumerate(toks):
        assert t in allowed_next(toks[:k])
    assert allowed_next(toks) == []
    assert allowed_next([]) [-1] == "<end>"
    assert "<1>" not in allowed_next(["<r1>"])                    # a row token needs a column first
    assert "<c5>" not in allowed_next(["<r1>"])                   # odd row: gaps sit on even columns
    assert "<c2>" not in allowed_next(["<r1>", "<c4>", "<7>"])    # columns only go right
    assert "<r1>" not in allowed_next(["<r1>", "<c4>", "<7>"])    # rows only go down


def test_bad_codes_are_refused():
    with pytest.raises(KolamError):
        from_grid(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.int8))   # a digit on a dot
    with pytest.raises(KolamError):
        sparse_grid([])
