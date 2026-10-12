import json

import numpy as np
import pytest

from kolam_press import KolamError, parse
from kolam_press.gapcode import (allowed_next, code_grid, from_cells, from_code, from_grid, from_tokens, to_cells,
                                 to_code, to_grid, to_tokens)
from kolam_press.generator import GenConfig, generate_sample
from kolam_press.systems import substitution, switching

SWITCHING_0010 = ("grid(5,5); c([0,2],[4,2]); c([1,1],[1,3]); c([2,0],[2,4]); c([3,1],[3,3]); "
                  "hl([0,2],W); hl([4,2],E); hr([0,2],E); hr([4,2],W)")


def same(a, b):
    return json.dumps(a.canonical().to_json()) == json.dumps(b.canonical().to_json())


def test_example_code():
    h = parse(SWITCHING_0010)
    assert to_code(h) == ["00000", "007500", "00100", "001100", "00100", "011110",
                          "00100", "001100", "00100", "008400", "00000"]
    assert to_tokens(to_code(h))[:7] == ["0", "0", "0", "0", "0", "|", "0"]
    assert same(from_code(to_code(h)), h)


@pytest.mark.parametrize("variant,dir_,digit", [
    ("h", "E", 3), ("hl", "E", 4), ("hr", "E", 5), ("h", "S", 3), ("hl", "S", 4), ("hr", "S", 5),
    ("h", "W", 6), ("hl", "W", 7), ("hr", "W", 8), ("h", "N", 6), ("hl", "N", 7), ("hr", "N", 8)])
def test_horn_digits(variant, dir_, digit):
    line = "c([0,1],[2,1])" if dir_ in "EW" else "c([1,0],[1,2])"      # the horn points away from its line
    h = parse(f"grid(3,3); {line}; {variant}([1,1],{dir_})")
    g = to_grid(h)
    assert sorted(set(g.ravel()) - {0, 1}) == [digit]
    assert same(from_grid(g), h)


def test_codes_of_generated_kolams_round_trip():
    hs = [parse(generate_sample(i, 3, GenConfig())[0]["program"]) for i in range(40)]
    for make, seed in ((substitution, 1), (switching, 2)):
        rng = np.random.default_rng(seed)
        hs += [make(rng)[0] for _ in range(4)]
    for h in hs:
        if not h.relations:
            continue
        code = to_code(h)
        assert len(code) == 2 * h.rows + 1 and [len(r) for r in code[:2]] == [h.cols, h.cols + 1]
        assert same(from_code(code), h)
        assert same(from_cells(to_cells(h)), h)
        assert from_tokens(to_tokens(code)) == code
        toks = to_tokens(code)
        assert all(t in allowed_next(toks[:k]) for k, t in enumerate(toks))


def test_cells_put_one_cell_on_every_dot():
    h = parse(SWITCHING_0010)
    c = to_cells(h)
    assert c.shape == (6, 6, 2)
    assert list(c[0, :, 0]) == [0, 0, 7, 5, 0, 0]       # dot row 0: left gaps (and the right border)
    assert c[5, 5].tolist() == [0, 0]                    # past the last dot: always empty


def test_constrained_decoding():
    toks = to_tokens(to_code(parse(SWITCHING_0010)))
    assert allowed_next(toks) == []
    assert "3" not in allowed_next([])                         # top border: only walls or horns facing up
    assert "<end>" not in allowed_next(list("00000|"))          # too short to end
    assert allowed_next(list("00000|007500")) == ["|"]          # row full
    assert "1" not in allowed_next(list("00000|"))               # left border
    assert "<end>" in allowed_next(list("000|0000|000|"))


def test_bad_codes_are_refused():
    with pytest.raises(KolamError):
        from_grid(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.int8))   # a digit on a dot
    with pytest.raises(KolamError):
        code_grid(["000", "0000"])                                           # even number of rows
    with pytest.raises(KolamError):
        code_grid(["000", "000", "000"])                                     # row 1 needs cols+1 digits
