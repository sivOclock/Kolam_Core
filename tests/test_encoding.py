import json

import numpy as np
import pytest
from conftest import kolam

from kolam_feed.encoding import (allowed_next, encoding_grid, from_cells, from_encoding, from_grid, from_tokens,
                                 to_cells, to_encoding, to_grid, to_tokens)
from kolam_feed.hypergraph import KolamError

C, H = "CROSSING_LOOP", "HORN"
# switching_0010: grid(5,5); c([0,2],[4,2]); c([1,1],[1,3]); c([2,0],[2,4]); c([3,1],[3,3]);
#                 hl([0,2],W); hl([4,2],E); hr([0,2],E); hr([4,2],W)
SWITCHING_0010 = kolam(5, 5, (C, [(r, 2) for r in range(5)]), (C, [(1, 1), (1, 2), (1, 3)]),
                       (C, [(2, c) for c in range(5)]), (C, [(3, 1), (3, 2), (3, 3)]),
                       (H, (0, 2), "LEFT_HORN", "W"), (H, (4, 2), "LEFT_HORN", "E"),
                       (H, (0, 2), "RIGHT_HORN", "E"), (H, (4, 2), "RIGHT_HORN", "W"))
CODE_0010 = ["00000", "007500", "00100", "001100", "00100", "011110", "00100", "001100", "00100", "008400", "00000"]


def same(a, b):
    return json.dumps(a.canonical().to_json()) == json.dumps(b.canonical().to_json())


def test_example_encoding():
    assert to_encoding(SWITCHING_0010) == CODE_0010
    assert to_tokens(CODE_0010)[:7] == ["0", "0", "0", "0", "0", "|", "0"]
    assert same(from_encoding(CODE_0010), SWITCHING_0010)


@pytest.mark.parametrize("variant,dir_,digit", [
    ("PLAIN_HORN", "E", 3), ("LEFT_HORN", "E", 4), ("RIGHT_HORN", "E", 5),
    ("PLAIN_HORN", "S", 3), ("LEFT_HORN", "S", 4), ("RIGHT_HORN", "S", 5),
    ("PLAIN_HORN", "W", 6), ("LEFT_HORN", "W", 7), ("RIGHT_HORN", "W", 8),
    ("PLAIN_HORN", "N", 6), ("LEFT_HORN", "N", 7), ("RIGHT_HORN", "N", 8)])
def test_horn_digits(variant, dir_, digit):
    line = [(0, 1), (1, 1), (2, 1)] if dir_ in "EW" else [(1, 0), (1, 1), (1, 2)]   # the horn points away from it
    h = kolam(3, 3, (C, line), (H, (1, 1), variant, dir_))
    g = to_grid(h)
    assert sorted(set(g.ravel()) - {0, 1}) == [digit]
    assert same(from_grid(g), h)


def test_encodings_read_back_to_the_python_labels(kolams):
    """Every fixture encoding reads back to the exact canonical label the Python generator stored,
    in every layout (encoding, cells, tokens), and every token is allowed by constrained decoding."""
    for k in kolams:
        h = from_encoding(k["encoding"])
        assert [r.to_json() for r in h.relations] == k["relations"]
        assert to_encoding(h) == k["encoding"]
        assert len(k["encoding"]) == 2 * h.rows + 1 and [len(r) for r in k["encoding"][:2]] == [h.cols, h.cols + 1]
        assert same(from_cells(to_cells(h)), h)
        toks = to_tokens(k["encoding"])
        assert from_tokens(toks) == k["encoding"]
        assert all(t in allowed_next(toks[:i]) for i, t in enumerate(toks))


def test_cells_put_one_cell_on_every_dot():
    c = to_cells(SWITCHING_0010)
    assert c.shape == (6, 6, 2)
    assert list(c[0, :, 0]) == [0, 0, 7, 5, 0, 0]       # dot row 0: left gaps (and the right border)
    assert c[5, 5].tolist() == [0, 0]                    # past the last dot: always empty


def test_constrained_decoding():
    assert allowed_next(to_tokens(CODE_0010)) == []
    assert "3" not in allowed_next([])                         # top border: only walls or horns facing up
    assert "<end>" not in allowed_next(list("00000|"))          # too short to end
    assert allowed_next(list("00000|007500")) == ["|"]          # row full
    assert "1" not in allowed_next(list("00000|"))               # left border
    assert "<end>" in allowed_next(list("000|0000|000|"))


def test_bad_encodings_are_refused():
    with pytest.raises(KolamError):
        from_grid(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.int8))   # a digit on a dot
    with pytest.raises(KolamError):
        encoding_grid(["000", "0000"])                                    # even number of rows
    with pytest.raises(KolamError):
        encoding_grid(["000", "000", "000"])                              # row 1 needs cols+1 digits
