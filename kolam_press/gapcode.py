"""Gap code: the training form of a label (docs/hypergraph_kolams.md §12).

A kolam is one digit on every gap of its dot grid, border gaps included:

    0 wall   1 cross   2 tube
    3 4 5    plain / left / right horn, facing E or S (right or down)
    6 7 8    plain / left / right horn, facing W or N (left or up)

A horn always points through a wall gap, so it is written on that gap; the gap's orientation
says E/W or S/N. The digits are read back into the canonical hypergraph with `from_gates`, so
label <-> code is exact.

    code    the gap rows of the lattice, top to bottom, as digit strings: rows alternate
            cols digits (up/down gaps) and cols+1 digits (left/right gaps, both borders), 2*rows+1
            rows in all, e.g. ["00000", "007500", "00100", ...]. Stored in labels.jsonl.
    grid    the same digits on the (2*rows+1) x (2*cols+1) lattice (dots and centres 0)
    cells   the same digits one cell per dot: (rows+1) x (cols+1) x 2 = (left gap, top gap) of
            dot (r, c); cells past the last dot row/col hold 0 in the missing slot. For a model
            that predicts every gap at once.
    tokens  the code for a sequence model: each row's digits, then "|", then "<end>" (11 tokens)
"""
from __future__ import annotations

import numpy as np

from .hypergraph import Hypergraph, KolamError, Relation, from_gates, loop_gates
from .spec import CROSS, HORN, HORN_VARIANTS, JOIN, gate_between, horn_target

VOCAB = [str(d) for d in range(9)] + ["|", "<end>"]
FACING_OUT = {"N": "0678", "W": "0678", "S": "0345", "E": "0345"}   # digits a border gap may hold


def to_grid(h: Hypergraph) -> np.ndarray:
    g = np.zeros((2 * h.rows + 1, 2 * h.cols + 1), np.int8)
    for p, s in loop_gates(h).items():
        g[p] = 1 if s == CROSS else 2
    for r in h.relations:
        if r.type != HORN:
            continue
        v = r.vertices[0]
        p = gate_between(v, horn_target(v, r.dir))
        if g[p]:
            raise KolamError(f"horn {r.variant} on {list(v)} {r.dir}: its gap already holds {g[p]}")
        g[p] = 3 + HORN_VARIANTS.index(r.variant) + (0 if r.dir in ("E", "S") else 3)
    return g


def from_grid(g: np.ndarray) -> Hypergraph:
    rows, cols = (g.shape[0] - 1) // 2, (g.shape[1] - 1) // 2
    gates, horns = {}, []
    for i, j in zip(*np.nonzero(g)):
        i, j, x = int(i), int(j), int(g[i, j])
        if (i + j) % 2 == 0 or not 1 <= x <= 8:
            raise KolamError(f"digit {x} at {(i, j)}: digits 1-8 belong on gaps only")
        if x <= 2:
            gates[(i, j)] = CROSS if x == 1 else JOIN
            continue
        if i % 2:   # left/right gap: dots to its left (a) and right (b)
            a, b = ((i - 1) // 2, j // 2 - 1), ((i - 1) // 2, j // 2)
        else:       # up/down gap: dots above (a) and below (b)
            a, b = (i // 2 - 1, (j - 1) // 2), (i // 2, (j - 1) // 2)
        fwd = x <= 5
        d = ("E" if fwd else "W") if i % 2 else ("S" if fwd else "N")
        horns.append(Relation(HORN, (a if fwd else b,), HORN_VARIANTS[(x - 3) % 3], False, d))
    return from_gates(rows, cols, gates, horns)


def to_code(h: Hypergraph) -> list[str]:
    g = to_grid(h)
    return ["".join(str(int(x)) for x in g[i, (i + 1) % 2::2]) for i in range(g.shape[0])]


def code_grid(code: list[str]) -> np.ndarray:
    if len(code) < 3 or len(code) % 2 == 0:
        raise KolamError(f"a code has 2*rows+1 rows (3, 5, 7, ...), not {len(code)}")
    rows, cols = (len(code) - 1) // 2, len(code[0])
    g = np.zeros((2 * rows + 1, 2 * cols + 1), np.int8)
    for i, row in enumerate(code):
        if len(row) != cols + i % 2:
            raise KolamError(f"code row {i} has {len(row)} digits, wants {cols + i % 2}")
        g[i, (i + 1) % 2::2] = [int(x) for x in row]
    return g


def from_code(code: list[str]) -> Hypergraph:
    return from_grid(code_grid(code))


def to_cells(h: Hypergraph) -> np.ndarray:
    g = to_grid(h)
    rows, cols = h.rows, h.cols
    c = np.zeros((rows + 1, cols + 1, 2), np.int8)
    c[:rows, :, 0] = g[1::2, 0::2]     # left gap of dot (r, c); column cols = the right border
    c[:, :cols, 1] = g[0::2, 1::2]     # top gap of dot (r, c); row rows = the bottom border
    return c


def from_cells(c: np.ndarray) -> Hypergraph:
    rows, cols = c.shape[0] - 1, c.shape[1] - 1
    g = np.zeros((2 * rows + 1, 2 * cols + 1), np.int8)
    g[1::2, 0::2] = c[:rows, :, 0]
    g[0::2, 1::2] = c[:, :cols, 1]
    return from_grid(g)


def to_tokens(code: list[str]) -> list[str]:
    return [t for row in code for t in [*row, "|"]] + ["<end>"]


def from_tokens(tokens) -> list[str]:
    code, row = [], ""
    for t in tokens:
        if t == "<end>":
            break
        if t == "|":
            code.append(row)
            row = ""
        else:
            row += t
    return code


def allowed_next(prefix) -> list[str]:
    """Constrained decoding: the tokens that keep a partial code well formed. The first row sets
    cols; rows then alternate cols+1 and cols digits; <end> only after a full cols-digit row (never
    the first). Border gaps only hold walls or horns facing out of the grid."""
    rows_done, row = [], []
    for t in prefix:
        if t == "<end>":
            return []
        if t == "|":
            rows_done.append(row)
            row = []
        else:
            row.append(t)
    i = len(rows_done)
    digits = [str(d) for d in range(9)]
    if i == 0:                                    # top border: walls or horns facing N
        return [d for d in digits if d in FACING_OUT["N"]] + (["|"] if row else [])
    cols = len(rows_done[0])
    want = cols + i % 2
    if not row and i >= 3 and i % 2 == 1:         # just closed a cols-digit row: may end here
        end = ["<end>"]
    else:
        end = []
    if len(row) == want:
        return ["|"]
    if i % 2 and len(row) == 0:                   # left border
        ok = FACING_OUT["W"]
    elif i % 2 and len(row) == want - 1:          # right border
        ok = FACING_OUT["E"]
    else:
        ok = "".join(digits)
    return [d for d in digits if d in ok] + end
