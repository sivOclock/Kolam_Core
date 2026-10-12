"""Gap codes: the training form of a label (docs/hypergraph_kolams.md §12).

A kolam is fully given by one digit on every gap of its dot grid, border gaps included:

    0 wall   1 cross   2 tube
    3 4 5    plain / left / right horn, facing E or S (right or down)
    6 7 8    plain / left / right horn, facing W or N (left or up)

A horn always points through a wall gap, so it is written on that gap; the gap's orientation
says E/W or S/N. The digits are read back into the canonical hypergraph with `from_gates`, so
label <-> code is exact.

    grid    (2*rows+1) x (2*cols+1) int8 lattice: gaps hold digits, dots and centres hold 0
    sparse  the non-zero gaps only, in reading order: [(lattice row, lattice col, digit), ...].
            Walls and empty space are not written. The grid size is read back from the dots the
            items touch (dataset labels are cropped to their dots).
    tokens  sparse for a sequence model: <rK> once per lattice row, then <cJ> <D> pairs, then <end>
"""
from __future__ import annotations

import numpy as np

from .hypergraph import Hypergraph, KolamError, Relation, from_gates, loop_gates
from .spec import CROSS, HORN, HORN_VARIANTS, JOIN, gate_between, horn_target

MAX_DOTS = 12                  # largest grid side; lattice coordinates run 0 .. 2*MAX_DOTS
L = 2 * MAX_DOTS
VOCAB = ([f"<r{i}>" for i in range(L + 1)] + [f"<c{j}>" for j in range(L + 1)]
         + [f"<{d}>" for d in range(1, 9)] + ["<end>"])


def _dots(i: int, j: int):
    """The two dots on either side of gap (i, j): (up or left, down or right)."""
    if i % 2:
        return ((i - 1) // 2, j // 2 - 1), ((i - 1) // 2, j // 2)
    return (i // 2 - 1, (j - 1) // 2), (i // 2, (j - 1) // 2)


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
        a, b = _dots(i, j)
        fwd = x <= 5
        d = ("E" if fwd else "W") if i % 2 else ("S" if fwd else "N")
        horns.append(Relation(HORN, (a if fwd else b,), HORN_VARIANTS[(x - 3) % 3], False, d))
    return from_gates(rows, cols, gates, horns)


def to_sparse(h: Hypergraph) -> list[tuple[int, int, int]]:
    g = to_grid(h)
    return [(int(i), int(j), int(g[i, j])) for i, j in zip(*np.nonzero(g))]


def sparse_grid(items, rows: int | None = None, cols: int | None = None) -> np.ndarray:
    """The full grid of a sparse list; rows/cols default to the extent of the dots it touches."""
    if rows is None or cols is None:
        dots = set()
        for i, j, x in items:
            a, b = _dots(i, j)
            dots |= {a, b} if x <= 2 else {a if x <= 5 else b}
        if not dots:
            raise KolamError("empty code: no kolam")
        rows = max(v[0] for v in dots) + 1 if rows is None else rows
        cols = max(v[1] for v in dots) + 1 if cols is None else cols
    g = np.zeros((2 * rows + 1, 2 * cols + 1), np.int8)
    for i, j, x in items:
        if not (0 <= i <= 2 * rows and 0 <= j <= 2 * cols):
            raise KolamError(f"item {(i, j, x)} is outside a {rows} x {cols} grid")
        g[i, j] = x
    return g


def from_sparse(items, rows: int | None = None, cols: int | None = None) -> Hypergraph:
    return from_grid(sparse_grid(items, rows, cols))


def to_tokens(items) -> list[str]:
    out, row = [], None
    for i, j, x in items:
        if i != row:
            out.append(f"<r{i}>")
            row = i
        out += [f"<c{j}>", f"<{x}>"]
    return out + ["<end>"]


def from_tokens(tokens) -> list[tuple[int, int, int]]:
    items, row, col = [], None, None
    for t in tokens:
        if t == "<end>":
            break
        if t.startswith("<r"):
            row = int(t[2:-1])
        elif t.startswith("<c"):
            col = int(t[2:-1])
        else:
            items.append((row, col, int(t[1:-1])))
    return items


def allowed_next(prefix) -> list[str]:
    """Constrained decoding: the tokens that keep a partial sequence well formed. Rows go down,
    columns go right within a row, a column sits on a gap of its row (odd row: even col, even row:
    odd col), every column is followed by one digit. Any sequence built this way parses."""
    row = col = None
    last = None
    for t in prefix:
        if t == "<end>":
            return []
        if t.startswith("<r"):
            row, col = int(t[2:-1]), None
        elif t.startswith("<c"):
            col = int(t[2:-1])
        last = t[1]
    rows_after = [f"<r{i}>" for i in range(L + 1) if row is None or i > row]
    if last is None:
        return rows_after + ["<end>"]
    if last == "c":
        return [f"<{d}>" for d in range(1, 9)]
    cols_after = [f"<c{j}>" for j in range(L + 1) if (row + j) % 2 and (col is None or j > col)]
    if last == "r":
        return cols_after
    return cols_after + rows_after + ["<end>"]
