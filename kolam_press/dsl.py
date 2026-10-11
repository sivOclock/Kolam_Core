"""Brush DSL: write a kolam the way you would draw it with three brushes.

    grid(5, 5)                         # optional; else the smallest grid that fits
    c([0,0],[0,4])                     # CROSSING_LOOP brush along row 0 (endpoints expand)
    p([2,0],[2,2],[4,2])               # PARALLEL_LOOP brush with a corner at [2,2]
    c([1,1],[1,3],[3,3],[3,1],[1,1])   # repeat the first dot to close a ring
    h([4,4],S)                         # PLAIN_HORN on dot [4,4], tip pointing S   (alias h0, hp)
    hl([1,2],N)                        # LEFT_HORN: tip curves to its left          (alias h1)
    hr([3,0],W)                        # RIGHT_HORN                                 (alias h2)

Dots are anchor coordinates [row, col], 0-indexed; N/E/S/W = up/right/down/left on
screen. Loop waypoints must share a row or column with the next waypoint; the dots
in between are filled in.
Statements are separated by newlines or ';'. '#' starts a comment.
"""
from __future__ import annotations

import ast

from .hypergraph import Hypergraph, KolamError, Relation
from .spec import HORN, HORN_DIRS, HORN_VARIANTS, LOOP_TYPES, brushes

SHORT_NAMES = {"CROSSING_LOOP": "c", "PARALLEL_LOOP": "p",
               "PLAIN_HORN": "h", "LEFT_HORN": "hl", "RIGHT_HORN": "hr"}


def _point(node, where: str) -> tuple[int, int]:
    if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) == 2:
        vals = []
        for e in node.elts:
            if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) and isinstance(e.operand, ast.Constant):
                vals.append(-e.operand.value)
            elif isinstance(e, ast.Constant):
                vals.append(e.value)
            else:
                vals = None
                break
        if vals and all(isinstance(x, int) and not isinstance(x, bool) for x in vals):
            return (vals[0], vals[1])
    raise KolamError(f"{where}: expected a dot like [row, col], got {ast.unparse(node)!r}")


def _expand(waypoints: list[tuple[int, int]], where: str) -> list[tuple[int, int]]:
    out = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:]):
        if a == b:
            raise KolamError(f"{where}: waypoint {list(a)} repeated")
        if a[0] != b[0] and a[1] != b[1]:
            raise KolamError(f"{where}: {list(a)} -> {list(b)} is diagonal; brushes run along rows/columns"
                             " (add a corner waypoint)")
        dr = (b[0] > a[0]) - (b[0] < a[0])
        dc = (b[1] > a[1]) - (b[1] < a[1])
        cur = a
        while cur != b:
            cur = (cur[0] + dr, cur[1] + dc)
            out.append(cur)
    return out


def parse(text: str, strict: bool = True) -> Hypergraph:
    """Parse a brush program into a canonical, validated hypergraph (strict: a dataset label;
    otherwise any free drawing, see hypergraph.problems)."""
    names = brushes()["dsl"]
    grid = None
    rels: list[Relation] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        for stmt in raw.split("#", 1)[0].split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            where = f"line {lineno} `{stmt}`"
            try:
                node = ast.parse(stmt, mode="eval").body
            except SyntaxError as e:
                raise KolamError(f"{where}: {e.msg}") from None
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)) or node.keywords:
                raise KolamError(f"{where}: expected a brush call like c([0,0],[0,3])")
            fn = node.func.id
            if fn == "grid":
                vals = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, int)]
                if len(vals) != 2 or len(node.args) != 2:
                    raise KolamError(f"{where}: grid(rows, cols) takes two integers")
                grid = (vals[0], vals[1])
                continue
            if fn not in names:
                raise KolamError(f"{where}: unknown brush {fn!r}; use c, p, h, hl, hr (or h0/h1/h2)")
            kind = names[fn]
            if kind in HORN_VARIANTS:
                args = node.args
                if len(args) != 2 or not (isinstance(args[1], ast.Name) and args[1].id in HORN_DIRS):
                    raise KolamError(f"{where}: a horn is one dot and a direction, like {fn}([1,1],N)"
                                     " (N/E/S/W); the old two-dot horn h(A,B) is now p(A,B) + h(B,dir)")
                rels.append(Relation(HORN, (_point(args[0], where),), kind, False, args[1].id))
                continue
            pts = [_point(a, where) for a in node.args]
            if len(pts) < 2:
                raise KolamError(f"{where}: a loop brush needs at least two dots")
            closed = len(pts) > 2 and pts[-1] == pts[0]
            dots = _expand(pts, where)
            if closed:
                dots = dots[:-1]
            rels.append(Relation(kind, tuple(dots), None, closed))
    if not rels:
        raise KolamError("empty program")
    if grid is None:
        vs = [v for r in rels for v in r.vertices]
        grid = (max(v[0] for v in vs) + 1, max(v[1] for v in vs) + 1)
    return Hypergraph(grid[0], grid[1], tuple(rels)).canonical().check(strict)


def _waypoints(vs: list[tuple[int, int]], closed: bool) -> list[tuple[int, int]]:
    seq = vs + [vs[0]] if closed else vs
    out = [seq[0]]
    for i in range(1, len(seq) - 1):
        a, b, c = seq[i - 1], seq[i], seq[i + 1]
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            out.append(b)
    out.append(seq[-1])
    return out


def format_program(h: Hypergraph) -> str:
    """Canonical brush program for a hypergraph (inverse of parse)."""
    lines = [f"grid({h.rows}, {h.cols})"]
    for r in h.canonical().relations:
        if r.type == HORN:
            (a, b), = r.vertices
            lines.append(f"{SHORT_NAMES[r.variant]}([{a},{b}],{r.dir})")
            continue
        assert r.type in LOOP_TYPES
        pts = _waypoints(list(r.vertices), r.closed)
        lines.append(f"{SHORT_NAMES[r.type]}(" + ",".join(f"[{a},{b}]" for a, b in pts) + ")")
    return "\n".join(lines)
