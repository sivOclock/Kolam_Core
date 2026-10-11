"""Relation compiler: ordered hypergraph -> gate configuration -> lattice strokes.

Gate model (Gerdes mirror curves / Nagata N-lines / Sivakumar gating, refs [2],[4]):
the curve travels diagonally from gate to gate; each gate is CROSS, JOIN or SPLIT
(brushes.json). Brushes only set gates, so relations compose without special cases:

    CROSSING_LOOP  -> CROSS on the gap between consecutive dots
    PARALLEL_LOOP  -> JOIN  on the gap between consecutive dots
    HORN(A -> B)   -> JOIN  on the A-B gap + a tip template replacing B's far-side turn

Every trace is a closed stroke; `build_crossing_loop(n)` from the notebook is
exactly the trace of one CROSSING_LOOP along a row (tests/test_kolam_press.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .hypergraph import Hypergraph, KolamError, Relation
from .spec import (CROSS, CROSSING_LOOP, HORN, JOIN, PARALLEL_LOOP, PLAIN_HORN, LEFT_HORN,
                   RIGHT_HORN, SPLIT, Point, Vertex, gate_between, gate_flip, horn_far_turn,
                   horn_frame, horn_tip_points, is_gate, is_h_gate, loop_gate_state, to_lattice)

DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass
class HornTip:
    rel: int
    base: Vertex
    tip_dot: Vertex
    variant: str
    turn: tuple[Point, Point, Point]   # (left gate, far gate, right gate) of the tip dot
    points: list[Point]                 # control points that replace the far gate


@dataclass
class Config:
    rows: int
    cols: int
    gates: dict[Point, tuple[str, int]] = field(default_factory=dict)  # non-SPLIT gate -> (state, rel)
    tips: list[HornTip] = field(default_factory=list)
    ring_dots: set[Vertex] = field(default_factory=set)
    dot_rels: dict[Vertex, list[int]] = field(default_factory=dict)


@dataclass
class Compiled:
    hypergraph: Hypergraph
    config: Config
    strokes: list[list[Point]]          # closed control sequences (first point not repeated)
    owners: list[list[int]]             # relation index per control point (-1 = none)

    @property
    def crossings(self) -> list[Point]:
        return sorted(g for g, (s, _) in self.config.gates.items() if s == CROSS)

    @property
    def tip_points(self) -> list[Point]:
        return [t.points[0] for t in self.config.tips]

    def sequence_text(self) -> str:
        return "\n---\n".join(" → ".join(f"[{a},{b}]" for a, b in s + s[:1]) for s in self.strokes)


def build_config(h: Hypergraph) -> Config:
    cfg = Config(h.rows, h.cols)
    for i, r in enumerate(h.relations):
        for v in r.vertices:
            cfg.dot_rels.setdefault(v, []).append(i)
        cfg.ring_dots.update(r.vertices)
        if r.type == HORN:
            a, b = r.vertices
            cfg.gates[gate_between(a, b)] = (JOIN, i)
            cfg.tips.append(HornTip(i, a, b, r.variant, horn_far_turn(a, b), horn_tip_points(a, b, r.variant)))
        else:
            state = loop_gate_state(r.type)
            for a, b in r.edges():
                cfg.gates[gate_between(a, b)] = (state, i)
    return cfg


def _dot_gates(v: Vertex) -> set[Point]:
    li, lj = to_lattice(v)
    return {(li - 1, lj), (li + 1, lj), (li, lj - 1), (li, lj + 1)}


def trace(cfg: Config) -> list[list[Point]]:
    """Follow the gate rules from every unvisited diagonal link until each stroke closes."""
    H, W = 2 * cfg.rows, 2 * cfg.cols
    gates = sorted((a, b) for a in range(H + 1) for b in range(W + 1) if (a + b) % 2 == 1)
    seen: set = set()
    strokes = []
    for g in gates:
        for d in DIRS:
            q = (g[0] + d[0], g[1] + d[1])
            if not (0 <= q[0] <= H and 0 <= q[1] <= W) or (g, q) in seen:
                continue
            seq = [g]
            cur, dd = g, d
            while True:
                nxt = (cur[0] + dd[0], cur[1] + dd[1])
                seen.add((cur, nxt))
                seen.add((nxt, cur))
                cur = nxt
                state = cfg.gates.get(cur, (SPLIT, -1))[0]
                f = gate_flip(state, cur)
                dd = (dd[0] * f[0], dd[1] * f[1])
                if cur == g and dd == d:
                    break
                seq.append(cur)
                if len(seq) > 4 * (H + 1) * (W + 1):
                    raise KolamError("trace did not close")
            strokes.append(seq)
    unused = {(r, c) for r in range(cfg.rows) for c in range(cfg.cols)} - cfg.ring_dots
    bare = [_dot_gates(v) for v in unused]
    return [s for s in strokes if not (len(s) == 4 and set(s) in bare)]


def _turn_owner(cfg: Config, g: Point, p: Point, q: Point) -> int:
    """Relation that owns a control point: the gate's relation, else the dot the strand turns around."""
    if g in cfg.gates:
        return cfg.gates[g][1]
    s = (p[0] + q[0], p[1] + q[1])
    if s[0] % 2 or s[1] % 2:
        return -1
    li, lj = s[0] // 2, s[1] // 2
    if li % 2 == 0 or lj % 2 == 0:
        return -1
    dot = ((li - 1) // 2, (lj - 1) // 2)
    rels = cfg.dot_rels.get(dot, [])
    if len(rels) <= 1:
        return rels[0] if rels else -1
    # dot in two relations: take the one whose gate on this dot is nearest
    best, best_d = rels[0], 99
    for gg in _dot_gates(dot):
        if gg in cfg.gates and cfg.gates[gg][1] in rels:
            dist = abs(gg[0] - g[0]) + abs(gg[1] - g[1])
            if dist < best_d:
                best, best_d = cfg.gates[gg][1], dist
    return best


def apply_tips(cfg: Config, strokes: list[list[Point]]) -> tuple[list[list[Point]], list[list[int]]]:
    out, owners = [], []
    # two horns can share a far gap (tip dots facing each other), so key by the whole turn
    tips = {(t.turn[1], frozenset((t.turn[0], t.turn[2]))): t for t in cfg.tips}
    for s in strokes:
        k = len(s)
        pts, own = [], []
        for i, g in enumerate(s):
            p, q = s[i - 1], s[(i + 1) % k]
            t = tips.get((g, frozenset((p, q))))
            if t is not None:
                pts += t.points
                own += [t.rel] * len(t.points)
            else:
                pts.append(g)
                own.append(_turn_owner(cfg, g, p, q))
        out.append(pts)
        owners.append(own)
    return out, owners


def compile_hypergraph(h: Hypergraph, check: bool = True, strict: bool = True) -> Compiled:
    """strict: the hypergraph must be a dataset label (decodes back exactly); otherwise any free
    drawing that can be drawn (hypergraph.problems)."""
    if check:
        h.check(strict)
    cfg = build_config(h)
    raw = trace(cfg)
    strokes, owners = apply_tips(cfg, raw)
    applied = sum(len(s) for s in strokes) - sum(len(s) for s in raw)
    if applied != sum(len(t.points) - 1 for t in cfg.tips):
        raise KolamError("a horn tip was not placed")
    return Compiled(h, cfg, strokes, owners)


# ------------------------------------------------------------------ decoding
def observe(cfg: Config) -> tuple[dict[Point, str], dict[Point, Vertex]]:
    """What an image shows: the state of every non-SPLIT gate, and each tip point with its dot."""
    return ({g: s for g, (s, _) in cfg.gates.items()},
            {t.points[0]: t.tip_dot for t in cfg.tips})


def _gate_dots(g: Point) -> tuple[Vertex, Vertex]:
    if is_h_gate(g):
        a, b = (g[0], g[1] - 1), (g[0], g[1] + 1)
    else:
        a, b = (g[0] - 1, g[1]), (g[0] + 1, g[1])
    return ((a[0] - 1) // 2, (a[1] - 1) // 2), ((b[0] - 1) // 2, (b[1] - 1) // 2)


def decode(rows: int, cols: int, gates: dict[Point, str], tips: dict[Point, Vertex]) -> Hypergraph:
    """Rebuild the canonical hypergraph from observed gate states and tips."""
    rels: list[Relation] = []
    horn_gates = set()
    for tp, b in tips.items():
        joins = [g for g in _dot_gates(b) if gates.get(g) == JOIN]
        if len(joins) != 1:
            raise KolamError(f"tip at {tp} has {len(joins)} joined gaps on its dot")
        g = joins[0]
        x, y = _gate_dots(g)
        a = x if y == b else y
        o, f, l = horn_frame(a, b)
        off = (tp[0] - o[0], tp[1] - o[1])
        variant = {f: PLAIN_HORN, (f[0] + l[0], f[1] + l[1]): LEFT_HORN,
                   (f[0] - l[0], f[1] - l[1]): RIGHT_HORN}.get(off)
        if variant is None:
            raise KolamError(f"tip at {tp} does not match a horn template")
        horn_gates.add(g)
        rels.append(Relation(HORN, (a, b), variant))
    for state, rtype in ((CROSS, CROSSING_LOOP), (JOIN, PARALLEL_LOOP)):
        adj: dict[Vertex, list[Vertex]] = {}
        for g, s in gates.items():
            if s == state and g not in horn_gates:
                a, b = _gate_dots(g)
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
        seen: set = set()
        for start in sorted(adj):
            if start in seen:
                continue
            comp, stack = [], [start]
            while stack:
                v = stack.pop()
                if v in seen:
                    continue
                seen.add(v)
                comp.append(v)
                stack += adj[v]
            if any(len(adj[v]) > 2 for v in comp):
                raise KolamError(f"{rtype} gaps branch at {[list(v) for v in comp if len(adj[v]) > 2]}")
            ends = [v for v in comp if len(adj[v]) == 1]
            closed = not ends
            cur = min(ends) if ends else min(comp)
            order, prev = [cur], None
            while len(order) < len(comp):
                nxt = [w for w in adj[cur] if w != prev and w not in order]
                prev, cur = cur, min(nxt)
                order.append(cur)
            rels.append(Relation(rtype, tuple(order), None, closed))
    return Hypergraph(rows, cols, tuple(rels)).canonical()
