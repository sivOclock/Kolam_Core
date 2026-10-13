"""Relation compiler: ordered hypergraph -> gate configuration -> lattice strokes.

Gate model (Gerdes mirror curves / Nagata N-lines / Sivakumar gating, refs [2],[4]):
the curve travels diagonally from gate to gate; each gate is CROSS, JOIN or SPLIT
(brushes.json). Brushes only set gates, so relations compose without special cases:

    CROSSING_LOOP  -> CROSS on the gap between consecutive dots
    PARALLEL_LOOP  -> JOIN  on the gap between consecutive dots
    HORN(A, dir)   -> no gate; a tip that geometry.drawn() grows out of the line in front of A

Every trace is a closed stroke; `build_crossing_loop(n)` from the notebook is
exactly the trace of one CROSSING_LOOP along a row (tests/test_kolam_feed.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .hypergraph import Hypergraph, KolamError, Relation, from_gates, horn_is_long
from .spec import (CROSS, HORN, HORN_DIRS, HORN_VARIANTS, SPLIT, Point, Vertex, gate_between, gate_flip,
                   horn_target, horn_tip, loop_gate_state, to_lattice)

DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass
class HornTip:
    rel: int
    dot: Vertex
    dir: str
    variant: str
    long: bool                          # crescent into an empty spot, else short (§10.6)
    point: tuple[float, float]          # lattice position of the sharp tip


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
    def tip_points(self) -> list[tuple[float, float]]:
        return [t.point for t in self.config.tips]

    def sequence_text(self) -> str:
        return "\n---\n".join(" → ".join(f"[{a},{b}]" for a, b in s + s[:1]) for s in self.strokes)


def build_config(h: Hypergraph) -> Config:
    cfg = Config(h.rows, h.cols)
    for i, r in enumerate(h.relations):
        for v in r.vertices:
            cfg.dot_rels.setdefault(v, []).append(i)
        cfg.ring_dots.update(r.vertices)
        if r.type == HORN:
            v, long = r.vertices[0], horn_is_long(h, r)
            cfg.tips.append(HornTip(i, v, r.dir, r.variant, long, horn_tip(v, r.dir, r.variant, long)))
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


def compile_hypergraph(h: Hypergraph, check: bool = True, strict: bool = True) -> Compiled:
    """strict: the hypergraph must be a dataset label (decodes back exactly); otherwise any free
    drawing that can be drawn (hypergraph.problems)."""
    if check:
        h.check(strict)
    cfg = build_config(h)
    strokes = trace(cfg)
    owners = [[_turn_owner(cfg, g, s[i - 1], s[(i + 1) % len(s)]) for i, g in enumerate(s)] for s in strokes]
    return Compiled(h, cfg, strokes, owners)


# ------------------------------------------------------------------ decoding
def observe(cfg: Config) -> tuple[dict[Point, str], list[tuple[Vertex, tuple[float, float]]]]:
    """What an image shows: the state of every non-SPLIT gate, and each horn as (the dot whose line
    it grows out of, where its tip is)."""
    return {g: s for g, (s, _) in cfg.gates.items()}, [(t.dot, t.point) for t in cfg.tips]


def decode(rows: int, cols: int, gates: dict[Point, str],
           tips: list[tuple[Vertex, tuple[float, float]]]) -> Hypergraph:
    """Rebuild the canonical hypergraph from observed gate states and horns (dot, tip position)."""
    loops = from_gates(rows, cols, gates)
    rels = list(loops.relations)
    # horns: the tip's place relative to its dot gives direction, variant and size; the size must then
    # agree with the dots in use (long exactly when the spot it points at is empty)
    horns = []
    for v, tp in tips:
        cs = [(v, d, var, long) for d in HORN_DIRS for var in HORN_VARIANTS for long in (True, False)
              if max(abs(x - y) for x, y in zip(horn_tip(v, d, var, long), tp)) < 1e-6]
        if len(cs) != 1:
            raise KolamError(f"tip at {tp} on dot {list(v)} matches {len(cs)} horns")
        horns.append(cs[0])
    used = {v for r in rels for v in r.vertices} | {v for v, *_ in horns}
    for v, d, var, long in horns:
        if long != (horn_target(v, d) not in used):
            raise KolamError(f"horn on {list(v)} pointing {d}: its size does not match the dots in use")
        rels.append(Relation(HORN, (v,), var, False, d))
    return from_gates(rows, cols, gates, [r for r in rels if r.type == HORN])
