"""Ordered typed hypergraph: the label of every kolam (docs/hypergraph_kolams.md §1-3, §6).

Labels are canonical: one picture, one label. A picture shows its gap codes (which gaps
cross, run parallel or are walls) and its horns, not how the lines were drawn, so the
canonical label is read off the gaps (`from_gates`): every chain of crossing (or parallel)
gaps is split into lines by one fixed rule. A line goes on through a dot with 2 such gaps
(also around a corner); at a dot with 3 or 4 it goes straight through and the side branch
is a line of its own. Lines are stored from their smaller end, closed rings rotated/
reflected to their smallest form. A HORN is one dot, the direction its tip points
(N/E/S/W) and its variant, all visible, so they are kept exactly.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .spec import (CROSS, CROSSING_LOOP, HORN, HORN_DIRS, HORN_VARIANTS, JOIN, LEFT_HORN,
                   PARALLEL_LOOP, RELATION_TYPES, RIGHT_HORN, Point, Vertex, adjacent, gate_between, horn_dir,
                   horn_target, horn_tip, loop_gate_state)

TYPE_ORDER = {CROSSING_LOOP: 0, PARALLEL_LOOP: 1, HORN: 2}


class KolamError(ValueError):
    pass


@dataclass(frozen=True)
class Relation:
    type: str
    vertices: tuple[Vertex, ...]
    variant: str | None = None   # HORN only
    closed: bool = False         # loops only: last vertex links back to the first
    dir: str | None = None       # HORN only: where its tip points, N/E/S/W

    def canonical(self) -> "Relation":
        if self.type == HORN:
            return self
        vs = list(self.vertices)
        if not self.closed:
            best = min(vs, vs[::-1])
        else:
            k = len(vs)
            cands = []
            for seq in (vs, vs[::-1]):
                cands += [seq[i:] + seq[:i] for i in range(k)]
            best = min(cands)
        return Relation(self.type, tuple(best), None, self.closed)

    def edges(self) -> list[tuple[Vertex, Vertex]]:
        """Consecutive dot pairs whose gate this relation sets."""
        vs = self.vertices
        pairs = list(zip(vs, vs[1:]))
        if self.closed:
            pairs.append((vs[-1], vs[0]))
        return pairs

    def sort_key(self):
        return (TYPE_ORDER[self.type], self.variant or "", self.closed, self.vertices, self.dir or "")

    def to_json(self) -> dict:
        d = {"type": self.type}
        if self.variant:
            d["variant"] = self.variant
        d["vertices"] = [list(v) for v in self.vertices]
        if self.dir:
            d["dir"] = self.dir
        if self.closed:
            d["closed"] = True
        return d

    @staticmethod
    def from_json(d: dict) -> "Relation":
        return Relation(d["type"], tuple(tuple(v) for v in d["vertices"]),
                        d.get("variant"), bool(d.get("closed", False)), d.get("dir"))


@dataclass(frozen=True)
class Hypergraph:
    rows: int
    cols: int
    relations: tuple[Relation, ...] = field(default_factory=tuple)

    @property
    def vertices(self) -> list[Vertex]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def used_vertices(self) -> set[Vertex]:
        return {v for r in self.relations for v in r.vertices}

    def canonical(self) -> "Hypergraph":
        """The label of this kolam's picture: loops re-read from the gaps they set (`from_gates`).
        Different programs that draw the same picture get the same canonical label."""
        if any(level == "error" for level, _ in problems(self)):     # cannot be drawn: tidy only
            rels = sorted({r.canonical() for r in self.relations}, key=Relation.sort_key)
            return Hypergraph(self.rows, self.cols, tuple(rels))
        return from_gates(self.rows, self.cols, loop_gates(self), [r for r in self.relations if r.type == HORN])

    def to_json(self) -> dict:
        return {
            "grid": {"rows": self.rows, "cols": self.cols},
            "vertices": [list(v) for v in self.vertices],
            "relations": [r.to_json() for r in self.relations],
        }

    @staticmethod
    def from_json(d: dict) -> "Hypergraph":
        g = d["grid"]
        return Hypergraph(g["rows"], g["cols"], tuple(Relation.from_json(r) for r in d["relations"]))

    def validate(self, strict: bool = True) -> list[str]:
        return validate(self, strict)

    def check(self, strict: bool = True) -> "Hypergraph":
        errs = validate(self, strict)
        if errs:
            raise KolamError("; ".join(errs))
        return self


def horn_is_long(h: Hypergraph, r: Relation) -> bool:
    """A horn is a long crescent when the spot it points at holds no used dot, else short (§10.6)."""
    return horn_target(r.vertices[0], r.dir) not in h.used_vertices()


def problems(h: Hypergraph) -> list[tuple[str, str]]:
    """Everything wrong with a hypergraph, as (level, message):

    error  it cannot be drawn: off the grid, dots that are not 4-neighbours, one gap given two
           different states, a horn pointing along a brush line, horn tips on top of each other
    label  it draws, but is not a dataset label: a horn pointing at a dot in use (a short stub, no
           room for a crescent); horns meeting in one spot that curl different ways, or that grow
           from the same inner corner of a parallel tube
    (validate(strict=True) also asks for the canonical form.)
    """
    out: list[tuple[str, str]] = []
    if not (1 <= h.rows <= 32 and 1 <= h.cols <= 32):
        return [("error", f"grid {h.rows}x{h.cols} out of range 1..32")]

    def inside(v):
        return 0 <= v[0] < h.rows and 0 <= v[1] < h.cols

    gate_state: dict = {}      # gap -> (state, first relation)
    horns = []
    for i, r in enumerate(h.relations):
        name = f"relation {i} ({r.type})"

        def err(m, level="error"):
            out.append((level, f"{name}: {m}"))
        if r.type not in RELATION_TYPES:
            err("unknown type")
            continue
        bad = [v for v in r.vertices if not inside(v)]
        if bad:
            err(f"vertices {bad} outside the {h.rows}x{h.cols} grid")
            continue
        if len(set(r.vertices)) != len(r.vertices):
            err("repeated vertex")
            continue
        if r.type == HORN:
            if r.variant not in HORN_VARIANTS or r.dir not in HORN_DIRS or len(r.vertices) != 1 or r.closed:
                err(f"a horn is one dot, a direction in {HORN_DIRS} and a variant in {HORN_VARIANTS}")
                continue
            horns.append((i, r))
            continue
        if r.variant is not None or r.dir is not None:
            err("loops have no variant or direction")
        if len(r.vertices) < 2:
            err("a loop needs at least 2 vertices")
            continue
        if r.closed and (len(r.vertices) < 4 or not adjacent(r.vertices[-1], r.vertices[0])):
            err("a closed ring needs >= 4 dots with the last next to the first")
            continue
        state = loop_gate_state(r.type)
        for a, b in r.edges():
            if not adjacent(a, b):
                err(f"{list(a)} and {list(b)} are not neighbours (diagonal gaps have no gate)")
                break
            g = gate_between(a, b)
            if g in gate_state and gate_state[g][0] != state:
                err(f"the gap between {list(a)} and {list(b)} is already a {gate_state[g][0]} of relation "
                    f"{gate_state[g][1]}; one gap can't be both")
            gate_state.setdefault(g, (state, i))
    # horns: grown out of the wall in front of their dot. Tips stay apart, except that horns pointing
    # into the same empty spot meet there (crescents, §10.6), all curling the same way
    used = h.used_vertices()
    seen, tips, meets = {}, [], {}
    for i, r in horns:
        v, d = r.vertices[0], r.dir
        name = f"relation {i} (HORN)"
        if (v, d) in seen:
            out.append(("error", f"{name}: dot {list(v)} already has a horn pointing {d} (relation {seen[v, d]})"))
            continue
        seen[v, d] = i
        g = gate_between(v, horn_target(v, d))
        if g in gate_state:
            out.append(("error", f"{name}: points {d} along the brush line of relation {gate_state[g][1]}; "
                                 "a horn points at a wall (a gap no loop crosses)"))
        t = horn_target(v, d)
        long = t not in used
        if not long:
            out.append(("label", f"{name}: points {d} at dot {list(t)}, which is in use; a horn points at "
                                 "an empty spot or off the grid (else it is only a short stub)"))
        p = horn_tip(v, d, r.variant, long)
        for q, j, tj in tips:
            if long and tj == t:
                continue            # meeting in one empty spot
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= 1:
                out.append(("error", f"{name}: tip too close to the tip of horn {j}"))
        tips.append((p, i, t if long else None))
        if long:
            for j, u, var in meets.get(t, []):
                if var != r.variant:
                    out.append(("label", f"{name}: meets horn {j} in spot {list(t)} but curls another way; "
                                         "horns meeting in one spot curl the same way"))
                d = (v[0] + u[0] - t[0], v[1] + u[1] - t[1])     # at 90 degrees: the dot across the corner
                if d != t and all(gate_state.get(gate_between(x, d), ("",))[0] == JOIN for x in (u, v)):
                    out.append(("label", f"{name}: meets horn {j} at the inner corner of the parallel tube round "
                                         f"{list(d)}; both would grow from the same short line"))
            meets.setdefault(t, []).append((i, v, r.variant))
    return out


def validate(h: Hypergraph, strict: bool = True) -> list[str]:
    """Problems as messages: what makes it impossible to draw; with strict (a dataset label) also
    whether it is the canonical label of its picture."""
    errs = [m for level, m in problems(h) if strict or level == "error"]
    if strict and not errs and h != h.canonical():
        errs.append("not the canonical label of its picture (use .canonical(): loops are read off the gaps)")
    return errs


# ---------------------------------------------------------------- gaps -> canonical lines
STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))   # N E S W


def loop_gates(h: Hypergraph) -> dict[Point, str]:
    """The state every loop sets on its gaps (non-SPLIT gaps only)."""
    out: dict = {}
    for r in h.relations:
        if r.type != HORN:
            for a, b in r.edges():
                out.setdefault(gate_between(a, b), loop_gate_state(r.type))
    return out


def from_gates(rows: int, cols: int, gates: dict[Point, str], horns=()) -> Hypergraph:
    """The canonical hypergraph of a picture with these gap states and horns. Each chain of CROSS
    (or JOIN) gaps is split into lines: a line goes on through a dot with 2 such gaps (corners too),
    goes straight through a dot with 3 or 4 (the side branch is a line of its own), and stops before
    it would visit a dot twice. A cycle of lines through dots with 2 gaps each is a closed ring."""
    rels: list[Relation] = []
    for state, rtype in ((CROSS, CROSSING_LOOP), (JOIN, PARALLEL_LOOP)):
        nb: dict[Vertex, set[int]] = {}
        for g, st in gates.items():
            if st != state:
                continue
            if g[0] % 2:                                   # dots left/right of the gap
                a, b, d = ((g[0] - 1) // 2, g[1] // 2 - 1), ((g[0] - 1) // 2, g[1] // 2), 1
            else:                                          # dots above/below
                a, b, d = (g[0] // 2 - 1, (g[1] - 1) // 2), (g[0] // 2, (g[1] - 1) // 2), 2
            nb.setdefault(a, set()).add(d)
            nb.setdefault(b, set()).add((d + 2) % 4)

        def pair(v, d):
            ds = nb[v]
            if len(ds) == 2:
                return next(x for x in ds if x != d)
            return (d + 2) % 4 if len(ds) > 2 and (d + 2) % 4 in ds else None

        used: set = set()

        def edge(v, d):
            w = (v[0] + STEPS[d][0], v[1] + STEPS[d][1])
            return w, frozenset((v, w))

        def trace(v, d):
            chain, cur = [v], v
            while True:
                w, e = edge(cur, d)
                if w in chain:
                    if w == v and len(chain) >= 4 and pair(v, (d + 2) % 4) == d0:
                        used.add(e)
                        return chain, True
                    return chain, False                    # would visit a dot twice: stop here
                used.add(e)
                chain.append(w)
                nd = pair(w, (d + 2) % 4)
                if nd is None or edge(w, nd)[1] in used:
                    return chain, False
                cur, d = w, nd

        ends = sorted((v, d) for v in nb for d in nb[v] if pair(v, d) is None)
        rest = sorted((v, d) for v in nb for d in nb[v])
        for v, d in ends + rest:
            if edge(v, d)[1] not in used:
                d0 = d
                chain, closed = trace(v, d)
                rels.append(Relation(rtype, tuple(chain), None, closed).canonical())
    rels += [r.canonical() for r in horns]
    return Hypergraph(rows, cols, tuple(sorted(set(rels), key=Relation.sort_key)))


def gate_codes(h: Hypergraph) -> dict[str, list[str]]:
    """The answer key's gap digits (spec §10.1): 0 wall, 1 cross, 2 parallel. "h": rows x (cols-1)
    gaps between left/right neighbours, "v": (rows-1) x cols gaps between up/down neighbours."""
    code = {CROSS: "1", JOIN: "2"}
    g = loop_gates(h)
    return {"h": ["".join(code.get(g.get((2 * r + 1, 2 * c + 2)), "0") for c in range(h.cols - 1)) for r in range(h.rows)],
            "v": ["".join(code.get(g.get((2 * r + 2, 2 * c + 1)), "0") for c in range(h.cols)) for r in range(h.rows - 1)]}


def from_gate_codes(rows: int, cols: int, codes: dict[str, list[str]], horns=()) -> Hypergraph:
    """Inverse of gate_codes (plus the horns): the canonical label."""
    state = {"1": CROSS, "2": JOIN}
    gates = {(2 * r + 1, 2 * c + 2): state[x] for r, row in enumerate(codes["h"]) for c, x in enumerate(row) if x in state}
    gates |= {(2 * r + 2, 2 * c + 1): state[x] for r, row in enumerate(codes["v"]) for c, x in enumerate(row) if x in state}
    return from_gates(rows, cols, gates, horns)


# ---------------------------------------------------------------- symmetry
def transform_vertex(v: Vertex, op: str, rows: int, cols: int) -> Vertex:
    r, c = v
    return {
        "id": (r, c),
        "mirror_lr": (r, cols - 1 - c),
        "mirror_ud": (rows - 1 - r, c),
        "rot180": (rows - 1 - r, cols - 1 - c),
        "rot90": (c, rows - 1 - r),        # square grids only
        "rot270": (cols - 1 - c, r),
        "transpose": (c, r),
        "antitranspose": (cols - 1 - c, rows - 1 - r),
    }[op]


REFLECTIONS = {"mirror_lr", "mirror_ud", "transpose", "antitranspose"}


def transform_relation(rel: Relation, op: str, rows: int, cols: int) -> Relation:
    vs = tuple(transform_vertex(v, op, rows, cols) for v in rel.vertices)
    variant, d = rel.variant, rel.dir
    if rel.type == HORN:
        if op in REFLECTIONS and variant in (LEFT_HORN, RIGHT_HORN):
            variant = RIGHT_HORN if variant == LEFT_HORN else LEFT_HORN
        f = horn_dir(d)                                  # the direction turns with the grid
        a, b = vs[0], transform_vertex((rel.vertices[0][0] + f[0], rel.vertices[0][1] + f[1]), op, rows, cols)
        d = next(k for k in HORN_DIRS if horn_dir(k) == (b[0] - a[0], b[1] - a[1]))
    return Relation(rel.type, vs, variant, rel.closed, d).canonical()


# ---------------------------------------------------------------- isomorphism (§6)
def _alignments(rel: Relation):
    vs = list(rel.vertices)
    if rel.type == HORN:
        return [vs]
    if not rel.closed:
        return [vs, vs[::-1]]
    k = len(vs)
    return [seq[i:] + seq[:i] for seq in (vs, vs[::-1]) for i in range(k)]


def _rel_sig(r: Relation):
    return (r.type, r.variant, r.closed, len(r.vertices), r.dir)


def iso_hash(h: Hypergraph) -> tuple:
    """Cheap invariant: equal for isomorphic hypergraphs (used to bucket before exact checks)."""
    incid: dict = {}
    for r in h.relations:
        for k, v in enumerate(r.vertices):
            role = k if r.type == HORN else -1
            incid.setdefault(v, []).append((_rel_sig(r), role))
    per_vertex = sorted(tuple(sorted(x, key=repr)) for x in incid.values())
    return (tuple(sorted(Counter(_rel_sig(r) for r in h.relations).items(), key=repr)),
            tuple(map(repr, per_vertex)))


def is_isomorphic(h1: Hypergraph, h2: Hypergraph) -> bool:
    """Exact ordered typed hypergraph isomorphism over the dots the relations use."""
    if Counter(map(_rel_sig, h1.relations)) != Counter(map(_rel_sig, h2.relations)):
        return False
    if len(h1.used_vertices()) != len(h2.used_vertices()):
        return False
    rels1 = sorted(h1.relations, key=lambda r: -len(r.vertices))
    rels2 = list(h2.relations)
    used = [False] * len(rels2)
    fwd: dict = {}
    bwd: dict = {}

    def extend(i: int) -> bool:
        if i == len(rels1):
            return True
        r1 = rels1[i]
        for j, r2 in enumerate(rels2):
            if used[j] or _rel_sig(r2) != _rel_sig(r1):
                continue
            for target in _alignments(r2):
                added = []
                ok = True
                for a, b in zip(r1.vertices, target):
                    if a in fwd:
                        if fwd[a] != b:
                            ok = False
                            break
                    elif b in bwd:
                        ok = False
                        break
                    else:
                        fwd[a] = b
                        bwd[b] = a
                        added.append(a)
                if ok:
                    used[j] = True
                    if extend(i + 1):
                        return True
                    used[j] = False
                for a in added:
                    del bwd[fwd.pop(a)]
        return False

    return extend(0)
