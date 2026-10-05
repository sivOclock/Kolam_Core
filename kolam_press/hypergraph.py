"""Ordered typed hypergraph: the label of every kolam (docs/hypergraph_kolams.md §1-3, §6).

Labels are canonical: a loop's start dot and drawing direction cannot be seen in
an image, so open chains are stored from their smaller end, and closed rings are
rotated/reflected to their smallest form. A HORN's order (A -> B) and variant are
visible, so they are kept exactly.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .spec import (CROSSING_LOOP, HORN, HORN_VARIANTS, LEFT_HORN, LOOP_TYPES, PARALLEL_LOOP,
                   RELATION_TYPES, RIGHT_HORN, Vertex, adjacent, gate_between, horn_tip_points)

TYPE_ORDER = {CROSSING_LOOP: 0, PARALLEL_LOOP: 1, HORN: 2}


class KolamError(ValueError):
    pass


@dataclass(frozen=True)
class Relation:
    type: str
    vertices: tuple[Vertex, ...]
    variant: str | None = None   # HORN only
    closed: bool = False         # loops only: last vertex links back to the first

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
        return (TYPE_ORDER[self.type], self.variant or "", self.closed, self.vertices)

    def to_json(self) -> dict:
        d = {"type": self.type}
        if self.variant:
            d["variant"] = self.variant
        d["vertices"] = [list(v) for v in self.vertices]
        if self.closed:
            d["closed"] = True
        return d

    @staticmethod
    def from_json(d: dict) -> "Relation":
        return Relation(d["type"], tuple(tuple(v) for v in d["vertices"]),
                        d.get("variant"), bool(d.get("closed", False)))


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
        rels = sorted({r.canonical() for r in self.relations}, key=Relation.sort_key)
        return Hypergraph(self.rows, self.cols, tuple(rels))

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

    def validate(self) -> list[str]:
        return validate(self)

    def check(self) -> "Hypergraph":
        errs = validate(self)
        if errs:
            raise KolamError("; ".join(errs))
        return self


def validate(h: Hypergraph) -> list[str]:
    """Structural rules that make the hypergraph compile to a proper kolam with a unique label."""
    errs: list[str] = []
    if not (1 <= h.rows <= 32 and 1 <= h.cols <= 32):
        errs.append(f"grid {h.rows}x{h.cols} out of range 1..32")
        return errs

    def inside(v):
        return 0 <= v[0] < h.rows and 0 <= v[1] < h.cols

    gate_owner: dict = {}
    loop_members = {CROSSING_LOOP: {}, PARALLEL_LOOP: {}}
    horn_bases, horn_targets, tips = {}, {}, {}
    for i, r in enumerate(h.relations):
        name = f"relation {i} ({r.type})"
        if r.type not in RELATION_TYPES:
            errs.append(f"{name}: unknown type")
            continue
        bad = [v for v in r.vertices if not inside(v)]
        if bad:
            errs.append(f"{name}: vertices {bad} outside the {h.rows}x{h.cols} grid")
            continue
        if len(set(r.vertices)) != len(r.vertices):
            errs.append(f"{name}: repeated vertex")
            continue
        if r.type == HORN:
            if r.variant not in HORN_VARIANTS or len(r.vertices) != 2 or r.closed:
                errs.append(f"{name}: a horn needs exactly 2 vertices and a variant in {HORN_VARIANTS}")
                continue
        else:
            if r.variant is not None:
                errs.append(f"{name}: loops have no variant")
            if len(r.vertices) < 2:
                errs.append(f"{name}: a loop needs at least 2 vertices")
                continue
            if r.closed and (len(r.vertices) < 4 or not adjacent(r.vertices[-1], r.vertices[0])):
                errs.append(f"{name}: a closed ring needs >= 4 dots with the last next to the first")
                continue
        for a, b in r.edges():
            if not adjacent(a, b):
                errs.append(f"{name}: {list(a)} and {list(b)} are not neighbours (diagonal gaps have no gate)")
                break
            g = gate_between(a, b)
            if g in gate_owner:
                errs.append(f"{name}: gap between {list(a)} and {list(b)} already used by relation {gate_owner[g]}")
            gate_owner[g] = i
        if r.type in LOOP_TYPES:
            for v in r.vertices:
                if v in loop_members[r.type]:
                    errs.append(f"{name}: dot {list(v)} already in {r.type} relation {loop_members[r.type][v]}"
                                " (same-brush loops must not share dots)")
                loop_members[r.type][v] = i
        else:
            a, b = r.vertices
            horn_bases.setdefault(a, i)
            if b in horn_targets:
                errs.append(f"{name}: dot {list(b)} is already the tip of horn {horn_targets[b]}")
            horn_targets[b] = i
            p = horn_tip_points(a, b, r.variant)[0]
            # tips need room: a plain tip (in a gap) next to a curled tip (in the centre beside
            # that gap) leaves their strands < 0.2 apart
            for q, j in tips.items():
                if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= 1:
                    errs.append(f"{name}: tip too close to the tip of horn {j}")
            tips[p] = i
    for b, i in horn_targets.items():
        users = [j for j, r in enumerate(h.relations) if j != i and b in r.vertices]
        if users:
            errs.append(f"relation {i} (HORN): tip dot {list(b)} must not be used by relations {users}")
    return errs


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
    variant = rel.variant
    if op in REFLECTIONS and variant in (LEFT_HORN, RIGHT_HORN):
        variant = RIGHT_HORN if variant == LEFT_HORN else LEFT_HORN
    return Relation(rel.type, vs, variant, rel.closed).canonical()


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
    return (r.type, r.variant, r.closed, len(r.vertices))


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
