"""Brush spec (brushes.json) and lattice helpers.

Anchor (row, col) -> lattice (2*row+1, 2*col+1). Gates are the lattice points
with exactly one odd coordinate; centers have both even. See brushes.json.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

Vertex = tuple[int, int]   # anchor dot (row, col)
Point = tuple[int, int]    # lattice point (li, lj)

CROSS, JOIN, SPLIT = "CROSS", "JOIN", "SPLIT"
CROSSING_LOOP, PARALLEL_LOOP, HORN = "CROSSING_LOOP", "PARALLEL_LOOP", "HORN"
PLAIN_HORN, LEFT_HORN, RIGHT_HORN = "PLAIN_HORN", "LEFT_HORN", "RIGHT_HORN"
LOOP_TYPES = (CROSSING_LOOP, PARALLEL_LOOP)
RELATION_TYPES = (CROSSING_LOOP, PARALLEL_LOOP, HORN)
HORN_VARIANTS = (PLAIN_HORN, LEFT_HORN, RIGHT_HORN)


@lru_cache(maxsize=1)
def brushes() -> dict:
    return json.loads((Path(__file__).with_name("brushes.json")).read_text())


def loop_gate_state(rtype: str) -> str:
    return brushes()["relations"][rtype]["gate_state"]


def tip_template(variant: str) -> list[tuple[int, int]]:
    return [tuple(p) for p in brushes()["relations"][HORN]["tip_templates"][variant]]


def gate_flip(state: str, gate: Point) -> tuple[int, int]:
    kind = "h_gate" if is_h_gate(gate) else "v_gate"
    return tuple(brushes()["gate_states"][state][kind])


def to_lattice(v: Vertex) -> Point:
    return (2 * v[0] + 1, 2 * v[1] + 1)


def is_anchor(p: Point) -> bool:
    return p[0] % 2 == 1 and p[1] % 2 == 1


def is_gate(p: Point) -> bool:
    return (p[0] + p[1]) % 2 == 1


def is_h_gate(p: Point) -> bool:
    return p[0] % 2 == 1 and p[1] % 2 == 0


def adjacent(a: Vertex, b: Vertex) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def gate_between(a: Vertex, b: Vertex) -> Point:
    """Lattice gate between two 4-adjacent anchors."""
    return (a[0] + b[0] + 1, a[1] + b[1] + 1)


def left_of(d: tuple[int, int]) -> tuple[int, int]:
    """Rotate a unit direction 90deg counter-clockwise on screen (rows grow downward)."""
    return (-d[1], d[0])


def horn_frame(a: Vertex, b: Vertex) -> tuple[Point, tuple[int, int], tuple[int, int]]:
    """(B's lattice point, fwd, left) for HORN(A -> B)."""
    fwd = (b[0] - a[0], b[1] - a[1])
    return to_lattice(b), fwd, left_of(fwd)


def horn_tip_points(a: Vertex, b: Vertex, variant: str) -> list[Point]:
    o, f, l = horn_frame(a, b)
    return [(o[0] + u * f[0] + v * l[0], o[1] + u * f[1] + v * l[1]) for u, v in tip_template(variant)]


def horn_far_turn(a: Vertex, b: Vertex) -> tuple[Point, Point, Point]:
    """(left gate, far gate, right gate) of B: the turn the tip template replaces."""
    o, f, l = horn_frame(a, b)
    return ((o[0] + l[0], o[1] + l[1]), (o[0] + f[0], o[1] + f[1]), (o[0] - l[0], o[1] - l[1]))
