"""Brush spec (../brushes.json, shared with kolam_gen and kolam_studio) and lattice helpers.

Anchor (row, col) -> lattice (2*row+1, 2*col+1). Gates are the lattice points
with exactly one odd coordinate; centers have both even. See brushes.json.
"""
from __future__ import annotations

import json
import math
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
HORN_DIRS = ("N", "E", "S", "W")


@lru_cache(maxsize=1)
def brushes() -> dict:
    return json.loads((Path(__file__).resolve().parents[1] / "brushes.json").read_text())


def loop_gate_state(rtype: str) -> str:
    return brushes()["relations"][rtype]["gate_state"]


def horn_dir(name: str) -> tuple[int, int]:
    return tuple(brushes()["relations"][HORN]["dirs"][name])


def horn_shape(long: bool) -> dict:
    return brushes()["relations"][HORN]["grow"]["long" if long else "short"]


def horn_curl(variant: str) -> int:
    """+1 LEFT_HORN, 0 PLAIN_HORN, -1 RIGHT_HORN."""
    return brushes()["relations"][HORN]["variants"][variant]


def gate_flip(state: str, gate: Point) -> tuple[int, int]:
    kind = "h_gate" if is_h_gate(gate) else "v_gate"
    return tuple(brushes()["gate_states"][state][kind])


def to_lattice(v: Vertex) -> Point:
    return (2 * v[0] + 1, 2 * v[1] + 1)


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


def horn_target(v: Vertex, d: str) -> Vertex:
    """The spot a horn on dot v points at (may be off the grid)."""
    f = horn_dir(d)
    return (v[0] + f[0], v[1] + f[1])


def horn_tip(v: Vertex, d: str, variant: str, long: bool) -> tuple[float, float]:
    """Lattice point of a horn's tip (brushes.json HORN.grow): reach along fwd, turned by the curl."""
    o, f = to_lattice(v), horn_dir(d)
    l, sh, s = left_of(f), horn_shape(long), horn_curl(variant)
    a = s * sh["curl"]
    cu, cv = sh["reach"] * math.cos(a), sh["reach"] * math.sin(a)
    return (o[0] + cu * f[0] + cv * l[0], o[1] + cu * f[1] + cv * l[1])
