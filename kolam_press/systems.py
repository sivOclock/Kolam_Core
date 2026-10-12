"""Experimental generators that write the gate grid directly (docs/hypergraph_kolams.md §10.1).

Each one sets gap states (plus crescent horns); the label is read off the gaps (`from_gates`), so
every sample is an exact canonical label, checked like the main generator's.

  substitution  a 2D L-system: every dot is rewritten into a k x k block of dots, generation after
                generation, the block chosen by the dot's number of open gaps and the seam between
                two blocks by the gap between their parent dots (Siromoney's kolam array grammars,
                Prusinkiewicz's L-system kolams)
  switching     start from a full sikku grid (every gap a crossing) and switch gaps, one symmetric orbit
                at a time, keeping a switch only when it merges strokes, down to one line
                (Sivakumar & Sivakumar 2026; Gerdes' mirror curves)
  wfc           wave function collapse: fill the grid with dot tiles (a dot's 4 gap states) seen in a
                real kolam, with their frequencies, so the local style is copied from a photo

    python -m kolam_press systems -n 100 -o data/systems
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .compiler import Config, compile_hypergraph, trace
from .dsl import format_program, parse
from .generator import GROUPS, generate_dataset
from .gapcode import to_sparse
from .geometry import check_geometry
from .hypergraph import Hypergraph, Relation, from_gates, gate_codes, loop_gates, problems, transform_relation
from .render import random_style, render_kolam
from .spec import CROSS, HORN, HORN_DIRS, HORN_VARIANTS, JOIN, SPLIT, gate_between, horn_target

STATES = (SPLIT, CROSS, JOIN)
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


# ------------------------------------------------------------------ gate-grid helpers
def lat_op(p, op: str, rows: int, cols: int):
    """A lattice point under a symmetry of the rows x cols dot grid (as hypergraph.transform_vertex)."""
    R, C = 2 * rows, 2 * cols
    i, j = p
    return {"id": (i, j), "mirror_lr": (i, C - j), "mirror_ud": (R - i, j), "rot180": (R - i, C - j),
            "rot90": (j, R - i), "rot270": (C - j, i), "transpose": (j, i), "antitranspose": (C - j, R - i)}[op]


def orbits(points, ops, rows, cols) -> list[list]:
    seen, out = set(), []
    for p in points:
        if p in seen:
            continue
        o = sorted({lat_op(p, op, rows, cols) for op in ops})
        seen.update(o)
        out.append(o)
    return out


def inner_gates(mask: np.ndarray) -> list:
    rows, cols = mask.shape
    out = []
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and c + 1 < cols and mask[r, c + 1]:
                out.append(gate_between((r, c), (r, c + 1)))
            if mask[r, c] and r + 1 < rows and mask[r + 1, c]:
                out.append(gate_between((r, c), (r + 1, c)))
    return out


def dot_gates(v):
    """The 4 gaps of dot v, in N E S W order."""
    li, lj = 2 * v[0] + 1, 2 * v[1] + 1
    return [(li - 1, lj), (li, lj + 1), (li + 1, lj), (li, lj - 1)]


def shape_mask(rng, n: int) -> tuple[np.ndarray, str]:
    """A D4-symmetric dot layout on an n x n grid: square, diamond (1-3-5-3-1), octagon or plus."""
    kind = ["square", "diamond", "octagon", "plus"][rng.choice(4, p=[0.3, 0.35, 0.2, 0.15])]
    r, c = np.mgrid[0:n, 0:n]
    ctr = (n - 1) / 2
    d = np.abs(r - ctr) + np.abs(c - ctr)
    if kind == "diamond":
        m = d <= n // 2
    elif kind == "octagon":
        m = d <= n // 2 + n // 4
    elif kind == "plus":
        w = max(1, n // 6)
        m = (np.abs(r - ctr) <= w + 0.5) | (np.abs(c - ctr) <= w + 0.5)
    else:
        m = np.ones((n, n), bool)
    return m, kind


def strokes_of(rows, cols, gates: dict, used) -> list:
    cfg = Config(rows, cols)
    cfg.gates = {g: (s, 0) for g, s in gates.items() if s != SPLIT}
    cfg.ring_dots = set(used)
    return trace(cfg)


def used_dots(rows, cols, gates: dict) -> set:
    return {(r, c) for r in range(rows) for c in range(cols)
            if any(gates.get(g, SPLIT) != SPLIT for g in dot_gates((r, c)))}


def add_horns(rng, rows, cols, gates: dict, ops, rate: float, base_horns=()) -> list:
    """Crescent horns on dots of the kolam pointing at empty spots or off the grid, whole symmetric
    orbits at a time, keeping only those that pass every label rule."""
    loops = {g: s for g, s in gates.items() if s != SPLIT}
    used = used_dots(rows, cols, loops)
    cands = [(v, d) for v in sorted(used) for d in HORN_DIRS
             if horn_target(v, d) not in used and loops.get(gate_between(v, horn_target(v, d)), SPLIT) == SPLIT]
    horns = list(base_horns)
    cap = max(1, int(0.12 * len(used)))
    for i in rng.permutation(len(cands)):
        if len(horns) >= cap or rng.random() > rate:
            continue
        v, d = cands[i]
        rel = Relation(HORN, (v,), HORN_VARIANTS[rng.integers(3)], False, d)
        orbit = sorted({transform_relation(rel, op, rows, cols) for op in ops} - set(horns), key=Relation.sort_key)
        trial = horns + orbit
        if not problems(from_gates(rows, cols, loops, trial)):
            horns = trial
    return horns


# ------------------------------------------------------------------ 1. substitution (2D L-system)
def _palindrome(rng, k: int, probs) -> tuple:
    half = [STATES[rng.choice(3, p=probs)] for _ in range((k + 1) // 2)]
    return tuple(half + half[: k // 2][::-1])


def substitution(rng) -> tuple[Hypergraph, dict]:
    s, k = [(2, 2), (3, 2), (4, 2), (2, 3), (3, 3)][rng.integers(5)]
    ops = GROUPS["d4"]
    # seed: a small symmetric gate grid
    mask = np.ones((s, s), bool)
    gates = {}
    for o in orbits(inner_gates(mask), ops, s, s):
        st = STATES[rng.choice(3, p=[0.2, 0.5, 0.3])]
        gates.update({g: st for g in o})
    # rules: a D4-symmetric block per number of open gaps, a palindromic seam per parent gap state
    open_p = [0.15, 0.55, 0.3]
    if k == 2:
        motif = {deg: (STATES[rng.choice(3, p=open_p)],) for deg in range(5)}
    else:
        motif = {deg: (STATES[rng.choice(3, p=open_p)], STATES[rng.choice(3, p=open_p)]) for deg in range(5)}
    seam = {SPLIT: _palindrome(rng, k, [0.7, 0.15, 0.15]), CROSS: _palindrome(rng, k, [0.2, 0.55, 0.25]),
            JOIN: _palindrome(rng, k, [0.2, 0.25, 0.55])}
    n, gens = s, 0
    while n * k <= 12:
        gates = _rewrite(gates, n, k, motif, seam)
        n, gens = n * k, gens + 1
    horns = add_horns(rng, n, n, gates, ops, rate=float(rng.uniform(0.1, 0.5)))
    rule = {"seed": s, "k": k, "generations": gens,
            "block": {str(d): list(m) for d, m in motif.items()}, "seam": {s_: list(p) for s_, p in seam.items()}}
    return from_gates(n, n, {g: x for g, x in gates.items() if x != SPLIT}, horns), {"symmetry": "d4", "rule": rule}


def _rewrite(gates: dict, n: int, k: int, motif: dict, seam: dict) -> dict:
    out = {}
    for r in range(n):
        for c in range(n):
            deg = sum(gates.get(g, SPLIT) != SPLIT for g in dot_gates((r, c)))
            m = motif[deg]
            r0, c0 = r * k, c * k
            for i in range(k):
                for j in range(k):
                    a = (r0 + i, c0 + j)
                    for b in ((r0 + i, c0 + j + 1), (r0 + i + 1, c0 + j)):
                        if b[0] >= r0 + k or b[1] >= c0 + k:
                            continue
                        if k == 2:
                            st = m[0]
                        else:   # the 4 gaps touching the block's centre dot, then the 8 round its edge
                            st = m[0] if (1, 1) in ((a[0] - r0, a[1] - c0), (b[0] - r0, b[1] - c0)) else m[1]
                        out[gate_between(a, b)] = st
            if c + 1 < n:
                p = seam[gates.get(gate_between((r, c), (r, c + 1)), SPLIT)]
                for i in range(k):
                    out[gate_between((r0 + i, c0 + k - 1), (r0 + i, c0 + k))] = p[i]
            if r + 1 < n:
                p = seam[gates.get(gate_between((r, c), (r + 1, c)), SPLIT)]
                for j in range(k):
                    out[gate_between((r0 + k - 1, c0 + j), (r0 + k, c0 + j))] = p[j]
    return out


# ------------------------------------------------------------------ 2. gate switching (one stroke)
def switching(rng) -> tuple[Hypergraph, dict]:
    n = int(rng.integers(5, 12))
    mask, shape = shape_mask(rng, n)
    sym = ["d4", "d2", "c4"][rng.choice(3, p=[0.5, 0.3, 0.2])]
    ops = GROUPS[sym]
    dots = {(r, c) for r in range(n) for c in range(n) if mask[r, c]}
    gates = {g: CROSS for g in inner_gates(mask)}
    orbs = orbits(list(gates), ops, n, n)
    join_p = float(rng.choice([0.0, 0.15, 0.35]))
    target = 1 if rng.random() < 0.75 else int(rng.integers(2, 5))
    strokes = strokes_of(n, n, gates, dots)
    stale = 0
    while len(strokes) > target and stale < 150:
        owner: dict = {}
        for k_, s in enumerate(strokes):
            for g in s:
                owner.setdefault(g, set()).add(k_)
        cands = [o for o in orbs if any(len(owner.get(g, ())) == 2 for g in o)]   # two strokes meet here
        if not cands:
            break
        o = cands[rng.integers(len(cands))]
        new = [x for x in STATES if x != gates[o[0]]]
        st = JOIN if (JOIN in new and rng.random() < join_p) else (SPLIT if SPLIT in new else CROSS)
        old = {g: gates[g] for g in o}
        gates.update({g: st for g in o})
        alive = all(any(gates.get(g, SPLIT) != SPLIT for g in dot_gates(v)) for v in dots)
        trial = strokes_of(n, n, gates, dots) if alive else strokes
        if alive and len(trial) < len(strokes):
            strokes, stale = trial, 0
        else:
            gates.update(old)
            stale += 1
    horns = add_horns(rng, n, n, gates, ops, rate=float(rng.uniform(0.0, 0.6)))
    return (from_gates(n, n, {g: x for g, x in gates.items() if x != SPLIT}, horns),
            {"symmetry": sym, "shape": shape, "target_strokes": target})


# ------------------------------------------------------------------ 3. wave function collapse
def photo_tiles() -> dict:
    """Dot tiles (N, E, S, W gap states) of the photographed sikku kolam with all 8 symmetries."""
    h = parse((EXAMPLES / "sikku_diamond.kolam").read_text())
    g = loop_gates(h)
    count: dict = {}
    for v in h.used_vertices():
        t = tuple(g.get(x, SPLIT) for x in dot_gates(v))
        for q in range(4):
            for tt in (t, (t[0], t[3], t[2], t[1])):           # a mirror swaps E and W
                key = tt[-q:] + tt[:-q] if q else tt
                count[key] = count.get(key, 0) + 1
    return count


def wfc(rng, tiles: dict) -> tuple[Hypergraph, dict]:
    keys = list(tiles)
    w = np.array([tiles[t] for t in keys], float)
    for _ in range(80):
        n = int(rng.integers(5, 12))
        mask, shape = shape_mask(rng, n)
        sym = ["d4", "d2", "c4"][rng.choice(3, p=[0.5, 0.3, 0.2])]
        ops = GROUPS[sym]
        dots = [(r, c) for r in range(n) for c in range(n) if mask[r, c]]
        free = set(inner_gates(mask))
        fixed: dict = {}
        tile_of: dict = {}

        def options(v):
            gs = dot_gates(v)
            return [i for i, t in enumerate(keys)
                    if all((t[s] == SPLIT) if gs[s] not in free else fixed.get(gs[s], t[s]) == t[s] for s in range(4))]

        ok = True
        while ok and len(tile_of) < len(dots):
            opts = {v: options(v) for v in dots if v not in tile_of}
            if any(not o for o in opts.values()):
                ok = False
                break
            v = min(opts, key=lambda u: (len(opts[u]), rng.random()))       # least freedom first
            o = opts[v]
            t = keys[o[rng.choice(len(o), p=w[o] / w[o].sum())]]
            for g, st in zip(dot_gates(v), t):
                for op in ops:
                    gg = lat_op(g, op, n, n)
                    if gg in free:
                        if fixed.setdefault(gg, st) != st:
                            ok = False
            for op in ops:                                                  # the orbit of v is now set too
                li, lj = lat_op((2 * v[0] + 1, 2 * v[1] + 1), op, n, n)
                tile_of[((li - 1) // 2, (lj - 1) // 2)] = True
        if not ok:
            continue
        gates = {g: s for g, s in fixed.items() if s != SPLIT}
        if used_dots(n, n, gates) != set(dots):
            continue
        horns = add_horns(rng, n, n, gates, ops, rate=float(rng.uniform(0.0, 0.6)))
        return from_gates(n, n, gates, horns), {"symmetry": sym, "shape": shape}
    raise RuntimeError("wfc: no tiling in 80 tries")


# ------------------------------------------------------------------ dataset output
def write_system(name: str, make, out: Path, n: int, seed: int, size: int = 384) -> Path:
    """n samples of one system in the main dataset format (labels.jsonl + images/), plus sheet.png."""
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    seen, recs = set(), []
    rng = np.random.default_rng(seed)
    tries = 0
    while len(recs) < n:
        tries += 1
        if tries > 50 * n:
            raise RuntimeError(f"{name}: only {len(recs)} samples in {tries} tries")
        h, meta = make(rng)
        if not h.relations or h.validate():
            continue
        h = Hypergraph(*_crop(h))
        key = json.dumps(h.to_json())
        if key in seen:
            continue
        c = compile_hypergraph(h)
        geo = check_geometry(c)
        if not geo["ok"]:
            continue
        seen.add(key)
        i = len(recs)
        style = random_style(rng, size)
        rec = {"id": f"{name}_{i:04d}", "image": f"images/{name}_{i:04d}.png", "system": name,
               "grid": {"rows": h.rows, "cols": h.cols}, "program": format_program(h), "gaps": gate_codes(h),
               "sparse": [list(t) for t in to_sparse(h)],
               "hypergraph": h.to_json(), "lattice_sequence": [[list(p) for p in s + s[:1]] for s in c.strokes],
               "stats": {"relations": len(h.relations), "strokes": geo["n_strokes"], "crossings": geo["n_crossings"],
                         "coverage": round(len(h.used_vertices()) / (h.rows * h.cols), 3), **meta},
               "style": style.to_json()}
        render_kolam(c, style).save(out / rec["image"])
        recs.append(rec)
    with (out / "labels.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    contact_sheet(out, recs)
    return out


def _crop(h: Hypergraph):
    used = h.used_vertices()
    r0, c0 = min(v[0] for v in used), min(v[1] for v in used)
    r1, c1 = max(v[0] for v in used), max(v[1] for v in used)
    rels = tuple(Relation(r.type, tuple((v[0] - r0, v[1] - c0) for v in r.vertices), r.variant, r.closed, r.dir)
                 for r in h.relations)
    return r1 - r0 + 1, c1 - c0 + 1, Hypergraph(r1 - r0 + 1, c1 - c0 + 1, rels).canonical().relations


def contact_sheet(out: Path, recs: list, cols: int = 10, tile: int = 200) -> None:
    rows = (len(recs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * tile), (10, 10, 12))
    for i, r in enumerate(recs):
        im = Image.open(out / r["image"]).convert("RGB").resize((tile - 4, tile - 4), Image.LANCZOS)
        sheet.paste(im, ((i % cols) * tile + 2, (i // cols) * tile + 2))
    sheet.save(out / "sheet.png")


def generate_systems(out: str | Path, n: int = 100, seed: int = 0, log=print) -> None:
    out = Path(out)
    log("random (main generator)")
    generate_dataset(out / "random", n, seed=seed, log=lambda *_: None)
    contact_sheet(out / "random", [json.loads(x) for x in (out / "random" / "labels.jsonl").open()])
    tiles = photo_tiles()
    for name, make in (("substitution", substitution), ("switching", switching),
                       ("wfc", lambda rng: wfc(rng, tiles))):
        log(name)
        write_system(name, make, out / name, n, seed)
