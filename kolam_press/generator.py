"""Synthetic data generator (docs/hypergraph_kolams.md §5): structure first.

    grid -> propose brush strokes (+ symmetric copies) -> validate -> fill bare dots
         -> compile -> geometric check -> render -> image + exact hypergraph label

Every sample is reproducible from (seed, index).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .compiler import compile_hypergraph
from .dsl import format_program
from .geometry import check_geometry
from .hypergraph import Hypergraph, Relation, gate_codes, transform_relation, validate
from .render import Style, random_style, render_kolam, render_panel
from .spec import CROSSING_LOOP, HORN, HORN_DIRS, HORN_VARIANTS, PARALLEL_LOOP, gate_between, horn_target

GROUPS = {
    "none": ["id"],
    "mirror_lr": ["id", "mirror_lr"],
    "mirror_ud": ["id", "mirror_ud"],
    "rot180": ["id", "rot180"],
    "d2": ["id", "mirror_lr", "mirror_ud", "rot180"],
    "c4": ["id", "rot90", "rot180", "rot270"],
    "d4": ["id", "rot90", "rot180", "rot270", "mirror_lr", "mirror_ud", "transpose", "antitranspose"],
}
SQUARE_ONLY = {"c4": "rot180", "d4": "d2"}
STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass
class GenConfig:
    min_size: int = 3
    max_size: int = 8
    square_prob: float = 0.5
    symmetry: dict = field(default_factory=lambda: {
        "none": 0.25, "mirror_lr": 0.15, "mirror_ud": 0.1, "rot180": 0.1, "d2": 0.2, "c4": 0.1, "d4": 0.1})
    brush_weights: dict = field(default_factory=lambda: {CROSSING_LOOP: 0.4, PARALLEL_LOOP: 0.3, HORN: 0.3})
    tree_prob: float = 0.4        # brush lines form a tree -> one stroke per tree (single-line kolams)
    pure_prob: float = 0.15       # single-brush kolams
    ring_prob: float = 0.15       # loop proposal is a closed rectangle ring (free mode only)
    mix_prob: float = 0.3         # crossing and parallel loops may share dots
    share_prob: float = 0.35      # a loop may run through dots of loops of its own brush ('+' and 'T' dots)
    straight_bias: float = 0.65
    max_horn_frac: float = 0.2    # at most this many horns per grid dot (they are cheap to add)
    max_chain: int = 8
    coverage: tuple = (0.7, 1.0)  # fraction of dots inside the kolam
    fill_prob: float = 0.85       # try to cover leftover bare dots
    crop_prob: float = 1.0        # drop border rows/cols with no kolam on them (unused dots are not
                                  # drawn, so an uncropped grid's size could not be read from the image)
    max_proposals: int = 300
    image_size: int = 384
    augment: bool = True


class _State:
    def __init__(self, rows: int, cols: int, tree: bool = False):
        self.rows, self.cols, self.tree = rows, cols, tree
        self.rels: list[Relation] = []
        self.gates: set = set()
        self.members = {CROSSING_LOOP: set(), PARALLEL_LOOP: set()}
        self.horns: set = set()        # (dot, dir)
        self.used: set = set()

    def add(self, rels):
        for r in rels:
            self.rels.append(r)
            self.used.update(r.vertices)
            for a, b in r.edges():
                self.gates.add(gate_between(a, b))
            if r.type == HORN:
                self.horns.add((r.vertices[0], r.dir))
            else:
                self.members[r.type].update(r.vertices)

    def rebuild(self, rels):
        self.__init__(self.rows, self.cols, self.tree)
        self.add(rels)

    def inside(self, v):
        return 0 <= v[0] < self.rows and 0 <= v[1] < self.cols

    def coverage(self):
        return len(self.used) / (self.rows * self.cols)


def brush_line_components(rels) -> tuple[bool, int]:
    """(is the brush-line graph a forest?, number of connected pieces). For a forest every
    piece compiles to exactly one closed stroke (each new line merges two curves)."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    forest = True
    for r in rels:
        for v in r.vertices:
            find(v)
        for a, b in r.edges():
            ra, rb = find(a), find(b)
            if ra == rb:
                forest = False
            parent[ra] = rb
    return forest, len({find(v) for v in parent})


def _loop_ok(st: _State, t: str, mix: bool, share: bool = False):
    other = PARALLEL_LOOP if t == CROSSING_LOOP else CROSSING_LOOP

    def ok(v):
        return st.inside(v) and (share or v not in st.members[t]) and (mix or v not in st.members[other])
    return ok


def _piece_of(rels) -> dict:
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x
    for r in rels:
        for v in r.vertices:
            find(v)
        for a, b in r.edges():
            parent[find(a)] = find(b)
    return {v: find(v) for v in list(parent)}


def _walk(rng, st: _State, start, want, ok, cfg: GenConfig, piece: dict):
    path = [start]
    d = None
    while len(path) < want:
        cur = path[-1]
        cands = []
        for s in STEPS:
            nx = (cur[0] + s[0], cur[1] + s[1])
            if not (ok(nx) and nx not in path and gate_between(cur, nx) not in st.gates):
                continue
            # tree mode: only bare dots, except one final step that joins a different piece
            if st.tree and nx in st.used and piece.get(nx) == piece.get(start):
                continue
            cands.append(s)
        if not cands:
            break
        if d in cands and rng.random() < cfg.straight_bias:
            s = d
        else:
            s = cands[rng.integers(len(cands))]
        path.append((cur[0] + s[0], cur[1] + s[1]))
        d = s
        if st.tree and path[-1] in st.used:
            break
    return path


def _propose_chain(rng, st: _State, t: str, cfg: GenConfig):
    attach = st.tree and bool(st.used)
    ok = _loop_ok(st, t, attach or rng.random() < cfg.mix_prob, rng.random() < cfg.share_prob)
    pool = [(r, c) for r in range(st.rows) for c in range(st.cols) if ok((r, c))]
    if attach:                 # tree mode: grow out of a dot of the existing kolam
        pool = [v for v in pool if v in st.used]
    if not pool:
        return None
    if not st.tree and rng.random() < cfg.ring_prob:
        ring = _propose_ring(rng, st, ok, cfg)
        if ring:
            return Relation(t, tuple(ring), None, True)
    want = int(rng.integers(2, cfg.max_chain + 1))
    piece = _piece_of(st.rels) if st.tree else {}
    best: list = []
    for _ in range(3):         # keep the longest of a few walks so short stubs stay rare
        path = _walk(rng, st, pool[rng.integers(len(pool))], want, ok, cfg, piece)
        if len(path) > len(best):
            best = path
        if len(best) >= want:
            break
    if len(best) < 2:
        return None
    return Relation(t, tuple(best))


def _propose_ring(rng, st: _State, ok, cfg: GenConfig):
    for _ in range(10):
        h = int(rng.integers(2, min(st.rows, cfg.max_chain // 2 + 1) + 1))
        w = int(rng.integers(2, min(st.cols, cfg.max_chain // 2 + 1) + 1))
        if h > st.rows or w > st.cols:
            continue
        r0 = int(rng.integers(0, st.rows - h + 1))
        c0 = int(rng.integers(0, st.cols - w + 1))
        ring = ([(r0, c) for c in range(c0, c0 + w)] + [(r, c0 + w - 1) for r in range(r0 + 1, r0 + h)]
                + [(r0 + h - 1, c) for c in range(c0 + w - 2, c0 - 1, -1)]
                + [(r, c0) for r in range(r0 + h - 2, r0, -1)])
        pairs = list(zip(ring, ring[1:] + ring[:1]))
        if all(ok(v) for v in ring) and all(gate_between(a, b) not in st.gates for a, b in pairs):
            return ring
    return None


def _propose_horn(rng, st: _State, target=None):
    """A horn on a dot of the kolam (or on the bare dot `target`), pointing at a wall of that dot."""
    if target is not None:
        pool = [target]
    else:
        pool = sorted(st.used) if st.used and (st.tree or rng.random() < 0.85) else \
            [(r, c) for r in range(st.rows) for c in range(st.cols)]
    if not pool:
        return None
    for _ in range(6):
        v = pool[rng.integers(len(pool))]
        dirs = [d for d in HORN_DIRS if (v, d) not in st.horns
                and gate_between(v, horn_target(v, d)) not in st.gates]
        if dirs:
            return Relation(HORN, (v,), HORN_VARIANTS[rng.integers(3)], False, dirs[rng.integers(len(dirs))])
    return None


def _propose_seed(rng, st: _State, t: str, ops, cfg: GenConfig):
    """A straight chain that the symmetry maps onto itself, so a symmetric tree starts as one piece."""
    cands = []
    for L in range(2, cfg.max_chain + 1):
        for r in range(st.rows):
            for c in range(st.cols - L + 1):
                cands.append(Relation(t, tuple((r, c + k) for k in range(L))))
        for c in range(st.cols):
            for r in range(st.rows - L + 1):
                cands.append(Relation(t, tuple((r + k, c) for k in range(L))))
    cands = [x for x in cands if len(_orbit(x, ops, st.rows, st.cols)) == 1]
    return cands[rng.integers(len(cands))] if cands else None


def _snake(rng, rows: int, cols: int, t: str) -> Relation:
    """Serpentine through every dot: one brush line, one stroke, full coverage."""
    order = []
    if rng.random() < 0.5:
        for r in range(rows):
            cs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            order += [(r, c) for c in cs]
    else:
        for c in range(cols):
            rs = range(rows) if c % 2 == 0 else range(rows - 1, -1, -1)
            order += [(r, c) for r in rs]
    return Relation(t, tuple(order))


def _crop(h: Hypergraph) -> Hypergraph:
    used = h.used_vertices()
    if not used:
        return h
    r0, r1 = min(v[0] for v in used), max(v[0] for v in used)
    c0, c1 = min(v[1] for v in used), max(v[1] for v in used)
    rels = tuple(Relation(r.type, tuple((v[0] - r0, v[1] - c0) for v in r.vertices), r.variant, r.closed, r.dir)
                 for r in h.relations)
    return Hypergraph(r1 - r0 + 1, c1 - c0 + 1, rels).canonical()


def _propose_extension(rng, st: _State, v):
    """Grow an existing open chain by one dot into the bare dot v."""
    opts = []
    for r in st.rels:
        if r.type == HORN or r.closed or v not in {*_nbrs(r.vertices[0]), *_nbrs(r.vertices[-1])}:
            continue
        if v in st.members[r.type]:
            continue
        if r.vertices[-1] in _nbrs(v) and gate_between(r.vertices[-1], v) not in st.gates:
            opts.append((r, Relation(r.type, r.vertices + (v,))))
        if r.vertices[0] in _nbrs(v) and gate_between(r.vertices[0], v) not in st.gates:
            opts.append((r, Relation(r.type, (v,) + r.vertices)))
    return opts[rng.integers(len(opts))] if opts else None


def _nbrs(v):
    return [(v[0] + s[0], v[1] + s[1]) for s in STEPS]


def _orbit(rel: Relation, ops, rows, cols):
    return {transform_relation(rel, op, rows, cols) for op in ops}


def _try_set(st: _State, ops, add: Relation, remove: Relation | None = None) -> bool:
    keep = list(st.rels)
    if remove is not None:
        gone = _orbit(remove, ops, st.rows, st.cols)
        keep = [r for r in keep if r not in gone]
    new = sorted(_orbit(add, ops, st.rows, st.cols) - set(keep), key=Relation.sort_key)
    trial = keep + new
    if validate(Hypergraph(st.rows, st.cols, tuple(trial)), strict=False):   # draws; made canonical at the end
        return False
    if st.tree and not brush_line_components(trial)[0]:
        return False
    if remove is None:
        st.add(new)
    else:
        st.rebuild(trial)
    return True


def sample_hypergraph(rng: np.random.Generator, cfg: GenConfig) -> tuple[Hypergraph, dict]:
    m = int(rng.integers(cfg.min_size, cfg.max_size + 1))
    n = m if rng.random() < cfg.square_prob else int(rng.integers(cfg.min_size, cfg.max_size + 1))
    names = list(cfg.symmetry)
    sym = names[rng.choice(len(names), p=np.array(list(cfg.symmetry.values())) / sum(cfg.symmetry.values()))]
    if m != n:
        sym = SQUARE_ONLY.get(sym, sym)
    ops = GROUPS[sym]
    types = list(cfg.brush_weights)
    base = np.array([cfg.brush_weights[t] for t in types], float)
    if rng.random() < cfg.pure_prob:
        w = np.eye(len(types))[rng.choice(len(types), p=base / base.sum())]
    else:
        w = 0.5 * base / base.sum() + 0.5 * rng.dirichlet(np.ones(len(types)))
    tree = bool(rng.random() < cfg.tree_prob)
    if tree and w[types.index(HORN)] == 1:     # horns alone cannot grow a tree
        tree = False
    target = float(rng.uniform(*cfg.coverage))
    st = _State(m, n, tree)
    pure_loop = [t for t in (CROSSING_LOOP, PARALLEL_LOOP) if w[types.index(t)] == 1]
    if tree and pure_loop:      # one brush, one line: a serpentine through every dot
        snake = _snake(rng, m, n, pure_loop[0])
        if len(_orbit(snake, ops, m, n)) > 1:
            sym, ops = "none", GROUPS["none"]
        _try_set(st, ops, snake)
    elif tree and len(ops) > 1:
        loops = [t for t in (CROSSING_LOOP, PARALLEL_LOOP) if w[types.index(t)] > 0]
        seed = _propose_seed(rng, st, loops[rng.integers(len(loops))], ops, cfg)
        if seed is not None:
            _try_set(st, ops, seed)
    horn_cap = max(1, round(cfg.max_horn_frac * m * n))
    for _ in range(cfg.max_proposals):
        if st.coverage() >= target:
            break
        t = types[rng.choice(len(types), p=w / w.sum())]
        if t == HORN and ((tree and not st.used) or len(st.horns) >= horn_cap):
            continue
        rel = _propose_horn(rng, st) if t == HORN else _propose_chain(rng, st, t, cfg)
        if rel is not None:
            _try_set(st, ops, rel)
    if rng.random() < cfg.fill_prob:
        # cover leftover bare dots: grow a neighbouring chain, put a horn on it, or a 2-dot loop
        allowed = [t for t, x in zip(types, w) if x > 0]
        bare = [(r, c) for r in range(m) for c in range(n) if (r, c) not in st.used]
        for i in rng.permutation(len(bare)):
            v = bare[i]
            if v in st.used:
                continue
            ext = _propose_extension(rng, st, v)
            if ext and _try_set(st, ops, ext[1], remove=ext[0]):
                continue
            for j in rng.permutation(len(allowed)):
                if v in st.used:
                    break
                t = allowed[j]
                if t == HORN:   # a lone dot is its own piece, so not in tree mode
                    rel = None if tree or len(st.horns) >= horn_cap else _propose_horn(rng, st, target=v)
                else:
                    ok = _loop_ok(st, t, True)
                    nb = [u for u in _nbrs(v) if ok(u) and gate_between(u, v) not in st.gates
                          and (not tree or u in st.used)]
                    rel = Relation(t, (v, nb[rng.integers(len(nb))])) if nb and ok(v) else None
                if rel is not None:
                    _try_set(st, ops, rel)
    forest, pieces = brush_line_components(st.rels)
    meta = {"symmetry": sym, "tree_mode": tree, "forest": forest, "pieces": pieces,
            "target_coverage": round(target, 3), "coverage": round(st.coverage(), 3),
            "brush_mix": {t: round(float(x), 3) for t, x in zip(types, w / w.sum())}}
    h = Hypergraph(m, n, tuple(st.rels)).canonical()
    if rng.random() < cfg.crop_prob:
        h = _crop(h)
    meta["coverage"] = round(len(h.used_vertices()) / (h.rows * h.cols), 3)
    return h, meta


def generate_sample(index: int, seed: int, cfg: GenConfig, attempt: int = 0):
    key = [seed, index] + ([attempt] if attempt else [])
    rng = np.random.default_rng(np.random.SeedSequence(key))
    rejected = []
    for _ in range(50):
        h, meta = sample_hypergraph(rng, cfg)
        if not h.relations:
            rejected.append("empty")
            continue
        c = compile_hypergraph(h)
        geo = check_geometry(c)
        if geo["ok"]:
            break
        rejected.append("; ".join(geo["issues"]))
    else:
        raise RuntimeError(f"sample {index}: no valid kolam in 50 tries: {rejected[-3:]}")
    style = random_style(rng, cfg.image_size) if cfg.augment else Style(size=cfg.image_size)
    rec = {
        "id": f"kolam_{index:06d}",
        "image": f"images/kolam_{index:06d}.png",
        "seed": key,
        "grid": {"rows": h.rows, "cols": h.cols},
        "program": format_program(h),
        "gaps": gate_codes(h),
        "hypergraph": h.to_json(),
        "lattice_sequence": [[list(p) for p in s + s[:1]] for s in c.strokes],
        "stats": {"relations": len(h.relations), "strokes": geo["n_strokes"], "crossings": geo["n_crossings"],
                  "min_dot_clearance": geo["min_dot_clearance"],
                  "min_strand_clearance": geo["min_strand_clearance"], **meta},
        "style": style.to_json(),
        "rejected_before": rejected,
    }
    return rec, c, style


def _worker(args):
    index, seed, cfg, out, preview, attempt = args
    rec, c, style = generate_sample(index, seed, cfg, attempt)
    render_kolam(c, style).save(out / rec["image"])
    if index < preview:
        render_panel(c, rec["program"], train_style=style, title=rec["id"]).save(out / "preview" / f"{rec['id']}.png")
    return rec


def _label_key(rec: dict) -> str:
    return json.dumps([rec["grid"], rec["hypergraph"]["relations"]])


def generate_dataset(out: str | Path, n: int, seed: int = 0, cfg: GenConfig | None = None,
                     workers: int = 0, preview: int = 0, unique: bool = True, log=print) -> Path:
    cfg = cfg or GenConfig()
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    if preview:
        (out / "preview").mkdir(exist_ok=True)
    jobs = [(i, seed, cfg, out, preview, 0) for i in range(n)]
    seen: set = set()
    redrawn = 0
    t0 = time.time()
    workers = workers or max(1, (mp.cpu_count() or 2) - 1)
    recs = []
    with (out / "labels.jsonl").open("w") as f:
        if workers == 1:
            it = map(_worker, jobs)
        else:
            pool = mp.get_context("fork").Pool(workers)
            it = pool.imap(_worker, jobs, chunksize=8)
        for k, rec in enumerate(it, 1):
            attempt = 0
            while unique and _label_key(rec) in seen and attempt < 50:   # same kolam twice: draw again
                attempt += 1
                redrawn += 1
                rec = _worker((k - 1, seed, cfg, out, preview, attempt))
            seen.add(_label_key(rec))
            f.write(json.dumps(rec) + "\n")
            recs.append(rec)
            if k % 500 == 0 or k == n:
                log(f"  {k}/{n} samples  ({time.time() - t0:.1f}s)")
        if workers != 1:
            pool.close()
            pool.join()
    manifest = {"n": n, "seed": seed, "unique": unique, "redrawn_duplicates": redrawn,
                "config": asdict(cfg), "seconds": round(time.time() - t0, 2),
                "brushes": "kolam_press/brushes.json"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out
