"""Spline geometry + geometric validation of compiled kolams.

Closed strokes use the engine's chained quadratic Bezier wrapped around the loop:
segment i runs mid(P[i-1],P[i]) -> mid(P[i],P[i+1]) with control P[i]. (The
notebook's open version starts *at* P[0], which leaves a corner at the seam of a
closed loop.) A repeated control point gives a sharp corner: the horn tip.
All lengths are in lattice units (neighbouring dots are 2 apart).

drawn() is the one place that turns compiled strokes into what is drawn (docs §10):
capsules for parallel runs, no inner loop in a box, and horns grown out of the line.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .compiler import Compiled, Config, HornTip
from .spec import (CROSS, JOIN, Point, horn_curl, horn_dir, horn_shape, horn_target, is_h_gate, left_of,
                   to_lattice)


def per_segment(pts, n: int) -> int:
    """Samples per control point so a stroke gets about n per gate step (1.41), however dense it is."""
    P = np.asarray(pts, dtype=float)
    step = float(np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).mean())
    return max(2, int(round(n * step / 1.41)))


def sample_stroke(pts, n: int = 12) -> np.ndarray:
    P = np.asarray(pts, dtype=float)
    a = (np.roll(P, 1, axis=0) + P) / 2
    b = (P + np.roll(P, -1, axis=0)) / 2
    t = np.linspace(0.0, 1.0, n, endpoint=False)[None, :, None]
    curve = (1 - t) ** 2 * a[:, None] + 2 * (1 - t) * t * P[:, None] + t ** 2 * b[:, None]
    return curve.reshape(-1, 2)


CAPSULE_W = 0.75     # parallel walls: distance from the line of dots (a crossing loop's distance)
CAPSULE_LEAN = 0.5   # a wall point feeding straight into a crossing gap leans in to this, so the X has room


def capsule_points(cfg: Config, s: list[Point], w: float = CAPSULE_W, with_source: bool = False):
    """Control points that draw parallel runs as capsules (docs/hypergraph_kolams.md §10.4).

    Every point of the line on a parallel run sits on a straight wall at distance w from the dots,
    or on a circle of radius w around a dot where the line turns (with an extra point at each
    45 degrees, so turns are round). Other points are unchanged. Drawing only: the stroke's gate
    sequence (the label side) is not touched. A dot with both crossing and parallel gaps is drawn
    the same way (§10.5), so a wall runs straight into the crossing; a wall point that feeds
    straight into a crossing gap leans in to CAPSULE_LEAN so that X has room. with_source=True also returns,
    per output point, the index of the input point it came from."""
    k = len(s)
    join_dots: set = set()
    for g, (st, _) in cfg.gates.items():
        if st == JOIN:
            join_dots |= ({(g[0], g[1] - 1), (g[0], g[1] + 1)} if is_h_gate(g) else {(g[0] - 1, g[1]), (g[0] + 1, g[1])})
    pts, around = [], []
    for i, g in enumerate(s):
        g = tuple(g)
        p, q = s[i - 1], s[(i + 1) % k]
        st = cfg.gates.get(g)
        if st and st[0] == JOIN:                              # wall point beside a parallel gap
            feeds_x = any(cfg.gates.get(tuple(n), ("SPLIT",))[0] == CROSS for n in (p, q))
            off = min(w, CAPSULE_LEAN) if feeds_x else w
            if is_h_gate(g):
                side = 1 if p[0] > g[0] else -1
                pts.append((g[0] + side * off, float(g[1])))
            else:
                side = 1 if p[1] > g[1] else -1
                pts.append((float(g[0]), g[1] + side * off))
            around.append(None)
            continue
        d = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)            # the dot the line turns around here
        if st is None and d[0] % 2 == 1 and d[1] % 2 == 1 and (int(d[0]), int(d[1])) in join_dots:
            pts.append((d[0] + (g[0] - d[0]) * w, d[1] + (g[1] - d[1]) * w))
            around.append((d, (g[0] - d[0], g[1] - d[1])))
        else:
            pts.append((float(g[0]), float(g[1])))
            around.append(None)
    out, src = [], []
    for i in range(k):
        out.append(pts[i])
        src.append(i)
        a, b = around[i], around[(i + 1) % k]
        if a and b and a[0] == b[0]:                          # turning on one dot's circle: add the 45 degree point
            d, u, v = a[0], a[1], b[1]
            n = ((u[0] + v[0]) ** 2 + (u[1] + v[1]) ** 2) ** 0.5
            out.append((d[0] + w * (u[0] + v[0]) / n, d[1] + w * (u[1] + v[1]) / n))
            src.append(i)
    return (out, src) if with_source else out


def drawn(c: Compiled) -> tuple[list[list[tuple[float, float]]], list[list[int]]]:
    """What is actually drawn for a compiled kolam (docs §10): the small loop inside a box of
    parallel gaps is dropped, parallel runs become capsules, and horns grow out of the line.
    Returns control points and the relation owning each point. Labels (c.strokes, the lattice
    sequence) are not changed."""
    strokes, owners = [], []
    for s, own in zip(c.strokes, c.owners):
        if is_box_hole(c.config, s):
            continue
        pts, src = capsule_points(c.config, s, with_source=True)
        strokes.append(pts)
        owners.append([own[i] for i in src])
    if c.config.tips:
        strokes, owners = grow_horns(c.config.tips, strokes, owners)
    return strokes, owners


HORN_SAMPLES = 10   # a stroke that carries a horn is redrawn through this many points per segment


def grow_horns(tips: list[HornTip], strokes, owners):
    """Grow every horn out of the line in front of its dot (docs §10.6). The stroke it grows from is
    replaced by dense points on its curve (drawing the same curve), which are then pulled toward the
    tip with a weight that fades to 0 smoothly, so the horn blends into whatever line it grows from."""
    dense: dict[int, np.ndarray] = {}
    owns: dict[int, list[int]] = {}
    dirs = {(t.dot, horn_dir(t.dir)) for t in tips}
    for t in tips:
        o = np.array(to_lattice(t.dot), float)
        f = np.array(horn_dir(t.dir), float)
        l = np.array(left_of(horn_dir(t.dir)), float)
        front = o + 0.75 * f

        def gap(k):
            P = dense[k] if k in dense else sample_stroke(strokes[k], 4)
            return np.linalg.norm(P - front, axis=1).min()
        k = min(range(len(strokes)), key=gap)
        if k not in dense:
            dense[k] = sample_stroke(strokes[k], HORN_SAMPLES)
            owns[k] = [x for x in owners[k] for _ in range(HORN_SAMPLES)]
        clip = {sd: (t.dot, (int(sd * l[0]), int(sd * l[1]))) in dirs for sd in (1, -1)}
        dense[k], owns[k] = _grow(dense[k], owns[k], t, o, f, l, clip)
    for k in dense:
        strokes[k] = [tuple(p) for p in dense[k]]
        owners[k] = owns[k]
    return strokes, owners


def _grow(D: np.ndarray, own: list[int], t: HornTip, o, f, l, clip: dict):
    sh = horn_shape(t.long)
    n = len(D)
    u, v = (D - o) @ f, (D - o) @ l
    c = int(np.argmin(np.where(u > 0, np.hypot(u - 0.75, v), np.inf)))     # the line just in front
    seg = np.linalg.norm(np.roll(D, -1, axis=0) - D, axis=1)
    pos = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    ahead = (pos - pos[c]) % seg.sum()
    a = np.where(ahead <= seg.sum() - ahead, ahead, ahead - seg.sum())    # signed arc length from c
    if v[(c + 1) % n] < v[c]:
        a = -a                                                            # now a > 0 runs to the left
    half = {}
    for sd in (1, -1):                       # a horn beside this one on the same dot: stop at 45 degrees
        lim = sh["half"]
        if clip[sd]:
            past = (np.sign(a) == sd) & (sd * v > u)
            if past.any():
                lim = min(lim, float(np.abs(a[past]).min()))
        half[sd] = lim
    side = np.where(a >= 0, 1, -1)
    hs = np.where(side > 0, half[1], half[-1])
    win = np.abs(a) < hs
    w = np.where(win, (1 - np.minimum(np.abs(a) / hs, 1)) ** sh["q"], 0.0)
    T = np.array(t.point, float)
    lean = sh["open"] * side - horn_curl(t.variant) * sh["bend"]
    out = D + w[:, None] * (T - D) + (w * (1 - w) * lean)[:, None] * l
    own = [t.rel if w[i] > 0.15 else own[i] for i in range(n)]       # coloured view: the horn proper
    out = np.vstack([out[:c], T, T, out[c + 1:]])                         # the tip, twice: a sharp corner
    return out, own[:c] + [t.rel, t.rel] + own[c + 1:]


def is_box_hole(cfg: Config, s: list[Point]) -> bool:
    """The small loop inside 4 dots whose 4 gaps are all parallel: not drawn, so the 4 dots form a
    filled box (docs §10.4)."""
    return (len(s) == 4 and all(tuple(g) in cfg.gates and cfg.gates[tuple(g)][0] == JOIN for g in s)
            and sum(g[0] for g in s) % 8 == 0 and sum(g[1] for g in s) % 8 == 0)


def _seg_intersections(A0, A1, B0, B1):
    """Proper intersection points of segment pairs (vectorised); NaN where none."""
    r = A1 - A0
    s = B1 - B0
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    qp = B0 - A0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / den
        u = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / den
    hit = (np.abs(den) > 1e-12) & (t >= 0) & (t < 1) & (u >= 0) & (u < 1)
    return A0 + np.where(hit, t, 0.0)[:, None] * r, hit


MEET_R = 0.5   # around a spot where horns meet, their flanks may come close or touch (docs §10.6)


def meeting_spots(c: Compiled) -> list[tuple[int, int]]:
    """Lattice points of the empty spots that two or more horns point into: their tips meet there."""
    n = Counter(to_lattice(horn_target(t.dot, t.dir)) for t in c.config.tips if t.long)
    return sorted(p for p, k in n.items() if k > 1)


def check_geometry(c: Compiled, n: int = 11, dot_clear: float = 0.4, strand_clear: float = 0.2,
                   cross_r: float = 0.3, arc_skip: float = 0.9, meet_r: float = MEET_R) -> dict:
    """Is this a proper kolam? Curves never touch drawn dots (rings pass 0.71 from their dot), strands
    keep apart except at their designed crossings and where horns meet, every CROSS gap shows exactly
    one X, and the only corners are horn tips."""
    rows, cols = c.config.rows, c.config.cols
    issues: list[str] = []
    pts, sid, arc, nxt = [], [], [], []
    lengths = []
    off = 0
    shown, _ = drawn(c)
    for k, s in enumerate(shown):
        P = sample_stroke(s, per_segment(s, n))
        seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
        pts.append(P)
        sid.append(np.full(len(P), k))
        arc.append(np.concatenate([[0], np.cumsum(seg)[:-1]]))
        lengths.append(seg.sum())
        nxt.append(off + (np.arange(len(P)) + 1) % len(P))
        off += len(P)
    if not pts:
        return {"ok": False, "issues": ["no strokes"], "n_strokes": 0}
    P = np.vstack(pts)
    S = np.concatenate(sid)
    A = np.concatenate(arc)
    N = np.concatenate(nxt)
    L = np.asarray(lengths)[S]

    # 1. inside the lattice box (a crescent pointing off the grid may reach one unit past it)
    m = 1.0 + 1e-6
    if P.min() < -m or P[:, 0].max() > 2 * rows + m or P[:, 1].max() > 2 * cols + m:
        issues.append("curve leaves the dot grid")

    # 2. never touch a drawn dot (dots no relation uses are not drawn)
    near = np.stack([np.clip(2 * np.round((P[:, 0] - 1) / 2) + 1, 1, 2 * rows - 1),
                     np.clip(2 * np.round((P[:, 1] - 1) / 2) + 1, 1, 2 * cols - 1)], axis=1)
    lit = np.zeros((rows, cols), bool)
    for r, cc in c.config.ring_dots:
        lit[r, cc] = True
    dd = np.linalg.norm(P - near, axis=1)
    dd[~lit[((near[:, 0] - 1) // 2).astype(int), ((near[:, 1] - 1) // 2).astype(int)]] = np.inf
    min_dot = float(dd.min())
    if min_dot < dot_clear:
        i = int(dd.argmin())
        issues.append(f"curve passes {min_dot:.2f} from dot at lattice {tuple(near[i].astype(int))}")

    # 3. candidate close pairs
    crosses = np.asarray(c.crossings, dtype=float).reshape(-1, 2)
    tips = np.asarray(c.tip_points, dtype=float).reshape(-1, 2)
    Pf = P.astype(np.float32)
    ii_all, jj_all = [], []
    rc = 0.45
    for s0 in range(0, len(P), 1024):
        blk = Pf[s0:s0 + 1024]
        dx = blk[:, 0, None] - Pf[None, :, 0]          # same sums as ((a - b) ** 2).sum(-1), 4x faster
        dy = blk[:, 1, None] - Pf[None, :, 1]
        d2 = dx * dx + dy * dy
        ii, jj = np.nonzero(d2 < rc * rc)
        ii = ii + s0
        keep = jj > ii
        ii_all.append(ii[keep])
        jj_all.append(jj[keep])
    I = np.concatenate(ii_all)
    J = np.concatenate(jj_all)
    dist = np.linalg.norm(P[I] - P[J], axis=1)
    same = S[I] == S[J]
    sep = np.abs(A[I] - A[J])
    sep = np.minimum(sep, L[I] - sep)
    local = same & (sep < arc_skip)

    def near_any(Q, R, r):
        if len(R) == 0:
            return np.zeros(len(Q), bool)
        return (np.linalg.norm(Q[:, None, :] - R[None, :, :], axis=-1) < r).any(1)

    meets = np.asarray(meeting_spots(c), dtype=float).reshape(-1, 2)
    at_cross = near_any(P[I], crosses, cross_r) & near_any(P[J], crosses, cross_r)
    at_cross |= near_any(P[I], meets, meet_r) & near_any(P[J], meets, meet_r)
    bad = (dist < strand_clear) & ~local & ~at_cross
    clear_pairs = dist[~local & ~at_cross]
    min_strand = float(clear_pairs.min()) if len(clear_pairs) else float("inf")
    if bad.any():
        k = int(np.argmax(bad))
        issues.append(f"strands {dist[k]:.2f} apart near lattice ({P[I[k]][0]:.1f},{P[I[k]][1]:.1f})")

    # 4. crossings: every intersection sits on a CROSS gap, one per gap
    adj = (N[I] == J) | (N[J] == I)
    X, hit = _seg_intersections(P[I], P[N[I]], P[J], P[N[J]])
    hit &= ~adj & ~near_any(X, meets, meet_r)       # flanks of meeting horns may touch near their spot
    Xs = np.unique(np.round(X[hit], 4), axis=0)      # an X on a sample point is found from both sides
    found = 0
    if len(Xs):
        if len(crosses):
            dc = np.linalg.norm(Xs[:, None, :] - crosses[None, :, :], axis=-1)
            owner = dc.argmin(1)
            ok = dc.min(1) < cross_r
            if (~ok).any():
                issues.append(f"{int((~ok).sum())} unplanned strand intersection(s)")
            counts = np.bincount(owner[ok], minlength=len(crosses))
        else:
            issues.append(f"{len(Xs)} unplanned strand intersection(s)")
            counts = np.zeros(0, int)
        found = int(ok.sum()) if len(crosses) else 0
    else:
        counts = np.zeros(len(crosses), int)
    if len(crosses) and (counts != 1).any():
        issues.append(f"{int((counts != 1).sum())} crossing gap(s) without exactly one X")

    # 5. no corners except horn tips
    for k, s in enumerate(shown):
        Q = pts[k]
        v0 = Q - np.roll(Q, 1, axis=0)
        v1 = np.roll(Q, -1, axis=0) - Q
        cosang = (v0 * v1).sum(1) / (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1) + 1e-12)
        sharp = cosang < np.cos(np.radians(50))
        if sharp.any():
            if len(tips) == 0 or not near_any(Q[sharp], tips, 0.05).all():
                issues.append("corner away from a horn tip")
                break

    return {
        "ok": not issues,
        "issues": issues,
        "n_strokes": len(shown),
        "n_crossings": len(crosses),
        "n_crossings_found": found,
        "min_dot_clearance": round(min_dot, 4),
        "min_strand_clearance": round(min_strand, 4),
        "length": round(float(sum(lengths)), 3),
    }
