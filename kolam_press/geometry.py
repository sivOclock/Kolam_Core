"""Spline geometry + geometric validation of compiled kolams.

Closed strokes use the engine's chained quadratic Bezier wrapped around the loop:
segment i runs mid(P[i-1],P[i]) -> mid(P[i],P[i+1]) with control P[i]. (The
notebook's open version starts *at* P[0], which leaves a corner at the seam of a
closed loop.) A repeated control point gives a sharp corner: the horn tip.
All lengths are in lattice units (neighbouring dots are 2 apart).
"""
from __future__ import annotations

import numpy as np

from .compiler import Compiled


def sample_stroke(pts, n: int = 12) -> np.ndarray:
    P = np.asarray(pts, dtype=float)
    a = (np.roll(P, 1, axis=0) + P) / 2
    b = (P + np.roll(P, -1, axis=0)) / 2
    t = np.linspace(0.0, 1.0, n, endpoint=False)[None, :, None]
    curve = (1 - t) ** 2 * a[:, None] + 2 * (1 - t) * t * P[:, None] + t ** 2 * b[:, None]
    return curve.reshape(-1, 2)


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


def check_geometry(c: Compiled, n: int = 11, dot_clear: float = 0.4, strand_clear: float = 0.2,
                   cross_r: float = 0.3, arc_skip: float = 0.9) -> dict:
    """Is this a proper kolam? Curves never touch dots (rings pass 0.71 from their dot, a curled
    horn's inner flank 0.45), strands keep apart except at their
    designed crossings, every CROSS gap shows exactly one X, and the only corners are horn tips."""
    rows, cols = c.config.rows, c.config.cols
    issues: list[str] = []
    pts, sid, arc, nxt = [], [], [], []
    lengths = []
    off = 0
    for k, s in enumerate(c.strokes):
        P = sample_stroke(s, n)
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

    # 1. inside the lattice box
    if P.min() < -1e-6 or P[:, 0].max() > 2 * rows + 1e-6 or P[:, 1].max() > 2 * cols + 1e-6:
        issues.append("curve leaves the dot grid")

    # 2. never touch a dot
    near = np.stack([np.clip(2 * np.round((P[:, 0] - 1) / 2) + 1, 1, 2 * rows - 1),
                     np.clip(2 * np.round((P[:, 1] - 1) / 2) + 1, 1, 2 * cols - 1)], axis=1)
    dd = np.linalg.norm(P - near, axis=1)
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
        d2 = ((blk[:, None, :] - Pf[None, :, :]) ** 2).sum(-1)
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

    at_cross = near_any(P[I], crosses, cross_r) & near_any(P[J], crosses, cross_r)
    bad = (dist < strand_clear) & ~local & ~at_cross
    clear_pairs = dist[~local & ~at_cross]
    min_strand = float(clear_pairs.min()) if len(clear_pairs) else float("inf")
    if bad.any():
        k = int(np.argmax(bad))
        issues.append(f"strands {dist[k]:.2f} apart near lattice ({P[I[k]][0]:.1f},{P[I[k]][1]:.1f})")

    # 4. crossings: every intersection sits on a CROSS gap, one per gap
    adj = (N[I] == J) | (N[J] == I)
    X, hit = _seg_intersections(P[I], P[N[I]], P[J], P[N[J]])
    hit &= ~adj
    Xs = X[hit]
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
    for k, s in enumerate(c.strokes):
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
        "n_strokes": len(c.strokes),
        "n_crossings": len(crosses),
        "n_crossings_found": found,
        "min_dot_clearance": round(min_dot, 4),
        "min_strand_clearance": round(min_strand, 4),
        "length": round(float(sum(lengths)), 3),
    }
