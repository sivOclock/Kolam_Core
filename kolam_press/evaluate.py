"""Dataset evaluation: is the generated data correct, unambiguous, diverse and readable from pixels?

Checks every sample independently of the generator:
  structure   label validates, is canonical, and its program re-parses to the same label
  compile     recompiling the label reproduces the stored lattice sequence
  geometry    curves avoid dots, strands keep apart, every CROSS gap shows one X, corners only at tips
  decode      the gate states + tips visible in the drawing decode back to the exact label
  render      stored images re-render byte-identically from (label, style)
  injective   different labels never give the same picture (fixed style render)
and reports diversity (unique labels, isomorphism classes) and distributions.

The pixel probe is a tiny MLP over 9x9 local patches at each gap/centre (grid
position given). If it decodes whole kolams exactly, the labels are carried by the pixels.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .compiler import compile_hypergraph, decode, observe
from .dsl import parse
from .generator import brush_line_components
from .geometry import check_geometry
from .hypergraph import Hypergraph, is_isomorphic, iso_hash
from .render import Style, render_kolam
from .spec import CROSS, CROSSING_LOOP, HORN, JOIN, PARALLEL_LOOP, PLAIN_HORN, is_h_gate, to_lattice


def load(dataset: str | Path) -> list[dict]:
    with open(Path(dataset) / "labels.jsonl") as f:
        return [json.loads(line) for line in f]


def _style(d: dict) -> Style:
    return Style(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


# ------------------------------------------------------------------ pixel probe
GATE_CLASSES = ["SPLIT", "CROSS", "JOIN", "TIP0", "TIP1"]
CENTER_CLASSES = ["NONE", "TIP_NW", "TIP_NE", "TIP_SW", "TIP_SE"]
_DIAG = {(-1, -1): "TIP_NW", (-1, 1): "TIP_NE", (1, -1): "TIP_SW", (1, 1): "TIP_SE"}
_OFF = np.linspace(-0.8, 0.8, 9)


def _ink(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L"), dtype=np.float32)
    d = np.abs(g - np.median(g))
    return np.clip(d / max(np.percentile(d, 99.5), 1.0), 0, 1)


def _bilinear(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = a.shape
    x = np.clip(x - 0.5, 0, w - 1.001)
    y = np.clip(y - 0.5, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * a[y0, x0] + fx * (1 - fy) * a[y0, x0 + 1]
            + (1 - fx) * fy * a[y0 + 1, x0] + fx * fy * a[y0 + 1, x0 + 1])


def probe_features(rec: dict, img: Image.Image):
    """Patches (9x9 ink samples) at every gap and centre, oriented along the gap's dot line."""
    rows, cols = rec["grid"]["rows"], rec["grid"]["cols"]
    st = rec["style"]
    size = img.size[0]
    scale = size / (max(2 * rows, 2 * cols) + 2 * st["margin"])
    ox, oy = (size - 2 * cols * scale) / 2, (size - 2 * rows * scale) / 2
    ink = _ink(img)
    U, V = np.meshgrid(_OFF, _OFF, indexing="ij")
    gates, centers = [], []
    for li in range(2 * rows + 1):
        for lj in range(2 * cols + 1):
            if (li + lj) % 2 == 1:
                gates.append((li, lj))
            elif li % 2 == 0:
                centers.append((li, lj))
    feats = []
    for pts, kind in ((gates, "gate"), (centers, "center")):
        out = []
        for li, lj in pts:
            if kind == "gate" and not is_h_gate((li, lj)):
                r, c = li + U, lj + V      # u runs along the dot line (rows for a vertical gap)
            else:
                r, c = li + V, lj + U
            out.append(_bilinear(ink, ox + c * scale, oy + r * scale).ravel())
        feats.append(np.asarray(out, dtype=np.float32))
    return gates, feats[0], centers, feats[1]


def probe_targets(h: Hypergraph, gates, centers):
    c = compile_hypergraph(h, check=False)
    states, tips = observe(c.config)
    g_lab = []
    for g in gates:
        if g in tips:
            b = to_lattice(tips[g])
            side = (b[1] > g[1]) if is_h_gate(g) else (b[0] > g[0])
            g_lab.append(GATE_CLASSES.index("TIP1" if side else "TIP0"))
        else:
            g_lab.append(GATE_CLASSES.index(states.get(g, "SPLIT")))
    c_lab = []
    for p in centers:
        if p in tips:
            b = to_lattice(tips[p])
            c_lab.append(CENTER_CLASSES.index(_DIAG[(b[0] - p[0], b[1] - p[1])]))
        else:
            c_lab.append(0)
    return np.array(g_lab), np.array(c_lab)


def probe_decode(rows, cols, gates, g_pred, centers, c_pred) -> Hypergraph:
    states, tips = {}, {}
    for g, k in zip(gates, g_pred):
        name = GATE_CLASSES[k]
        if name in (CROSS, JOIN):
            states[g] = name
        elif name.startswith("TIP"):
            s = 1 if name == "TIP1" else -1
            b = (g[0], g[1] + s) if is_h_gate(g) else (g[0] + s, g[1])
            tips[g] = ((b[0] - 1) // 2, (b[1] - 1) // 2)
    inv = {v: k for k, v in _DIAG.items()}
    for p, k in zip(centers, c_pred):
        if k:
            d = inv[CENTER_CLASSES[k]]
            tips[p] = ((p[0] + d[0] - 1) // 2, (p[1] + d[1] - 1) // 2)
    return decode(rows, cols, states, tips)


def _mlp_fit(X, y, k, hidden=96, epochs=400, lr=3e-3, seed=0):
    """One-hidden-layer MLP, full-batch Adam, class-balanced. Small on purpose: a probe, not a model."""
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = ((X - mu) / sd).astype(np.float32)
    params = [rng.normal(0, np.sqrt(2 / Z.shape[1]), (Z.shape[1], hidden)).astype(np.float32),
              np.zeros(hidden, np.float32),
              rng.normal(0, np.sqrt(1 / hidden), (hidden, k)).astype(np.float32),
              np.zeros(k, np.float32)]
    m = [0 * p for p in params]
    v = [0 * p for p in params]
    Y = np.eye(k, dtype=np.float32)[y]
    wc = (1.0 / np.maximum(np.bincount(y, minlength=k), 1))[y]
    wc = (wc / wc.mean()).astype(np.float32)[:, None]
    for t in range(1, epochs + 1):
        W1, b1, W2, b2 = params
        H = np.maximum(Z @ W1 + b1, 0)
        L = H @ W2 + b2
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        P /= P.sum(1, keepdims=True)
        G = (P - Y) * wc / len(Z)
        GH = (G @ W2.T) * (H > 0)
        for i, g in enumerate((Z.T @ GH, GH.sum(0), H.T @ G, G.sum(0))):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            params[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    W1, b1, W2, b2 = params
    return lambda X_: (np.maximum(((X_ - mu) / sd) @ W1 + b1, 0) @ W2 + b2).argmax(1)


def run_probe(dataset: Path, recs: list[dict], n: int, train_frac: float = 0.7, log=print) -> dict:
    recs = recs[:n]
    rows = []
    for r in recs:
        h = Hypergraph.from_json(r["hypergraph"])
        img = Image.open(dataset / r["image"]).convert("RGB")
        gates, Xg, centers, Xc = probe_features(r, img)
        yg, yc = probe_targets(h, gates, centers)
        rows.append((r, h, gates, Xg, yg, centers, Xc, yc))
    k = int(len(rows) * train_frac)
    tr, te = rows[:k], rows[k:]
    fg = _mlp_fit(np.vstack([x[3] for x in tr]), np.concatenate([x[4] for x in tr]), len(GATE_CLASSES))
    fc = _mlp_fit(np.vstack([x[6] for x in tr]), np.concatenate([x[7] for x in tr]), len(CENTER_CLASSES))
    g_ok = g_n = c_ok = c_n = exact = 0
    conf_g = Counter()
    by_style = Counter()
    by_style_ok = Counter()
    for r, h, gates, Xg, yg, centers, Xc, yc in te:
        pg, pc = fg(Xg), fc(Xc)
        g_ok += int((pg == yg).sum())
        g_n += len(yg)
        c_ok += int((pc == yc).sum())
        c_n += len(yc)
        for a, b in zip(yg[pg != yg], pg[pg != yg]):
            conf_g[f"{GATE_CLASSES[a]}->{GATE_CLASSES[b]}"] += 1
        try:
            ok = probe_decode(h.rows, h.cols, gates, pg, centers, pc) == h
        except Exception:
            ok = False
        exact += ok
        kind = "blur/noise" if (r["style"]["blur"] > 0 or r["style"]["noise"] > 0) else "clean"
        by_style[kind] += 1
        by_style_ok[kind] += ok
    res = {
        "train_images": len(tr), "test_images": len(te),
        "gap_accuracy": round(g_ok / max(g_n, 1), 5), "gaps_tested": g_n,
        "centre_accuracy": round(c_ok / max(c_n, 1), 5), "centres_tested": c_n,
        "exact_kolam_decode": round(exact / max(len(te), 1), 4),
        "exact_by_style": {s: round(by_style_ok[s] / by_style[s], 4) for s in by_style},
        "top_gap_confusions": conf_g.most_common(5),
    }
    log(f"  probe: gaps {res['gap_accuracy']:.4f}, centres {res['centre_accuracy']:.4f}, "
        f"exact kolams {res['exact_kolam_decode']:.4f}")
    return res


# ------------------------------------------------------------------ main evaluation
def evaluate(dataset: str | Path, probe: bool = True, probe_n: int = 1500, rerender_n: int = 200,
             iso_n: int = 3000, log=print) -> dict:
    dataset = Path(dataset)
    recs = load(dataset)
    N = len(recs)
    fails = Counter()
    examples: dict[str, str] = {}
    stats = Counter()
    dist = {k: Counter() for k in ("grid", "relations", "strokes", "symmetry", "loop_len", "horn_variant",
                                   "brush_dots", "style")}
    canon_hashes = Counter()
    img_hash_to_label: dict[str, str] = {}
    collisions = 0
    min_dot, min_strand = 9.0, 9.0

    def fail(key, rec, msg=""):
        fails[key] += 1
        examples.setdefault(key, f"{rec['id']}: {msg}"[:300])

    for i, rec in enumerate(recs):
        h = Hypergraph.from_json(rec["hypergraph"])
        if h.validate():
            fail("structure_invalid", rec, "; ".join(h.validate()))
            continue
        if h != h.canonical():
            fail("not_canonical", rec)
        try:
            if parse(rec["program"]) != h:
                fail("program_mismatch", rec)
        except Exception as e:  # noqa: BLE001
            fail("program_mismatch", rec, str(e))
        c = compile_hypergraph(h)
        if [[list(p) for p in s + s[:1]] for s in c.strokes] != rec["lattice_sequence"]:
            fail("compile_not_deterministic", rec)
        g = check_geometry(c)
        if not g["ok"]:
            fail("geometry", rec, "; ".join(g["issues"]))
        min_dot = min(min_dot, g["min_dot_clearance"])
        min_strand = min(min_strand, g["min_strand_clearance"])
        forest, pieces = brush_line_components(h.relations)
        if forest:
            stats["forest_kolams"] += 1
            if g["n_strokes"] != pieces:
                fail("stroke_count_theorem", rec, f"{g['n_strokes']} strokes for {pieces} tree(s)")
        gates, tips = observe(c.config)
        try:
            if decode(h.rows, h.cols, gates, tips) != h:
                fail("decode_mismatch", rec)
        except Exception as e:  # noqa: BLE001
            fail("decode_mismatch", rec, str(e))
        if i < rerender_n:
            a = np.asarray(Image.open(dataset / rec["image"]).convert("RGB"))
            b = np.asarray(render_kolam(c, _style(rec["style"])))
            if a.shape != b.shape or not np.array_equal(a, b):
                fail("render_not_deterministic", rec)
        key = json.dumps(rec["hypergraph"]["relations"]) + f"{h.rows}x{h.cols}"
        canon_hashes[key] += 1
        fixed = render_kolam(c, Style(size=192, supersample=2)).tobytes()
        ih = hashlib.sha1(fixed).hexdigest()
        if ih in img_hash_to_label and img_hash_to_label[ih] != key:
            collisions += 1
        img_hash_to_label.setdefault(ih, key)

        # distributions
        dist["grid"][f"{h.rows}x{h.cols}"] += 1
        dist["relations"][len(h.relations)] += 1
        dist["strokes"][g["n_strokes"]] += 1
        dist["symmetry"][rec["stats"]["symmetry"]] += 1
        st = rec["style"]
        dist["style"]["light bg" if sum(st["bg"]) > 450 else "dark bg"] += 1
        stats["one_stroke"] += g["n_strokes"] == 1
        stats["crossings"] += g["n_crossings"]
        stats["coverage_sum"] += rec["stats"]["coverage"]
        stats["full_coverage"] += rec["stats"]["coverage"] >= 0.999
        types = {r.type for r in h.relations}
        stats["all_three_brushes"] += len(types) == 3
        stats["single_brush"] += len(types) == 1
        members = {t: {v for r in h.relations if r.type == t for v in r.vertices} for t in (CROSSING_LOOP, PARALLEL_LOOP)}
        stats["kolams_with_shared_dots"] += bool(members[CROSSING_LOOP] & members[PARALLEL_LOOP])
        for r in h.relations:
            stats[r.type] += 1
            if r.type == HORN:
                dist["horn_variant"][r.variant] += 1
                dist["brush_dots"]["HORN"] += 2
            else:
                dist["loop_len"][len(r.vertices)] += 1
                dist["brush_dots"][r.type] += len(r.vertices)
                stats["closed_rings"] += r.closed
        if (i + 1) % 1000 == 0:
            log(f"  checked {i + 1}/{N}")

    # isomorphism classes (§6) on a subset
    sub = [Hypergraph.from_json(r["hypergraph"]) for r in recs[:iso_n]]
    buckets: dict = {}
    classes = 0
    for h in sub:
        reps = buckets.setdefault(iso_hash(h), [])
        if not any(is_isomorphic(h, o) for o in reps):
            reps.append(h)
            classes += 1

    report = {
        "dataset": str(dataset), "samples": N,
        "checks": {k: {"failures": fails.get(k, 0), "example": examples.get(k)} for k in
                   ("structure_invalid", "not_canonical", "program_mismatch", "compile_not_deterministic",
                    "geometry", "decode_mismatch", "render_not_deterministic", "stroke_count_theorem")},
        "forest_kolams_checked": stats["forest_kolams"],
        "rerendered_images": min(rerender_n, N),
        "label_to_image_collisions": collisions,
        "unique_labels": len(canon_hashes), "duplicate_label_samples": N - len(canon_hashes),
        "isomorphism_classes": {"checked": len(sub), "classes": classes},
        "min_dot_clearance": round(min_dot, 3), "min_strand_clearance": round(min_strand, 3),
        "means": {
            "relations": round(sum(k * v for k, v in dist["relations"].items()) / N, 2),
            "strokes": round(sum(k * v for k, v in dist["strokes"].items()) / N, 2),
            "crossings": round(stats["crossings"] / N, 2),
            "coverage": round(stats["coverage_sum"] / N, 3),
        },
        "fractions": {
            "one_stroke": round(stats["one_stroke"] / N, 3),
            "full_coverage": round(stats["full_coverage"] / N, 3),
            "all_three_brushes": round(stats["all_three_brushes"] / N, 3),
            "single_brush": round(stats["single_brush"] / N, 3),
            "kolams_with_shared_dots": round(stats["kolams_with_shared_dots"] / N, 3),
        },
        "relation_counts": {t: stats[t] for t in (CROSSING_LOOP, PARALLEL_LOOP, HORN)},
        "closed_rings": stats["closed_rings"],
        "distributions": {k: dict(sorted(v.items(), key=lambda kv: str(kv[0]))) for k, v in dist.items()},
    }
    if probe:
        report["pixel_probe"] = run_probe(dataset, recs, probe_n, log=log)
    (dataset / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (dataset / "report.md").write_text(report_markdown(report))
    return report


def report_markdown(r: dict) -> str:
    L = [f"# Dataset report — `{r['dataset']}`", "", f"**{r['samples']} samples**", "",
         "## Correctness (independent re-check of every sample)", "", "| check | failures |", "|---|---|"]
    for k, v in r["checks"].items():
        L.append(f"| {k} | {v['failures']} |")
    L += ["", f"- re-rendered byte-identical: {r['rerendered_images']} images checked",
          f"- different labels with identical pictures: **{r['label_to_image_collisions']}**",
          f"- stroke_count_theorem: tree-shaped brush lines give one stroke per tree, checked on "
          f"{r['forest_kolams_checked']} kolams",
          f"- min dot clearance {r['min_dot_clearance']} / min strand clearance {r['min_strand_clearance']} "
          "(lattice units; dots are 2 apart)", "",
          "## Diversity", "",
          f"- unique labels: {r['unique_labels']} / {r['samples']} (duplicates: {r['duplicate_label_samples']})",
          f"- isomorphism classes (§6, translation etc. merged): {r['isomorphism_classes']['classes']} / "
          f"{r['isomorphism_classes']['checked']}",
          f"- means: {r['means']}", f"- fractions: {r['fractions']}",
          f"- relations: {r['relation_counts']}, closed rings: {r['closed_rings']}", ""]
    for k, v in r["distributions"].items():
        L.append(f"- **{k}**: " + ", ".join(f"{a}: {b}" for a, b in v.items()))
    if "pixel_probe" in r:
        p = r["pixel_probe"]
        L += ["", "## Pixel probe (tiny MLP on 9x9 local patches, grid position given)", "",
              f"- trained on {p['train_images']} images, tested on {p['test_images']}",
              f"- gap state accuracy: {p['gap_accuracy']} over {p['gaps_tested']} gaps",
              f"- centre (curled tip) accuracy: {p['centre_accuracy']} over {p['centres_tested']} centres",
              f"- **whole kolam decoded exactly: {p['exact_kolam_decode']}** (by style: {p['exact_by_style']})",
              f"- top confusions: {p['top_gap_confusions']}"]
    return "\n".join(L) + "\n"
