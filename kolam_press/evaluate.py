"""Dataset evaluation: is the generated data correct, unambiguous, diverse and readable from pixels?

Checks every sample independently of the generator:
  structure   label validates, is canonical, and its program re-parses to the same label
  compile     recompiling the label reproduces the stored lattice sequence
  geometry    curves avoid dots, strands keep apart, every CROSS gap shows one X, corners only at tips
  decode      the gate states + tips visible in the drawing decode back to the exact label
  render      stored images re-render byte-identically from (label, style)
  injective   different labels never give the same picture (fixed style render)
and reports diversity (unique labels, isomorphism classes) and distributions.
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
from .spec import CROSSING_LOOP, HORN, PARALLEL_LOOP


def load(dataset: str | Path) -> list[dict]:
    with open(Path(dataset) / "labels.jsonl") as f:
        return [json.loads(line) for line in f]


def _style(d: dict) -> Style:
    return Style(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


# ------------------------------------------------------------------ main evaluation
def evaluate(dataset: str | Path, rerender_n: int = 200, iso_n: int = 3000, log=print) -> dict:
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
                dist["brush_dots"]["HORN"] += 1
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
    return "\n".join(L) + "\n"
