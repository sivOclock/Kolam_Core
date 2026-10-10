# AGENTS.md — Kolam Labs

Source of truth for semantics: `docs/hypergraph_kolams.md`. Read this first, follow it exactly.

Current focus: data-generation pipeline + training the model (`todo.md`).
The kolam engine / UI / slate-chalk design is archaic and out of scope — it lives
in `archive/` (plus stale `README.md` sections describing it). Do not revive it,
do not import from it, do not style new code to match it.

## What this project is

Build a dataset of kolam patterns and train an image-to-sequence model that reads
a kolam image and outputs its structural encoding.

Pipeline (hypergraph doc §4–5, §7):

```text
Anchor grid → ordered hypergraph → relation compiler → lattice sequence → renderer → image
```

The ML task is `image → ordered typed hypergraph`. Every synthetic image must have
an exact hypergraph label. The lattice sequence is a compiled intermediate, not
the primary label.

## The 3 forms (only these — see docs/hypergraph_kolams.md §2)

1. `HORN` — exactly 2 vertices, ordered. Variants: `PLAIN_HORN`, `LEFT_HORN`, `RIGHT_HORN`.
   All are type `HORN` (JSON: `{"type":"HORN","variant":"LEFT_HORN","vertices":[...]}`).
2. `CROSSING_LOOP` — multiple vertices in cyclic order.
3. `PARALLEL_LOOP` — multiple vertices in structural order.

Rules:
- Never use `NORMAL_LOOP` (old name for `CROSSING_LOOP`). Never use `HORN_LEFT` / `HORN_RIGHT`.
- Vertices are anchor dots only (`V = {A, B, C, ...}`). Midpoints / lattice coords are
  not part of the learned representation.
- Ordering matters. `HORN(A → B)` ≠ `HORN(B → A)`. Cyclic order matters for `CROSSING_LOOP`.

## Repo layout (current)

```text
Kolam_labs/
├── AGENTS.md
├── README.md / todo.md
├── docs/hypergraph_kolams.md      # semantic spec — read this first (§8 = brush compiler, §9 = DSL + label JSON)
├── docs/references.md
├── docs/results/                  # gallery + evaluation report of the current generator
├── kolam_press/                   # data generator (Python package)
│   ├── brushes.json               # shared brush spec: gate states + horn tip templates
│   ├── spec.py / hypergraph.py    # lattice helpers; labels, validation, isomorphism
│   ├── dsl.py                     # brush DSL: c(...), p(...), h/hl/hr(...)
│   ├── compiler.py                # hypergraph -> gates -> closed lattice strokes; decode
│   ├── geometry.py / render.py    # closed chained Bezier, geometric checks; PIL renderer
│   └── generator.py / evaluate.py # random symmetric kolams; dataset correctness + diversity checks
├── examples/*.kolam               # brush programs
├── tests/                         # pytest
└── archive/                       # ARCHAIC — old engine, UI, slate-chalk design
    ├── kolam_engine/              # do not touch, do not import, do not revive
    ├── kolam_engine_demo.ipynb    # original crossing-loop research (reference copied into tests/)
    └── pulli kolam encoder demo.html  # legacy — do not touch
```

## Core implementation facts (from code)

- Lattice: `m × n` anchors → `(2m+1) × (2n+1)` lattice. `isAnchor(li,lj) = li%2==1 && lj%2==1`.
  Anchor `(row,col)` → lattice `(2*row+1, 2*col+1)`.
- Path visits midpoints, never anchors.
- Sequence format (compiled intermediate, not the label): ordered `[li,lj]`, strokes
  separated by `---`. `[1,1] → [1,2] → [2,2] --- [3,1] → [3,2]`.
  Token for training: `li * (2n+1) + lj`.
- Spline math (reuse for the renderer, not the UI): chained quadratic Bézier —
  `quadraticCurveTo(coords[i], mid(coords[i],coords[i+1]))`. Same algorithm in
  notebook `build_chained_bezier`. Keep geometry deterministic.
- Hypergraph-doc isomorphism (§6) is the definition: ordered typed hypergraph
  isomorphism — exact vertex-mapping preserving type/arity/order/variant.
  The old engine's WL certificate was only a practical proxy. Do not conflate the two.
- `kolam_press/compiler.py` is the relation compiler (gate model, hypergraph doc §8). One
  `CROSSING_LOOP` along a row must keep reproducing the notebook's `build_crossing_loop(n)`
  (reference copy in `tests/test_kolam_press.py`).
- Notebook prototype: `build_crossing_loop(n_dots)` — n dots → n-1 crossings at
  `(1,2k+2)`, each visited 2× from perpendicular diagonals (forward ↘/↗, backward ↙/↖),
  closed loop. This is the reference `CROSSING_LOOP` compiler. JSON sample in last cell
  uses `{"type":"CROSSING_LOOP",...}`.

## Commands

```bash
pip install -r requirements.txt                       # numpy, pillow, pytest
python -m pytest tests -q                             # must stay green
python -m kolam_press draw -f examples/showcase.kolam -o out.png --view panel
python -m kolam_press generate -n 3000 -o data/kolams_v1 --preview 16   # data/ is gitignored
python -m kolam_press evaluate data/kolams_v1         # writes report.md; exit 1 on any failure
```
(The old `kolam_engine` npm app is archived — no `npm install` / `dev` / `build` workflow.)

## Conventions for edits

- Docs: only `docs/hypergraph_kolams.md` is active. Don't resurrect deleted doc names
  (`NORMAL_LOOP`, `HORN_LEFT`, `HORN_RIGHT`, `dot_port_grammar_summary.md`, `ref_papers/`).
- Code: new generator / training code lives outside `archive/`. Never import from `archive/`.
- When adding generator code: structure-first per hypergraph doc §5
  (grid → relations → compose → validate → compile → render). Put `CROSSING_LOOP` /
  `HORN` / `PARALLEL_LOOP` compilers where both TS engine and Python exporter can
  share the spec — don't hardcode sequences for dataset output.
