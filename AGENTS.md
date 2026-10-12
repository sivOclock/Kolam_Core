# AGENTS.md — Kolam Labs

Source of truth for semantics: `docs/hypergraph_kolams.md`. Read this first, follow it exactly.

Current focus: data-generation pipeline + training the model (`todo.md`).
The old kolam engine / UI lives in `archive/` and is out of scope: do not revive its code
and never import from it. Its chalkboard look (board texture, wooden frame, chalk brush,
Schoolbell labels, chalk palette) is reused on purpose by `kolam_studio/`, the brush
drawing UI — keep new UI work consistent with that.

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
The training target is the label's gap code (`sparse` in `labels.jsonl`, spec §12): the non-zero
gaps as `(lattice row, lattice col, digit)`, digits 1 cross, 2 tube, 3–8 horns; it converts to and
from the hypergraph exactly (`kolam_press/gapcode.py`).

## The 3 forms (only these — see docs/hypergraph_kolams.md §2)

1. `HORN` — exactly 1 vertex plus a direction `N`/`E`/`S`/`W` (where its tip points). Variants:
   `PLAIN_HORN` (straight), `LEFT_HORN`, `RIGHT_HORN` (curve a little left/right). All are type
   `HORN` (JSON: `{"type":"HORN","variant":"LEFT_HORN","vertices":[[r,c]],"dir":"N"}`; DSL
   `hl([r,c],N)`). The old two-dot horn `HORN(A → B)` is retired (= `p(A,B)` + a horn on B).
2. `CROSSING_LOOP` — multiple vertices in cyclic order.
3. `PARALLEL_LOOP` — multiple vertices in structural order.

Rules:
- Never use `NORMAL_LOOP` (old name for `CROSSING_LOOP`). Never use `HORN_LEFT` / `HORN_RIGHT`.
- Vertices are anchor dots only (`V = {A, B, C, ...}`). Midpoints / lattice coords are
  not part of the learned representation.
- Ordering matters. A horn's direction matters (`HORN(A, N)` ≠ `HORN(A, S)`). Cyclic order matters
  for `CROSSING_LOOP`.

## Repo layout (current)

```text
Kolam_labs/
├── AGENTS.md
├── README.md / todo.md
├── docs/hypergraph_kolams.md      # semantic spec — read this first (§8 brush compiler, §9 DSL + label JSON, §10 drawing primitives + horns, §11 generator systems, §12 gap codes)
├── docs/references.md
├── docs/results/                  # gallery + evaluation report of the current generator
├── kolam_press/                   # data generator (Python package)
│   ├── brushes.json               # shared brush spec: gate states + horn directions and shapes
│   ├── spec.py / hypergraph.py    # lattice helpers; labels, validation, isomorphism
│   ├── dsl.py                     # brush DSL: c(...), p(...), h/hl/hr([r,c],N)
│   ├── compiler.py                # hypergraph -> gates -> closed lattice strokes; decode
│   ├── geometry.py / render.py    # closed chained Bezier, geometric checks; PIL renderer
│   ├── generator.py / evaluate.py # random symmetric kolams; dataset correctness + diversity checks
│   ├── gapcode.py                 # gap codes: label <-> grid / sparse / tokens, constrained decoding (spec §12)
│   └── systems.py                 # substitution (2D L-system), switching, WFC generators (spec §11)
├── kolam_studio/                  # brush drawing UI (Vite + TS, no backend): draw → formula → kolam
│   └── src/kolam/                 # TS port of model + compiler + drawn(); reads kolam_press/brushes.json
├── examples/*.kolam               # brush programs (sikku_diamond = a photographed kolam, free drawing)
├── refs/                          # reference sheets: primitives, mixed options, example kolams
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
- Gate codes (spec §10.1): every gap is `0` wall (`SPLIT`), `1` cross (`CROSS`, brush `c`) or `2`
  parallel (`JOIN`, brush `p`); border gaps are `0`. A kolam = two grids of these digits.
- 4-gate model only: a dot has 4 gates (above, right, below, left) and links run along rows and
  columns. Diagonal links were tried and removed — do not add them back. Diagonal-looking kolams,
  if needed, come from a diamond dot layout (the grid turned 45°).
- Drawing primitives (spec §10): 21 shapes around one dot (6 crossing, 5 parallel, 10 mixed).
  Crossing look is final. Parallel is a **capsule**: straight walls 0.75 from the dots and round
  ends/corners of the same radius, as thick and round as a crossing loop, no pinch at gaps
  (`geometry.capsule_points`); 4 dots whose 4 gaps are all parallel form a filled box (inner loop
  not drawn, `geometry.is_box_hole`). Mixed (crossing + parallel at one dot): the dot is drawn
  like a tube end, so the wall runs straight into the ✕, leaning in to 0.5 right at the ✕ so it
  has room (spec §10.5). `geometry.drawn()` is what render.py and check_geometry use; the studio
  draws the same through `kolam_studio/src/kolam/drawn.ts` (a port; keep them in step). One rule behind it all: the
  line stays 0.75 from its dots and runs, and leaves only to cross at a crossing gap.
  Drawing changes never change labels, programs or gate codes.
- Horns (spec §10.6): set no gate; `geometry.drawn()` grows each one out of the line in front of its
  dot with a weight that fades smoothly (no lump). Pointing at an empty spot or off the grid →
  crescent, tip on that spot (reach 2.0). Dataset labels allow only crescents (spec §8.4 rule 7);
  pointing at a used dot gives a short stub (1.05), allowed in free drawing only. 2–4 horns may meet
  in one empty spot (tips touch; same variant; not two off one parallel inner corner, rule 8). Shapes live in `brushes.json` `HORN.grow`. Unused dots are not drawn (so the generator
  always crops). `observe` gives each horn as (dot, tip point); `decode` reads them back.
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
cd kolam_studio && npm install && npm run dev        # brush drawing UI on http://localhost:5173
```
(The old `kolam_engine` npm app is archived; `kolam_studio/` is the only npm app.)

## Conventions for edits

- Docs: only `docs/hypergraph_kolams.md` is active. Don't resurrect deleted doc names
  (`NORMAL_LOOP`, `HORN_LEFT`, `HORN_RIGHT`, `dot_port_grammar_summary.md`, `ref_papers/`).
- Code: new generator / training code lives outside `archive/`. Never import from `archive/`.
- `kolam_studio/src/kolam/` mirrors `kolam_press` (hypergraph, DSL, compiler). Python is the
  source of truth: when the brush model changes, change both and check that both compile the
  same programs to the same canonical text and traced strokes. The same goes for drawing:
  `kolam_studio/src/kolam/drawn.ts` must draw what `geometry.drawn()` draws (spec §10).
- Free drawing vs dataset labels: `strict=True` (default) enforces every label rule; the studio
  uses `strict=False`, which only refuses what cannot be drawn (`hypergraph.problems`).
- When adding generator code: structure-first per hypergraph doc §5
  (grid → relations → compose → validate → compile → render). Put `CROSSING_LOOP` /
  `HORN` / `PARALLEL_LOOP` compilers where both TS engine and Python exporter can
  share the spec — don't hardcode sequences for dataset output.
