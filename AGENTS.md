# AGENTS.md — Kolam Labs

Source of truth for semantics: `docs/hypergraph_kolams.md`. Read this first, follow it exactly.

Current focus: training model B (grid) and shipping it in the browser — the plan is `todo.md`.
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
kolam_gen (Rust): generators → gaps + horns → checks → encoding shards
kolam_feed (Python): encoding → hypergraph → strokes → image, rendered on the fly while training
```

The ML task is `image → ordered typed hypergraph`. Every synthetic image must have
an exact hypergraph label. The lattice sequence is a compiled intermediate, not
the primary label.
The training target is the label's kolam encoding (`encoding` in the kolam_gen shards, spec §12): a
plain grid of one digit per gap (0 wall, 1 cross, 2 tube, 3–8 horns), rows of `cols` and `cols+1`
digits alternating; it converts to and from the hypergraph exactly (`kolam_feed/encoding.py`).

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
├── docs/hypergraph_kolams.md      # semantic spec — read this first (§8 brush compiler, §9 DSL + label JSON, §10 drawing primitives + horns, §11 generator systems, §12 kolam encoding)
├── docs/references.md
├── docs/results/                  # gallery + evaluation report of the first (Python) generator
├── brushes.json                   # shared brush spec: gate states + horn directions and shapes
├── kolam_gen/                     # Rust: data generation (spec §11) → encoding shards
│   └── src/                       # model.rs (labels, rules, symmetry, encoding), geometry.rs (trace, drawn, check),
│                                  #   random.rs + systems.rs (the four systems), main.rs (generate / check / big)
├── kolam_feed/                    # Python: reading, rendering, checking, feeding the model
│   ├── spec.py / hypergraph.py    # lattice helpers; labels, validation, canonical form (from_gates)
│   ├── encoding.py                # kolam encoding <-> label; grid / cells / tokens; constrained decoding (spec §12)
│   ├── compiler.py                # hypergraph -> gates -> closed lattice strokes; observe / decode
│   ├── geometry.py / render.py    # closed chained Bezier, drawn(), check_geometry; PIL renderer + random styles
│   ├── data.py                    # KolamDataset (renders on the fly), NotKolamDataset, collate
│   ├── model.py / train.py        # models A (seq) and B (grid) + is-a-kolam head; train, eval, export
│   ├── watch.py / watch.html      # live training page
│   └── verify.py                  # independent re-check of kolam_gen output
├── scripts/gpu/                   # launch.sh + jobs.sh: overnight training on the GPU laptop (todo.md)
├── kolam_studio/                  # Vite + TS, no backend: index.html = photo reader (main page, model in the browser);
│                                  #   studio.html = brush drawing UI: draw → formula → kolam;
│                                  #   encodings.html: paste kolam encodings → one chalkboard each
│   └── src/kolam/                 # TS port of model + DSL + compiler + drawn() + encoding; reads ../brushes.json
├── refs/                          # reference sheets: primitives, mixed options, example kolams
├── tests/                         # pytest; fixtures/kolams.jsonl = 180 reference kolams from the first generator
└── archive/                       # ARCHAIC — old engine, UI, slate-chalk design
    ├── kolam_engine/              # do not touch, do not import, do not revive
    ├── kolam_engine_demo.ipynb    # original crossing-loop research (`build_crossing_loop`)
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
- `kolam_feed/compiler.py` (and its Rust twin `kolam_gen/src/geometry.rs`) is the relation compiler
  (gate model, hypergraph doc §8). One `CROSSING_LOOP` along a row must keep reproducing the
  notebook's `build_crossing_loop(n)` (`tests/test_kolam_feed.py`).
- Notebook prototype: `build_crossing_loop(n_dots)` — n dots → n-1 crossings at
  `(1,2k+2)`, each visited 2× from perpendicular diagonals (forward ↘/↗, backward ↙/↖),
  closed loop. This is the reference `CROSSING_LOOP` compiler. JSON sample in last cell
  uses `{"type":"CROSSING_LOOP",...}`.

## Commands

```bash
pip install -r requirements.txt                       # numpy, pillow, pytest
python -m pytest tests -q                             # must stay green
cd kolam_gen && cargo build --release                # the generator (all four systems)
kolam_gen/target/release/kolam_gen generate -n 1000000 -o data/enc_1m   # shards, resumable, no images; data/ is gitignored
kolam_gen/target/release/kolam_gen big --system substitution --size 64 -o big.jsonl   # one kolam, any size
python -m kolam_feed verify  data/enc_1m --n 5000    # independent Python re-check; exit 1 on failure
python -m kolam_feed preview data/enc_1m -o sheet.png   # rendered training images
cd kolam_studio && npm install && npm run dev        # brush drawing UI on http://localhost:5173
python -m kolam_feed train data/enc_400k_20 --model grid --size 384 --aug real --not-kolam data/not_kolam --real-kolams data/real_kolams_line --out runs/b
python -m kolam_feed.watch runs/b                    # live training page: http://localhost:8800
python -m kolam_feed export runs/pilot_seq/last.pt  # ONNX into kolam_studio/public/model: the reader (index.html) reads photos in the browser
cd kolam_studio && npm run build                    # dist/ = static site (upload as is, e.g. cPanel; .htaccess sets .wasm/.onnx types)
```
(The old `kolam_engine` npm app is archived; `kolam_studio/` is the only npm app.)

## Conventions for edits

- Docs: only `docs/hypergraph_kolams.md` is active. Don't resurrect deleted doc names
  (`NORMAL_LOOP`, `HORN_LEFT`, `HORN_RIGHT`, `dot_port_grammar_summary.md`, `ref_papers/`).
- Code: new generator / training code lives outside `archive/`. Never import from `archive/`.
- Two implementations of the kolam rules, kept in step: `kolam_gen/` (Rust: generates and checks every
  kolam) and `kolam_feed/` (Python: reads, renders, re-checks independently). Change a rule (spec, label
  rules, compiler, `drawn()`, geometry check) in both; `tests/test_kolam_gen.py` and
  `tests/test_kolam_feed.py` hold both to the same answers on `tests/fixtures/kolams.jsonl`, and
  `kolam_feed verify` checks generated encodings. Generators live only in Rust.
- `kolam_studio/src/kolam/` mirrors the model too (hypergraph, DSL, compiler, encoding): when the brush
  model changes, change it as well. The same goes for drawing: `kolam_studio/src/kolam/drawn.ts` must
  draw what `geometry.drawn()` draws (spec §10).
- Free drawing vs dataset labels: `strict=True` (default) enforces every label rule; the studio
  uses `strict=False`, which only refuses what cannot be drawn (`hypergraph.problems`).
- When adding generator code (in `kolam_gen/`): structure-first per hypergraph doc §5
  (grid → relations → compose → validate → compile → check). Read gate states and horn shapes
  from `brushes.json` — don't hardcode sequences for dataset output.
