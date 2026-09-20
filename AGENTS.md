# AGENTS.md — Kolam Labs

Source of truth for semantics: `docs/hypergraph_kolams.md`.
Source of truth for implementation: `kolam_engine/src/` (TypeScript + Paper.js + Vite).

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

## Repo layout

```text
Kolam_labs/
├── AGENTS.md
├── README.md / todo.md / references.md
├── docs/hypergraph_kolams.md      # semantic spec — read this first
├── kolam_engine/                  # ACTIVE app (Vite + TS + paper.js)
│   ├── index.html                 # encoder page → src/main.ts
│   ├── challenge.html             # challenge page → src/challenge.ts
│   └── src/
│       ├── main.ts                # encoder wiring
│       ├── challenge.ts           # 3 levels + live WL validation + graph viz
│       ├── modules/schema.ts      # GridConfig, LatticePoint, AnchorDot, isAnchor()
│       ├── modules/canvas.ts      # lattice math, drawGrid, auto-fit spacing
│       ├── modules/draw.ts        # snap-to-midpoint tool, onLive/onCommit
│       ├── modules/decoder.ts     # decodeLive (static) + decodeLoop (animated)
│       ├── modules/renderer.ts    # chalkStroke (seeded), board texture
│       ├── modules/isomorphism.ts # WL certificate, areIsomorphic()
│       ├── modules/seq-parser.ts  # textarea text → [number,number][][]
│       ├── modules/seq-input.ts   # keystroke helper (currently UNUSED by main.ts)
│       ├── modules/spline.ts      # legacy anchor-to-anchor demo loop
│       ├── modules/motif.ts       # STUB (1 line) — do not import
│       └── styles/theme.ts        # chalk color system
├── kolam_press/kolam_engine_demo.ipynb  # Python mirror; build_crossing_loop(n) prototype
├── kolam_engine_v0/               # DEAD — do not touch, do not import
└── pulli kolam encoder demo.html  # LEGACY — do not touch
```

## Core implementation facts (from code)

- Lattice: `m × n` anchors → `(2m+1) × (2n+1)` lattice. `isAnchor(li,lj) = li%2==1 && lj%2==1`
  (`kolam_engine/src/modules/schema.ts`). Anchor `(row,col)` → lattice `(2*row+1, 2*col+1)`.
- Path visits midpoints, never anchors. Draw tool snaps to nearest non-anchor within
  `spacing/2` (`draw.ts:nearest`).
- Sequence format: ordered `[li,lj]`, strokes separated by `---`.
  `[1,1] → [1,2] → [2,2] --- [3,1] → [3,2]`. Parser drops strokes with < 2 points
  (`seq-parser.ts`). Token for training: `li * (2n+1) + lj`.
- Spline: chained quadratic Bézier — `quadraticCurveTo(coords[i], mid(coords[i],coords[i+1]))`
  (`decoder.ts:buildSplinePts`, `renderer.ts:buildPath`). Same algorithm in notebook
  `build_chained_bezier`.
- Chalk: point-sampled, seeded LCG so same geometry = same grain
  (`renderer.ts:chalkStroke`, `seededRand`). Seed from first point coords.
- Isomorphism (engine-level, NOT the hypergraph definition): undirected graph from
  sequence → 3-round WL, `certificate()` compare (`isomorphism.ts`). Invariant to
  position / direction / stroke order. Challenge page uses `areIsomorphic()` live.
- Hypergraph-doc isomorphism (§6) is ordered typed hypergraph isomorphism — exact
  vertex-mapping preserving type/arity/order/variant. WL is a practical proxy in the
  engine, not the definition. Do not conflate the two.
- `kolam_press` prototype: `build_crossing_loop(n_dots)` — n dots → n-1 crossings at
  `(1,2k+2)`, each visited 2× from perpendicular diagonals (forward ↘/↗, backward ↙/↖),
  closed loop. This is the reference `CROSSING_LOOP` compiler. JSON sample in last cell
  uses `{"type":"CROSSING_LOOP",...}`.

## Commands

```bash
cd kolam_engine
npm install
npm run dev      # encoder at /, challenge at /challenge.html
npm run build    # tsc + vite build (base './')
```

No tests, no linter, no Python package yet. Notebook runs standalone (numpy + matplotlib).

## Conventions for edits

- Docs: only `docs/hypergraph_kolams.md` is active. Don't resurrect deleted doc names
  (`NORMAL_LOOP`, `HORN_LEFT`, `HORN_RIGHT`, `dot_port_grammar_summary.md`, `ref_papers/`).
- Code: Paper.js scopes are `any` (no TS interfaces available) — follow existing pattern.
  Canvas sizing: never set `canvas.width/height` directly for Paper.js canvases; use
  `scope.view.viewSize` (see `canvas.ts:sizeCanvas`). Graph-viz canvases (2D ctx) are
  the exception — they set width/height with DPR.
- `draw.ts` owns stroke state (`finished`, `currentPts`). `decoder.ts` owns animation
  frame (`_animId`, `stopDecode()`). `seq-parser.ts` is the only textarea parser.
- Touch handling in `draw.ts`/`decoder.ts` is deliberate (tap vs drag, no textarea
  focus-steal on coarse pointers). Don't "simplify" it away.
- When adding generator code: structure-first per hypergraph doc §5
  (grid → relations → compose → validate → compile → render). Put `CROSSING_LOOP` /
  `HORN` / `PARALLEL_LOOP` compilers where both TS engine and Python exporter can
  share the spec — don't hardcode sequences in `challenge.ts` style for dataset output.
