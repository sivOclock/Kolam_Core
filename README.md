# Kolam Labs

**Goal:** build a dataset of kolam images, then train a model that reads a kolam image and
outputs its structure as an **ordered typed hypergraph** (`docs/hypergraph_kolams.md`).

```text
brush program  →  ordered hypergraph  →  gate compiler  →  lattice strokes  →  renderer  →  image
c(...) p(...) h(...)    (exact label)        (§8)          [li,lj] … ---       chained Bézier
```

Every image is generated from its label, so every label is exact.

## The three brushes

You draw a kolam the way you would with three brushes, through the dots of a grid
(`[row, col]`, 0-indexed):

| brush | call | what it draws |
|---|---|---|
| `CROSSING_LOOP` | `c([0,0],[0,4])` | a chain of dots; the line **crosses** (✕) in every gap along the chain |
| `PARALLEL_LOOP` | `p([2,0],[2,2],[4,2])` | a chain of dots; the strands run **parallel** around them, with no crossings |
| `HORN` | `h([1,1],N)` · `hl(…,N)` · `hr(…,N)` | a sharp tip grown out of dot `[1,1]`'s line, pointing N/E/S/W: straight / curving a little left / right (aliases `h0` `h1` `h2`). It points at an empty spot or off the grid and is drawn as a crescent; up to 4 horns may meet in one spot |

Waypoints on the same row or column are filled in. Repeat the first dot to close a ring:
`c([1,1],[1,3],[3,3],[3,1],[1,1])`.

```bash
pip install -r requirements.txt
python -m kolam_press draw "c([0,0],[0,3]); p([1,0],[1,3]); hl([1,3],E)" -o kolam.png
python -m kolam_press draw -f my.kolam -o my.png --view panel --seq   # a program from a file
```

![brush program | kolam coloured by brush | training image](docs/results/examples.jpg)

How it works, from hypergraph doc §8:
- Each brush only sets the state of the gaps between neighbouring dots: `CROSS`, `JOIN`
  or `SPLIT`.
- The curve is traced from gate to gate (the Gerdes / Nagata mirror-curve model).
- A horn sets no gap: the drawing grows the line in front of its dot into a sharp tip (§10.6).

A crossing brush along a row reproduces the notebook's `build_crossing_loop(n)` exactly.

## Generating data

```bash
python -m kolam_press generate -n 3000 -o data/kolams_v4 --preview 16
python -m kolam_press evaluate data/kolams_v4       # writes report.md, exits 1 on any failure
python -m kolam_press gallery  data/kolams_v4 -o gallery.png
```

The generator works structure first:
1. Pick a grid (3–8 dots a side) and a symmetry (none, mirror, 180° turn, D2, C4 or D4).
2. Lay random brush strokes with their symmetric copies, keeping only valid ones.
3. In about 40% of kolams (tree mode), grow the brush lines as a tree, which gives
   single-stroke kolams.
4. Cover leftover dots by extending chains or adding horns (at most one horn per 5 dots). Horns
   only point at empty spots, so each is a crescent; several may meet in one spot.
5. Compile, check the geometry, render with a random look (chalk, ink or colour; line
   width, dots, blur, noise, hand jitter), and re-draw any duplicate.

Each line of `labels.jsonl` holds:
- the image path;
- `program`, the canonical brush program;
- `hypergraph`, the label in the spec JSON format;
- `encoding`, the same label as a kolam encoding (the training target);
- `lattice_sequence`, the compiled strokes;
- stats and the render style.

![36 generated training images](docs/results/gallery.jpg)

### Four generator systems

`generate` is the first of four ways to make kolams. All four only choose what each gap between two
dots does (wall, cross or tube), then add crescent horns, and every sample passes the same checks.

![the four systems, step by step](refs/systems_explained.png)

- **Random** (the painter): lays brush strokes one at a time with their symmetric copies.
- **Substitution** (the fractal grower, a 2D L-system): every dot becomes a block of dots by a rule
  book made up for each kolam, generation after generation, so patterns sit inside patterns.
- **Switching** (the one-line maker): starts with every gap a crossing and flips gaps only when two
  lines merge into one, the way real sikku kolams are drawn as one unbroken line.
- **WFC** (the student): cuts a real kolam into dot pieces and fills a new grid with them like a jigsaw.

`python -m kolam_press systems -n 100 -o data/systems` writes 100 of each with a contact sheet.
Details and first results: spec §11.

### The training target: the kolam encoding

Every kolam is one digit on each gap between two dots: `0` wall, `1` cross, `2` tube, `3`–`8` a horn
(plain / left / right, facing right-or-down or left-or-up). The label for training (`encoding`) is just
those digits as a grid, row by row, nothing else:

```text
00000
007500     ← two horns on the top dot
00100
001100
00100
011110     ← the long middle line
...
```

A model that predicts every gap at once outputs exactly this, at any grid size; a sequence model
writes it with 11 tokens (`0`–`8`, row end, end) under constrained decoding. It converts to and from
the hypergraph exactly.

![a kolam as gap digits](refs/grid_representation.png)

Why this form, how lines are read back, and tokenizer pitfalls: spec §12 and `refs/grid_why.png`.

### At scale: `kolam_gen` (Rust) → encodings → `kolam_feed` (Python) → the model

For training, only encodings are stored; images are rendered while training, a new look each epoch.

```bash
cd kolam_gen && cargo build --release && cd ..
kolam_gen/target/release/kolam_gen generate -n 1000000 -o data/enc_1m     # ~2,100 kolams/s on 8 cores
python -m kolam_feed verify  data/enc_1m --n 5000                          # Python re-check, exit 1 on failure
python -m kolam_feed preview data/enc_1m -o sheet.png
```

- `kolam_gen` mixes all four systems (default `--mix random=0.55,substitution=0.15,switching=0.15,wfc=0.15`),
  checks every kolam with the same label rules and geometry check as Python, drops duplicates (drawn
  again), and writes `shard_00000.jsonl`, … (50k kolams each, about 150 bytes per kolam). A stopped run
  resumes: finished shards are kept. Every kolam is reproducible from its `seed` (seed, index, attempt).
- It is a port; Python stays the source of truth. On 700 kolams (83 failing), Rust and Python give the
  same geometry verdict and the same stats on every one (`tests/test_kolam_gen.py` keeps checking).
- `kolam_feed.KolamDataset(dir, size=448, transform=None)` is a map-style dataset: item `i` is
  `{"image", "encoding", "text", "tokens", "system"}`, rendered by `kolam_press` in a random style from
  (seed, epoch, i); `set_epoch(e)` before each epoch for new looks. Batch with
  `DataLoader(ds, collate_fn=kolam_feed.collate)` (lists, PIL images kept for a VLM processor), or pass a
  `transform` that makes tensors. About 17 images/s per process at 448 px.

## Is the data good? (3000-sample run, `docs/results/report.md`)

This run predates the crescent-only horn rule (spec §8.4 rule 7): about half its horns are short
stubs. It also predates the kolam encoding: its labels have no `encoding` field. Regenerate before training.

Every sample is re-checked from its label alone, independently of the generator:

| check (all 3000 samples) | failures |
|---|---|
| label valid and canonical; its brush program re-parses to the same label | 0 |
| recompiling the label reproduces the stored lattice sequence | 0 |
| geometry (see below) | 0 |
| the gate states + horns visible in the drawing decode back to the exact label | 0 |
| a sample of 200 images re-renders byte-identically from (label, style) | 0 |
| different labels that give the same picture | 0 |
| tree-shaped brush lines give one stroke per tree (§8.6; 2208 such kolams) | 0 |

The geometry check requires that:
- curves stay ≥ 0.46 from every drawn dot;
- strands stay ≥ 0.2 apart except at their crossings (dots are 2 apart);
- every crossing gap shows exactly one ✕;
- the only corners are horn tips.

**Diversity**
- 3000 / 3000 unique labels; 2887 isomorphism classes (§6).
- Grids: all 36 shapes from 3×3 to 8×8, plus cropped smaller ones (always cropped: unused dots
  are not drawn).
- 7 symmetry types; 28% single-stroke; 60% use all three brushes.
- 1000 closed rings.
- Horns: 5131 straight, 4509 left, 4653 right.
- 2120 dark-background images, 880 light.

**Labels are readable from pixels** (measured on the earlier run, before the one-dot horns and
capsule parallels). A one-off tiny MLP probe (since removed from the code) reading a 9×9 grid of ink samples around each gap (trained on
1050 images, tested on 450; grid position given) gets:
- 99.87% of gaps (crossing / parallel / split / tip) right;
- 99.995% of horn-tip centres right;
- the whole hypergraph exact on 92.9% of kolams.

Its misses are spread evenly across resolutions and styles. They are small
rounded-versus-pointed differences that a real CNN should resolve. Full report:
`docs/results/report.md`; more previews: `docs/results/dataset_previews.jpg`.

## Layout

```text
kolam_press/   brushes.json (shared spec) · spec · hypergraph · dsl · compiler · geometry · render · generator · systems · encoding · evaluate
kolam_gen/     Rust port of the generators + checks: encodings at scale, in shards
kolam_feed/    Python: encoding shards → images rendered on the fly → the model; verify
tests/         pytest: kolam encoding round trips, constrained decoding, Rust ↔ Python parity, kolam_feed
docs/          hypergraph_kolams.md (spec, §8 = brushes) · references.md · results/
archive/       old TypeScript engine/UI — archived, not used
```

## Next

- Train the image → hypergraph model on this data, with the kolam encoding (`labels.jsonl:encoding`,
  spec §12) as the target.
- Run a VLM baseline against the same evaluator.
