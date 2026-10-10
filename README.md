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
| `HORN` | `h(A,B)` · `hl(A,B)` · `hr(A,B)` | grows from A around B into a sharp tip: plain / curling left / curling right (aliases `h0` `h1` `h2`) |

Waypoints on the same row or column are filled in. Repeat the first dot to close a ring:
`c([1,1],[1,3],[3,3],[3,1],[1,1])`.

```bash
pip install -r requirements.txt
python -m kolam_press draw "c([0,0],[0,3]); p([1,0],[1,3]); hl([2,1],[2,2])" -o kolam.png
python -m kolam_press draw -f examples/flower.kolam -o flower.png --view panel --seq
```

![brush program | kolam coloured by brush | training image](docs/results/examples.jpg)

How it works, from hypergraph doc §8:
- Each brush only sets the state of the gaps between neighbouring dots: `CROSS`, `JOIN`
  or `SPLIT`.
- The curve is traced from gate to gate (the Gerdes / Nagata mirror-curve model).
- A horn swaps one rounded turn for a sharp tip template.

A crossing brush along a row reproduces the notebook's `build_crossing_loop(n)` exactly.

## Generating data

```bash
python -m kolam_press generate -n 3000 -o data/kolams_v1 --preview 16
python -m kolam_press evaluate data/kolams_v1       # writes report.md, exits 1 on any failure
python -m kolam_press gallery  data/kolams_v1 -o gallery.png
```

The generator works structure first:
1. Pick a grid (3–8 dots a side) and a symmetry (none, mirror, 180° turn, D2, C4 or D4).
2. Lay random brush strokes with their symmetric copies, keeping only valid ones.
3. In about 40% of kolams (tree mode), grow the brush lines as a tree, which gives
   single-stroke kolams.
4. Cover leftover dots by extending chains or adding horns.
5. Compile, check the geometry, render with a random look (chalk, ink or colour; line
   width, dots, blur, noise, hand jitter), and re-draw any duplicate.

Each line of `labels.jsonl` holds:
- the image path;
- `program`, the canonical brush program;
- `hypergraph`, the label in the spec JSON format;
- `lattice_sequence`, the compiled strokes;
- stats and the render style.

![36 generated training images](docs/results/gallery.jpg)

## Is the data good? (3000-sample run, `docs/results/report.md`)

Every sample is re-checked from its label alone, independently of the generator:

| check (all 3000 samples) | failures |
|---|---|
| label valid and canonical; its brush program re-parses to the same label | 0 |
| recompiling the label reproduces the stored lattice sequence | 0 |
| geometry (see below) | 0 |
| the gate states + horn tips visible in the drawing decode back to the exact label | 0 |
| a sample of 200 images re-renders byte-identically from (label, style) | 0 |
| different labels that give the same picture | 0 |
| tree-shaped brush lines give one stroke per tree (§8.6; 2350 such kolams) | 0 |

The geometry check requires that:
- curves stay ≥ 0.44 from every dot;
- strands stay ≥ 0.23 apart except at their crossings (dots are 2 apart);
- every crossing gap shows exactly one ✕;
- the only corners are horn tips.

**Diversity**
- 3000 / 3000 unique labels; 2841 isomorphism classes (§6).
- Grids: all 36 shapes from 3×3 to 8×8, plus a few cropped smaller ones.
- 7 symmetry types; 29% single-stroke; 55% use all three brushes.
- 800 closed rings.
- Horns: 6132 plain, 5023 left, 5214 right.
- 2134 dark-background images, 866 light.

**Labels are readable from pixels.** A one-off tiny MLP probe (since removed from the code) reading a 9×9 grid of ink samples around each gap (trained on
1050 images, tested on 450; grid position given) gets:
- 99.87% of gaps (crossing / parallel / split / tip) right;
- 99.995% of horn-tip centres right;
- the whole hypergraph exact on 92.9% of kolams.

Its misses are spread evenly across resolutions and styles. They are small
rounded-versus-pointed differences that a real CNN should resolve. Full report:
`docs/results/report.md`; more previews: `docs/results/dataset_previews.jpg`.

## Layout

```text
kolam_press/   brushes.json (shared spec) · spec · hypergraph · dsl · compiler · geometry · render · generator · evaluate
examples/      brush programs (showcase, flower, one_stroke)
tests/         pytest: notebook crossing-loop equivalence, mirror ↔ left/right on pixels, DSL, decode, determinism
docs/          hypergraph_kolams.md (spec, §8 = brushes) · references.md · results/
archive/       old TypeScript engine/UI — archived, not used
```

## Next

- Train the image → hypergraph model on this data. A good first target is the brush
  program text (`labels.jsonl:program`).
- Run a VLM baseline against the same evaluator.
