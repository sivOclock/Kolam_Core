# Ordered Hypergraph Kolam Data Model

A Kolam is represented as an **ordered, typed hypergraph**:

$$
H = (V, E)
$$

---

## 1. Vertices

Vertices are the **anchor dots only**.

```text
V = {A, B, C, D, ...}
```

> Midpoints and lattice coordinates are **not part of the learned representation**.

---

## 2. Typed Ordered Relations

There are exactly **3 relation types**: `HORN`, `CROSSING_LOOP`, and `PARALLEL_LOOP`.

Each relation connects one or more anchor vertices and has a specific type:

```text
HORN(A, N)

CROSSING_LOOP([A, B, C, D])

PARALLEL_LOOP([A, B, C, D])
```

### Horn Relations

- Form **1 of 3**: `HORN`.
- A horn involves exactly **1 vertex**: the dot it grows from, plus the **direction** its tip
  points (`N`, `E`, `S`, `W` = up, right, down, left on screen).
- There are three horn variants — `PLAIN_HORN`, `LEFT_HORN`, `RIGHT_HORN` — all of type `HORN`.
  All three point the same way; `LEFT_HORN` and `RIGHT_HORN` curve a little to their left or right.

| Variant | Notation | Shape |
|---|---|---|
| `PLAIN_HORN` | `PLAIN_HORN(A, N)` | straight |
| `LEFT_HORN` | `LEFT_HORN(A, N)` | curves to the left of its direction |
| `RIGHT_HORN` | `RIGHT_HORN(A, N)` | curves to the right |

The **direction matters**: `HORN(A, N)` and `HORN(A, S)` are different horns. (The old two-dot
horn `HORN(A → B)` is retired: it drew the same thing as a parallel link A–B plus a one-dot horn
on B.)

---

### Crossing Loop

- Form **2 of 3**: `CROSSING_LOOP`.
- Connects **multiple vertices**.
- Vertices are stored in **cyclic order**.
- The cyclic ordering determines how the loop is constructed.

```text
CROSSING_LOOP([A, B, C, D])
```

---

### Parallel Loop

- Form **3 of 3**: `PARALLEL_LOOP`.
- Vertices are stored in the required **structural order**.
- The ordering determines the parallel-loop construction.

```text
PARALLEL_LOOP([A, B, C, D])
```

---

## 3. Kolam Representation

A complete Kolam is represented as:

```text
H = {
    vertices: [A, B, C, D, ...],

    relations: [
        CROSSING_LOOP([A, B, C, D]),
        LEFT_HORN(A, N),
        RIGHT_HORN(C, E),
        PARALLEL_LOOP([A, C, G, H])
    ]
}
```

The **ordered hypergraph is the semantic representation**.

The lattice sequence is a *compiled* representation — not the primary structural label.

> [!WARNING]
> **This schema is provisional.** The Kolam representation defined here is expected to evolve as new structural insights, optimizations, and design decisions emerge. Treat the current form as a working baseline, not a finalized specification.

---

## 4. Deterministic Compilation

The pipeline is:

```text
Ordered Hypergraph
        ↓
Relation compiler
        ↓
Lattice sequence [li, lj][][]
        ↓
Existing Kolam renderer
        ↓
Kolam image
```

The compiler handles:

- Midpoint generation
- Spline / control points
- Traversal
- Exact lattice coordinates
- Rendering geometry

> The ML model does **not** need to learn these low-level details.

---

## 5. Synthetic Data Generation

Generate **structure first**:

```text
Anchor grid
    ↓
Select valid ordered relations
    ↓
Compose relations
    ↓
Validate the hypergraph
    ↓
Ordered hypergraph = exact ground truth
    ↓
Deterministic compiler
    ↓
Rendered Kolam image
```

Each training sample contains:

```json
{
  "image": "kolam_00001.png",
  "hypergraph": {
    "vertices": ["A", "B", "C", "D", "E"],
    "relations": [
      {
        "type": "CROSSING_LOOP",
        "vertices": ["A", "B", "C", "D"]
      },
      {
        "type": "HORN",
        "variant": "LEFT_HORN",
        "vertices": ["A"],
        "dir": "N"
      }
    ]
  }
}
```

---

## 6. Isomorphism

Structural equivalence is defined through **ordered typed hypergraph isomorphism**.

Two Kolams are equivalent when there exists a mapping between their vertices that preserves:

| Property | Description |
|---|---|
| Vertex correspondence | Bijective vertex mapping |
| Relation type | `CROSSING_LOOP`, `PARALLEL_LOOP`, `HORN` (variants `PLAIN_HORN`, `LEFT_HORN`, `RIGHT_HORN`) |
| Relation arity | Same number of vertices per relation |
| Ordering | Vertex sequence within each relation |
| Cyclic ordering | Where applicable (e.g., `CROSSING_LOOP`) |
| Horn variation | `PLAIN_HORN`, `LEFT_HORN`, or `RIGHT_HORN` |
| Horn direction | `N`, `E`, `S` or `W` |

> WL certificates are **not required** as the definition of equivalence. Exact ordered-hypergraph isomorphism is the structural comparison.

---

## 7. Core Principle

**Generate structure → compile structure → render image.**

```text
Hypergraph → Kolam
```

is **deterministic**.

The eventual learning task becomes:

```text
Kolam image
      ↓
Vision model
      ↓
Ordered typed hypergraph
```

Every synthetic image therefore has an **exact, automatically generated structural label**.

---

## 8. Brush Compilation (gate model) — implemented in `kolam_press/`

This section pins down the *primitive formation* of each of the 3 forms so the
compiler in §4 is exact. The shared, language-neutral spec is
`kolam_press/brushes.json`.

### 8.1 Lattice, gates, centres

- Anchor `(row, col)` → lattice `(2*row+1, 2*col+1)`.
- A **gate** is a lattice point with exactly one odd coordinate: the gap between two
  4-adjacent dots (or between a dot and the border).
- A **centre** (both coordinates even) is the empty space between 4 dots.
- The curve moves diagonally `(±1, ±1)` from gate to gate. It never visits an anchor or a
  centre. (Horn tips are added when drawing, §10.6; they are not on the gate path.)

```text
  ·   g   ·   g   ·        • = anchor dot  (odd, odd)
  g   •   g   •   g        g = gate        (exactly one odd coordinate)
  ·   g   ·   g   ·        · = centre      (even, even)
  g   •   g   •   g
  ·   g   ·   g   ·
```

### 8.2 Gate states

Each gate is in one of three states. Unassigned gates and border gates are `SPLIT`.

| state | at the gap | used by |
|---|---|---|
| `CROSS` | strands pass straight through → an ✕ | `CROSSING_LOOP` |
| `JOIN` | mirror along the dot–dot line → strands run parallel around both dots | `PARALLEL_LOOP` |
| `SPLIT` | wall between the dots → each strand turns back around its own dot | default |

```text
     CROSS             JOIN              SPLIT
   •  ╲╱  •         • ‾‾‾‾‾ •          • ) ( •
      ╱╲              _____
  pass straight     bounce off the     bounce off a wall
  through: an ✕     dot–dot line       between the dots
```

In `brushes.json` each state is a sign flip `(fr, fc)` applied to the incoming
direction `(dr, dc)`, given separately for horizontal gates (dots left/right) and
vertical gates (dots above/below). An untouched dot is surrounded by four `SPLIT`
gates, so it traces a small circle; `compiler.trace` drops those, leaving bare dots
bare.

Every trace closes, so a kolam is always a set of closed strokes. This is the
mirror-curve / N-line / gating model of refs [2] and [4]. One `CROSSING_LOOP` along a
row reproduces the notebook's `build_crossing_loop(n)` exactly (tested for n = 2…9).

### 8.3 The three brushes

| brush | DSL | formation |
|---|---|---|
| `CROSSING_LOOP([v0…vk])` | `c(...)` | chain of 4-adjacent dots; `CROSS` on the gap between consecutive dots |
| `PARALLEL_LOOP([v0…vk])` | `p(...)` | chain of 4-adjacent dots; `JOIN` on the gap between consecutive dots |
| `HORN(A, dir)` | `h` / `hl` / `hr` (aliases `h0` / `h1` / `h2`) | no gate; a horn grown out of the line in front of A when drawing (§10.6) |

- **Closed rings.** Repeating the first dot at the end of a loop call
  (`c([1,1],[1,3],[3,3],[3,1],[1,1])`) closes the chain. The label then carries
  `"closed": true`.
- **Horns** set no gate, so they never change which strands join, cross or bounce: the
  traced strokes (and the lattice sequence) are the same with or without them. A horn is
  drawn by growing the line in front of its dot into a sharp tip (§10.6). The gap it points
  at must be a wall (no loop through it).
  - Frame at A: `fwd = dir`, `left` = `fwd` turned 90° counter-clockwise on screen.
  - Mirroring a kolam swaps `LEFT_HORN` and `RIGHT_HORN` and mirrors the direction; rotating
    it turns the direction. Mirroring is checked on pixels in `tests/`.

### 8.4 Composition rules (what makes a label valid and unique)

There are two levels (`hypergraph.problems`):
- **Free drawing** (`strict=False`, used by `kolam_studio`): only what cannot be drawn is refused:
  dots off the grid, steps between dots that are not 4-neighbours, one gap given two different
  states, and broken horns (rules 5–6). Free drawings may still have short horns (rule 7) and the
  meetings that rule 8 refuses.
- **Dataset label** (`strict=True`, the default; generator and evaluator): all rules below, so the
  picture decodes back to exactly one label.

| # | rule | why |
|---|---|---|
| 1 | Loops have ≥ 2 dots. A closed ring needs ≥ 4 dots, with the last dot next to the first. | the smallest cycle of 4-neighbours is a 2×2 square |
| 2 | Consecutive dots are 4-neighbours. The DSL fills in the dots between waypoints on the same row or column. | diagonal dots have no gate between them |
| 3 | Each gap is set by at most one relation. | a gate holds exactly one state |
| 4 | The label is the one read off the gaps (`hypergraph.from_gates`): where lines of one brush meet at a dot ('+', 'T'), a line goes straight through and the side branch is a line of its own. Crossing and parallel loops may share dots. | a picture shows its gaps, not how the lines were drawn; one fixed splitting rule gives one label per picture, even for dense sikku kolams |
| 5 | A horn points at a wall: the gap in its direction is not set by any loop. One horn per dot and direction. | it grows out of the line that turns in front of its dot |
| 6 | Horn tips are more than 1 lattice unit apart, except horns pointing into the same empty spot: they meet there (§10.6). | tips must not touch by accident |
| 7 | A horn points at an empty spot or off the grid, never at a dot in use. | only then is there room for a crescent; at a used dot it can only be a short stub |
| 8 | Horns meeting in one spot (2 to 4 of them) all have the same variant, and two at 90° do not both grow from the inner corner of one parallel tube (the dot across the corner joined to both by `JOIN` gaps). | mixed curls miss each other and make a loop; two horns on one short inner-corner line collapse it into a leaf |

Under these rules the gate states and tips visible in a drawing decode back to exactly
one canonical hypergraph (`compiler.decode`). The generator also checks every sample
geometrically (`geometry.check_geometry`):
- curves stay ≥ 0.4 from every dot;
- strands stay ≥ 0.2 apart, except at their crossings;
- every `CROSS` gap shows exactly one ✕;
- the only corners are horn tips.

### 8.5 Label conventions

- Vertex ids are anchor coordinates `[row, col]`. `vertices` lists every dot of the grid.
- Labels are **canonical**: an image shows neither a loop's start dot nor its drawing
  direction.
  - Open chains are stored from their smaller end.
  - Closed rings are rotated/reflected to their smallest form.
  - A horn keeps its exact dot, direction and variant, because all three are visible.
- §6 isomorphism is still the structural comparison (`hypergraph.is_isomorphic`).

### 8.6 Stroke count

If the brush lines (consecutive-dot links of all relations) form a **forest**, each
tree compiles to exactly **one** closed stroke, for any mix of `CROSS` and `JOIN`. The
reason: each new link joins the two separate curves that run beside its gap. A tree of
brush lines is therefore a single-stroke kolam. Conversely, a link that closes a cycle
in the brush lines splits a curve in two, which is why a closed ring gives two
strokes. This was checked on every forest-shaped sample generated.

### 8.7 How brushes combine

Brushes never interact directly. Each one only writes gate states (plus, for a horn,
one tip), and the curve is traced once over the whole grid. There is no special case
for any pair of brushes: where two strokes meet, the picture is decided by the gate
states alone, and the only place two strands intersect is a `CROSS` gate.

Brushes meet at **dots**, never at gaps (rule 3). A dot has four gaps; different
brushes may own different gaps of the same dot, and the curve weaves through all of
them. Compiler output on a 3×3 grid:

| program | gates set | strokes | ✕ |
|---|---|---|---|
| `c([0,0],[0,2])` | 2 `CROSS` | 1 | 2 |
| `p([0,0],[0,2])` | 2 `JOIN` | 1 | 0 |
| `c([0,0],[0,2]); p([0,1],[2,1])` (share dot `[0,1]`) | 2 `CROSS` + 2 `JOIN` | 1 | 2 |
| … `+ hl([2,1],S)` | no gate; a crescent off the grid below `[2,1]` | 1 | 2 |
| `c([0,0],[0,2]); p([1,0],[1,2])` (no shared dot) | 2 `CROSS` + 2 `JOIN` | 2 | 2 |
| `c([0,0],[0,1],[1,1],[1,0],[0,0])` (ring) | 4 `CROSS` | 2 | 4 |

The stroke counts follow §8.6: the shared-dot kolams are one tree, the two separate
rows are two trees, and the ring is a cycle.

### 8.8 Decoding

Because everything visible lives in gate states and horns, a drawing decodes back to its
label (`compiler.decode`; the evaluator runs it on every sample). What a picture shows
(`compiler.observe`): the state of every gate, and for each horn the dot whose line it grows out
of and where its tip is.

1. **Loops.** The `CROSS` gates, grouped into connected chains of dots, are the crossing loops;
   the `JOIN` chains are the parallel loops. A chain with no ends is a closed ring. A branching
   chain is an error, which rule 4 rules out.
2. **Horns.** The tip's place relative to its dot gives exactly one direction, variant and size
   (§10.6). The size must agree with the dots in use: long exactly when the spot it points at is
   empty. Meeting horns share a spot but not a dot, so each is still read from its own dot. (The tip alone is not enough: a straight crescent's tip lands on the empty spot, which up
   to 3 neighbouring dots could point at; the line it grows out of tells which.)
3. The result is put in canonical form (§8.5).

## 9. Brush DSL and label JSON (`kolam_press/dsl.py`, `hypergraph.py`)

The hypergraph is the label; the brush DSL is a compact text form of the same thing,
and the natural output for a sequence model (about 6× shorter than the JSON).

```text
                 dsl.parse            compiler          render
  brush program ──────────▶ Hypergraph ───────▶ strokes ──────▶ image
                ◀──────────  (label)
              dsl.format_program
```

### 9.1 DSL syntax

```text
grid(5, 5)                          # optional; else the smallest grid that fits
c([0,0],[0,2])                      # CROSSING_LOOP through [0,0] [0,1] [0,2]
p([0,1],[2,1])                      # PARALLEL_LOOP, sharing dot [0,1] with the crossing loop
c([2,2],[2,4],[4,4],[4,2],[2,2])    # waypoints turn corners; repeat the first dot to close a ring
h([0,3],N)                          # PLAIN_HORN on dot [0,3], tip pointing up  (aliases h0, hp)
hl([2,0],W)                         # LEFT_HORN: curves to its left             (alias h1)
hr([3,0],S)                         # RIGHT_HORN                                (alias h2)
```

- Dots are `[row, col]`, 0-indexed. Statements are separated by newlines or `;`;
  `#` starts a comment.
- Consecutive loop waypoints share a row or column; the dots in between are filled in.
- A horn is one dot and a direction (`N`/`E`/`S`/`W`). The old two-dot form `h(A,B)` is refused
  with a hint: write `p(A,B); h(B,dir)`.
- `parse` returns a canonical, validated hypergraph, or raises `KolamError` with the
  line and the broken rule (e.g. "is diagonal; brushes run along rows/columns").
- `format_program` writes the canonical program, collapsing straight runs to their
  endpoints and corners. `parse(format_program(h)) == h` for every label.

### 9.2 Label JSON

`labels.jsonl` stores `program` (canonical DSL) and `hypergraph` (this JSON) for every
image. The program `c([0,0],[0,2]); hl([2,1],S)` on a 3×3 grid is:

```json
{"grid": {"rows": 3, "cols": 3},
 "vertices": [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]],
 "relations": [
   {"type": "CROSSING_LOOP", "vertices": [[0,0],[0,1],[0,2]]},
   {"type": "HORN", "variant": "LEFT_HORN", "vertices": [[2,1]], "dir": "S"}
 ]}
```

A closed ring adds `"closed": true`. Relations are sorted (crossing loops, parallel
loops, horns), so equal kolams give byte-equal labels.

Each record also stores `code`, the same label as a gap code (§12): for the example above
`["000", "0110", "000", "0000", "000", "0000", "040"]` (the horn sits on the bottom border gap). It is
the training target; the hypergraph is rebuilt from it exactly.

### 9.3 What `hypergraph.py` provides

| function | job | used by |
|---|---|---|
| `Relation.canonical`, `Hypergraph.canonical` | one label per picture (§8.5) | everything |
| `validate` | the composition rules of §8.4 | DSL, generator, evaluator |
| `transform_relation` | mirror / rotate a relation (a horn's direction turns with it); reflections swap `LEFT_HORN` ↔ `RIGHT_HORN` | symmetric generation |
| `is_isomorphic`, `iso_hash` | exact §6 isomorphism, with a cheap pre-bucketing hash | diversity report, model scoring |
| `to_json`, `from_json` | the label format above | dataset I/O |

## 10. Drawing primitives (visual spec)

The gate model (§8) fixes which strands join, cross or bounce. This section fixes how they
**look**. Everything here is drawing only: labels, programs, gate states and the lattice
sequence are unchanged by it.

### 10.1 Gate codes

Every gap between two neighbouring dots has one of three states, written as a digit:

| code | state (`brushes.json`) | brush | at the gap |
|---|---|---|---|
| `0` | `SPLIT` | none | wall: each strand turns back around its own dot |
| `1` | `CROSS` | `c` | the strands cross (✕) |
| `2` | `JOIN` | `p` | the strands run alongside, enclosing both dots |

A kolam on an `m × n` grid is fully described by two grids of these digits: the gaps between
left/right neighbours (`m × (n−1)`) and between up/down neighbours (`(m−1) × n`). Border gaps
are always `0`.

The model is the **4-gate model**: a dot has 4 gates (above, right, below, left), shared with
its neighbours, and every link runs along a row or a column. Diagonal links between dots were
tried and dropped (they need gates at the centres, shared by 4 dots, and triangles); a diamond
(1-3-5-3-1) dot layout, i.e. the same grid turned 45°, is the way to get diagonal-looking kolams
if needed later.

### 10.2 Primitives around one dot

The look around a dot is fixed by its 4 gates. Up to rotation and mirror there are **21**
primitives (named after the one-stroke gating paper, references [4], by how many gates are not walls):

| | gates not `0` | crossing only (`1`) | parallel only (`2`) | mixed `1` + `2` |
|---|---|---|---|---|
| bare | 0 | 1 (shared) | | |
| drop | 1 | 1 | 1 | |
| door (2 adjacent) | 2 | 1 | 1 | 1 |
| eye (2 opposite) | 2 | 1 | 1 | 1 |
| fan | 3 | 1 | 1 | 4 |
| diamond | 4 | 1 | 1 | 4 |
| **total** | | **6** | **5** | **10** |

Reference sheets: `refs/primitives_parallel_straight.png` (crossing and parallel primitives,
side by side), `refs/mixed_options.png` (the mixed primitives, options A and B) and
`refs/examples/` (288 generated crossing + parallel kolams in the final drawing).

### 10.3 Crossing — final

The chained quadratic Bézier through the gate points (§4, `geometry.sample_stroke`), as
implemented. A crossing loop passes **0.75** from its dot above/below and beside it and **0.71**
diagonally; strands cross at the gap.

### 10.4 Parallel — final

A parallel run is drawn as a **capsule** with the same thickness and roundness as a crossing
loop, with no pinch at the gaps:

- **Straight walls at `w = 0.75`** from the line of dots (the crossing loop's distance): the
  control point at each parallel gap is moved off the gap by `w` toward the strand's own side,
  and the wall gaps the strand passes in the middle of a run are pulled in to `w` from their dot.
- **Round ends and outer corners**: where the line turns around a dot of a parallel run, its
  control points sit on a circle of radius `w` around that dot, with an extra point at each 45°
  so the turn is round. Measured: walls 0.75 along the whole run, ends 0.70–0.72 (crossing:
  0.71–0.75), closest approach to a dot 0.69.
- **Box**: when the 4 gaps around one centre are all parallel, the 4 dots form a filled box:
  the small inner loop around that centre is not drawn, leaving one rounded square.
- Inner corners (where two walls of an L meet) are tight; that is intended.

Geometry: `geometry.capsule_points()` and `geometry.is_box_hole()`, combined in
`geometry.drawn()`, which both `render.py` and `check_geometry` use, so the checks test what is
drawn. The studio draws the same through `kolam_studio/src/kolam/drawn.ts`, a port of `drawn()`
(its drawn points match Python's to 1e-13).

### 10.5 Mixed (crossing and parallel at one dot) — final

A dot with both a `1` and a `2` gap is drawn **like the end of a tube** (option B in
`refs/mixed_options.png`): its turns sit on the same radius-`w` circle as the capsule's ends, and the
line leaves that circle straight into the crossing's ✕ (the crossing gap stays where crossing
puts it). This needs no extra rule: `geometry.capsule_points()` applies to every dot touched by a
parallel gap, mixed or not.

Why it is seamless: walls, round ends and crossing loops all sit at the same distance (0.75) from
their dots, so a wall meets the circle tangentially and leaves it into the ✕ without a bump. The
rule behind all of §10: **the line stays 0.75 from its dots and runs, and leaves that distance
only to cross at a crossing gap.** (The rejected option A, giving the meeting dot a crossing-style
ring, puts its points a full unit from the dot and bulges where the tube meets the crossing.)

**Lean into a crossing.** A wall point that feeds straight into a crossing gap (the point just
before or after the ✕ in the stroke) sits at `CAPSULE_LEAN = 0.5` instead of 0.75, so the ✕ has a
clear angle. Without it, a crossing between two parallel tubes side by side (a "ladder") has only
the 0.5 channel between their walls and its strands cross at ~28°, nearly touching. With it,
every ladder passes the geometry check, and horn-free generated kolams went from 112/399 failing
to 0/288 (`refs/examples/`).

### 10.6 Horns — final

A horn is **one dot, a direction (`N`/`E`/`S`/`W`) and a curve (`L`/`S`/`R` = `LEFT_HORN`,
`PLAIN_HORN`, `RIGHT_HORN`)**. All three point the same way; left and right bend a little. It sets
no gate, so it never changes the strokes: it is drawn by **growing the line in front of its dot**
into a sharp tip (`geometry.grow_horns`, numbers in `brushes.json` `HORN.grow`):

- The stroke that passes in front of the dot is redrawn through dense points on its own curve
  (the same curve), and the points within `half` (arc length) of the front are pulled toward the
  tip `T` with weight `w = (1 − t)^q`, `t = arc / half`. The weight fades to 0 with zero slope, so
  the horn **blends into whatever line it grows from** (crossing loop, capsule end, tube wall) with
  no lump where it starts. A term `w(1 − w)` opens the two flanks and bends the body for `L`/`R`.
  `T` is repeated, giving the one sharp corner.
- **Crescents only in the dataset.** If the spot the horn points at holds no used dot (an empty
  spot, or off the grid), it is a **crescent**: the tip lands on that spot (reach 2.0), as in
  classic sikku kolams (`refs/photo_remake.png`). Pointing at a dot some relation uses, the lines
  are only 0.5 apart, so it can only be a **short** stub (reach 1.05) that reads as a kink. Labels
  forbid that (rule 7); free drawings still draw it.

  | | reach | half | q | open | curl | bend |
  |---|---|---|---|---|---|---|
  | long (crescent) | 2.0 | 1.8 | 2 | 0.35 | 0.1 | 1.3 |
  | short | 1.05 | 1.0 | 2 | 0.35 | 0.3 | 0.7 |

- Several horns may share a dot. Two horns at 90° on one dot each stop at the 45° line between
  them, so their flanks never overlap.
- **Meeting horns.** Two to four horns from different dots may point into the same empty spot:
  facing each other or at 90°. Each keeps its own crescent, and their tips meet on the spot (plain
  horns touch; left/left or right/right pass close by, a pinwheel for four). They must all curl
  the same way, and two at 90° must not both grow from the inner corner of one parallel tube
  (rule 8). `check_geometry` lets strands come close or touch within `MEET_R = 0.5` of a meeting
  spot (`geometry.meeting_spots`). Reference sheets: `refs/horns_facing.png` (the look, option A
  chosen) and `refs/horns_meeting.png` (90°, three and four into one spot).
- **Unused dots are not drawn** (as in real kolams, and so a crescent's tip does not land on a
  dot). The generator therefore always crops empty border rows/columns, so the grid size can be
  read from the picture; the canvas grows to fit crescents that point off the grid.
- Labels: `{"type": "HORN", "variant": "LEFT_HORN", "vertices": [[r, c]], "dir": "N"}`; DSL
  `hl([r,c],N)`. Rules 5–8 of §8.4; decoding in §8.8.

Reference sheets: `refs/horns_final.png` (L/S/R in six contexts) and `refs/photo_remake.png` (a photographed sikku kolam rebuilt by the engine:
its gates read off the pixels plus 8 crescents, now `examples/sikku_diamond.kolam`).

The photo kolam is a valid dataset label: its crossing lines branch at most dots, and rule 4 reads
them off the gaps. The experimental generators in `kolam_press/systems.py` make kolams of this kind (§11).

### 10.7 Open

1. **Old datasets**: images rendered before §10 use old looks, labels with two-dot horns are no
   longer valid, and `data/kolams_v2` (and `docs/results/`) still has short horns (rule 7, 54% of
   its horns); regenerate them.
2. **Generator systems**: four ways to make kolams (§11); not yet mixed into `generate`.

## 11. Generator systems (`kolam_press/generator.py`, `kolam_press/systems.py`)

Picture: `refs/systems_explained.png` (one row per system, step by step).

All four systems choose **gap digits** (§10.1): every gap between two neighbouring dots is
`0` wall, `1` cross or `2` tube. Once every gap is set, the kolam is fully decided; the label is read
off the gaps (§8.4 rule 4). Crescent horns (§10.6) are added last, in whole symmetric orbits, and
only if every label rule still holds. Each sample is then checked like any other: strict
`validate`, program re-parse, decode from the drawing, `check_geometry`, no duplicates.

```bash
python -m kolam_press generate -n 3000 -o data/kolams_v3      # 1. random
python -m kolam_press systems  -n 100  -o data/systems        # all four, 100 each, sheet.png per system
```

### 11.1 Random: the painter (`generator.py`)

- Pick a grid (3–8 dots a side) and a symmetry (none, mirror, 180° turn, D2, C4, D4).
- Lay brush strokes one at a time (a crossing line, a tube line or a horn), each copied at once to
  its symmetric places; a stroke that breaks a rule is thrown out.
- About 40% of kolams grow their brush lines as a tree, which gives one-stroke kolams.
- Stop at 70–100% of the dots covered, then cover leftover dots by extending lines or adding horns.
- Like a person drawing freehand, motif by motif: the most mixed kolams (tubes, crossings and
  horns together) and the most open space.

### 11.2 Substitution: the fractal grower (2D L-system)

- Start from a small seed: 2×2 to 4×4 dots with D4-symmetric gaps.
- Invent a rule book for this sample:
  - **block rule**: a dot with N open gaps (N = 0–4) becomes a given k×k block of dots
    (k = 2 or 3; for k = 3 the 4 gaps round the centre and the 8 round the edge are set apart);
  - **seam rule**: where two parent dots had a wall, cross or tube between them, the gaps between
    their blocks follow a given palindromic pattern, so symmetry is kept.
- Apply the rule book again and again (2×2 → 4×4 → 8×8) while the grid is at most 12 dots a side.
- The same rule works at every scale, so patterns sit inside patterns. Each sample invents a new rule
  book, so no two look alike. Ornate and surprising; rarely one stroke.
- After Siromoney's kolam array grammars and Prusinkiewicz's L-system kolams.

### 11.3 Switching: the one-line maker

- Pick a dot layout (square, diamond, octagon or plus; 5–11 dots a side) and a symmetry (D4, D2, C4).
- Start with **every gap a cross**: the full sikku net, which is usually several closed lines.
- Pick a symmetric set of gaps where two different lines meet and flip it (to a wall, or sometimes
  to a tube). Keep the flip only if the number of lines went down and no dot was left bare.
- Repeat until one line (75% of samples aim for that, the rest for 2–4 lines) or 150 failed flips.
- Real sikku kolams work this way: one unbroken line round every dot is the classic ideal.
- After Sivakumar & Sivakumar 2026 (gate switching) and Gerdes' mirror curves.
- Weak spot: under D4 one flip changes 4–8 gaps at once, which can split lines as well as join them,
  so it often gets stuck.

### 11.4 WFC: the student (wave function collapse)

- Cut a real kolam (`examples/sikku_diamond.kolam`, the photographed one) into **dot pieces**:
  a dot plus what its 4 gaps do. With the 8 turns and mirrors there are 15 kinds, each counted.
- Pick a dot layout and symmetry as in switching, then fill the grid like a jigsaw:
  the dot with the **fewest pieces that still fit** its neighbours goes first, gets a piece drawn by
  how often it appears in the photo, and is copied to its symmetric places.
- If no piece fits somewhere, start again (up to 80 tries).
- Every local piece comes from a real kolam, so the style carries over into a new design.
- Weak spot: it only knows its source. The photo has no tubes, so WFC never makes tubes; each new
  source kolam adds new pieces.

### 11.5 First run (100 each, seed 0)

All 400 pass every check above.

| system | how it creates | best at | one stroke | weak spot |
|---|---|---|---|---|
| random | places brush strokes | variety, horns, open layouts | 37% | less traditional |
| substitution | repeats a rule at every scale | ornate, nested, surprising | 1% | rarely one line |
| switching | merges lines | real sikku, one-line kolams | 27% | stuck under D4 |
| wfc | copies local pieces of a real kolam | looking human | 4% | only its source's pieces (no tubes) |

Substitution averages 10.7 horns a kolam and 83 of 100 have tubes; switching 16 of 100; WFC none.
`evaluate`'s exact isomorphism check is too slow on large symmetric kolams; fix it before evaluating
these sets.

## 12. Gap code: the training form of the label (`kolam_press/gapcode.py`)

Pictures: `refs/grid_representation.png` (the digits), `refs/representation_examples.png`
(one kolam in every form), `refs/grid_why.png` (why, how lines are read back, tokenizers).

The hypergraph stays the label (§3). For training it is written as its **gap code**, an exact
equivalent: one digit on every gap of the dot grid, border gaps included, laid out as a plain grid.
Lines and tubes are already gap states (§10.1); a horn always points through a wall gap, so it is
written on that gap.

### 12.1 The 9 digits

| digit | gap |
|---|---|
| `0` | wall: nothing passes (also all empty space) |
| `1` | cross |
| `2` | tube |
| `3` `4` `5` | plain / left / right horn facing E or S (right or down) |
| `6` `7` `8` | plain / left / right horn facing W or N (left or up) |

A horn has 3 curls × 4 directions, but the gap fixes the axis: a left/right gap can only hold an E or
W horn, an up/down gap only an S or N horn. So the digit only says the curl and forward (E, S) or back
(W, N): `5` on a left/right gap is `hr` facing E, on an up/down gap `hr` facing S. Every gap holds at
most one thing: a horn needs a wall gap, and its target spot is empty, so no horn comes from the other
side; horns meeting in one spot (§10.6) come in through different gaps.

### 12.2 The code

The gap rows of the lattice (§8.1), top to bottom. They alternate: `cols` digits (the up/down gaps
above a dot row, or below the last one) and `cols + 1` digits (the left/right gaps of a dot row, both
borders included); `2·rows + 1` rows, `2·rows·cols + rows + cols` digits (5×5 dots: 60, 12×12: 312).
Nothing else is stored: no addresses, no size, since the row count and row lengths give
`rows × cols`. `labels.jsonl` keeps it as a list of digit strings in `code`. The program
`grid(5,5); c([0,2],[4,2]); c([1,1],[1,3]); c([2,0],[2,4]); c([3,1],[3,3]); hl([0,2],W);
hl([4,2],E); hr([0,2],E); hr([4,2],W)` (one drawn line, 8 relations) is:

```text
00000        up/down gaps above dot row 0 (top border)
007500       dot row 0: 7 = hl facing W, 5 = hr facing E
00100        the line from dot [0,2] down
001100       dot row 1: c([1,1],[1,3])
00100
011110       dot row 2: c([2,0],[2,4])
00100
001100       dot row 3
00100
008400       dot row 4: 8 = hr facing W, 4 = hl facing E
00000        bottom border
```

Two more layouts of the same digits:

- **grid**: the `(2·rows+1) × (2·cols+1)` lattice, digits on the gaps, dots and centres `0`.
- **cells**: one cell per dot, `(rows+1) × (cols+1) × 2` = (left gap, top gap) of dot `(r, c)`; the
  extra last column holds the right border, the extra last row the bottom border, and the slots past
  the last dot are always `0`. A plain rectangle, one output per dot, for a model that predicts every
  gap at once.

### 12.3 Reading it back

1. Row count and lengths give `rows × cols`.
2. Lines from the `1`s and the `2`s, separately, with the `from_gates` rule (§8.4 rule 4): at a dot
   with 2 open gaps the line goes on (round corners too); with 3 or 4 it goes straight through and a
   side branch is a line of its own; it ends at a dot with 1; a ring of 2-gap dots is a closed loop.
3. Each `3`–`8` is a horn on the dot beside its gap, facing out through it.
4. That is the canonical hypergraph; program, JSON and picture follow.

`to_code`/`from_code`, `to_grid`/`from_grid`, `to_cells`/`from_cells`, `to_tokens`/`from_tokens` are
exact inverses (`tests/test_gapcode.py`; all 400 samples of the first systems run round-trip).

### 12.4 Models

**All gaps at once (the main model).** The model outputs a 9-way choice per gap, on the cells layout
(two choices per dot). Every digit is local: wall, cross, tube or horn is visible in the pixels around
its gap (about one dot spacing; two for a crescent). A fully convolutional model therefore reads any
grid size with the same weights: train on small kolams and crops of big ones (each gap's digit inside a
crop is still right), run on 50×50. Everything global (joining gaps into lines, splitting, the
hypergraph) is done by `from_gates`, at any size. On a free image the dot lattice is found first (dot
heatmap, fit the regular grid); then each gap is read at the midpoint between two dots.

**Writing tokens (a sequence model or a fine-tuned vision-language model).** The code is written row by
row: each row's digits, then `|`, then `<end>`: 11 tokens, `0`–`8`, `|`, `<end>`. Decoding is
**constrained** (`allowed_next`): the first row sets `cols`, rows then alternate `cols + 1` and `cols`
digits, `<end>` only after a full `cols`-digit row, and border gaps only hold walls or horns facing out
of the grid (top and left: `0` `6` `7` `8`; right: `0` `3` `4` `5`). Every output parses into a grid; the
label rules (§8.4) are then checked by `validate`. Training is plain next-token prediction.

### 12.5 Why this form

Tokens per kolam on the first systems run (100 each), average / max:

| system | program, own tokens | **code** (digits + row ends) | sparse (address + digit, rows grouped) |
|---|---|---|---|
| random | 39 / 123 | **83 / 161** | 70 / 220 |
| switching | 67 / 160 | **177 / 287** | 210 / 496 |
| wfc | 75 / 186 | **168 / 287** | 125 / 328 |
| substitution | 140 / 426 | **176 / 337** | 238 / 602 |

- **The program is shorter but harder to learn.** A position is counted across the image (`[7,3]`)
  instead of read at one spot; the model must pick an order and split lines at branches; one mistake can
  shift or end a line and all after it. In the code, one mistake is one wrong gap.
- **Dropping the zeros does not pay.** Each kept item must then carry its address; on average that costs
  about what the zeros saved, and in the worst case much more (602 vs 337). The code's length is fixed
  by the grid size, its positions are implicit, it has 11 tokens instead of 59, and it is the very
  output of the all-at-once model.
- **Tokenizers.** A text tokenizer must not see the code as plain digits. GPT-4's `cl100k` cuts the
  rows `00000|007500|00100|…` into `000` `00` `|` `007` `500` `|` `001` `00` …: `007` is wall, wall,
  left horn in one token, so one gap gets different tokens depending on its neighbours. Spacing the
  digits doubles the length (`0`, ` ` per digit). Use our own 11 tokens, or add them as special tokens
  to a pretrained vision-language model (as PaliGemma does with its location tokens).
- **Size.** Written as tokens, a 50×50 kolam is ~5,200 tokens, and a sequence model does not generalise
  well past the lengths it was trained on. The all-at-once model has no such limit.

**Not chosen**: the sparse list (above); per-dot tiles with all 4 gaps (each gap stored twice, the copies
can disagree); the lattice stroke sequence (longest, stroke order arbitrary); folding by symmetry (2–8×
smaller but fails on kolams without symmetry and makes the model guess the group first).
