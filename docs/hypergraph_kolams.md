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
  states, and broken horns (rules 5–6).
- **Dataset label** (`strict=True`, the default; generator and evaluator): all rules below, so the
  picture decodes back to exactly one label.

| # | rule | why |
|---|---|---|
| 1 | Loops have ≥ 2 dots. A closed ring needs ≥ 4 dots, with the last dot next to the first. | the smallest cycle of 4-neighbours is a 2×2 square |
| 2 | Consecutive dots are 4-neighbours. The DSL fills in the dots between waypoints on the same row or column. | diagonal dots have no gate between them |
| 3 | Each gap is set by at most one relation. | a gate holds exactly one state |
| 4 | Loops of the same brush never share a dot. Crossing and parallel loops may share dots. | two same-brush chains through one dot would branch there, and the picture could not say which chain is which; different brushes stay distinguishable by their gate states |
| 5 | A horn points at a wall: the gap in its direction is not set by any loop. One horn per dot and direction. | it grows out of the line that turns in front of its dot |
| 6 | Horn tips are more than 1 lattice unit apart. | two crescents into one empty spot would meet there |

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
   empty. (The tip alone is not enough: a straight crescent's tip lands on the empty spot, which up
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
- **Size follows the drawing, not the label.** If the spot the horn points at holds no used dot
  (an empty spot, or off the grid), it is a **crescent**: the tip lands on that spot (reach 2.0),
  as in classic sikku kolams (`refs/photo_remake.png`). If it points at a dot some relation uses,
  it is **short** (reach 1.05), so it stays clear of that dot's loop.

  | | reach | half | q | open | curl | bend |
  |---|---|---|---|---|---|---|
  | long (crescent) | 2.0 | 1.8 | 2 | 0.35 | 0.1 | 1.3 |
  | short | 1.05 | 1.0 | 2 | 0.35 | 0.3 | 0.7 |

- Several horns may share a dot. Two horns at 90° on one dot each stop at the 45° line between
  them, so their flanks never overlap.
- **Unused dots are not drawn** (as in real kolams, and so a crescent's tip does not land on a
  dot). The generator therefore always crops empty border rows/columns, so the grid size can be
  read from the picture; the canvas grows to fit crescents that point off the grid.
- Labels: `{"type": "HORN", "variant": "LEFT_HORN", "vertices": [[r, c]], "dir": "N"}`; DSL
  `hl([r,c],N)`. Rules 5–6 of §8.4; decoding in §8.8.

Reference sheets: `refs/horns_final.png` (L/S/R in six contexts) and `refs/photo_remake.png` (a photographed sikku kolam rebuilt by the engine:
its gates read off the pixels plus 8 crescents, now `examples/sikku_diamond.kolam`).

Not yet: the photo kolam is a free drawing, not a dataset label, because its crossing chains
branch at most dots (rule 4). The generator only makes such dense sikku kolams if rule 4 is
relaxed, e.g. by labelling a kolam by its gate codes (§10.1) rather than by chains.

### 10.7 Open

1. **Old datasets**: images rendered before §10 use old looks, and labels with two-dot horns are no
   longer valid; regenerate them (`docs/results/` has been).
2. **Dense sikku kolams** (crossing chains branching at most dots, like `examples/sikku_diamond.kolam`)
   are free drawings only, because of rule 4; see the end of §10.6.
