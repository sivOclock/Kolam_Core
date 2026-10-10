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
HORN(A → B)

CROSSING_LOOP([A, B, C, D])

PARALLEL_LOOP([A, B, C, D])
```

### Horn Relations

- Form **1 of 3**: `HORN`.
- Horn relations involve exactly **2 vertices**.
- There are three horn variants — `PLAIN_HORN`, `LEFT_HORN`, `RIGHT_HORN` — all of type `HORN`:

| Variant | Notation |
|---|---|
| `PLAIN_HORN` | `PLAIN_HORN(A → B)` |
| `LEFT_HORN` | `LEFT_HORN(A → B)` |
| `RIGHT_HORN` | `RIGHT_HORN(A → B)` |

The vertex **ordering matters** because the horn has a specific orientation and tip direction.

> `HORN(A → B)` and `HORN(B → A)` are **not** automatically equivalent.

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
        LEFT_HORN(A → E),
        RIGHT_HORN(C → F),
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
        "vertices": ["A", "E"]
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
- The curve moves diagonally `(±1, ±1)` from gate to gate. It never visits an anchor,
  and it visits a centre only at a horn tip.

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
| `JOIN` | mirror along the dot–dot line → strands run parallel around both dots | `PARALLEL_LOOP`, `HORN` |
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
| `HORN(A → B)` | `h` / `hl` / `hr` (aliases `h0` / `h1` / `h2`) | `JOIN` on the A–B gap, plus a sharp tip replacing B's far-side turn |

- **Closed rings.** Repeating the first dot at the end of a loop call
  (`c([1,1],[1,3],[3,3],[3,1],[1,1])`) closes the chain. The label then carries
  `"closed": true`.
- **Horn tips.** These are defined in B's frame: `fwd = B − A`, and `left` = `fwd`
  turned 90° counter-clockwise on screen. A repeated control point gives the sharp tip.
  - `PLAIN_HORN`: tip at the far gate `B + fwd`, pointing straight ahead.
  - `LEFT_HORN`: tip at the centre `B + fwd + left`, curling left of the direction of travel.
  - `RIGHT_HORN`: tip at the centre `B + fwd − left`, curling right.
  - Mirroring a kolam swaps `LEFT_HORN` and `RIGHT_HORN`; rotating it does not. This is
    checked on pixels in `tests/`.

### 8.4 Composition rules (what makes a label valid and unique)

| # | rule | why |
|---|---|---|
| 1 | Loops have ≥ 2 dots. A closed ring needs ≥ 4 dots, with the last dot next to the first. | the smallest cycle of 4-neighbours is a 2×2 square |
| 2 | Consecutive dots are 4-neighbours. The DSL fills in the dots between waypoints on the same row or column. | diagonal dots have no gate between them |
| 3 | Each gap is set by at most one relation. | a gate holds exactly one state |
| 4 | Loops of the same brush never share a dot. Crossing and parallel loops may share dots. | two same-brush chains through one dot would branch there, and the picture could not say which chain is which; different brushes stay distinguishable by their gate states |
| 5 | A horn's tip dot `B` belongs to that horn only. | the tip must identify its horn when decoding |
| 6 | Horn tips are more than 1 lattice unit apart. | a plain tip beside a curled tip leaves strands < 0.2 apart |

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
  - A horn keeps its exact order and variant, because both are visible.
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
| … `+ hl([2,1],[2,2])` | + 1 `JOIN` + tip at centre `(4,6)` | 1 | 2 |
| `c([0,0],[0,2]); p([1,0],[1,2])` (no shared dot) | 2 `CROSS` + 2 `JOIN` | 2 | 2 |
| `c([0,0],[0,1],[1,1],[1,0],[0,0])` (ring) | 4 `CROSS` | 2 | 4 |

The stroke counts follow §8.6: the shared-dot kolams are one tree, the two separate
rows are two trees, and the ring is a cycle.

### 8.8 Decoding

Because everything visible lives in gate states and tip positions, a drawing decodes
back to its label (`compiler.decode`; the evaluator runs it on every sample):

1. **Horns first.** Each tip sits beside its dot `B`. `B` has exactly one `JOIN` gate
   (rule 5), whose other dot is the base `A`. The tip's offset in `B`'s frame
   (`fwd`, `fwd + left`, `fwd − left`) gives `PLAIN_HORN`, `LEFT_HORN` or `RIGHT_HORN`.
2. **Loops.** The remaining `CROSS` gates, grouped into connected chains of dots, are
   the crossing loops; the remaining `JOIN` chains are the parallel loops. A chain
   with no ends is a closed ring. A branching chain is an error, which rule 4 rules out.
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
h([1,3],[0,3])                      # PLAIN_HORN base -> tip  (aliases h0, hp)
hl([1,0],[2,0])                     # LEFT_HORN               (alias h1)
hr([3,1],[3,0])                     # RIGHT_HORN              (alias h2)
```

- Dots are `[row, col]`, 0-indexed. Statements are separated by newlines or `;`;
  `#` starts a comment.
- Consecutive loop waypoints share a row or column; the dots in between are filled in.
- `parse` returns a canonical, validated hypergraph, or raises `KolamError` with the
  line and the broken rule (e.g. "is diagonal; brushes run along rows/columns").
- `format_program` writes the canonical program, collapsing straight runs to their
  endpoints and corners. `parse(format_program(h)) == h` for every label.

### 9.2 Label JSON

`labels.jsonl` stores `program` (canonical DSL) and `hypergraph` (this JSON) for every
image. The program `c([0,0],[0,2]); hl([1,1],[2,1])` on a 3×3 grid is:

```json
{"grid": {"rows": 3, "cols": 3},
 "vertices": [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]],
 "relations": [
   {"type": "CROSSING_LOOP", "vertices": [[0,0],[0,1],[0,2]]},
   {"type": "HORN", "variant": "LEFT_HORN", "vertices": [[1,1],[2,1]]}
 ]}
```

A closed ring adds `"closed": true`. Relations are sorted (crossing loops, parallel
loops, horns), so equal kolams give byte-equal labels.

### 9.3 What `hypergraph.py` provides

| function | job | used by |
|---|---|---|
| `Relation.canonical`, `Hypergraph.canonical` | one label per picture (§8.5) | everything |
| `validate` | the composition rules of §8.4 | DSL, generator, evaluator |
| `transform_relation` | mirror / rotate a relation; reflections swap `LEFT_HORN` ↔ `RIGHT_HORN` | symmetric generation |
| `is_isomorphic`, `iso_hash` | exact §6 isomorphism, with a cheap pre-bucketing hash | diversity report, model scoring |
| `to_json`, `from_json` | the label format above | dataset I/O |
