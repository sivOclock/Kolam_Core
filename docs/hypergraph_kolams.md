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
