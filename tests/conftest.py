import json
from pathlib import Path

import pytest

from kolam_feed.hypergraph import Hypergraph, Relation

FIXTURES = Path(__file__).with_name("fixtures") / "kolams.jsonl"


@pytest.fixture(scope="session")
def kolams():
    """180 kolams with the answers of the retired Python generator (random + the 3 gate-grid systems,
    21 failing the geometry check): encoding, canonical relations, program, geometry stats."""
    return [json.loads(line) for line in FIXTURES.open()]


def kolam(rows, cols, *rels):
    """A canonical hypergraph from (type, vertices[, variant, dir]) tuples; a loop that repeats its
    first dot is a closed ring."""
    out = []
    for r in rels:
        if r[0] == "HORN":
            out.append(Relation("HORN", (r[1],), r[2], False, r[3]))
        else:
            vs = list(r[1])
            closed = len(vs) > 2 and vs[0] == vs[-1]
            out.append(Relation(r[0], tuple(vs[:-1] if closed else vs), None, closed))
    return Hypergraph(rows, cols, tuple(out)).canonical()
