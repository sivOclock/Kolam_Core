"""kolam_feed: kolam encodings (from kolam_gen) -> images rendered on the fly -> the model.

    from kolam_feed import KolamDataset, collate
    ds = KolamDataset("data/enc_1m", size=448)
    item = ds[0]          # {"image": PIL.Image, "encoding": [...], "text": "...", "tokens": [...]}

The Python side of the kolam model (docs/hypergraph_kolams.md): read an encoding into its canonical
hypergraph (encoding, hypergraph), compile it to strokes (compiler), draw it (geometry.drawn, render)
with a fresh random style per (seed, epoch, index), and check it independently of the Rust generator
(geometry.check_geometry, compiler.decode, verify). Constrained decoding for a sequence model:
encoding.allowed_next.
"""
from .compiler import Compiled, compile_hypergraph, decode, observe
from .data import KolamDataset, collate, load_encodings, target_text
from .encoding import allowed_next, from_encoding, from_tokens, to_encoding, to_tokens
from .geometry import check_geometry, drawn
from .hypergraph import Hypergraph, KolamError, Relation
from .render import Style, random_style, render_kolam
from .verify import verify

__all__ = [
    "Compiled", "Hypergraph", "KolamDataset", "KolamError", "Relation", "Style", "allowed_next", "check_geometry",
    "collate", "compile_hypergraph", "decode", "drawn", "from_encoding", "from_tokens", "load_encodings", "observe",
    "random_style", "render_kolam", "target_text", "to_encoding", "to_tokens", "verify",
]
