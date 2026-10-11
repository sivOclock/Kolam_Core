"""kolam_press: brush-program -> ordered hypergraph -> kolam data generator.

    from kolam_press import parse, compile_hypergraph, render_kolam
    h = parse("c([0,0],[0,3]); hl([1,0],S)")
    img = render_kolam(compile_hypergraph(h))
"""
from .compiler import Compiled, compile_hypergraph, decode, observe
from .dsl import format_program, parse
from .hypergraph import Hypergraph, KolamError, Relation, is_isomorphic
from .geometry import check_geometry
from .render import Style, render_brush, render_kolam, render_panel

__all__ = [
    "Compiled", "Hypergraph", "KolamError", "Relation", "Style", "check_geometry",
    "compile_hypergraph", "decode", "format_program", "is_isomorphic", "observe", "parse",
    "render_brush", "render_kolam", "render_panel",
]
