from typing import Optional

from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.utils import VertexType
from pyzx.rewrite_rules.basicrules import strong_comp
from pyzx.simplify import spider_simp, id_simp


def css_encoder(stabilizers: list[list[int]],
                logicals: list[list[int]],
                backend: Optional[str] = None) -> BaseGraph[VT, ET]:
    n, k = len(stabilizers) and len(stabilizers[0]), len(logicals) and len(logicals[0])

    num_internals_vertices = len(stabilizers) + len(logicals)
    spacing = [(k + .5) * (n - 1) / num_internals_vertices for k in range(num_internals_vertices)]

    encoder = Graph(backend)
    inputs: list[VT] = []
    outputs: list[VT] = []
    x_vertices, z_vertices = [], []

    for i, s in enumerate(logicals):
        v1 = encoder.add_vertex(ty=VertexType.BOUNDARY, qubit=spacing[i], row=0)
        v2 = encoder.add_vertex(ty=VertexType.Z, qubit=spacing[i], row=1)
        encoder.add_edge((v1, v2))
        inputs.append(v1)
        z_vertices.append(v2)

    for i in range(n):
        v1 = encoder.add_vertex(ty=VertexType.X, qubit=i, row=3)
        v2 = encoder.add_vertex(ty=VertexType.BOUNDARY, qubit=i, row=4)
        encoder.add_edge((v1, v2))
        x_vertices.append(v1)
        outputs.append(v2)

    for i, s in enumerate(stabilizers):
        v2 = encoder.add_vertex(ty=VertexType.Z, qubit=spacing[len(logicals) + i], row=1)
        z_vertices.append(v2)

    for i, v in enumerate(logicals + stabilizers):
        for j, val in enumerate(v):
            if val == 1:
                encoder.add_edge((z_vertices[i], x_vertices[j]))

    encoder.set_inputs(tuple(inputs))
    encoder.set_outputs(tuple(outputs))

    return encoder


def css_zero_state(stabilizers: list[list[int]],
                   backend: Optional[str] = None) -> BaseGraph[VT, ET]:
    n = len(stabilizers[0])

    num_internals_vertices = len(stabilizers)
    spacing = [(k + .5) * (n - 1) / num_internals_vertices for k in range(num_internals_vertices)]

    zero_state = Graph(backend)
    inputs: list[VT] = []
    outputs: list[VT] = []
    x_vertices, z_vertices = [], []

    for i in range(n):
        v1 = zero_state.add_vertex(ty=VertexType.X, qubit=i, row=3)
        v2 = zero_state.add_vertex(ty=VertexType.BOUNDARY, qubit=i, row=4)
        zero_state.add_edge((v1, v2))
        x_vertices.append(v1)
        outputs.append(v2)

    for i, s in enumerate(stabilizers):
        v2 = zero_state.add_vertex(ty=VertexType.Z, qubit=spacing[i], row=1)
        z_vertices.append(v2)
        for j, val in enumerate(s):
            if val == 1:
                zero_state.add_edge((z_vertices[i], x_vertices[j]))

    zero_state.set_inputs(tuple(inputs))
    zero_state.set_outputs(tuple(outputs))

    return zero_state


def _first_red_boundary(graph: BaseGraph[VT, ET]) -> tuple[Optional[VT], Optional[VT]]:
    for o in graph.outputs():
        for n in graph.neighbors(o):
            if graph.type(n) == VertexType.X:
                return n, o
    return None, None


def _last_green_neigh(graph: BaseGraph[VT, ET], v: VT) -> tuple[Optional[VT], Optional[VT]]:
    green_neighs = [n for n in graph.neighbors(v) if graph.type(n) == VertexType.Z]
    last = max(green_neighs, key=graph.qubit) if green_neighs else None
    boundary_nodes = green_neighs and [n for n in graph.neighbors(last) if graph.type(n) == VertexType.BOUNDARY]
    boundary = max(boundary_nodes, key=graph.qubit) if boundary_nodes else None
    return last, boundary


def _find_neighbouring_output(graph: BaseGraph[VT, ET], v: VT) -> Optional[VT]:
    for o in graph.outputs():
        if v in graph.neighbors(o):
            return o
    return None


def _rearrange_bipartite_css_state(g: BaseGraph[VT, ET]) -> None:
    output_row = max(g.row(o) for o in g.outputs())
    for type in [VertexType.X, VertexType.Z]:
        vs = [(v, _find_neighbouring_output(g, v)) for v in g.vertices() if g.type(v) == type]
        vs = list(sorted(vs, key=lambda x: g.qubit(x[1])))
        for i, (w, o) in enumerate(vs):
            g.set_row(w, output_row - len(vs) - 1 + i)
            if o is not None:
                g.set_qubit(w, g.qubit(o))


def css_state_to_bipartite(g: BaseGraph):
    """
    Transformes a CSS state in the usual normal form to
    :param g:
    :return:
    """
    v0, b0 = _first_red_boundary(g)
    v1, b1 = _last_green_neigh(g, v0)
    is_nice_bipartite_css_state = b0 is not None and b1 is not None and g.qubit(b1) < g.qubit(b0)
    while not is_nice_bipartite_css_state:
        strong_comp(g, v0,v1)
        spider_simp(g)
        id_simp(g)
        v0, b0 = _first_red_boundary(g)
        v1, b1 = _last_green_neigh(g, v0)
        is_nice_bipartite_css_state = b0 is not None and b1 is not None and g.qubit(b1) < g.qubit(b0)

    _rearrange_bipartite_css_state(g)


def bipartite_css_zero_state(stabilizers: list[list[int]],
                             backend: Optional[str] = None) -> BaseGraph[VT, ET]:
    state = css_zero_state(stabilizers, backend)
    css_state_to_bipartite(state)
    return state
