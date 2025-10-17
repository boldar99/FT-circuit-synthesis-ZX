from typing import Optional

from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.utils import VertexType


def css_encoder(stabilizers: list[list[int]],
                logicals: list[list[int]],
                backend: Optional[str] = None) -> BaseGraph[VT, ET]:
    n, k = len(stabilizers[0]), len(logicals[0])

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
