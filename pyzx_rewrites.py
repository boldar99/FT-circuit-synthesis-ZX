"""
PyZX rewrite rules for simplifying graphs.
"""

from pyzx import VertexType, EdgeType
from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.rewrite_rules.basicrules import (
    fuse as _fuse,
    color_change as _color_change,
)


def check_fuse_degree1(g: BaseGraph[VT, ET], v: VT) -> bool:
    """Check if a degree-1 spider can be fused with its neighbor."""
    if g.vertex_degree(v) != 1:
        return False
    v_type = g.type(v)
    if v_type not in (VertexType.X, VertexType.Z):
        return False
    neighbor = list(g.neighbors(v))[0]
    return g.type(neighbor) == v_type


def fuse_degree1(g: BaseGraph[VT, ET], v: VT) -> bool:
    """Fuse a degree-1 spider with its neighbor if same color."""
    if not check_fuse_degree1(g, v):
        return False
    return _fuse(g, list(g.neighbors(v))[0], v)


def check_cc_degree1(g: BaseGraph[VT, ET], v: VT) -> bool:
    """Check if degree-1 spider is connected by Hadamard edge."""
    if g.vertex_degree(v) != 1:
        return False
    neighbor = list(g.neighbors(v))[0]
    edge = g.edge(v, neighbor)
    return g.edge_type(edge) == EdgeType.HADAMARD


def cc_degree1(g: BaseGraph[VT, ET], v: VT) -> bool:
    """Color change a degree-1 spider connected by Hadamard edge."""
    if not check_cc_degree1(g, v):
        return False
    return _color_change(g, v)


def degree1_simp(g: BaseGraph[VT, ET]) -> None:
    """Fuses and color-change degree-1 spiders."""
    changed = True

    while changed:
        changed = False
        vertices = list(g.vertices())

        for v in vertices:
            if v not in g.vertices():
                continue

            if check_fuse_degree1(g, v):
                if fuse_degree1(g, v):
                    changed = True
                    break
            elif check_cc_degree1(g, v):
                if cc_degree1(g, v):
                    changed = True
                    break


def compress_rows(g: BaseGraph[VT, ET]) -> None:
    """Reposition vertices to compress the graph visualization.

    Changes the row positions of vertices so that the representation is as
    compact as possible, without having vertices share row positions unless
    they originally shared the same row index.
    """
    # Collect all vertices with their current row positions
    row_to_vertices = {}
    for v in g.vertices():
        original_row = g.row(v)
        if original_row not in row_to_vertices:
            row_to_vertices[original_row] = []
        row_to_vertices[original_row].append(v)

    # Sort the unique row values
    sorted_rows = sorted(row_to_vertices.keys())

    # Assign new compact row indices
    # For each original row group, assign the next available row index
    new_row = 0
    for original_row in sorted_rows:
        vertices_at_row = row_to_vertices[original_row]
        # All vertices that originally shared a row get the same new row
        for v in vertices_at_row:
            g.set_row(v, new_row)
        # Move to next row index
        new_row += 1
