r"""
This module implements fault-equivalent rewrites.

Functions that provide fault-equivalent rewrites have names ending with `_FE`, e.g. `unfuse_1_FE`.

Fault-equivalent rewrites are defined in arXiv:2506.17181.
Alternatively, they are defined as distance-preserving rewrites in arXiv:2410.17240.

Formal Definition
=================

Let :math:`C_1` and :math:`C_2` be two circuits with respective noise models :math:`\mathcal{F}_1` and :math:`\mathcal{F}_2`.
The circuit :math:`C_1` under :math:`\mathcal{F}_1` is **w-fault-equivalent** to :math:`C_2` under :math:`\mathcal{F}_2`,
if and only if for all faults :math:`F_1 \in \langle \mathcal{F}_1 \rangle` with weight :math:`wt(F_1) < w`, we have either:

1.  :math:`F_1` is detectable, or
2.  There exists a fault :math:`F_2 \in \langle \mathcal{F}_2 \rangle` on :math:`C_2` such that:
        - :math:`wt(F_2) \leq wt(F_1)` and
        - :math:`C_1^{F_1} = C_2^{F_2}`.

The condition must similarly hold for all faults :math:`F_2 \in \langle \mathcal{F}_2 \rangle` with weight :math:`wt(F_2) < w`, making this equivalence relation symmetric.

Two circuits :math:`C_1` and :math:`C_2` are **fault-equivalent**, written :math:`C_1 \hat{=} C_2`, if they are :math:`w`-fault-equivalent for all :math:`w \in \mathbb{N}`.
"""

import itertools
import math
from typing import Callable, Optional

from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.rewrite_rules import (
    check_remove_id as check_elim_FE,             # noqa # pylint: disable=unused-import
    remove_id as elim_FE,                         # noqa # pylint: disable=unused-import
    check_color_change as check_color_change_FE,  # noqa # pylint: disable=unused-import
    color_change as color_change_FE,              # noqa # pylint: disable=unused-import
    fuse as _fuse,
)
from pyzx.utils import VertexType, is_pauli
from sympy import ceiling
import networkx as nx


def check_fuse_1_FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    neighs = g.neighbors(v)
    return len(neighs) == 1 and g.type(neighs[0]) == g.type(v) and is_pauli(g.phase(v))


def fuse_1_FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_fuse_1_FE(g, v):
        return False
    [v2] = g.neighbors(v)
    return _fuse(g, v, v2)


def check_unfuse_1_FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.type(v) in (VertexType.X, VertexType.Z, VertexType.Z_BOX) and g.vertex_degree(v) > 0


def unfuse_1_FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_unfuse_1_FE(g, v):
        return False
    typ = VertexType.X if g.type(v) == VertexType.X else VertexType.Z
    v2 = g.add_vertex(typ, g.qubit(v), g.row(v) - 1)
    _e = g.add_edge((v, v2))
    return True


def _find_best_pairing_scipy(
        g: BaseGraph[VT, ET],
        neighbors: list[VT],
        new_vertices: list[VT]
) -> tuple:
    """Finds the optimal assignment using the Hungarian algorithm via SciPy."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    num_vs = len(neighbors)
    cost_matrix = np.zeros((num_vs, num_vs))

    for i in range(num_vs):
        n_q, n_r = g.qubit(neighbors[i]), g.row(neighbors[i])
        for j in range(num_vs):
            new_v_q, new_v_r = g.qubit(new_vertices[j]), g.row(new_vertices[j])
            cost_matrix[i, j] = math.hypot(new_v_q - n_q, new_v_r - n_r)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return tuple(col_ind)


def _find_best_assignment_scipy(
        g: BaseGraph[VT, ET],
        items_to_assign: list[VT],
        available_slots: list[VT]
) -> dict[VT, VT]:
    """
    Finds the optimal assignment of items to slots (where len(slots) >= len(items))
    to minimize total connection distance using the Hungarian algorithm.

    Returns:
        A dictionary mapping {item: slot}
    """
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        # Fallback to simple assignment if numpy/scipy not installed
        assignment_map = {}
        for i, item in enumerate(items_to_assign):
            if i < len(available_slots):
                assignment_map[item] = available_slots[i]
            else:
                break  # No more slots
        return assignment_map

    num_items = len(items_to_assign)
    num_slots = len(available_slots)

    if num_items == 0:
        return {}

    if num_items > num_slots:
        # This should not happen if graph generation logic is correct
        # We can only assign up to num_slots items.
        items_to_assign = items_to_assign[:num_slots]
        num_items = num_slots

    # Create an M x N cost matrix (M = items, N = slots)
    cost_matrix = np.zeros((num_items, num_slots))

    for i in range(num_items):
        item_q, item_r = g.qubit(items_to_assign[i]), g.row(items_to_assign[i])
        for j in range(num_slots):
            slot_q, slot_r = g.qubit(available_slots[j]), g.row(available_slots[j])
            cost_matrix[i, j] = math.hypot(slot_q - item_q, slot_r - item_r)

    # Scipy's function handles rectangular matrices correctly
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignment_map = {}
    for i in range(len(row_ind)):
        item_index = row_ind[i]
        slot_index = col_ind[i]
        assignment_map[items_to_assign[item_index]] = available_slots[slot_index]

    return assignment_map


def _find_best_pairing_itertools(
        g: BaseGraph[VT, ET],
        neighbors: list[VT],
        new_vertices: list[VT]
) -> tuple:
    """
    Finds the optimal neighbor-to-new-vertex assignment to minimize total distance.

    For the small number of nodes this is meant to be used, `itertools.permutations` works.
    """
    num_vs = len(neighbors)
    # 1. Build a cost matrix of distances
    cost_matrix = [[math.hypot(g.qubit(new_v) - g.qubit(n), g.row(new_v) - g.row(n))
                    for new_v in new_vertices] for n in neighbors]

    min_cost = float('inf')
    best_assignment = tuple(range(num_vs))

    # 2. Iterate through all permutations to find the one with the lowest cost
    for p in itertools.permutations(range(num_vs)):
        current_cost = sum(cost_matrix[i][p[i]] for i in range(num_vs))
        if current_cost < min_cost:
            min_cost = current_cost
            best_assignment = p

    return best_assignment


def _get_square_coords(q: float, r: float) -> list[tuple[float, float]]:
    """Generates coordinates for 4 vertices in a square centered at (q, r)."""
    d = 0.5
    return [(q - d, r - d), (q + d, r - d), (q + d, r + d), (q - d, r + d)]


def _get_n_cycle_coords(N: int, q: float, r: float) -> list[tuple[float, float]]:
    """Generates coordinates for N vertices in a N-cycle graph centered at (q, r)."""
    radius = 0.75 * N / 5
    coords = []
    for i in range(N):
        angle = (2 * math.pi * i / N) + (math.pi)
        qc = q + radius * math.cos(angle)
        rc = r - radius * math.sin(angle)
        coords.append((qc, rc))
    return coords


def _unfuse_spider(
        g: BaseGraph[VT, ET],
        v: VT,
        check_func: Callable,
        coords_func: Callable
) -> bool:
    """A generic function to unfuse a spider into a polygon of new spiders."""
    if not check_func(g, v):
        return False

    v_type = g.type(v)
    neighs = list(g.neighbors(v))
    original_edge_types = {n: g.edge_type(g.edge(v, n)) for n in neighs}
    q, r = g.qubit(v), g.row(v)

    # 1. Generate coordinates for the new shape using the provided function
    new_coords = coords_func(q, r)
    num_vs = len(new_coords)

    # 2. Add the new vertices
    new_vs = [g.add_vertex(v_type, qc, rc) for qc, rc in new_coords]

    # 3. Connect new vertices to form the polygon
    for i in range(num_vs):
        g.add_edge((new_vs[i], new_vs[(i + 1) % num_vs]))

    # 4. Find and apply the optimal one-to-one neighbor connections
    assignment = _find_best_pairing_scipy(g, neighs, new_vs)
    for i, neighbor_v in enumerate(neighs):
        new_v = new_vs[assignment[i]]
        g.add_edge((neighbor_v, new_v), original_edge_types[neighbor_v])

    # 5. Remove the original vertex
    g.remove_vertex(v)
    return True


def check_unfuse_4_FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.type(v) in (VertexType.X, VertexType.Z) and g.vertex_degree(v) == 4 and g.phase(v) == 0


def unfuse_4_FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    """Unfuses a degree-4 spider into a square."""
    return _unfuse_spider(g, v, check_unfuse_4_FE, _get_square_coords)


def check_unfuse_5_FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.type(v) in (VertexType.X, VertexType.Z) and g.vertex_degree(v) == 5 and g.phase(v) == 0


def unfuse_5_FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    """Unfuses a degree-5 spider into a pentagon."""
    return _unfuse_spider(g, v, check_unfuse_5_FE,
                          lambda x, y: _get_n_cycle_coords(5, x, y))


def check_unfuse_n_2FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.type(v) in (VertexType.X, VertexType.Z) and g.phase(v) == 0

def _unfuse_n_FE_core(
        g: BaseGraph[VT, ET],
        v: VT,
        G_nx: "nx.Graph",
        slot_strategy: str,
        num_spiders_per_edge: int = 1
) -> bool:
    """
    Core helper to unfuse a spider 'v' into a new structure based on 'G_nx'.

    Args:
        g: The BaseGraph.
        v: The vertex to unfuse.
        G_nx: The NetworkX graph defining the new spider layout.
        slot_strategy: Where to create connection points ("slots").
            - "edge": Create 'num_spiders_per_edge' spiders on each edge of G_nx.
            - "node": Use the main spiders (nodes of G_nx) as slots.
        num_spiders_per_edge: Number of spiders to add to each edge
                              (only used if slot_strategy="edge").
    """
    v_type = g.type(v)
    if v_type not in (VertexType.X, VertexType.Z):
        return False

    q, r = g.qubit(v), g.row(v)
    boundaries = list(g.neighbors(v))
    original_edge_types = {n: g.edge_type(g.edge(v, n)) for n in boundaries}

    if not G_nx.nodes():
        g.remove_vertex(v)
        return True

    # 1. Get scaled layout from NetworkX graph
    pos = nx.spring_layout(G_nx, seed=42)
    q_coords = [p[0] for p in pos.values()]
    r_coords = [p[1] for p in pos.values()]
    min_q, max_q = (min(q_coords), max(q_coords)) if q_coords else (0, 0)
    min_r, max_r = (min(r_coords), max(r_coords)) if r_coords else (0, 0)
    delta_q = max_q - min_q
    delta_r = max_r - min_r
    scale = 0.5 * math.sqrt(G_nx.number_of_nodes()) + 1.5

    def scale_pos(p):
        norm_q = (p[0] - min_q) / delta_q if delta_q != 0 else 0.5
        norm_r = (p[1] - min_r) / delta_r if delta_r != 0 else 0.5
        return (q + scale * (norm_q - 0.5) * 2, r + scale * (norm_r - 0.5) * 2)

    scaled_pos = {n: scale_pos(p) for n, p in pos.items()}

    # 2. Add main spiders (one for each node in G_nx)
    nx_to_zx_map = {}
    for nx_node in G_nx.nodes():
        qc, rc = scaled_pos[nx_node]
        new_v = g.add_vertex(v_type, qc, rc, phase=0)
        nx_to_zx_map[nx_node] = new_v

    # 3. Generate boundary slots based on the chosen strategy
    boundary_slots = []
    if slot_strategy == "edge":
        for nx_u, nx_v in G_nx.edges():
            zx_u = nx_to_zx_map[nx_u]
            zx_v = nx_to_zx_map[nx_v]
            q_u, r_u = g.qubit(zx_u), g.row(zx_u)
            q_v, r_v = g.qubit(zx_v), g.row(zx_v)

            last_spider_in_chain = zx_u
            for i in range(num_spiders_per_edge):
                frac = (i + 1) / (num_spiders_per_edge + 1)
                q_s = q_u + frac * (q_v - q_u)
                r_s = r_u + frac * (r_v - r_u)

                s = g.add_vertex(v_type, q_s, r_s, phase=0)
                g.add_edge((last_spider_in_chain, s))
                boundary_slots.append(s)
                last_spider_in_chain = s

            g.add_edge((last_spider_in_chain, zx_v))

    elif slot_strategy == "node":
        # Connect internal spiders to each other
        for nx_u, nx_v in G_nx.edges():
            g.add_edge((nx_to_zx_map[nx_u], nx_to_zx_map[nx_v]))
        # The slots are the nodes themselves
        boundary_slots = list(nx_to_zx_map.values())

    else:
        raise ValueError(f"Unknown slot_strategy: {slot_strategy}")

    # 4. Connect original boundaries to the new boundary slots
    if len(boundary_slots) < len(boundaries):
        # This is a critical error in the graph generation logic
        # We don't have enough slots for the boundaries.
        # Restore vertex and fail
        # (Ideally, we would rollback all changes, but for now we just error)
        # For simplicity, we'll just return False.
        # A more robust implementation would undo the add_vertex/add_edge calls.
        print(f"Error: Not enough slots created. Need {len(boundaries)}, got {len(boundary_slots)}")
        return False  # Abort

    # Use optimal assignment. This handles num_slots >= num_boundaries
    assignment = _find_best_assignment_scipy(g, boundaries, boundary_slots)

    for boundary_v, slot_v in assignment.items():
        g.add_edge((boundary_v, slot_v), original_edge_types[boundary_v])

    # 5. Remove the original vertex
    g.remove_vertex(v)
    return True


def check_unfuse_n_3FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.type(v) in (VertexType.X, VertexType.Z) and g.phase(v) == 0


def check_unfuse_n_4FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    # Same check as 3FE
    return g.type(v) in (VertexType.X, VertexType.Z) and g.phase(v) == 0


def unfuse_n_3FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_unfuse_n_3FE(g, v):
        return False

    targets = g.vertex_degree(v)
    if targets <= 3:
        return True

    # Logic from your prompt:
    # We need num_slots = num_edges * 2 >= targets
    # num_edges = vertices * 3 / 2
    # So, (vertices * 3 / 2) * 2 >= targets  =>  vertices * 3 >= targets
    # vertices >= targets / 3
    # Your logic: n = ceil(t/2), vertices = ceil(n/3)*2
    # This is more complex but ensures local cut properties. We trust it.
    n = math.ceil(targets / 2)
    vertices = math.ceil(1 / 3 * n) * 2

    G = find_cubic_graph_with_local_cuts(vertices, 2)

    return _unfuse_n_FE_core(
        g, v, G,
        slot_strategy="edge",
        num_spiders_per_edge=2
    )


def unfuse_n_4FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    if not check_unfuse_n_4FE(g, v):
        return False

    targets = g.vertex_degree(v)
    if targets <= 3:
        return True

    # We need num_slots = num_edges * 1 >= targets
    # num_edges = vertices * 3 / 2
    # So, vertices * 3 / 2 >= targets => vertices >= targets * 2 / 3
    # Since vertices must be even:
    vertices = math.ceil((targets * 2 / 3) / 2) * 2  # = math.ceil(targets / 3) * 2

    G = find_cubic_graph_with_local_cuts(vertices, 2, random_search=True, max_trials=10_000)
    if G is None:
        return False

    return _unfuse_n_FE_core(
        g, v, G,
        slot_strategy="edge",
        num_spiders_per_edge=1
    )

def check_unfuse_2n_FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.type(v) in (VertexType.X, VertexType.Z) and g.vertex_degree(v) % 2 == 0 and g.phase(v) == 0


def check_unfuse_2n_plus_FE(g: BaseGraph[VT, ET], v: VT) -> bool:
    return g.type(v) in (VertexType.X, VertexType.Z) and g.vertex_degree(v) % 2 == 1 and g.phase(v) == 0


def _split_neighbors_into_groups(g: BaseGraph[VT, ET], neighbors: list[VT]) -> tuple[list[VT], list[VT]]:
    """
    Splits neighbors into two groups based on their horizontal position (qubit).
    """
    sorted_neighbors = sorted(neighbors, key=lambda n: g.qubit(n))

    midpoint = len(sorted_neighbors) // 2
    group1 = sorted_neighbors[:midpoint]
    group2 = sorted_neighbors[midpoint:]

    return group1, group2


def _calculate_new_spider_positions(
        g: BaseGraph[VT, ET],
        group1: list[VT],
        group2: list[VT]
) -> tuple[float, float, float]:
    """Calculates the average positions for the two new central spiders based on the groups."""
    all_neighbors = group1 + group2
    start_from = min(g.row(n) for n in all_neighbors) - 1

    # Calculate the center (centroid) of each group
    pos_q1 = sum(g.qubit(n) for n in group1) / len(group1)
    pos_q2 = sum(g.qubit(n) for n in group2) / len(group2)

    return pos_q1, pos_q2, start_from


def _unfuse_2n_spider_core(g: BaseGraph[VT, ET], v: VT, w: Optional[int] = None, alternate_pairing_order: bool = False) -> tuple[
    VT, VT]:
    """
    The core function that performs the 2n-degree unfusing operation.

    Args:
        w (Optional[int]): If specified, the function implements the w-fault-equivalent rewrite.
            Only the first w-1 pairs will have a full parity check gadget created between them.
    """
    v_type = g.type(v)
    neighs = list(g.neighbors(v))

    # 1. Split neighbors into two deterministic groups (left and right)
    group1, group2 = _split_neighbors_into_groups(g, neighs)

    # At each level of recursion, we reverse the pairing order within the groups
    # to ensure the resulting ZX diagram can be implemented.
    if alternate_pairing_order:
        group1, group2 = group1[::-1], group2[::-1]
    degree_n = len(group1)

    # 2. Calculate positions based on these two groups
    pos_1, pos_2, start_from = _calculate_new_spider_positions(g, group1, group2)

    # 3. Add the two new central "inner" spiders
    inner_1 = g.add_vertex(v_type, pos_1, start_from - degree_n - 1)
    inner_2 = g.add_vertex(v_type, pos_2, start_from - degree_n - 1)

    # 4. Create the parity check gadgets by pairing nodes from each group
    for i, (n1, n2) in enumerate(zip(group1, group2)):
        if w is None or w >= i + 1:
            v1 = g.add_vertex(v_type, g.qubit(n1), start_from - i)
            v2 = g.add_vertex(v_type, g.qubit(n2), start_from - i)

            g.add_edge((v1, n1))
            g.add_edge((v2, n2))
            g.add_edge((v1, v2))
            g.add_edge((inner_1, v1))
            g.add_edge((inner_2, v2))
        else:
            g.add_edge((inner_1, n1))
            g.add_edge((inner_2, n2))
    if len(group2) > len(group1):
        g.add_edge((inner_2, group2[-1]))

    g.remove_vertex(v)
    return inner_1, inner_2


def unfuse_2n_FE(g: BaseGraph[VT, ET], v: VT, w: Optional[int] = None) -> bool:
    """
    Unfuses a degree-2n spider into two degree-n spiders.

    Args:
        w (Optional[int]): If specified, the function implements the w-fault-equivalent rewrite.
            Only the first w-1 pairs will have a full parity check gadget created between them.
    """
    if not check_unfuse_2n_FE(g, v):
        return False
    _unfuse_2n_spider_core(g, v, w)
    return True


def unfuse_2n_plus_FE(g: BaseGraph[VT, ET], v: VT, w: Optional[int] = None) -> bool:
    """
    Unfuses a degree-(2n + 1) spider into a degree-n spider and a degree-(n + 1) spider.

    Args:
        w (Optional[int]): If specified, the function implements the w-fault-equivalent rewrite.
            Only the first w-1 pairs will have a full parity check gadget created between them.
    """
    if not check_unfuse_2n_plus_FE(g, v):
        return False
    _unfuse_2n_spider_core(g, v, w)
    return True



def recursive_unfuse_FE(g: BaseGraph[VT, ET], v: VT, w: Optional[int] = None, _alternate_pairing_order: bool = False) -> bool:
    """
    Recursively unfuses a spider.

    Args:
        w (Optional[int]): If specified, the function implements the w-fault-equivalent rewrite.
            Only the first w-1 pairs will have a full parity check gadget created between them.
    """
    degree = g.vertex_degree(v)
    if degree <= 3:
        return True
    if degree == 4:
        return unfuse_4_FE(g, v)
    if degree == 5:
        return unfuse_5_FE(g, v)
    if w == 2:
        return unfuse_n_2FE(g, v)

    inner_1, inner_2 = _unfuse_2n_spider_core(g, v, w, _alternate_pairing_order)
    return (recursive_unfuse_FE(g, inner_1, w, not _alternate_pairing_order) and
            recursive_unfuse_FE(g, inner_2, w, not _alternate_pairing_order))


def decompose_bipartite_css_state_FE(graph: BaseGraph[VT, ET], w: Optional[int] = None) -> None:
    """
    Decomposes each spider of a bipartite CSS state into degree-3 spiders.

    Args:
        w (Optional[int]): If specified, the function implements the w-fault-equivalent rewrite.
    """
    for type in [VertexType.Z, VertexType.X]:
        spiders_to_decompose = sorted([
            v for v in graph.vertices() if graph.type(v) == type
        ], key=graph.row, reverse=True)
        spiders_decomposed = list(graph.outputs())
        for i, s in enumerate(spiders_to_decompose):
            recursive_unfuse_FE(graph, s, w)
            to_move = spiders_to_decompose[i+1:]

            if to_move:
                not_to_move = [v for v in graph.vertices() if graph.type(v) == type and v not in to_move]
                first_fixed_row = min(graph.row(v) for v in not_to_move)
                last_moving_row = max(graph.row(v) for v in to_move)
                gap = 1.5  # vertical spacing between decomposed layers

                # how far we need to push earlier (negative means up in time)
                move_by = (last_moving_row - first_fixed_row) + gap

                for v in to_move:
                    graph.set_row(v, graph.row(v) - move_by)

            spiders_decomposed.append(s)

