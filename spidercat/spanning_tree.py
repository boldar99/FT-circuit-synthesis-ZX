import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mypy.checkexpr import defaultdict
from networkx.utils import UnionFind

from spidercat.draw import draw_spanning_forest_solution
from spidercat.utils import ed


def build_trivial_spanning_forest(G: nx.Graph, M: dict[tuple[int, int], int]) -> nx.Graph:
    """
    Creates a forest (collection of trees) graph using only unmarked edges.
    """
    forest = nx.Graph()
    forest.add_nodes_from(G.nodes())
    ds = UnionFind(G.nodes())
    for u, v in G.edges():
        if M.get(ed(u, v), 0) == 0:
            if ds[u] != ds[v]:
                forest.add_edge(u, v)
                ds.union(u, v)
    return forest


def match_forest_leaves_to_marked_edges(
        forest: nx.Graph,
        markings: dict[tuple[int, int], int]
) -> dict[int, list[tuple[int, int]]]:
    """
    Matches nodes to marked edges using Maximum Bipartite Matching for leaves,
    followed by a greedy extension for internal nodes.

    Strategy:
    1. Construct a bipartite graph: (Forest Leaves) <-> (Marked Edge Slots).
    2. Solve Maximum Bipartite Matching to prioritize satisfying leaves.
    3. Use internal nodes (and remaining leaves) to cover any remaining marks.
    """

    matches = defaultdict(list)
    leaves = {n for n, d in forest.degree() if d <= 1}

    # 2. Build Bipartite Graph
    B = nx.Graph()
    mark_slot_to_edge = {}

    for edge, count in markings.items():
        u, v = edge
        if count <= 0 or edge in forest.edges():
            continue

        # Create 'count' number of slots for this edge
        for i in range(count):
            # Unique ID for the bipartite node
            mark_node_id = f"mark_{edge}_{i}"
            mark_slot_to_edge[mark_node_id] = edge

            # Add to Bipartite Graph (Right side)
            B.add_node(mark_node_id, bipartite=1)

            # Add edges to adjacent LEAVES (Left side)
            if u in leaves:
                B.add_edge(u, mark_node_id)
            if v in leaves:
                B.add_edge(v, mark_node_id)

    # 3. Compute Maximum Bipartite Matching
    bipartite_leaves = [n for n in leaves if n in B]
    matching_result = nx.bipartite.maximum_matching(B, top_nodes=bipartite_leaves)

    remaining_markings = markings.copy()

    # 4. Process Matching Results (Phase 1)
    for node, matched_partner in matching_result.items():
        if node in leaves:
            mark_id = matched_partner
            edge = mark_slot_to_edge[mark_id]

            matches[node].append(edge)

            # Decrement the remaining count for this edge
            remaining_markings[edge] -= 1
            if remaining_markings[edge] == 0:
                del remaining_markings[edge]

    # 5. Extend with Internal Nodes (Phase 2)
    is_internal = {n: (forest.degree(n) > 1) for n in forest.nodes()}

    for edge, count in list(remaining_markings.items()):
        u, v = edge
        if count <= 0 or edge in forest.edges():
            continue

        for _ in range(count):
            candidates = []
            if u in forest.nodes: candidates.append(u)
            if v in forest.nodes: candidates.append(v)

            if not candidates:
                continue

            candidates.sort(key=lambda n: not is_internal.get(n, False))

            chosen_node = candidates[0]
            matches[chosen_node].append(edge)

    return dict(matches)


def analyze_forest_metrics(forest: nx.Graph):
    """
    Computes diameter and node eccentricities for every component in the forest.

    Returns:
        component_map (dict): {node_id: component_id}
        diameters (dict): {component_id: diameter_int}
        eccentricities (dict): {node_id: eccentricity_int}
        sizes (dict): {node_id: component_size}
    """
    component_map = {}
    diameters = {}
    eccentricities = {}
    sizes = {} # New: track number of nodes per component

    for component_nodes in nx.connected_components(forest):
        tree = forest.subgraph(component_nodes)
        eccs = nx.eccentricity(tree)
        diam = nx.diameter(tree, e=eccs)

        comp_id = min(component_nodes)
        diameters[comp_id] = diam
        sizes[comp_id] = len(component_nodes) # Store size

        for node in component_nodes:
            component_map[node] = comp_id
            eccentricities[node] = eccs[node]

    return component_map, diameters, eccentricities, sizes


def find_best_merge_edge(
        G: nx.Graph,
        forest: nx.Graph,
        markings: dict[tuple[int, int], int],
        metrics: tuple,
        balance_factor: float = 0.5
) -> tuple[tuple[int, int], int]:
    """
    Identifies the edge that minimizes the resulting tree diameter.

    Args:
        G: The full graph (source of valid edges).
        forest: The current spanning forest.
        markings: Used for tie-breaking (prefer lower marks).
        metrics: The output from analyze_forest_metrics.

    Returns:
        (best_edge, new_diameter)
    """
    comp_map, diameters, eccs, sizes = metrics

    best_edge = None
    min_score = float('inf')
    min_mark_cost = float('inf')

    for u, v in G.edges():
        # 1. Skip if edge already exists in forest
        if forest.has_edge(u, v):
            continue
        comp_u, comp_v = comp_map[u], comp_map[v]
        if comp_u == comp_v:
            continue

        # 1. Diameter Calculation
        potential_diameter = max(diameters[comp_u], diameters[comp_v], eccs[u] + 1 + eccs[v])

        # 2. Size Penalty
        # We penalize the creation of large components
        combined_size = sizes[comp_u] + sizes[comp_v]

        # 3. Composite Score
        # Adjusting the weight of combined_size balances the trees
        score = potential_diameter + (balance_factor * combined_size)

        mark_count = markings.get((u, v), markings.get((v, u), 0))

        if score < min_score:
            min_score = score
            best_edge = (u, v)
            min_mark_cost = mark_count
        elif score == min_score:
            if mark_count < min_mark_cost:
                best_edge = (u, v)
                min_mark_cost = mark_count

    return best_edge


def build_min_diameter_spanning_tree(
        G: nx.Graph,
        initial_forest: nx.Graph,
        markings: dict[tuple[int, int], int],
        max_num_trees: int = 1,
        balance_factor: float = 0.5,
) -> nx.Graph:
    """
    Iteratively merges trees in the forest by selecting edges that
    minimize diameter growth, until a single spanning tree remains.
    """
    # Work on a copy to avoid mutating the original forest input
    current_forest = initial_forest.copy()

    while True:
        # Check if fully connected (1 component)
        if nx.number_connected_components(current_forest) <= max_num_trees:
            break

        # 1. Refresh Metrics (O(N))
        metrics = analyze_forest_metrics(current_forest)

        # 2. Find Best Merge (O(E_candidates))
        best_edge = find_best_merge_edge(G, current_forest, markings, metrics, balance_factor)

        if best_edge is None:
            print("Warning: Graph is disconnected, cannot merge further.")
            break

        # 3. Merge
        current_forest.add_edge(*best_edge)
        # Optional: Print progress
        # print(f"Merged {best_edge}. New Max Diameter: {new_diam}")

    return current_forest


def find_min_height_roots(forest: nx.Graph) -> dict[int, int]:
    """
    Identifies the ideal root for each component in the forest to minimize tree height.

    Args:
        forest: The forest graph containing one or more trees.

    Returns:
        dict: {Tree_ID: Ideal_Root_Node}
        (Tree_ID is the minimum node ID in that component, used as a stable key)
    """
    ideal_roots = {}

    # Iterate over each distinct tree in the forest
    for component in nx.connected_components(forest):
        # Create a view of the single tree
        tree = forest.subgraph(component)

        # nx.center finds nodes with minimum eccentricity (min height)
        # A tree can have 1 or 2 centers.
        centers = nx.center(tree)

        # Tie-breaker: If there are 2 centers, pick the one with the lower ID
        # for deterministic behavior.
        best_root = min(centers)

        # Use the min node ID of the component as a stable identifier for the tree
        tree_id = min(component)
        ideal_roots[tree_id] = best_root

    return ideal_roots




if __name__ == "__main__":
    grf = nx.heawood_graph()
    pos = nx.kamada_kawai_layout(grf)
    from spidercat.markings import GraphMarker

    mrkr = GraphMarker(grf)
    M = mrkr.find_solution(T=6)

    forest = build_trivial_spanning_forest(grf, M)
    matchings = match_forest_leaves_to_marked_edges(forest, M)
    roots = find_min_height_roots(forest)
    draw_spanning_forest_solution(grf, forest, M, matchings, roots)


    spacing_tree = build_min_diameter_spanning_tree(grf, forest, M)
    matchings = match_forest_leaves_to_marked_edges(spacing_tree, M)
    roots = find_min_height_roots(spacing_tree)
    draw_spanning_forest_solution(grf, spacing_tree, M, matchings, roots)
    plt.show()
