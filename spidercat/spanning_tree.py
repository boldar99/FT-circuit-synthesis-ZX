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
    Matches nodes to marked edges using a randomized greedy strategy.

    Strategy:
    1. Leaves get 'first dibs' on adjacent marked edges.
    2. Internal nodes fill in any remaining required marks.
    """

    # 1. Setup Data Structures
    # Track how many marks are left to be filled for each edge
    remaining_marks = {e: count for e, count in markings.items() if count > 0}

    # Build a quick lookup: Node -> List of adjacent marked edges
    # This lets us quickly see which markings a node *could* satisfy
    node_to_marked_edges = {n: [] for n in forest.nodes()}
    for (u, v) in remaining_marks:
        if u in node_to_marked_edges:
            node_to_marked_edges[u].append((u, v))
        if v in node_to_marked_edges:
            node_to_marked_edges[v].append((u, v))

    matches = defaultdict(list)

    # 2. Identify Nodes
    leaves = [n for n, d in forest.degree() if d <= 1]
    internal = [n for n, d in forest.degree() if d > 1]

    # Helper function to perform the matching logic until exhaustion
    def try_match_nodes(candidate_nodes):
        while True:
            progress = False
            for node in candidate_nodes:
                # Find valid adjacent markings that still need filling
                options = [
                    edge for edge in node_to_marked_edges[node]
                    if remaining_marks.get(edge, 0) > 0
                ]

                if not options:
                    continue

                chosen_edge = options[0]  # Can be randomized
                matches[node].append(chosen_edge)

                remaining_marks[chosen_edge] -= 1
                if remaining_marks[chosen_edge] == 0:
                    del remaining_marks[chosen_edge]

                progress = True

            # If we went through the whole group and found nothing to do, stop.
            if not progress:
                break

    # 3. Execute Phases
    # Phase 1: Leaves consume everything they can
    try_match_nodes(leaves)
    try_match_nodes(internal)

    return dict(matches)


import networkx as nx


def analyze_forest_metrics(forest: nx.Graph):
    """
    Computes diameter and node eccentricities for every component in the forest.

    Returns:
        component_map (dict): {node_id: component_id}
        diameters (dict): {component_id: diameter_int}
        eccentricities (dict): {node_id: eccentricity_int}
    """
    component_map = {}
    diameters = {}
    eccentricities = {}

    # Iterate over each tree in the forest
    for i, component_nodes in enumerate(nx.connected_components(forest)):
        # Create a view/subgraph for calculation
        tree = forest.subgraph(component_nodes)

        # Calculate all eccentricities for this tree at once
        # (nx.eccentricity is efficient for trees)
        eccs = nx.eccentricity(tree)

        # The diameter of a tree is simply the max eccentricity
        diam = nx.diameter(tree, e=eccs)

        # Store metrics
        # We use the min(node) as a stable ID for the component
        comp_id = min(component_nodes)
        diameters[comp_id] = diam

        for node in component_nodes:
            component_map[node] = comp_id
            eccentricities[node] = eccs[node]

    return component_map, diameters, eccentricities


def find_best_merge_edge(
        G: nx.Graph,
        forest: nx.Graph,
        markings: dict[tuple[int, int], int],
        metrics: tuple
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
    comp_map, diameters, eccs = metrics

    best_edge = None
    min_new_diameter = float('inf')
    min_mark_cost = float('inf')

    # Iterate over all possible edges in the full graph
    for u, v in G.edges():
        # 1. Skip if edge already exists in forest
        if forest.has_edge(u, v):
            continue

        # 2. Check component connectivity
        comp_u = comp_map[u]
        comp_v = comp_map[v]

        # Skip if they are already in the same tree (would form a cycle)
        if comp_u == comp_v:
            continue

        # 3. Calculate Potential Diameter
        # Formula: max(diam(T1), diam(T2), ecc(u) + 1 + ecc(v))
        diam_u = diameters[comp_u]
        diam_v = diameters[comp_v]

        potential_diameter = max(
            diam_u,
            diam_v,
            eccs[u] + 1 + eccs[v]
        )

        # 4. Selection Logic (Minimizing Diameter)
        # We get the mark count for tie-breaking
        mark_count = markings.get((u, v))
        if mark_count is None:
            mark_count = markings.get((v, u), 0)

        # Update best found
        if potential_diameter < min_new_diameter:
            min_new_diameter = potential_diameter
            best_edge = (u, v)
            min_mark_cost = mark_count
        elif potential_diameter == min_new_diameter:
            # Tie-breaker: choose edge with fewer marks
            if mark_count < min_mark_cost:
                best_edge = (u, v)
                min_mark_cost = mark_count

    return best_edge, min_new_diameter


def build_min_diameter_spanning_tree(
        G: nx.Graph,
        initial_forest: nx.Graph,
        markings: dict[tuple[int, int], int]
) -> nx.Graph:
    """
    Iteratively merges trees in the forest by selecting edges that
    minimize diameter growth, until a single spanning tree remains.
    """
    # Work on a copy to avoid mutating the original forest input
    current_forest = initial_forest.copy()

    while True:
        # Check if fully connected (1 component)
        if nx.number_connected_components(current_forest) == 1:
            break

        # 1. Refresh Metrics (O(N))
        metrics = analyze_forest_metrics(current_forest)

        # 2. Find Best Merge (O(E_candidates))
        best_edge, new_diam = find_best_merge_edge(G, current_forest, markings, metrics)

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
    nx.draw(grf, pos=pos)
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
