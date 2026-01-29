from itertools import combinations

import networkx as nx
from pysat.card import CardEnc, EncType
from pysat.solvers import Glucose3


def match_path_ends_to_marked_edges(
        G: nx.Graph,
        path_cover: list[list[int]],
        markings: dict[tuple[int, int], int]
) -> dict[int, int]:
    """
     matches path endpoints to incident marked edges using a maximum bipartite matching.

    Args:
        G: The original graph.
        path_cover: List of paths (each path is a list of node IDs).
        markings: Dictionary mapping edges (tuples) to number of marks.

    Returns:
        A dictionary mapping path end nodes to their assigned marked edge.
        {end_node: (u, v)}
    """

    # 1. Identify all end nodes
    end_nodes = []
    path_edges = set()

    for path in path_cover:
        if not path:
            continue
        # Both ends are endpoints
        # Note: In a path cover, a path could be a single isolated node (len=1)
        if len(path) == 1:
            end_nodes.append(path[0])
        else:
            # ONLY add the "end" node (path[-1])
            end_nodes.append(path[-1])

            # Collect path edges to exclude them
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                path_edges.add(tuple(sorted((u, v))))

    end_nodes_set = set(end_nodes)

    # 2. Identify relevant marked edges
    # Filter markings to those > 0 and NOT in path_edges?
    # Usually markings on path edges are for SWAPs or similar, markings OFF path are interactions.
    # The requirement is "dictionary of end nodes to edges with markings".
    # We'll assume we want the off-path edges.

    relevant_marked_edges = []
    for edge, marks in markings.items():
        if marks > 0:
            u, v = sorted(edge)
            if (u, v) not in path_edges:
                relevant_marked_edges.append((u, v))

    # 3. Build Bipartite Graph
    B = nx.Graph()

    # Left Nodes: End Nodes (integers)
    # Right Nodes: Marked Edges (tuples)
    # To avoid ID collision if edges are somehow integers (unlikely), we keeps types distinct.
    # But networkx nodes can be anything hashable.

    B.add_nodes_from(end_nodes, bipartite=0)
    B.add_nodes_from(relevant_marked_edges, bipartite=1)

    for edge in relevant_marked_edges:
        u, v = edge
        # If u is an end node, add edge in B
        if u in end_nodes_set:
            B.add_edge(u, edge)
        # If v is an end node, add edge in B
        if v in end_nodes_set:
            B.add_edge(v, edge)

    # 4. Compute Maximum Matching
    # top_nodes = end_nodes
    try:
        matching = nx.bipartite.maximum_matching(B, top_nodes=end_nodes)
    except:
        # Fallback for empty graph or other issues
        matching = {}

    # 5. Extract result
    # matching contains both u->v and v->u. We only want end_node -> neighbor
    result = {}
    for node in end_nodes:
        if node in matching:
            # edge is (u, v) where one is node
            edge = matching[node]
            neighbor = edge[0] if edge[1] == node else edge[1]
            result[node] = neighbor

    return result


def find_all_path_covers(G: nx.Graph, n_paths: int | None = None):
    """
    Uses a SAT solver (Glucose3) to find all unique undirected path covers.
    - Constraints: Degree 1 or 2 for every vertex (covers all vertices).
    - Lazy Logic: Eliminates cycles only when they are proposed by the solver.

    Args:
        G: NetworkX graph to find path covers for.
        n_paths: If specified, only return covers with exactly this many paths.
                   Use n_paths=1 to find only Hamiltonian paths.

    Returns:
        List of path covers, where each cover is a list of paths.
        Each path is an ordered list of nodes from one endpoint to the other.
    """
    num_vertices = G.number_of_nodes()
    nodes = list(G.nodes())

    # 1. Map edges to SAT variables
    undirected_edges = sorted([tuple(sorted(e)) for e in G.edges()])
    edge_to_id = {e: i + 1 for i, e in enumerate(undirected_edges)}
    id_to_edge = {i + 1: e for i, e in enumerate(undirected_edges)}
    num_edge_vars = len(undirected_edges)

    solver = Glucose3()

    # 2. Cardinality constraint: For k paths on n vertices, we need n-k edges
    # If n_paths is specified, require exactly n - n_paths edges
    if n_paths is not None:
        target_edges = num_vertices - n_paths

        # Impossible to have more paths than vertices (requires negative edges)
        if target_edges < 0:
            return

        # Equals constraint: exactly target_edges must be true
        edge_vars = list(range(1, num_edge_vars + 1))
        equals_clauses = CardEnc.equals(
            lits=edge_vars, bound=target_edges,
            top_id=num_edge_vars, encoding=EncType.seqcounter
        )
        for clause in equals_clauses:
            solver.add_clause(clause)

    # 3. Degree Constraints: Every vertex degree must be 1 or 2
    # This ensures a Path Cover (collection of paths touching every vertex)
    for node in nodes:
        incident_vars = [edge_to_id[e] for e in undirected_edges if node in e]

        # Coverage: At least one edge per vertex
        if incident_vars:
            solver.add_clause(incident_vars)
        else:
            # If a vertex has no edges, it's an isolated path of length 0 (node-only)
            # Standard path cover theory allows this, but we'll assume edges exist here.
            pass

        # Structure: At most two edges per vertex (prevents forks/junctions)
        if len(incident_vars) >= 3:
            for combo in combinations(incident_vars, 3):
                solver.add_clause([-v for v in combo])

    N = 0

    # 4. Solving Loop
    while solver.solve():
        model = solver.get_model()
        # Filter to only edge variables (auxiliary vars from cardinality encoding are > num_edge_vars)
        selected_edges = [id_to_edge[idx] for idx in model if idx > 0 and idx <= num_edge_vars]

        # Lazy Cycle Elimination
        G_cover = nx.Graph(selected_edges)
        try:
            cycle = nx.find_cycle(G_cover)
            # Cycle found: block this specific set of edges (forming the cycle)
            cycle_edge_ids = [edge_to_id[tuple(sorted((u, v)))] for u, v in cycle]
            solver.add_clause([-eid for eid in cycle_edge_ids])
            continue
        except nx.NetworkXNoCycle:
            # Valid Path Cover Found!
            # Add isolated nodes to get accurate component count
            G_cover.add_nodes_from(nodes)
            num_paths = nx.number_connected_components(G_cover)

            # Convert edges to ordered paths directly
            paths = []
            for component_nodes in nx.connected_components(G_cover):
                subgraph = G_cover.subgraph(component_nodes)
                endpoints = [n for n, deg in subgraph.degree() if deg == 1]

                if len(endpoints) >= 2:
                    # Normal path: traverse from one endpoint to the other
                    path = nx.shortest_path(subgraph, source=endpoints[0], target=endpoints[1])
                elif len(endpoints) == 0:
                    # Isolated node (no edges)
                    path = list(component_nodes)
                else:
                    # Single endpoint (shouldn't happen in valid path cover)
                    path = list(component_nodes)
                paths.append(path)

            N += 1
            yield paths

            # Block this exact combination of edges to find the next unique solution
            # We need to block THIS EXACT SET, not just "at least one different"
            # because blocking selected edges would also block supersets.
            # The blocking clause: NOT(all selected AND none of unselected)
            # = (NOT all selected) OR (at least one unselected becomes selected)
            # = (-e1 OR -e2 OR ... OR -ek) OR (u1 OR u2 OR ... OR um)
            # where e_i are selected and u_j are unselected
            selected_edge_ids = [edge_to_id[e] for e in selected_edges]
            unselected_edge_ids = [eid for eid in range(1, num_edge_vars + 1)
                                   if eid not in selected_edge_ids]
            # Block clause: at least one selected edge is removed OR at least one new edge added
            blocking_clause = [-eid for eid in selected_edge_ids] + unselected_edge_ids
            solver.add_clause(blocking_clause)

            # Safety cap to avoid hanging on massive graphs
            if N >= 1000:
                print("[!] Capped at 1000 solutions.")
                break
