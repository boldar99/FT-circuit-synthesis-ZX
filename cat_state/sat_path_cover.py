import networkx as nx
import matplotlib.pyplot as plt
from pysat.solvers import Glucose3
from pysat.card import CardEnc, EncType
from itertools import combinations
from collections import defaultdict, Counter
from cat_state.markings import GraphMarker

def find_all_path_covers(G: nx.Graph, max_paths: int = None):
    """
    Uses a SAT solver (Glucose3) to find all unique undirected path covers.
    - Constraints: Degree 1 or 2 for every vertex (covers all vertices).
    - Lazy Logic: Eliminates cycles only when they are proposed by the solver.

    Args:
        G: NetworkX graph to find path covers for.
        max_paths: If specified, only return covers with at most this many paths.
                   Use max_paths=1 to find only Hamiltonian paths.

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
    # If max_paths is specified, require at least n - max_paths edges
    if max_paths is not None:
        min_edges = num_vertices - max_paths
        # AtLeast constraint: at least min_edges must be true
        edge_vars = list(range(1, num_edge_vars + 1))
        atleast_clauses = CardEnc.atleast(
            lits=edge_vars, bound=min_edges,
            top_id=num_edge_vars, encoding=EncType.seqcounter
        )
        for clause in atleast_clauses:
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

    all_path_covers = []

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

            # Filter by max_paths if specified
            if max_paths is None or num_paths <= max_paths:
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

                all_path_covers.append(paths)

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
            if len(all_path_covers) >= 1000:
                print("[!] Capped at 1000 solutions.")
                break

    return all_path_covers

def draw_path_cover(ax, G_base, pos, cover_paths, markings=None, node_size=200, label_font_size=8):
    # Base faint background
    edges = list(G_base.edges())
    nx.draw_networkx_nodes(G_base, pos, node_color="#ecf0f1", node_size=node_size, ax=ax)
    nx.draw_networkx_labels(G_base, pos, font_size=label_font_size, ax=ax)
    nx.draw_networkx_edges(G_base, pos, edgelist=edges, edge_color="gray", alpha=0.2, ax=ax)

    # Draw each path with a distinct color
    colors = plt.cm.tab10.colors
    for idx, path in enumerate(cover_paths):
        color = colors[idx % len(colors)]
        if len(path) >= 2:
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(G_base, pos, edgelist=path_edges, edge_color=[color], width=3, ax=ax)
        nx.draw_networkx_nodes(G_base, pos, nodelist=path, node_color=[color], node_size=int(node_size * 1.2), ax=ax)

    # Overlay tick marks from `markings` (if provided)
    if markings:
        edge_labels = {}
        for u, v in G_base.edges():
            num_marks = markings.get((u, v), markings.get((v, u), 0))
            if num_marks:
                edge_labels[(u, v)] = "  |  " * num_marks
        if edge_labels:
            nx.draw_networkx_edge_labels(G_base, pos, edge_labels=edge_labels, font_size=14, font_weight="bold", bbox=dict(alpha=0), ax=ax)

def run_and_visualize(G: nx.Graph, max_paths: int = None):
    """
    Main execution flow: Solve -> Print Statistics -> Sequential Visualization
    """
    num_vertices = G.number_of_nodes()
    edges = list(G.edges())

    # 1. Run Solver
    all_solutions = find_all_path_covers(G, max_paths=max_paths)

    # 2. Print All Statistics
    counts = Counter([len(cover) for cover in all_solutions])
    print("\n" + "="*45)
    print(f"{'PATH COVER STATISTICS':^45}")
    print(f"{'(N=' + str(num_vertices) + ' vertices)':^45}")
    print("="*45)
    print(f"Total Unique Solutions Found: {len(all_solutions)}")

    # Sort by path count (1 path = Hamiltonian)
    sorted_counts = sorted(counts.keys())
    for n_paths in sorted_counts:
        label = "★ HAMILTONIAN ★" if n_paths == 1 else f"{n_paths} separate paths"
        print(f" - {label:20} : {counts[n_paths]} solutions")
    print("="*45 + "\n")

    if not all_solutions:
        print("No valid covers found.")
        return

    # 3. Sequential Grouped Visualization
    grouped = defaultdict(list)
    for cover in all_solutions:
        grouped[len(cover)].append(cover)

    G_base = G

    # Layout choice: Spring layout handles arbitrary node counts well
    pos = nx.spring_layout(G, seed=1)

    for path_count in sorted_counts:
        covers = grouped[path_count][:6]  # Show up to 6 per group

        plt.figure(figsize=(15, 8))
        title = "HAMILTONIAN PATHS (1 Path)" if path_count == 1 else f"COVERS WITH {path_count} PATHS"
        plt.suptitle(f"{title}\nEach separate path is shown in a unique color", fontsize=16, fontweight='bold')

        for i, cover_paths in enumerate(covers):
            ax = plt.subplot(2, 3, i + 1)

            # Build markings for this cover using GraphMarker
            marker = GraphMarker(G_base, path_cover=cover_paths, max_marks=10)
            markings = marker.find_solution(6)
            print('---')

            node_size = 200
            draw_path_cover(ax, G_base, pos, cover_paths, markings=markings, node_size=node_size)

            ax.set_title(f"Solution {i+1}")
            ax.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        print(f"[*] Showing: {title}. Close current window to proceed.")
        plt.show()

# --- Peterson Graph Setup ---
V = 10
edges = [
    (0,1), (1,2), (2,3), (3,4), (4,0),  # Outer
    (5,7), (7,9), (9,6), (6,8), (8,5),  # Inner
    (0,5), (1,6), (2,7), (3,8), (4,9)   # Spokes
]

# # --- Graph Definition: Dodecahedron (20 vertices, 30 edges) ---
# V = 20
# edges = [
#     # Outer ring
#     (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
#     # Outer to Middle
#     (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
#     # Middle ring (decagon)
#     (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
#     (5, 14), (6, 10), (7, 11), (8, 12), (9, 13),
#     # Middle to Inner
#     (10, 15), (11, 16), (12, 17), (13, 18), (14, 19),
#     # Inner ring
#     (15, 16), (16, 17), (17, 18), (18, 19), (19, 15)
# ]

# Run the full process
if __name__ == "__main__":
    # G = nx.Graph()
    # G.add_nodes_from(range(V))
    # G.add_edges_from(edges)
    # G = nx.petersen_graph()  # Using built-in Petersen graph for convenience
    from cat_state.cat_graphs_random import generate_high_girth_cubic_graph
    # N = 8
    # T = 3
    # G = generate_high_girth_cubic_graph(N, T, max_tries=1_000_000)
    G = nx.dodecahedral_graph()
    run_and_visualize(G, max_paths=None)  # Set max_paths=1 for Hamiltonian paths onlyx
