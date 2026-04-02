import networkx as nx
import numpy as np

from spidercat.circuit_extraction import CatStateExtractor, StimBuilder
from spidercat.draw import draw_forest_on_graph, display_digraph
from spiderstate.utils import find_pivots_in_matrix, well_ordered_ft_cat_state_data


def match_edges(H: np.ndarray, non_pivots: list[int],
                z_digraphs: list[nx.DiGraph], x_digraphs: list[nx.DiGraph],
                z_candidates: list[list[int]], x_candidates: list[list[int]]) -> list[
    tuple[tuple[int, int], tuple[int, int]]]:
    # 1. Identify all required logical connections dictated by the parity check matrix
    edge_list = [
        (i, j)
        for i, r in enumerate(H)
        for j, x in enumerate(r[non_pivots])
        if x == 1
    ]

    # 2. Initialize a global tracking digraph to strictly monitor transitive dependencies
    tracker = nx.DiGraph()

    # Prefix node names to avoid collisions between independent cat states
    for i, D in enumerate(z_digraphs):
        tracker.add_edges_from((f"Z_{i}_{u}", f"Z_{i}_{v}") for u, v in D.edges())
    for j, D in enumerate(x_digraphs):
        tracker.add_edges_from((f"X_{j}_{u}", f"X_{j}_{v}") for u, v in D.edges())

    # Deep copy candidate pools so we can mutate them during the search
    z_pools = [[c for c in pool] for pool in z_candidates]
    x_pools = [[c for c in pool] for pool in x_candidates]

    # 3. Backtracking Constraint Solver
    def backtrack(remaining_edges, current_matches):
        if not remaining_edges:
            return current_matches

        # Heuristic: Sort remaining edges by fewest available candidate combinations.
        # This dramatically prunes the search tree by attacking bottlenecks first.
        remaining_edges = sorted(remaining_edges, key=lambda e: len(z_pools[e[0]]) * len(x_pools[e[1]]))
        i, j = remaining_edges[0]

        for z_val in list(z_pools[i]):
            for x_val in list(x_pools[j]):

                # Determine provisional cross-edges dictated by the inter-cat CNOT logic
                new_edges = []
                for u, _ in z_digraphs[i].in_edges(z_val):
                    new_edges.append((f"Z_{i}_{u}", f"X_{j}_{x_val}"))
                for u, _ in x_digraphs[j].in_edges(x_val):
                    new_edges.append((f"X_{j}_{u}", f"Z_{i}_{z_val}"))

                # Cycle Verification: Ensure adding these edges preserves the DAG
                added_edges_this_step = []
                cycle_found = False

                for src, dst in new_edges:
                    # If src == dst or a path already exists from dst to src, adding src->dst creates a cycle
                    if src == dst or nx.has_path(tracker, dst, src):
                        cycle_found = True
                        break
                    tracker.add_edge(src, dst)
                    added_edges_this_step.append((src, dst))

                if not cycle_found:
                    # State transition: Commit provisional match and dive deeper
                    z_pools[i].remove(z_val)
                    x_pools[j].remove(x_val)

                    result = backtrack(remaining_edges[1:], current_matches + [((i, j), (z_val, x_val))])
                    if result is not None:
                        return result  # Valid global state found

                    # State rollback: The branch hit a dead end
                    z_pools[i].append(z_val)
                    x_pools[j].append(x_val)

                # Cleanup the tracker if a cycle was found or if we rolled back
                tracker.remove_edges_from(added_edges_this_step)

        return None  # Trigger upstream backtracking

    final_matching = backtrack(edge_list, [])

    if final_matching is None:
        raise ValueError("No valid cycle-free edge matching exists for this matrix topology.")

    return final_matching


def flag_by_construction(H: np.ndarray, d: int):
    N = H.shape[1]
    t = (d - 1) // 2

    pivots, rows_without_pivots = find_pivots_in_matrix(H)
    pivots_perm = [row for row, col in sorted(pivots.items(), key=lambda item: item[1])]
    non_pivots = [p for p in range(N) if p not in pivots.values()]
    assert len(rows_without_pivots) == 0

    z_spiders = np.sum(H, axis=1)
    x_spiders = np.sum(H[:, non_pivots], axis=0) + 1

    z_data = [well_ordered_ft_cat_state_data(zs, t) for zs in z_spiders]
    x_data = [well_ordered_ft_cat_state_data(xs, t) for xs in x_spiders]
    z_graphs, x_graphs, z_trees, x_trees, z_mains, x_mains = [], [], [], [], [], []
    z_digraphs, x_digraphs = [], []
    z_candidates, x_candidates = [], []
    z_roots, x_roots = [], []
    for (G, F, roots, D, e) in z_data:
        nx.set_node_attributes(G, "Z", 'spider_type')
        z_graphs.append(G); z_trees.append(F); z_roots.append(roots)
        z_digraphs.append(D); z_mains.append(e)

        # Flatten topological generations into prioritized 1D candidate pools
        cands = []
        for layer in nx.topological_generations(D):
            cands.extend([l for l in layer if l != e and G.nodes[l].get("is_mark", False)])
        z_candidates.append(cands)

    for (G, F, roots, D, e) in x_data:
        nx.set_node_attributes(G, "X", 'spider_type')
        x_graphs.append(G); x_trees.append(F); x_roots.append(roots)
        x_digraphs.append(D); x_mains.append(e)

        cands = []
        for layer in nx.topological_generations(D):
            cands.extend([l for l in layer if l != e and G.nodes[l].get("is_mark", False)])
        x_candidates.append(cands)

    matched_edges = match_edges(H, non_pivots, z_digraphs, x_digraphs, z_candidates, x_candidates)

    z_node_mapping = {}
    x_node_mapping = {}
    global_G = nx.Graph()
    global_F = nx.Graph()
    global_roots = {}
    global_D = nx.DiGraph()
    global_primary_paths = {}
    i, j = 0, 0
    k = 0
    non_pivots_set = set(non_pivots)

    while i + j < N:
        curr_col = i + j
        is_non_pivot = curr_col in non_pivots_set

        if is_non_pivot:
            graph, trees, digraph = x_graphs[i], x_trees[i], x_digraphs[i]
            node_mapping, root, index = x_node_mapping, x_roots[i], i
        else:
            row = pivots_perm[j]
            graph, trees, digraph = z_graphs[row], z_trees[row], z_digraphs[row]
            node_mapping, root, index = z_node_mapping, z_roots[row], row
        for node, data in graph.nodes(data=True):
            node_mapping[(index, node)] = k
            global_G.add_node(k, **data)
            global_F.add_node(k)
            global_D.add_node(k)
            k += 1
        for u, v, data in graph.edges(data=True):
            u_prime = node_mapping[(index, u)]
            v_prime = node_mapping[(index, v)]
            global_G.add_edge(u_prime, v_prime, **data)
        for u, v, data in trees.edges(data=True):
            u_prime = node_mapping[(index, u)]
            v_prime = node_mapping[(index, v)]
            global_F.add_edge(u_prime, v_prime, **data)
        for u, v, data in digraph.edges(data=True):
            u_prime = node_mapping[(index, u)]
            v_prime = node_mapping[(index, v)]
            global_D.add_edge(u_prime, v_prime, **data)

        global_roots[i + j] = node_mapping[(index, root[0])]
        global_primary_paths[i + j] = nx.shortest_path(
            global_F,
            source=global_roots[i + j],
            target=node_mapping[(index, x_mains[i] if i + j in non_pivots else z_mains[pivots_perm[j]])]
        )
        if i + j in non_pivots: i += 1
        else: j += 1

    while matched_edges:
        (z_graph, x_graph), (z_val, x_val) = matched_edges.pop(0)
        global_G.add_edge(z_node_mapping[(z_graph, z_val)], x_node_mapping[(x_graph, x_val)], edge_type="cnot")
        global_G.nodes[z_node_mapping[(z_graph, z_val)]]["is_mark"] = False
        global_G.nodes[x_node_mapping[(x_graph, x_val)]]["is_mark"] = False
        if global_F.degree(z_node_mapping[(z_graph, z_val)]) == 1:
            global_G.nodes[z_node_mapping[(z_graph, z_val)]]["is_flag"] = True
        if global_F.degree(x_node_mapping[(x_graph, x_val)]) == 1:
            global_G.nodes[x_node_mapping[(x_graph, x_val)]]["is_flag"] = True



        for u, _ in z_digraphs[z_graph].in_edges(z_val):
            global_D.add_edge(z_node_mapping[(z_graph, u)], x_node_mapping[(x_graph, x_val)], edge_type="cnot")
        for u, _ in x_digraphs[x_graph].in_edges(x_val):
            global_D.add_edge(x_node_mapping[(x_graph, u)], z_node_mapping[(z_graph, z_val)], edge_type="cnot")


    extractor = CatStateExtractor(StimBuilder(), verbose=True)
    draw_forest_on_graph(global_G, global_F, figsize=(20, 20))
    display_digraph(global_D)
    circ = extractor.extract(global_G, global_F, global_roots, global_D, global_primary_paths)
    return circ


if __name__ == "__main__":
    H_x, d = np.array([
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ]), 4
    H_x = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1]
    ])
  #   H_x, d = np.array([
  #       [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  #       [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
  #       [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  #       [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  #       [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
  #       [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
  #       [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  #       [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  #       [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  #       [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  #       [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  # ]), 7
    # H_x, d = np.array([
    #     [1, 1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 1, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 1]
    # ]), 3
    # H_x, d = np.array([
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
    # ]), 5
    # H_x, d, name = np.array([
    #   [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, ],
    #   [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, ],
    #   [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, ],
    #   [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, ],
    #   [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, ],
    #   [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, ],
    #   [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, ],
    #   [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, ],
    #   [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, ],
    # ]), 6, "[[20, 2, 6]]"
    # H_x, d = np.array([
    #     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    #     [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    # ]), 4
    H_x, d = np.array([
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ]), 5
    circ = flag_by_construction(H_x, d)
