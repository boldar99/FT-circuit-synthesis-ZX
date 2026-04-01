import networkx as nx
import numpy as np

from spidercat.circuit_extraction import CatStateExtractor, StimBuilder
from spidercat.draw import draw_forest_on_graph, display_digraph
from spiderstate.utils import find_pivots_in_matrix, well_ordered_ft_cat_state_data


def match_edges(H: np.ndarray, non_pivots: list[int], z_orderings: list[list[list[int]]],
                x_orderings: list[list[list[int]]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edge_list = [
        (i, j)
        for i, r in enumerate(H)
        for j, x in enumerate(r[non_pivots])
        if x == 1
    ]

    # Deep copy and sort descending so pool[-1] is always the minimum
    z_pools = [[sorted(pool, reverse=True) for pool in circuit] for circuit in z_orderings]
    x_pools = [[sorted(pool, reverse=True) for pool in circuit] for circuit in x_orderings]

    edge_to_values = []

    while edge_list:
        min_cost = np.inf
        best_edge = None
        best_indices = (-1, -1)

        # 1. Evaluate all remaining edges
        for (i, j) in edge_list:
            # Find the minimum available integer across all valid pools for Z-circuit i
            z_k, z_min = -1, np.inf
            for k, pool in enumerate(z_pools[i]):
                if pool and pool[-1] < z_min:
                    z_min = pool[-1]
                    z_k = k

            # Find the minimum available integer across all valid pools for X-circuit j
            x_l, x_min = -1, np.inf
            for l, pool in enumerate(x_pools[j]):
                if pool and pool[-1] < x_min:
                    x_min = pool[-1]
                    x_l = l

            # Skip if either circuit has completely exhausted its pools
            if z_k == -1 or x_l == -1:
                continue

            cost = max(z_min, x_min)
            if cost < min_cost:
                min_cost = cost
                best_edge = (i, j)
                best_indices = (z_k, x_l)

        if best_edge is None:
            break  # Unroutable remaining edges

        # 2. Extract the minimum values from the chosen pools
        i, j = best_edge
        z_k, x_l = best_indices
        z_val = z_pools[i][z_k].pop()
        x_val = x_pools[j][x_l].pop()

        # 3. Record the actual values and remove the processed edge
        edge_to_values.append((best_edge, (z_val, x_val)))
        edge_list.remove(best_edge)

    return edge_to_values


def flag_by_construction(H: np.ndarray, d: int):
    N = H.shape[1]
    t = (d - 1) // 2

    pivots, rows_without_pivots = find_pivots_in_matrix(H)
    pivots_perm = [b for (_, b) in sorted((b, a) for (a, b) in pivots.items())]
    non_pivots = [p for p in range(N) if p not in pivots.values()]
    assert len(rows_without_pivots) == 0

    z_spiders = np.sum(H, axis=1)
    x_spiders = np.sum(H[:, non_pivots], axis=0) + 1

    z_data = [well_ordered_ft_cat_state_data(zs, t) for zs in z_spiders]
    x_data = [well_ordered_ft_cat_state_data(xs, t) for xs in x_spiders]
    z_graphs, x_graphs, z_trees, x_trees, z_mains, x_mains = [], [], [], [], [], []
    z_digraphs, x_digraphs, filtered_z_orderings, filtered_x_orderings = [], [], [], []
    z_roots, x_roots = [], []
    for (G, F, roots, D, e) in z_data:
        nx.set_node_attributes(G, "Z", 'spider_type')
        z_graphs.append(G)
        z_trees.append(F)
        z_roots.append(roots)
        z_digraphs.append(D)
        z_mains.append(e)
        filtered_z_orderings.append([])
        for layer in list(nx.topological_generations(D)):
            to_append = [l for l in layer if l != e and G.nodes[l].get("is_mark", False)]
            if to_append:
                filtered_z_orderings[-1].append(to_append)
    for (G, F, roots, D, e) in x_data:
        nx.set_node_attributes(G, "X", 'spider_type')
        x_graphs.append(G)
        x_trees.append(F)
        x_roots.append(roots)
        x_digraphs.append(D)
        x_mains.append(e)
        filtered_x_orderings.append([])
        for layer in list(nx.topological_generations(D)):
            to_append = [l for l in layer if l != e and G.nodes[l].get("is_mark", False)]
            if to_append:
                filtered_x_orderings[-1].append(to_append)

    matched_edges = match_edges(H, non_pivots, filtered_z_orderings, filtered_x_orderings)

    z_node_mapping = {}
    x_node_mapping = {}
    global_G = nx.Graph()
    global_F = nx.Graph()
    global_roots = {}
    global_D = nx.DiGraph()
    global_primary_paths = {}
    i, j = 0, 0
    k = 0
    while i + j < N:
        graph = x_graphs[i] if i + j in non_pivots else z_graphs[pivots_perm[j]]
        trees = x_trees[i] if i + j in non_pivots else z_trees[pivots_perm[j]]
        digraph = x_digraphs[i] if i + j in non_pivots else z_digraphs[pivots_perm[j]]
        node_mapping = x_node_mapping if i + j in non_pivots else z_node_mapping
        root = x_roots[i] if i + j in non_pivots else z_roots[pivots_perm[j]]
        index = i if i + j in non_pivots else pivots_perm[j]
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
            global_G,
            source=global_roots[i + j],
            target=node_mapping[(index, x_mains[i] if i + j in non_pivots else z_mains[pivots_perm[j]])]
        )
        if i + j in non_pivots: i += 1
        else: j += 1

    while matched_edges:
        (z_graph, x_graph), (z_val, x_val) = matched_edges.pop(0)
        global_G.add_edge(z_node_mapping[(z_graph, z_val)], x_node_mapping[(x_graph, x_val)], edge_type="intercat")
        global_G.nodes[z_node_mapping[(z_graph, z_val)]]["is_cnot"] = True
        global_G.nodes[x_node_mapping[(x_graph, x_val)]]["is_cnot"] = True
        global_G.nodes[z_node_mapping[(z_graph, z_val)]]["is_mark"] = False
        global_G.nodes[x_node_mapping[(x_graph, x_val)]]["is_mark"] = False

        for u, _ in z_digraphs[z_graph].in_edges(z_val):
            global_D.add_edge(z_node_mapping[(z_graph, u)], x_node_mapping[(x_graph, x_val)], edge_type="intercat")
        for u, _ in x_digraphs[x_graph].in_edges(x_val):
            global_D.add_edge(x_node_mapping[(x_graph, u)], z_node_mapping[(z_graph, z_val)], edge_type="intercat")


    extractor = CatStateExtractor(StimBuilder(), verbose=True)
    draw_forest_on_graph(global_G, global_F)
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
    # H_x = np.array([
    #     [1, 1, 1, 1, 0, 0, 0],
    #     [0, 1, 1, 0, 1, 1, 0],
    #     [0, 0, 1, 1, 0, 1, 1]
    # ])
    H_x, d = np.array([
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1]
    ]), 7
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
    H_x, d, name = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1]
    ]), 6, "[[20, 2, 6"
    circ = flag_by_construction(H_x, d)
