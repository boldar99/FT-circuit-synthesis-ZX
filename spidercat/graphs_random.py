import random

import matplotlib.pyplot as plt
import networkx as nx

from spidercat.nonlocal_cut import has_small_nonlocal_cut
from spidercat.utils import graph_exists_with_girth


def generate_3_regular_high_girth(n, target_girth, max_tries=100):
    """
    Generates a 3-regular graph with a minimum specified girth.
    Uses a greedy edge-addition approach with restarts.
    """
    if not graph_exists_with_girth(n, target_girth):
        return None

    for attempt in range(max_tries):
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # 1. Start with a Hamiltonian Cycle (Degree 2)
        # This guarantees connectivity and an initial girth of n
        nodes = list(range(n))
        random.shuffle(nodes)
        for i in range(n):
            G.add_edge(nodes[i], nodes[(i + 1) % n])

        # 2. Add the "Chords" to reach Degree 3
        unmet_nodes = list(range(n))
        random.shuffle(unmet_nodes)

        success = True
        while unmet_nodes:
            u = unmet_nodes.pop()

            # Potential candidates are other nodes that still need an edge
            # and are NOT currently neighbors of u
            candidates = [v for v in unmet_nodes if not G.has_edge(u, v)]
            random.shuffle(candidates)

            edge_added = False
            for v in candidates:
                # Optimized check: Is distance < target_girth - 1?
                try:
                    # We only care if there is a SHORT path
                    dist = nx.shortest_path_length(G, source=u, target=v)
                except nx.NetworkXNoPath:
                    dist = float('inf')

                if dist >= target_girth - 1:
                    G.add_edge(u, v)
                    # If v still needs an edge, it stays in unmet_nodes
                    # otherwise it was already popped or we remove it now
                    if G.degree(v) == 3:
                        unmet_nodes.remove(v)
                    edge_added = True
                    break

            if not edge_added:
                success = False
                break  # Failed this attempt, restart loop

        if success:
            return G

    return None


def generate_3regular_graph_with_no_nonlocal_t_cut(N, T, max_iter=10_000):
    """
    Constructs a 3-regular graph with no non-local cuts of size <= T.
    Strategy: Hill Climbing on Girth + Algebraic Connectivity.
    """
    target_girth = T + 1
    if N <= 2: return None
    if not graph_exists_with_girth(N, target_girth):
        return None
    if target_girth <= 5 and N == 10:
        return nx.generators.petersen_graph()
    if target_girth <= 6 and N == 14:
        return nx.generators.heawood_graph()
    if target_girth <= 6 and N == 18:
        return nx.generators.pappus_graph()

    G = generate_3_regular_high_girth(N, target_girth, max_tries=max_iter)
    if G is None:
        G = nx.random_regular_graph(3, N)
    girth = nx.girth(G)
    lambda_curr = nx.algebraic_connectivity(G, method='lanczos', tol=1e-4)

    lambda_threshold = 10 / 3 * T / N

    for i in range(max_iter):
        if girth > T and lambda_curr >= lambda_threshold:
            if not has_small_nonlocal_cut(G, T):
                return G
        try:
            G_new = G.copy()
            nx.connected_double_edge_swap(G_new, nswap=2)
            new_girth = nx.girth(G_new)

            if new_girth >= girth and girth <= T:
                G = G_new
                girth = new_girth
                continue
            if new_girth < girth:
                continue

            if girth > T:
                lambda_curr = nx.algebraic_connectivity(G, tol=1e-3)
                lambda_new = nx.algebraic_connectivity(G_new, tol=1e-3)

                if lambda_new > lambda_curr:
                    lambda_curr = lambda_new
                    G = G_new

        except nx.NetworkXError:
            continue

    if has_small_nonlocal_cut(G, T):
        return None
    else:
        return G


if __name__ == "__main__":
    G = generate_3regular_graph_with_no_nonlocal_t_cut(40, 7)
    if G is not None:
        nx.draw(G)
        plt.show()
        print("Found suitable graph")
    else:
        print("No graph found")
