import copy
import itertools

import matplotlib.pyplot as plt
import networkx as nx

from generate_cubic_graphs import generate_cubic_graphs_with_geng
from functools import lru_cache

def find_small_nonlocal_cut(G, T):
    """Return True if G has a cut of size ≤ T that is non-local."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    adj = {n: set(G[n]) for n in G.nodes()}

    # Enumerate subsets up to size n//2 (smaller side)
    for k in range(1, n // 2 + 1):
        for S in itertools.combinations(nodes, k):
            S = set(S)
            # 2. OPTIMIZATION: Direct Neighbor Check with Early Exit
            cut_size = 0
            possible = True

            for u in S:
                # Only check neighbors of nodes in S
                for v in adj[u]:
                    if v not in S:
                        cut_size += 1

                # CRITICAL: Stop immediately if we blow the budget
                if cut_size > T:
                    possible = False
                    break

            if possible:  # effectively: if cut_size <= T
                # Get smaller subgraph (S or complement)
                if k > n // 2:
                    S = set(nodes) - S
                subG = G.subgraph(S)
                # Check if it has a cycle
                if not nx.is_forest(subG):
                    return subG  # Found a non-local small cut
    return None


# def has_small_nonlocal_cut(G, T):
#     return find_small_nonlocal_cut(G, T) is not None


def has_small_nonlocal_cut(G, T):
    """
    EXACT Solver using Bounded Search.
    Returns True if there exists ANY subset S such that:
      1. Cut(S) <= T
      2. S contains a cycle (Induced Edges >= Nodes)
      3. |S| <= N/2 (Symmetry)
    """
    N = G.number_of_nodes()
    adj = {n: set(G[n]) for n in G.nodes()}
    visited_fingerprints = set()  # To avoid checking same set twice

    # We iterate possible subgraphs.
    # Since T is small, the search tree is shallow.

    def search(current_S, current_cut, current_internal_edges):
        # 1. Pruning: If Cut is too high, give up on this branch.
        # Strict bound: A single node addition can reduce cut by at most 3.
        # If we are at T+4, we need at least 2 perfect additions to get back to T.
        # For correctness, use a loose bound or exact check.
        # Optimization: Stop if cut > T + 2 (Very conservative but efficient)
        if current_cut > T + 2:
            return False

        # 2. Victory Condition
        if current_cut <= T:
            # Check if Non-Local (Contains Cycle)
            if current_internal_edges >= len(current_S):
                return True

        # 3. Size Limit (Symmetry)
        if len(current_S) >= N // 2:
            return False

        # 4. Expansion: Add neighbors
        # Candidates = Neighbors of S not in S
        candidates = set()
        for u in current_S:
            candidates.update(adj[u])
        candidates.difference_update(current_S)

        # Sort candidates by "Greediness" (how much they lower the cut)
        # This is critical for finding the answer fast.
        cand_list = []
        for v in candidates:
            # Edges to S
            k = len(adj[v].intersection(current_S))
            # Delta Cut = (3-k) - k = 3 - 2k
            delta = 3 - 2 * k
            cand_list.append((v, delta, k))

        # Sort: Prefer negative delta (reducing cut)
        cand_list.sort(key=lambda x: x[1])

        for v, delta, k in cand_list:
            new_S = current_S | {v}

            # Hashing to prevent redundancy
            # (Sorting tuple is faster than frozenset for small sets)
            fp = tuple(sorted(list(new_S)))
            if fp in visited_fingerprints:
                continue
            visited_fingerprints.add(fp)

            # Recurse
            if search(new_S, current_cut + delta, current_internal_edges + k):
                return True

        return False

    # Run search starting from every individual node
    # (Checking single nodes is fast and ensures coverage)
    for n in G.nodes():
        # Initial State: S={n}, Cut=3, Internal=0
        # Check pruning immediately for huge T
        if search({n}, 3, 0):
            return True

    return False

def find_cubic_graph_with_local_cuts(N, T, random_search=False, max_trials=100):
    """
    Finds a 3-regular graph on N vertices where all cuts of size ≤ T are local.
    If random_search=True, sample random graphs instead of enumerating all.
    """
    if random_search:
        for _ in range(max_trials):
            G = nx.random_regular_graph(3, N)
            if not has_small_nonlocal_cut(G, T):
                return G
        return None
    else:
        for G in generate_cubic_graphs_with_geng(N):
            if not has_small_nonlocal_cut(G, T):
                return G
        return None

def generate_all_cubic_graph_with_local_cuts(N, T):
    """
    Finds all 3-regular graph on N vertices where all cuts of size ≤ T are local.
    """
    for G in generate_cubic_graphs_with_geng(N):
        if not has_small_nonlocal_cut(G, T):
            yield G


def all_cubic_graph_with_local_cuts(N, T):
    """
    Finds all 3-regular graph on N vertices where all cuts of size ≤ T are local.
    """
    return list(generate_all_cubic_graph_with_local_cuts(N, T))


@lru_cache(maxsize=None)
def generate_high_girth_cubic_graph(N, T, max_tries=1_000_000) -> nx.Graph | None:
    """
    Generates a 3-regular graph with Girth > T.
    Uses edge-switching (rewiring) to break short cycles.
    """
    # 1. Target Girth
    target_girth = T + 1

    if N <= 2: return None
    if target_girth == 6 and N < 14: return None
    if target_girth == 7 and N < 22: return None
    if target_girth <= 5 and N == 10:
        return nx.generators.petersen_graph()
    if target_girth <= 6 and N == 14:
        return nx.generators.heawood_graph()
    if target_girth <= 6 and N == 18:
        return nx.generators.pappus_graph()

    # 2. Start with a random cubic graph
    G = nx.random_regular_graph(3, N)

    # 3. Hill Climbing / Rewiring Loop
    for _ in range(max_tries):
        current_girth = nx.girth(G)
        if current_girth > T:
            return G

        nx.connected_double_edge_swap(G, nswap=1, _window_threshold=1)

    return None


def algebraic_connectivity_heuristic(G):
    """
    Returns the second smallest eigenvalue of the Laplacian.
    Higher value = Better expansion (fewer bottlenecks).
    """
    return nx.algebraic_connectivity(G, method='lanczos', tol=1e-3)


def construct_cyclic_connected_graph(N, T, max_iter=1_000):
    """
    Constructs a 3-regular graph with no non-local cuts of size <= T.
    Strategy: Hill Climbing on Girth + Algebraic Connectivity.
    """
    target_girth = T + 1
    if N <= 2: return None
    if target_girth == 6 and N < 14: return None
    if target_girth == 7 and N < 22: return None
    if target_girth <= 5 and N == 10:
        return nx.generators.petersen_graph()
    if target_girth <= 6 and N == 14:
        return nx.generators.heawood_graph()
    if target_girth <= 6 and N == 18:
        return nx.generators.pappus_graph()


    lambda_curr = 0
    while lambda_curr < 0.1:
        G = nx.random_regular_graph(3, N)
        lambda_curr = nx.algebraic_connectivity(G, method='lanczos', tol=1e-4)
    girth = nx.girth(G)

    lambda_threshold = 3 * T / N

    for i in range(max_iter):
        # if i % 100 == 0:
        #     print(f"{i} iterations: {girth = }; {lambda_curr = }")
        if girth > T and lambda_curr >= lambda_threshold:
            if not has_small_nonlocal_cut(G, T):  # Your function
                # print(f"Girth: {girth}; lambda: {lambda_curr}")
                # print(f"Success at iter {i}: Girth {girth}")
                return G
            # else:
            #     nx.draw(G)
            #     plt.show()
            #     print(f"Girth: {girth}; lambda: {lambda_curr}")
            #     print(f"No success at iter {i}: Girth {girth}")

        try:
            G_new = copy.deepcopy(G)
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
    # G = find_cubic_graph_with_local_cuts(N=math.ceil((27 * 2 / 3) / 2) * 2, T=3, random_search=True, max_trials=10_000)
    G = construct_cyclic_connected_graph(20, 4)
    if G is not None:
        nx.draw(G)
        plt.show()
        print("Found suitable graph")
    else:
        print("No graph found")