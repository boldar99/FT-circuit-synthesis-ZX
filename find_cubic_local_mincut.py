import itertools
import math
import random
import networkx as nx
from generate_cubic_graphs import generate_cubic_graphs_with_geng
from functools import lru_cache

def find_small_nonlocal_cut(G, T):
    """Return True if G has a cut of size ≤ T that is non-local."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    # Enumerate subsets up to size n//2 (smaller side)
    for k in range(1, n // 2 + 1):
        for S in itertools.combinations(nodes, k):
            S = set(S)
            # Compute cut size
            cut_size = sum(1 for u, v in G.edges() if (u in S) ^ (v in S))
            if cut_size <= T:
                # Get smaller subgraph (S or complement)
                if k > n // 2:
                    S = set(nodes) - S
                subG = G.subgraph(S)
                # Check if it has a cycle
                if not nx.is_forest(subG):
                    return subG  # Found a non-local small cut
    return None


def has_small_nonlocal_cut(G, T):
    return find_small_nonlocal_cut(G, T) is not None


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
def generate_high_girth_cubic_graph(N, T, max_tries=100_000) -> nx.Graph | None:
    """
    Generates a 3-regular graph with Girth > T.
    Uses edge-switching (rewiring) to break short cycles.
    """
    # 1. Target Girth
    target_girth = T + 1

    if target_girth <= 5 and N == 10:
        return nx.generators.petersen_graph()
    if target_girth <= 6 and N == 14:
        return nx.generators.heawood_graph()
    if target_girth <= 6 and N == 18:
        return nx.generators.pappus_graph()
    # if target_girth <= 7 and N == 24:
    #     return nx.generators.mc()

    # Check Moore Bound feasibility (approximate)
    # Moore Bound for Girth 6 is N >= 14.
    # Moore Bound for Girth 7 is N >= 22.
    if target_girth == 6 and N < 14: return None
    if target_girth == 7 and N < 22: return None
    if N <= 2: return None

    # 2. Start with a random cubic graph
    G = nx.random_regular_graph(3, N)

    # 3. Hill Climbing / Rewiring Loop
    for _ in range(max_tries):
        current_girth = nx.girth(G)
        if current_girth > T:
            return G

        nx.connected_double_edge_swap(G, nswap=1, _window_threshold=1)

    return None


if __name__ == "__main__":
    G = find_cubic_graph_with_local_cuts(N=math.ceil((27 * 2 / 3) / 2) * 2, T=3, random_search=True, max_trials=10_000)
    if G is not None:
        print("Found suitable graph")
    else:
        print("No graph found")