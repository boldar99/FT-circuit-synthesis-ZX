import itertools
import random
import networkx as nx
from generate_cubic_graphs import generate_cubic_graphs_with_geng

def has_small_nonlocal_cut(G, T):
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
                    return True  # Found a non-local small cut
    return False


def find_cubic_graph_with_local_cuts(N, T, max_trials=100, random_search=False):
    """
    Find a 3-regular graph on N vertices where all cuts of size ≤ T are local.
    If random_search=True, sample random graphs instead of enumerating all.
    """
    if random_search:
        for _ in range(max_trials):
            Gs = list(generate_cubic_graphs_with_geng(N, randomize=True, max_graphs=1))
            if not Gs:
                continue
            G = Gs[0]
            if not has_small_nonlocal_cut(G, T):
                return G
        return None
    else:
        return [G for G in generate_cubic_graphs_with_geng(N) if not has_small_nonlocal_cut(G, T)]


if __name__ == "__main__":
    G = find_cubic_graph_with_local_cuts(N=10, T=4)
    if G is not None:
        print("Found suitable graph")
    else:
        print("No graph found")