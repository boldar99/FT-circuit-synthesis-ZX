import math

import networkx as nx
import sympy
from matplotlib import pyplot as plt

from spidercat.draw import draw_spanning_forest_solution
from spidercat.markings import find_marking_property_violation
from spidercat.spanning_tree import match_forest_leaves_to_marked_edges, find_min_height_roots


def _find_valid_l(n: int) -> int | None:
    valid_l = None
    lower_bound = n / 4 - 1
    upper_bound = n / 2

    # We search the range of integers strictly inside (lower_bound, upper_bound)
    start_search = int(math.floor(lower_bound)) + 1
    end_search = int(math.ceil(upper_bound))

    for l in range(start_search, end_search):
        # Check Coprimality
        if math.gcd(l, n) != 1:
            continue

        # Calculate modular inverse
        try:
            l_inv = pow(l, -1, n)
        except ValueError:
            continue

        # Check constraints on the inverse
        if lower_bound < l_inv < upper_bound:
            valid_l = l
            break
    return valid_l


def construct_special_marked_graph(n: int, l: int | None = None) -> nx.Graph | None:
    """
    Constructs a specific 3-regular graph with 2*n vertices based on modular arithmetic properties.

    The graph consists of a cycle of 2n vertices (v_0...v_{n-1}, w_0...w_{n-1})
    and chords connecting v_k to w_{l*k mod n}.

    Args:
        n: The parameter defining the half-size of the graph (total vertices = 2*n).
           Must be large enough (e.g., n >= 12) to find a valid 'l'.


    Returns:
        nx.Graph: The constructed 3-regular graph.

    Raises:
        ValueError: If no valid 'l' can be found for the given n.
    """
    # 1. Find a valid l such that:
    #    a) gcd(l, n) == 1
    #    b) n/4 - 1 < l < n/2
    #    c) n/4 - 1 < l^-1 < n/2  (where l^-1 is the modular inverse modulo n)

    valid_l = l or _find_valid_l(n)
    if valid_l is None:
        return None

    # 2. Construct the Graph
    # We map indices as follows:
    # v_k -> node k          (for k in 0..n-1)
    # w_k -> node n + k      (for k in 0..n-1)

    num_nodes = 2 * n
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    cycle_edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    G.add_edges_from(cycle_edges)

    # Connect v_k to w_{l*k mod n}
    # Source: k
    # Target: n + ((l * k) % n)
    chord_edges = []
    for k in range(n):
        u = k
        w_index = (valid_l * k) % n
        v = n + w_index
        chord_edges.append((u, v))

    G.add_edges_from(chord_edges)
    return G


def construct_prime_inverse_graph(p: int) -> tuple[nx.Graph | None, dict]:
    """
    Constructs a 3-regular graph for a prime p.
    Vertices are 2, ..., p-2.
    Edges:
      1. Cycle: (v, v+1) for v in 2..p-3, and (p-2, 2).
      2. Chords: (v, v^-1 mod p).
    Marks:
      Edges (3,4), (5,6), ..., (p-2, 2) are marked.
    """
    # 1. Validation
    # p must be a prime number >= 5 for this construction to have at least 2 nodes
    # and for p-2 to be odd (required for the marking logic).
    marks = {}
    if p < 5 or not sympy.isprime(p):
        return None, marks
    if p % 10 == 1 or p % 10 == 9:
        return None, marks

    G = nx.Graph()
    nodes = list(range(2, p - 1))  # 2 to p-2
    G.add_nodes_from(nodes)

    for u in range(2, p - 1):
        if u == p - 2:
            v = 2
        else:
            v = u + 1
        if u % 2 != 0:
            marks[(u, v)] = 1

        G.add_edge(u, v)

    for u in nodes:
        inv = pow(u, -1, p)
        if u < inv:
            G.add_edge(u, inv)
            if u == inv + 1 or u == inv - 1:
                return None, marks

    return G, marks


if __name__ == "__main__":
    N = 37
    G, M = construct_prime_inverse_graph(N)
    print(find_marking_property_violation(G, M, 100))
    nx.draw(G)
    plt.show()
