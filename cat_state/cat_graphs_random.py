import itertools

import matplotlib.pyplot as plt
import networkx as nx

from cubic_graphs import generate_cubic_graphs_with_geng
from functools import lru_cache
import random
import networkx as nx
from pysat.solvers import Glucose3
from pysat.card import CardEnc
from pysat.formula import IDPool, CNF


def has_small_nonlocal_cut(G: nx.Graph, T: int) -> bool:
    """
    Determines if a 3-regular graph G has a non-local cut of size <= T.

    A non-local cut (A, B) requires that G[A] and G[B] both contain at least one cycle.
    This is encoded as:
    1. A and B are non-empty sets.
    2. A contains a non-empty 2-Core (subset where every node has degree >= 2).
    3. B contains a non-empty 2-Core.
    4. The number of edges between A and B is <= T.
    """

    # Check if G is actually 3-regular (assumption of the prompt)
    if not all(d == 3 for _, d in G.degree()):
        raise ValueError("Graph must be 3-regular")

    v_list = list(G.nodes())
    n = len(v_list)

    # IDPool manages variable mapping automatically
    vpool = IDPool()

    # --- Variable Definitions ---
    # x_u: Partition variable. True -> A, False -> B
    # a_u: Core variable for A. True -> u is in the 2-Core of A
    # b_u: Core variable for B. True -> u is in the 2-Core of B
    def x(u):
        return vpool.id(f"x_{u}")

    def a(u):
        return vpool.id(f"a_{u}")

    def b(u):
        return vpool.id(f"b_{u}")

    cnf = CNF()

    # --- 1. Cut Size Constraints ---
    # We introduce auxiliary variables for cut edges to count them
    cut_edge_vars = []

    for u, v in G.edges():
        # d_uv is true if edge (u, v) is a cut edge (endpoints have different partitions)
        d_uv = vpool.id(f"cut_{u}_{v}")
        cut_edge_vars.append(d_uv)

        # Logic: If x(u) != x(v), then d_uv must be true.
        # We only need the implication (x_u != x_v) -> d_uv, because we are minimizing
        # (or limiting) the sum of d_uv. The solver won't set d_uv to True unnecessarily.

        # Clause 1: x(u) & !x(v) -> d_uv  <=>  !x(u) v x(v) v d_uv
        cnf.append([-x(u), x(v), d_uv])

        # Clause 2: !x(u) & x(v) -> d_uv  <=>  x(u) v !x(v) v d_uv
        cnf.append([x(u), -x(v), d_uv])

    # Constraint: Sum(cut_edge_vars) <= T
    # CardEnc.atmost generates the CNF clauses for cardinality constraint
    cnf.extend(CardEnc.atmost(lits=cut_edge_vars, bound=T, vpool=vpool))

    # --- 2. Core Constraints for Partition A ---
    # To enforce a cycle in A, we enforce a non-empty subset S_A where min_degree >= 2

    a_vars = []
    for u in v_list:
        a_u = a(u)
        a_vars.append(a_u)

        # Constraint: If u is in Core A, u must be in Partition A
        # a_u -> x_u  <=>  !a_u v x_u
        cnf.append([-a_u, x(u)])

        # Constraint: If u is in Core A, it must have at least 2 neighbors in Core A.
        # Since G is 3-regular, u has neighbors n1, n2, n3.
        # We forbid having 0 or 1 neighbor in Core A.
        # Equivalent to: We forbid having >= 2 neighbors NOT in Core A.
        # Pairs of neighbors cannot both be false.
        neighbors = list(G.neighbors(u))
        n1, n2, n3 = neighbors[0], neighbors[1], neighbors[2]

        # If a_u is True, then:
        # (a_n1 or a_n2) AND (a_n1 or a_n3) AND (a_n2 or a_n3)
        cnf.append([-a_u, a(n1), a(n2)])
        cnf.append([-a_u, a(n1), a(n3)])
        cnf.append([-a_u, a(n2), a(n3)])

    # Constraint: Core A must be non-empty
    cnf.append(a_vars)

    # --- 3. Core Constraints for Partition B ---
    # Mirror logic for B (where x_u is False)

    b_vars = []
    for u in v_list:
        b_u = b(u)
        b_vars.append(b_u)

        # Constraint: If u is in Core B, u must be in Partition B
        # b_u -> !x_u  <=>  !b_u v !x_u
        cnf.append([-b_u, -x(u)])

        # Degree constraint for B
        neighbors = list(G.neighbors(u))
        n1, n2, n3 = neighbors[0], neighbors[1], neighbors[2]

        cnf.append([-b_u, b(n1), b(n2)])
        cnf.append([-b_u, b(n1), b(n3)])
        cnf.append([-b_u, b(n2), b(n3)])

    # Constraint: Core B must be non-empty
    cnf.append(b_vars)

    # --- 4. Symmetry Breaking (Optional) ---
    # Fix the first node to be in Partition A to reduce search space by half
    cnf.append([x(v_list[0])])

    # --- Solve ---
    with Glucose3(bootstrap_with=cnf) as solver:
        result = solver.solve()
        if result:
            # If you need to extract the actual cut, you can query the model:
            # model = solver.get_model()
            # partition_A = [u for u in v_list if x(u) in model]
            # partition_B = [u for u in v_list if -x(u) in model]
            return True
        else:
            return False


# def has_small_nonlocal_cut(G, T):
#     return find_small_nonlocal_cut(G, T) is not None


def has_small_nonlocal_cut_(G, T):
    """
    Highly Optimized EXACT Solver using Bounded Search.

    Optimizations:
      1. Incremental Candidate Updates (O(1) instead of O(|S|))
      2. Bucket Sorting (O(N) instead of O(N log N))
      3. Mathematical Pruning (Provably correct bounds)
      4. Integer Mapping (Fast array lookups)
    """
    N = G.number_of_nodes()
    MAX_SIZE = N // 2

    # 1. PREPROCESSING: Map nodes to 0..N-1 for speed
    # (Set lookups on integers are faster than on objects/strings)
    mapping = {n: i for i, n in enumerate(G.nodes())}
    adj = [set() for _ in range(N)]
    for u, v in G.edges():
        ui, vi = mapping[u], mapping[v]
        adj[ui].add(vi)
        adj[vi].add(ui)

    # Global visited cache (Canonical Fingerprints)
    visited_fingerprints = set()

    def search(current_S_tuple, current_S_set, current_cut, current_internal, current_candidates):
        # 1. VICTORY CHECK
        if current_cut <= T:
            # Cycle Condition: Internal Edges >= Nodes
            if current_internal >= len(current_S_tuple):
                return True

        # 2. SIZE LIMIT
        current_size = len(current_S_tuple)
        if current_size >= MAX_SIZE:
            return False

        # 3. MATHEMATICAL PRUNING
        # Max reduction per node is 3 (filling a hole).
        # We have (MAX_SIZE - current_size) nodes left to add.
        # If best possible case still doesn't reach T, prune.
        needed_drop = current_cut - T
        if needed_drop > 0:
            max_possible_drop = 3 * (MAX_SIZE - current_size)
            if max_possible_drop < needed_drop:
                return False

        # 4. CANDIDATE SORTING (Bucket Sort)
        # Avoid generic .sort(). We know k is always 1, 2, or 3.
        # k=3 -> Delta -3 (Best)
        # k=2 -> Delta -1 (Good)
        # k=1 -> Delta +1 (Bad)
        priority_3 = []
        priority_1 = []
        priority_minus_1 = []

        for v in current_candidates:
            # Fast intersection count
            # (Checking 3 neighbors is faster than set.intersection allocation)
            k = 0
            for nbr in adj[v]:
                if nbr in current_S_set:
                    k += 1

            if k == 3:
                priority_3.append((v, -3, 3))
            elif k == 2:
                priority_1.append((v, -1, 2))
            else:
                priority_minus_1.append((v, 1, 1))

        # Chain lists (Implicit sort)
        ordered_candidates = priority_3 + priority_1 + priority_minus_1

        # 5. EXPANSION LOOP
        for v, delta, k in ordered_candidates:
            # Create new S
            # (Using tuple logic for hashing is the standard safe way)
            new_S_list = list(current_S_tuple)
            new_S_list.append(v)
            new_S_list.sort()
            new_fp = tuple(new_S_list)

            if new_fp in visited_fingerprints:
                continue
            visited_fingerprints.add(new_fp)

            # Incremental Candidate Update (Crucial Optimization)
            # New Candidates = (Old - {v}) U (Neighbors of v not in S)
            new_candidates = set(current_candidates)
            new_candidates.remove(v)
            for nbr in adj[v]:
                if nbr not in current_S_set:
                    new_candidates.add(nbr)

            # Recurse
            # We must pass a new set copy for current_S_set to maintain state
            new_S_set = current_S_set.copy()
            new_S_set.add(v)

            if search(new_fp, new_S_set, current_cut + delta, current_internal + k, new_candidates):
                return True

        return False

    for start_node in range(N):
        candidates = set(adj[start_node])

        fp = (start_node,)
        visited_fingerprints.add(fp)

        if search(fp, {start_node}, 3, 0, candidates):
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
    if N % 2 != 0: return None
    if target_girth == 6 and N < 14: return None
    if target_girth == 7 and N < 24: return None
    if target_girth == 8 and N < 30: return None
    if target_girth == 9 and N < 58: return None
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
    lambda_threshold = 10 / 3 * T / N

    for i in range(max_iter):
        if girth > T and lambda_curr >= lambda_threshold:
            if not has_small_nonlocal_cut(G, T):
                return G
            # else:
            #     nx.draw(G)
            #     plt.show()
            #     print(f"Girth: {girth}; lambda: {lambda_curr}")
            #     print(f"No success at iter {i}: Girth {girth}")

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
    # G = find_cubic_graph_with_local_cuts(N=math.ceil((27 * 2 / 3) / 2) * 2, T=3, random_search=True, max_trials=10_000)
    G = construct_cyclic_connected_graph(20, 4)
    if G is not None:
        nx.draw(G)
        plt.show()
        print("Found suitable graph")
    else:
        print("No graph found")