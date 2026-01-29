import matplotlib.pyplot as plt
import networkx as nx
from pysat.card import CardEnc
from pysat.formula import IDPool, CNF
from pysat.solvers import Glucose3


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


def generate_3regular_graph_with_no_nonlocal_t_cut(N, T, max_iter=1_000):
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
    G = generate_3regular_graph_with_no_nonlocal_t_cut(20, 4)
    if G is not None:
        nx.draw(G)
        plt.show()
        print("Found suitable graph")
    else:
        print("No graph found")
