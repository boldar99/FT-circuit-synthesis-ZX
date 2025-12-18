import warnings
from itertools import combinations

from cat_graphs_circular import random_circular_cubic_graph_with_no_T_nonlocal_cut
from cat_graphs_random import generate_high_girth_cubic_graph, has_small_nonlocal_cut, \
    construct_cyclic_connected_graph
from markings import verify_marking_property, find_marking_property_violation

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import stim

from cat_state.markings import GraphMarker


def density_lower_bound(t):
    with warnings.catch_warnings(action="ignore"):
        return np.where(t == 1, np.inf,
            (
                np.ceil((t + 3) / 2)
                * np.floor((t + 3) / 2)
            )
            /
            (
                np.ceil((t + 3) / 2)
                * np.floor((t + 3) / 2)
                +
                np.ceil((t - 3) / 2)
                * np.floor((t + 3) / 2)
                +
                np.floor((t - 3) / 2)
                * np.ceil((t + 3) / 2)
            )
        )


def minimum_E_and_V(n, t):
    density = density_lower_bound(t)
    E_nec = np.ceil(n / density).astype(int)
    remainder = E_nec % 3
    adjustment = (3 - remainder) % 3
    E_final = E_nec + adjustment
    V_final = (2 * E_final) // 3
    return E_final.tolist(), V_final.tolist()


def minimum_number_of_flags(n, t):
    t_alt =  np.floor(n / 2) - 1
    t = np.where(t < t_alt, t, t_alt)
    E, N = minimum_E_and_V(n, t)
    return (np.ceil(E - N + 2).astype(int) - 1).tolist()


def visualize_cat_state_base(G, ham_path, markings):
    plt.figure(figsize=(5, 5))
    pos = nx.circular_layout(G) # Kamada-Kawai usually looks best for regular graphs
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={e: "  |  " * num_marks for e, num_marks in markings.items()},
                                 font_size=18, font_weight='bold', bbox=dict(alpha=0))
    nx.draw_networkx_edges(
        G, pos=pos,
        edgelist=ham_path,
        edge_color='red', width=1.5
    )
    plt.show()


def find_all_hamiltonian_paths(graph):
    """
    Optimized for 3-regular graphs.
    Uses bitmasks and static adjacency tuples for maximum speed.
    """
    nodes = list(graph.nodes)
    n = len(nodes)

    # 1. Map nodes to integers 0..N-1
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 2. Create a static Adjacency Tuple (faster than lists)
    # Since it's 3-regular, every row has exactly 3 entries.
    adj = [None] * n
    for node in nodes:
        u = node_to_idx[node]
        # Convert neighbors to mapped integers
        neighbors = tuple(node_to_idx[v] for v in graph.neighbors(node))
        adj[u] = neighbors

    # Convert list of tuples to tuple of tuples for fastest read access
    adj = tuple(adj)

    # Pre-allocate path array to avoid list creation overhead
    path = [0] * n

    # Pre-compute bit powers to avoid bit-shifting in the tight loop
    powers = [1 << i for i in range(n)]

    def solve(u, pos, mask):
        path[pos] = u

        # Base Case: Path complete
        if pos == n - 1:
            # Yield the recovered node objects
            yield [nodes[i] for i in path]
            return

        # Recursive Step: Unrolled for performance
        # We iterate over the fixed tuple of neighbors
        for v in adj[u]:
            # Bitwise check: if (mask & 2^v) == 0
            if not (mask & powers[v]):
                yield from solve(v, pos + 1, mask | powers[v])

    # 3. Execution Strategy
    # Iterate through all start nodes
    for i in range(n):
        yield from solve(i, 0, powers[i])


def sorted_pair(v1, v2):
    return (v1, v2) if v1 < v2 else (v2, v1)


def extract_circuit(G, ham_path, marks: dict | list):
    circ = stim.Circuit()
    if isinstance(marks, dict):
        marks = {sorted_pair(v1, v2): int(v) for (v1, v2), v in marks.items()}
    else:
        marks = {sorted_pair(v1, v2): 1 for v1, v2 in marks}


    num_flags = G.number_of_edges() - len(ham_path)
    flag_dict = dict()

    v0, v1 = ham_path[0]
    neighbors_0 = tuple(set(G.neighbors(v0)) - {v1})
    flag_dict[sorted_pair(v0, neighbors_0[0])] = 0
    flag_dict[sorted_pair(v0, neighbors_0[1])] = 1

    circ.append("H", num_flags)
    circ.append("CNOT", [num_flags, 0])
    circ.append("CNOT", [num_flags, 1])

    next_free_flag = 2
    next_free_cat = num_flags + 1

    for _ in range(marks.get(sorted_pair(v0, v1), 0)):
        circ.append("CNOT", [num_flags, next_free_cat])
        next_free_cat += 1

    v_prev = v0
    v_current, v_next = None, None
    for v_current, v_next in ham_path[1:]:
        if len(set(G.neighbors(v_current)) - {v_prev, v_next}) != 1:
            pass
        [v_neighbor] = set(G.neighbors(v_current)) - {v_prev, v_next}
        link = sorted_pair(v_current, v_neighbor)

        if link not in flag_dict:
            circ.append("CNOT", [num_flags, next_free_flag])
            flag_dict[link] = next_free_flag
            next_free_flag += 1
        else:
            flag_qubit = flag_dict[link]

            for _ in range(marks.get(link, 0)):
                circ.append("CNOT", [flag_qubit, next_free_cat])
                next_free_cat += 1

            circ.append("CNOT", [num_flags, flag_qubit])
            circ.append("MR", flag_qubit)

        for _ in range(marks.get(sorted_pair(v_current, v_next), 0)):
            circ.append("CNOT", [num_flags, next_free_cat])
            next_free_cat += 1

        v_prev = v_current

    if len(ham_path) > 1:
        neighbors_last = tuple(set(G.neighbors(v_next)) - {v_current})
        link_penultimate = sorted_pair(v_next, neighbors_last[0])
        link_last = sorted_pair(v_next, neighbors_last[1])
        num_cat_legs = marks.get(link_penultimate, 0) + marks.get(link_last, 0)
        i = 0

        for _ in range(marks.get(link_penultimate, 0)):
            i += 1
            if i != num_cat_legs:
                circ.append("CNOT", [flag_dict[link_penultimate], next_free_cat])
                next_free_cat += 1
        for _ in range(marks.get(link_last, 0)):
            i += 1
            if i != num_cat_legs:
                circ.append("CNOT", [flag_dict[link_last], next_free_cat])
                next_free_cat += 1

        circ.append("CNOT", [num_flags, flag_dict[link_penultimate]])
        circ.append("CNOT", [num_flags, flag_dict[link_last]])
        circ.append("MR", flag_dict[link_penultimate])
        circ.append("MR", flag_dict[link_last])

    return circ


def unflagged_cat(n):
    circ = stim.Circuit()
    circ.append("H", 0)
    for i in range(1, n):
        circ.append("CNOT", [0, i])
    return circ


def one_flagged_cat(n):
    circ = stim.Circuit()
    circ.append("H", 1)
    circ.append("CNOT", [0, 1])
    for i in range(2, n+1):
        circ.append("CNOT", [0, i])
    circ.append("CNOT", [0, 1])
    circ.append("MR", 0)
    return circ

def cat_state_6():
    return stim.Circuit("""
        H 2
        CNOT 2 3 2 1 2 4 2 0 2 5 2 6 2 1 2 7 2 0
        MR 0 1 
    """)


def cat_state_FT_circular(num_marks, num_vertices, T, max_iter_graph=1_000, max_new_graphs=25, run_verification=False) -> stim.Circuit | None:
    for _ in range(max_new_graphs):
        G = random_circular_cubic_graph_with_no_T_nonlocal_cut(num_vertices, T, max_iter=max_iter_graph)
        if G is None:
            continue

        ham_cycle = [(i, (i + 1) % num_vertices) for i in range(num_vertices)]
        ham_path = ham_cycle[:-1]
        marker = GraphMarker(G, ham_path=ham_path, max_marks=num_marks)
        marks = marker.find_solution(T)
        if sum(marks.values()) == num_marks:
            break
    else:
        return None

    circ = extract_circuit(G, ham_path, marks)
    if run_verification:
        violations = find_marking_property_violation(G, marks, T)

        if violations is not None:
            print("Edges:", G.edges())
            print("H-path:", ham_path)
            print("Marks:", marks)
            print("Violations:", violations)
            raise AssertionError
    return circ


def cat_state_FT_random(num_marks, num_vertices, T, max_iter_graph=100_000, max_new_graphs=100) -> stim.Circuit | None:
    for _ in range(max_new_graphs):
        G = construct_cyclic_connected_graph(num_vertices, T, max_iter=max_iter_graph)
        if G is None or has_small_nonlocal_cut(G, T):
            return None

        marker = GraphMarker(G, ham_path=[], max_marks=num_marks)
        marks = marker.find_solution(T)
        if sum(marks.values()) == num_marks:
            p = next(find_all_hamiltonian_paths(G))
            ham_path = list(zip(p, p[1:]))
            marker = GraphMarker(G, ham_path=ham_path, max_marks=num_marks)
            marks = marker.find_solution(T)
            if sum(marks.values()) != num_marks:
                continue

            break
        else:
            continue
    else:
        return None

    violations = find_marking_property_violation(G, marks, T)

    if violations is not None:
        print("Violations:", violations)
        print("Edges:", G.edges())
        print("H-path:", ham_path)
        print("Marks:", marks)
        visualize_cat_state_base(G, ham_path, marks)
        raise AssertionError
    return extract_circuit(G, ham_path, marks)


def cat_state_FT(n, t) -> stim.Circuit | None:
    t_alt =  (np.floor(n / 2) - 1).astype(int)
    T = min(t, t_alt)

    if n < 1:
        raise ValueError
    if n <= 3:
        return unflagged_cat(n)
    if n <= 5 or T == 1:
        return one_flagged_cat(n)
    if n == 6:
        return cat_state_6()

    E, N = minimum_E_and_V(n, T)
    circ = cat_state_FT_circular(n, N, T, max_new_graphs=5, run_verification=False)
    if circ is not None:
        return circ
    # if T <= 5 and N < 30:
    #     return cat_state_FT_random(N, T)
    return None


if __name__ == "__main__":
    import time
    start_time = time.time()

    N = 31
    T = 7

    print("Theoretically optimal number of flags for given n and t (from actual circuit instances):")
    print()

    ns = range(2, N)
    print('t\\n |', end=' ')
    for f in ns:
        print(f if f > 9 else f' {f}', end=' ')
    print()
    print("-" * 3 * N)
    for t in range(1, T):
    # for t in range(T - 1, T):
        print(f"t={t} |", end=' ')
        for n in ns:
            circ = cat_state_FT(n, t)
            if circ is None:
                flag = "-"
            else:
                flag = circ.num_qubits - n
                if flag != minimum_number_of_flags(n, t):
                    flag = "?"
            print(flag if len(str(flag)) == 2 else f' {flag}', end=' ')
        print()

    print("--- %s seconds ---" % (time.time() - start_time))
