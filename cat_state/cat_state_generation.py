from __future__ import annotations

import typing
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from cat_graphs_circular import random_circular_cubic_graph_with_no_T_nonlocal_cut
from cat_graphs_random import has_small_nonlocal_cut, \
    construct_cyclic_connected_graph
from cat_state.circuit_extraction import find_all_hamiltonian_paths, extract_circuit, unflagged_cat, one_flagged_cat, \
    cat_state_6
from cat_state.markings import GraphMarker
from markings import find_marking_property_violation

if typing.TYPE_CHECKING:
    import stim


def density_lower_bound(t):
    with warnings.catch_warnings(action="ignore"):
        return np.where(
            t == 1, np.inf,
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
    t_alt = np.floor(n / 2) - 1
    t = np.where(t < t_alt, t, t_alt)
    E, N = minimum_E_and_V(n, t)
    return (np.ceil(E - N + 2).astype(int) - 1).tolist()


def visualize_cat_state_base(G, ham_path, markings, pos=None):
    plt.figure(figsize=(5, 5))
    pos = pos or nx.circular_layout(G)  # Kamada-Kawai usually looks best for regular graphs
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={e: "  |  " * num_marks for e, num_marks in markings.items()},
                                 font_size=18, font_weight='bold', bbox=dict(alpha=0))
    nx.draw_networkx_edges(
        G, pos=pos,
        edgelist=ham_path,
        edge_color='red', width=1.5
    )
    plt.show()


def cat_state_FT_circular(num_marks, num_vertices, T, max_iter_graph=1_000, max_new_graphs=25) -> tuple[nx.Graph, list[
    tuple], dict] | None:
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

    return G, ham_path, marks


def cat_state_FT_random(N, T, max_iter_graph=100_000, max_new_graphs=100) -> tuple[nx.Graph, list[tuple], dict] | None:
    try:
        for _ in range(max_new_graphs):
            G = construct_cyclic_connected_graph(N, T, max_iter=max_iter_graph)
            if G is None or has_small_nonlocal_cut(G, T):
                return None

            marker = GraphMarker(G, ham_path=[], max_marks=n)
            marks = marker.find_solution(T)
            if sum(marks.values()) == n:
                p = next(find_all_hamiltonian_paths(G))
                ham_path = list(zip(p, p[1:]))
                marker = GraphMarker(G, ham_path=ham_path, max_marks=n)
                marks = marker.find_solution(T)
                if sum(marks.values()) != n:
                    continue

                break
            else:
                continue
        else:
            return None
    except:
        return None

    return G, ham_path, marks


def cat_state_FT(n, t, allow_non_optimal=True, run_verification=False) -> stim.Circuit | None:
    t_alt = (np.floor(n / 2) - 1).astype(int)
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

    solution_triplet = cat_state_FT_circular(n, N, T, max_new_graphs=10)
    if solution_triplet is None:
        solution_triplet = cat_state_FT_random(N, T)
    if solution_triplet is None:
        return None

    if run_verification and solution_triplet is not None:
        G, H, M = solution_triplet
        violations = find_marking_property_violation(G, M, T)

        if violations is not None:
            print("Edges:", G.edges())
            print("H-path:", H)
            print("Marks:", M)
            print("Violations:", violations)
            print("pos =", nx.circular_layout(G))

            visualize_cat_state_base(G, H, M)
            raise AssertionError

    return extract_circuit(*solution_triplet)


if __name__ == "__main__":
    import time

    start_time = time.time()

    N = 20
    T = 6

    print("Theoretically optimal number of flags for given n and t (from actual circuit instances):")
    print()

    ns = range(2, N + 1)
    print('t\\n |', end=' ')
    for f in ns:
        print(f if f > 9 else f' {f}', end=' ')
    print()
    print("-" * 3 * (N + 1))
    for t in range(1, T + 1):
    # for t in range(T, T + 1):
        print(f"t={t} |", end=' ')
        for n in ns:
            circ = cat_state_FT(n, t, run_verification=False)
            if circ is None:
                flag = "-"
            else:
                flag = circ.num_qubits - n
                if flag != minimum_number_of_flags(n, t):
                    flag = "?"
            print(flag if len(str(flag)) == 2 else f' {flag}', end=' ')
        print()

    print("--- %s seconds ---" % (time.time() - start_time))
