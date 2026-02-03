from __future__ import annotations

import json
import typing
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
from joblib import delayed, Parallel
from mypy.checkexpr import defaultdict

from graphs_circular import random_circular_cubic_graph_with_no_T_nonlocal_cut
from graphs_random import has_small_nonlocal_cut, \
    generate_3regular_graph_with_no_nonlocal_t_cut
from markings import find_marking_property_violation
from spidercat.circuit_extraction import extract_circuit, unflagged_cat, one_flagged_cat, \
    cat_state_6, StimBuilder, extract_circuit_rooted
from spidercat.markings import GraphMarker
from spidercat.path_cover import find_all_path_covers, match_path_ends_to_marked_edges
from spidercat.spanning_tree import build_trivial_spanning_forest, build_min_diameter_spanning_tree, \
    match_forest_leaves_to_marked_edges, find_min_height_roots

if typing.TYPE_CHECKING:
    import stim

cwd = Path.cwd().joinpath("spidercat")
cwd = Path.cwd()


def init_circuits_folder():
    Path(f"{cwd}/circuits").mkdir(parents=True, exist_ok=True)
    Path(f"{cwd}/circuits_data").mkdir(parents=True, exist_ok=True)


def save_stim_circuit(circuit: stim.Circuit, t: int, n: int, p: int):
    file_name = f"{cwd}/circuits/cat_state_t{t}_n{n}_p{p}.stim"
    with open(file_name, "w") as f:
        circuit.to_file(f)


def save_stim_circuit_data(G: nx.Graph, H: list[list[int]], M: dict[tuple[int, int], int], matching, t: int, n: int,
                           p: int):
    file_name = f"{cwd}/circuits_data/cat_state_t{t}_n{n}_p{p}.json"
    with open(file_name, "w") as f:
        M_inv = defaultdict(list)
        for k, v in M.items():
            M_inv[v].append(k)
        json.dump(
            {"G.edges": list(G.edges()), "M_inv": dict(M_inv), "H": list(H.edges()), "matching": matching, "t": t, "n": n, "p": p},
            f,
        )


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


def minimum_number_of_flags(n, t, p=1):
    t_alt = np.floor(n / 2) - 1
    t = np.where(t < t_alt, t, t_alt)
    E, N = minimum_E_and_V(n, t)
    return (np.ceil(E - N + 2).astype(int) - 2 + p).tolist()


def cat_state_FT_circular(num_marks, num_vertices, T, p, max_iter_graph=1_000, max_new_graphs=25) -> tuple[
                                                                                                         nx.Graph, list[
                                                                                                             list[
                                                                                                                 int]], dict] | None:
    for _ in range(max_new_graphs):
        G = random_circular_cubic_graph_with_no_T_nonlocal_cut(num_vertices, T, max_iter=max_iter_graph)
        if G is None:
            continue

        marker = GraphMarker(G, max_marks=num_marks)
        marks = marker.find_solution(T)
        if sum(marks.values()) == num_marks:
            break
    else:
        return None

    forest = build_trivial_spanning_forest(G, marks)
    spacing_tree = build_min_diameter_spanning_tree(G, forest, marks, p)

    return G, spacing_tree, marks


def cat_state_FT_random(n, N, T, p, max_iter_graph=100_000, max_new_graphs=100) -> tuple[
                                                                                       nx.Graph, list[
                                                                                           list[int]], dict] | None:
    for _ in range(max_new_graphs):
        G = generate_3regular_graph_with_no_nonlocal_t_cut(N, T, max_iter=max_iter_graph)
        if G is None or has_small_nonlocal_cut(G, T):
            return None

        marker = GraphMarker(G, path_cover=[], max_marks=n)
        marks = marker.find_solution(T)
        if sum(marks.values()) == n:
            break
        else:
            continue
    else:
        return None

    forest = build_trivial_spanning_forest(G, marks)
    spacing_tree = build_min_diameter_spanning_tree(G, forest, marks, p)

    return G, spacing_tree, marks


def cat_state_FT(n, t, p, allow_non_optimal=True, run_verification=False) -> stim.Circuit | None:
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

    solution_triplet = cat_state_FT_circular(n, N, T, p, max_new_graphs=10)
    if solution_triplet is None:
        solution_triplet = cat_state_FT_random(n, N, T, p, max_new_graphs=100)
    if solution_triplet is None:
        return None

    G, H, M = solution_triplet
    if run_verification:
        violations = find_marking_property_violation(G, M, T)

        if violations is not None:
            print("Edges:", G.edges())
            print("H-path:", H)
            print("Marks:", M)
            print("Violations:", violations)
            print("pos =", nx.circular_layout(G))

            visualize_cat_state_base(G, H, M)
            raise AssertionError

    matchings = match_forest_leaves_to_marked_edges(H, M)
    roots = find_min_height_roots(H)

    save_stim_circuit_data(G, H, M, matchings, t, n, p)

    circ = extract_circuit_rooted(G, H, roots, M, matchings, verbose=False)
    # print(circ.diagram("timeline-text"))

    return circ


def process_cell(n, t, p, cwd, replace=False):
    # Check if file exists
    if not replace and Path(f"{cwd}/circuits/cat_state_t{t}_n{n}.stim").is_file():
        return " x "

    # Generate circuit
    circ = cat_state_FT(n, t, p, run_verification=False)

    # Handle failure to generate
    if circ is None:
        return " - "

    # Check flags
    num_flags = circ.num_qubits - n
    if num_flags != minimum_number_of_flags(n, t, p):
        # print(f'{n=}, {t=}, {p=}')
        # print("num_flags != minimum_number_of_flags(n, t, p)")
        # print("num_flags =", num_flags)
        # print("minimum_number_of_flags(n, t, p) =", minimum_number_of_flags(n, t, p))
        # print(circ.diagram("timeline-text"))
        # assert False
        return " ? "

    # Save and format success output
    # Matches original logic: 2 digits or space+digit, followed by space
    save_stim_circuit(circ, t, n, p)
    return f"{num_flags:>2} "

    # ---------------------------------------------------------


if __name__ == "__main__":
    import time

    start_time = time.time()

    init_circuits_folder()

    P = 4
    N = 30
    T = 4

    print("Generating cat-state preparation circuits with optimal number of flags for given n and t")
    print()

    print("Number of flags for given n and t:")
    print()

    ns = range(8, N + 1)

    for p in range(P, P + 1):
        print("NUM_PATHS: ", p)
        print('t\\n |', end=' ')
        for f in ns:
            print(f if f > 9 else f' {f}', end=' ')
        print()
        print("-" * 3 * (len(ns) + 2))

        # for t in range(3, T + 1):
        for t in range(T, T + 1):
            print(f"t={t} |", end=' ', flush=True)

            results_generator = (process_cell(n, t, p, cwd, True) for n in ns)

            # results_generator = Parallel(n_jobs=-2, return_as="generator")(
            #     delayed(process_cell)(n, t, p, cwd, replace=True) for n in ns
            # )
            for cell_str in results_generator:
                print(cell_str, end='', flush=True)
            print()
        print()
    print(f"Files saved to: {cwd}/circuits")
    print()
    print("--- %s seconds ---" % (time.time() - start_time))
