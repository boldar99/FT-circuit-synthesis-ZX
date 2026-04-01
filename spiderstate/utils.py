import itertools
from functools import lru_cache

import networkx as nx
import numpy as np
import stim

from spidercat.circuit_extraction import expand_graph_and_forest, build_traversal_digraph, CatStateExtractor, \
    StimBuilder, resolve_dag_by_removing_missing_link
from spidercat.mdsf import constrained_mdsf_generation
from spidercat.simulate import _layer_cnot_circuit
from spidercat.spanning_tree import find_min_height_degree_3_roots
from spidercat.utils import load_solution_triplet


def layer_stim_circuit(circuit: stim.Circuit, n):
    """
    Layers an unbatched list of operation triplets into non-interacting layers.
    Returns a list of layers, where each layer is a list of triplets.
    """
    before_cnots = []
    cnots = []
    after_cnots = []
    for name, targets, args in circuit.flattened_operations():
        if name == "CX":
            cnots += [(targets[i], targets[i + 1]) for i in range(0, len(targets), 2)]
        elif len(cnots) == 0:
            before_cnots.append((name, targets, args))
        else:
            after_cnots.append((name, targets, args))
    # new_cnots, fixed_output = ensure_last_flag_cnots_order(cnots, n, circuit.num_qubits - n)
    layered_cnots = _layer_cnot_circuit(cnots)
    layered_stim_triplets = [("CX", flatten(cnots), 0) for cnots in layered_cnots]

    return before_cnots + layered_stim_triplets + after_cnots


def flatten(ls: list) -> list:
    return list(itertools.chain(*ls))


def find_pivots_in_matrix(parity_matrix):
    r, c = parity_matrix.shape

    # Dictionary to store {row_index: pivot_column_index}
    pivots = {}
    # List to track any rows that do not have a valid pivot
    rows_without_pivots = []

    for i in range(r):
        # 1. Find all columns where the current row has a '1'
        candidate_cols = np.where(parity_matrix[i] == 1)[0]

        found_pivot = False
        for j in candidate_cols:
            # 2. Check if this column is a valid pivot (the sum of the column must be exactly 1)
            if np.sum(parity_matrix[:, j]) == 1:
                pivots[i] = int(j)
                found_pivot = True
                break  # We only need one pivot per row

        if not found_pivot:
            rows_without_pivots.append(i)

    return pivots, rows_without_pivots


def cat_state_circuit_in_dual_basis(circuit):
    ret = stim.Circuit()
    assert circuit[0].name == "H"
    hs = [t.value for t in circuit[0].targets_copy()]
    ret.append("H", [q for q in range(circuit.num_qubits) if q not in hs])
    for op in circuit[1:]:
        targets = op.targets_copy()
        if op.name == "CX":
            for i in range(0, len(targets), 2):
                ret.append("CX", [targets[i + 1], targets[i]])
        elif op.name == "M":
            ret.append("MX", targets)
        elif op.name == "R":
            ret.append("RX", targets)
        elif op.name == "DETECTOR":
            ret.append(op)
        else:
            raise NotImplementedError

    return ret

def load_ft_cat_state(n, t):
    with open(f"../spidercat/circuits/cat_state_t{t}_n{n}_p1.stim", "r") as f:
        return stim.Circuit(f.read())


def G_F_alt_for_t_0(N):
    G = nx.Graph()
    G.add_nodes_from([0])
    G.add_nodes_from(range(1, N + 1), is_mark=True)
    for i in range(N):
        G.add_edge(i, i + 1)
    F = G.copy()
    return G, F, 0


def G_F_alt_for_t_1(N):
    G = nx.Graph()
    G.add_nodes_from([0])
    G.add_nodes_from(range(2, 2 + N), is_mark=True)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    for i in range(N - 2):
        G.add_edge(2 + i, 4 + i)
    G.add_edge(N, N + 1)
    F = G.copy()
    F.remove_edge(N + 1, N)
    return G, F, 0


def G_F_n_6():
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_nodes_from(range(2, 8), is_mark=True)
    for i in range(3):
        G.add_edge(0, i + 2)
        G.add_edge(1, i + 5)
        G.add_edge(i + 2, i + 5)

    F = G.copy()
    F.remove_edge(0, 4)
    F.remove_edge(1, 5)
    return G, F, 0


def well_ordered_ft_cat_state_data(n, t):
    if n <= 3:
        G_alt, F_alt, root = G_F_alt_for_t_0(n)
        roots = [root]
        e = n
    elif t == 1 or n <= 5:
        G_alt, F_alt, root = G_F_alt_for_t_1(n)
        roots = [root]
    elif n == 6:
        G_alt, F_alt, root = G_F_n_6()
        roots = [root]
    else:
        grf, tree, M, matchings = load_solution_triplet(n, t, 1)
        G_alt, _ = expand_graph_and_forest(grf, tree, M, matchings, expand_flags=False)
        F_alt = constrained_mdsf_generation(G_alt, 1, seed=9001)
        F_alt = F_alt.copy()
        roots = find_min_height_degree_3_roots(F_alt)
    D = build_traversal_digraph(G_alt, F_alt, roots[0])
    _, edge, dependency_graph = resolve_dag_by_removing_missing_link(D)

    return G_alt, F_alt, roots, dependency_graph, edge[0][0] if len(edge) else e


def well_ordered_ft_cat_state(n, t):
    G_alt, F_alt, roots, dependency_graph, edge = well_ordered_ft_cat_state_data(n, t)
    extractor = CatStateExtractor(StimBuilder(), verbose=False)
    circ = extractor.extract(G_alt, F_alt, roots, dependency_graph)
    return circ, extractor.node_to_qubit[edge]
