import itertools
from collections import defaultdict
from pprint import pprint

import numpy as np
import stim

from spidercat.simulate import _layer_cnot_circuit


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
    new_cnots, fixed_output = ensure_last_flag_cnots_order(cnots, n, circuit.num_qubits - n)
    layered_cnots = _layer_cnot_circuit(cnots)
    layered_stim_triplets = [("CX", flatten(cnots), 0) for cnots in layered_cnots]

    return before_cnots + layered_stim_triplets + after_cnots


from collections import defaultdict


def ensure_last_flag_cnots_order(cnots: list[tuple[int, int]], num_qubits, num_flags):
    # Helper to determine if two non-flag CNOTs can move past each other
    ret = None

    def commutes(cnot1, cnot2):
        """
        Strict constraint: Operations can only move past each other if they
        are completely disjoint (share zero qubits).
        """
        c1, t1 = cnot1
        c2, t2 = cnot2

        # The set of qubits involved must be exactly 4, meaning no overlap.
        return len({c1, t1, c2, t2}) == 4

    result_cnots = list(cnots)  # Work on a copy to allow safe mutation
    last_cnot_per_qubit = {}
    first_cnot_per_qubit = {}
    num_cnots_per_qubit = defaultdict(int)

    for q in range(num_qubits + num_flags):
        for i, cnot in enumerate(result_cnots):
            if q in cnot:
                last_cnot_per_qubit[q] = i, cnot
                num_cnots_per_qubit[q] += 1
                if q not in first_cnot_per_qubit:
                    first_cnot_per_qubit[q] = i, cnot

    swapped_flags = set()

    for q in range(num_qubits):
        if q not in last_cnot_per_qubit:
            continue

        _, (c, n) = last_cnot_per_qubit[q]
        flag = None

        if q == c and n >= num_qubits:
            flag = n
        elif q == n and c >= num_qubits:
            flag = c

        if flag is not None and last_cnot_per_qubit[flag] == last_cnot_per_qubit[q]:
            assert num_cnots_per_qubit[flag] == 2

            if flag in swapped_flags:
                continue

            # Dynamically find the CURRENT indices to avoid the stale dictionary problem
            f_indices = [idx for idx, gate in enumerate(result_cnots) if gate[0] == flag or gate[1] == flag]
            idx1, idx2 = f_indices[0], f_indices[1]

            # 1. Commute the last CNOT (idx2) to the left
            while idx2 > idx1:
                if commutes(result_cnots[idx2], result_cnots[idx2 - 1]):
                    # Swap adjacent elements
                    result_cnots[idx2], result_cnots[idx2 - 1] = result_cnots[idx2 - 1], result_cnots[idx2]
                    idx2 -= 1
                else:
                    break  # Blocked by an interacting qubit

            # 2. If blocked, commute the first CNOT (idx1) to the right
            while idx1 < idx2 - 1:
                if commutes(result_cnots[idx1], result_cnots[idx1 + 1]):
                    # Swap adjacent elements
                    result_cnots[idx1], result_cnots[idx1 + 1] = result_cnots[idx1 + 1], result_cnots[idx1]
                    idx1 += 1
                else:
                    break  # Blocked by an interacting qubit

            # 3. Final Evaluation: Are they adjacent?
            if idx2 - idx1 == 1:
                # Rule 1 applies: commute the two CNOTs on the flag qubit
                result_cnots[idx1], result_cnots[idx2] = result_cnots[idx2], result_cnots[idx1]
                swapped_flags.add(flag)
            else:
                if ret is None:
                    ret = q
                else:
                    # Prioritize truth: Do not silently return a failed state
                    raise ValueError(
                        f"Flag {flag} CNOTs cannot be swapped. They are blocked by "
                        f"intermediate operations {result_cnots[idx1 + 1:idx2]} that do not commute."
                    )

    return result_cnots, ret




def flatten(ls: list) -> list:
    return list(itertools.chain(*ls))


def get_output_ordering(circuit, N):
    flat_cnots = []
    cnots = []
    for op, ixs, _ in circuit.flattened_operations():
        if op == "CX":
            for i in range(0, len(ixs), 2):
                flat_cnots.append(ixs[i])
                flat_cnots.append(ixs[i+1])
                cnots.append((ixs[i], ixs[i+1]))
    cnot_layers = _layer_cnot_circuit(cnots)
    flat_layers = [flatten(layer) for layer in cnot_layers]

    qubit_to_layer = {}
    for q in range(N):
        last_occurrence = 0
        for i, layer in enumerate(flat_layers):
            if q in layer:
                last_occurrence = i
        qubit_to_layer[q] = last_occurrence


    return qubit_to_layer


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


def circuit_in_dual_basis(circuit):
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


def get_connectivity_ordering(non_pivots: list[int], parity_matrix: np.ndarray, x_circuits: list[tuple[stim.Circuit, int]],
                              z_circuits: list[tuple[stim.Circuit, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    z_output_availability = [get_output_ordering(c, n) for c, n in z_circuits]
    x_output_availability = [get_output_ordering(c, n) for c, n in x_circuits]

    edge_list = [
        (i, j)
        for i, r in enumerate(parity_matrix)
        for j, x in enumerate(r[non_pivots])
        if x == 1
    ]

    edge_to_qubits = []
    while edge_list:
        min_cost = np.inf
        best_edge = (0, 0)
        best_edge_qubit_indices = (0, 0)
        for (i, j) in edge_list:
            zs = z_output_availability[i]
            xs = x_output_availability[j]
            zx_min_ix, zs_min = min(zs.items(), key=lambda p: p[1], default=-1)
            xs_min_ix, xs_min = min(xs.items(), key=lambda p: p[1], default=-1)
            cost = max(zs_min, xs_min)
            if cost < min_cost:
                min_cost = cost
                best_edge = (i, j)
                best_edge_qubit_indices = (zx_min_ix, xs_min_ix)

        edge_to_qubits.append((best_edge, best_edge_qubit_indices))
        edge_list.remove(best_edge)
        del z_output_availability[best_edge[0]][best_edge_qubit_indices[0]]
        del x_output_availability[best_edge[1]][best_edge_qubit_indices[1]]
    return edge_to_qubits


def num_layers_to_last_use_of_qubit(layered_operations, qubit) -> int:
    k = 0
    for i, (name, targets, _) in enumerate(layered_operations):
        if name == "CX" and qubit in targets:
            k = i
    return k


def remove_from_first_layer(layered_operations, qubit):
    targets = layered_operations[0][1]
    zi = targets.index(qubit)
    targets.pop(zi)
    ret = targets.pop(zi - zi % 2)
    if len(targets) == 0:
        layered_operations.pop(0)
    return ret


def flag_by_construction(parity_matrix, t):
    parity_matrix = np.asarray(parity_matrix)
    N = parity_matrix.shape[1]
    pivots, rows_without_pivots = find_pivots_in_matrix(parity_matrix)
    non_pivots = [p for p in range(N) if p not in pivots.values()]
    assert len(rows_without_pivots) == 0

    z_spiders = np.sum(parity_matrix, axis=1)
    x_spiders = np.sum(parity_matrix[:,non_pivots], axis=0) + 1
    z_circuits = [load_ft_cat_state(zs, t) for zs in z_spiders]
    x_circuits = [circuit_in_dual_basis(load_ft_cat_state(xs, t)) for xs in x_spiders]

    layered_z_circuits = [layer_stim_circuit(z_circ, z) for z_circ, z in zip(z_circuits, z_spiders)]
    layered_x_circuits = [layer_stim_circuit(x_circ, x) for x_circ, x in zip(x_circuits, x_spiders)]

    offsets = calculate_offsets(N, non_pivots, x_circuits, z_circuits, x_spiders, z_spiders)
    x_spider_offsets = [o for i, o in enumerate(offsets) if i in non_pivots]
    z_spider_offsets = [o for i, o in enumerate(offsets) if i not in non_pivots]

    edge_to_qubits = get_connectivity_ordering(non_pivots, parity_matrix, zip(x_circuits, x_spiders), zip(z_circuits, z_spiders))

    circuit = stim.Circuit()
    while edge_to_qubits:
        (z_spider_index, x_spider_index), (z_internal_qubit, x_internal_qubit) = edge_to_qubits.pop(0)

        layered_z_circuit = layered_z_circuits[z_spider_index]
        layered_x_circuit = layered_x_circuits[x_spider_index]

        z_offset = z_spider_offsets[z_spider_index]
        x_offset = x_spider_offsets[x_spider_index]

        z_k = num_layers_to_last_use_of_qubit(layered_z_circuit, z_internal_qubit)
        x_k = num_layers_to_last_use_of_qubit(layered_x_circuit, x_internal_qubit)

        for _ in range(z_k):
            name, targets, _ = layered_z_circuit.pop(0)
            targets = [t + z_offset for t in targets]
            circuit.append(name, targets)
        for _ in range(x_k):
            name, targets, _ = layered_x_circuit.pop(0)
            targets = [t + x_offset for t in targets]
            circuit.append(name, targets)

        c = remove_from_first_layer(layered_z_circuit, z_internal_qubit)
        n = remove_from_first_layer(layered_x_circuit, x_internal_qubit)
        circuit.append("CX", [c + z_offset, n + x_offset])

    for remaining_layers, offset in zip(layered_z_circuits + layered_x_circuits, z_spider_offsets + x_spider_offsets):
        for name, targets, _ in remaining_layers:
            targets = [stim.target_rec(t[1]) if isinstance(t, tuple) else t + offset for t in targets]
            circuit.append(name, targets)

    return circuit


def calculate_offsets(N, non_pivots: list[int], x_circuits: list[stim.Circuit], z_circuits: list[stim.Circuit], x_spiders, z_spiders) -> list[int]:
    o = 0
    offsets = []
    i, j = 0, 0
    while i + j < N:
        if i + j in non_pivots:
            flag_count = x_circuits[i].num_qubits - x_spiders[i]
            i += 1
        else:
            flag_count = z_circuits[j].num_qubits - z_spiders[j]
            j += 1
        offsets.append(o)
        o += flag_count + 1
    return offsets


def calculate_internal_qubit_mapping(layered_z_circuit, output_qubit):
    mapping = {output_qubit: 0}
    for i, f in enumerate(flags, 1):
        mapping[f] = i



if __name__ == "__main__":

    # # --- Example Usage ---
    # H_x = np.array([
    #     [1, 1, 1, 1, 0, 0, 0],
    #     [0, 1, 1, 0, 1, 1, 0],
    #     [0, 0, 1, 1, 0, 1, 1]
    # ])
    H_x = np.array([
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ])

    circ = stim.Circuit("""
        H 0
        CX 0 1 1 2 1 10 1 11 0 3 3 4 3 12 3 5 3 11
        M 11
        DETECTOR rec[-1]
        CX 3 6 3 7 7 13 3 14 0 8 0 9 9 13
        M 13
        DETECTOR rec[-1]
        CX 9 12
        M 12
        DETECTOR rec[-1]
        CX 0 14
        M 14
        DETECTOR rec[-1]
        CX 0 10
        M 10
        DETECTOR rec[-1]
    """)
    # pprint(layer_stim_circuit(circ))

    # print(circuit_in_dual_basis(circ))
    print(flag_by_construction(H_x, 3))
