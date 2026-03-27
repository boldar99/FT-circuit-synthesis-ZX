import itertools
from collections import defaultdict

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
    # new_cnots, fixed_output = ensure_last_flag_cnots_order(cnots, n, circuit.num_qubits - n)
    layered_cnots = _layer_cnot_circuit(cnots)
    layered_stim_triplets = [("CX", flatten(cnots), 0) for cnots in layered_cnots]

    return before_cnots + layered_stim_triplets + after_cnots


def flatten(ls: list) -> list:
    return list(itertools.chain(*ls))


def get_output_ordering(circuit, N):
    flat_cnots = []
    cnots = []
    for op, ixs, _ in circuit.flattened_operations():
        if op == "CX":
            for i in range(0, len(ixs), 2):
                flat_cnots.append(ixs[i])
                flat_cnots.append(ixs[i + 1])
                cnots.append((ixs[i], ixs[i + 1]))
    cnot_layers = _layer_cnot_circuit(cnots)
    flat_layers = [flatten(layer) for layer in cnot_layers]

    qubit_to_layer = {}
    for q in range(1, N):
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


def get_connectivity_ordering(non_pivots: list[int], parity_matrix: np.ndarray,
                              x_circuits: list[tuple[stim.Circuit, int]],
                              z_circuits: list[tuple[stim.Circuit, int]]) -> list[
    tuple[tuple[int, int], tuple[int, int]]]:
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
    circuit = stim.Circuit()

    parity_matrix = np.asarray(parity_matrix)
    N = parity_matrix.shape[1]
    pivots, rows_without_pivots = find_pivots_in_matrix(parity_matrix)
    non_pivots = [p for p in range(N) if p not in pivots.values()]
    assert len(rows_without_pivots) == 0

    z_spiders = np.sum(parity_matrix, axis=1)
    x_spiders = np.sum(parity_matrix[:, non_pivots], axis=0) + 1
    print(z_spiders)
    print(x_spiders)

    z_circuits = [load_ft_cat_state(zs, t) for zs in z_spiders]
    x_circuits = [circuit_in_dual_basis(load_ft_cat_state(xs, t)) for xs in x_spiders]

    layered_z_circuits = [layer_stim_circuit(z_circ, z) for z_circ, z in zip(z_circuits, z_spiders)]
    layered_x_circuits = [layer_stim_circuit(x_circ, x) for x_circ, x in zip(x_circuits, x_spiders)]

    offsets, global_offsets = calculate_offsets(N, non_pivots, x_circuits, z_circuits, x_spiders, z_spiders)
    x_spider_offsets = [o for i, o in enumerate(offsets) if i in non_pivots]
    z_spider_offsets = [o for i, o in enumerate(offsets) if i not in non_pivots]
    print(offsets)

    z_internal_mappings = [
        calculate_internal_qubit_mapping(zs, z_circuits[i].num_qubits - zs)
        for i, zs in enumerate(z_spiders)
    ]
    x_internal_mappings = [
        calculate_internal_qubit_mapping(xs, x_circuits[i].num_qubits - xs)
        for i, xs in enumerate(x_spiders)
    ]

    edge_to_qubits = get_connectivity_ordering(
        non_pivots, parity_matrix, zip(x_circuits, x_spiders), zip(z_circuits, z_spiders)
    )

    for i, lc in enumerate(layered_z_circuits):
        while lc[0][0] == "H":
            name, targets, _ = lc.pop(0)
            z_internal_mapping = z_internal_mappings[i]
            z_offset = z_spider_offsets[i]
            offset_targets = [global_offsets[z_internal_mapping[t] + z_offset] for t in targets if t in z_internal_mapping]
            circuit.append(name, offset_targets)

    while edge_to_qubits:
        (z_spider_index, x_spider_index), (z_internal_qubit, x_internal_qubit) = edge_to_qubits.pop(0)

        layered_z_circuit = layered_z_circuits[z_spider_index]
        layered_x_circuit = layered_x_circuits[x_spider_index]

        z_offset = z_spider_offsets[z_spider_index]
        x_offset = x_spider_offsets[x_spider_index]

        z_k = num_layers_to_last_use_of_qubit(layered_z_circuit, z_internal_qubit)
        x_k = num_layers_to_last_use_of_qubit(layered_x_circuit, x_internal_qubit)

        z_internal_mapping = z_internal_mappings[z_spider_index]
        x_internal_mapping = x_internal_mappings[x_spider_index]

        for _ in range(z_k):
            name, targets, _ = layered_z_circuit.pop(0)
            offset_targets = []
            if name == "CX":
                for i in range(0, len(targets), 2):
                    if targets[i] not in z_internal_mapping or targets[i + 1] not in z_internal_mapping:
                        continue
                    offset_targets.append(global_offsets[z_internal_mapping[targets[i]] + z_offset])
                    offset_targets.append(global_offsets[z_internal_mapping[targets[i + 1]] + z_offset])
            else:
                offset_targets = [global_offsets[z_internal_mapping[t] + z_offset] for t in targets if t in z_internal_mapping]

            circuit.append(name, offset_targets)
        for _ in range(x_k):
            name, targets, _ = layered_x_circuit.pop(0)
            offset_targets = []
            if name == "CX":
                for i in range(0, len(targets), 2):
                    if targets[i] not in x_internal_mapping or targets[i + 1] not in x_internal_mapping:
                        continue
                    offset_targets.append(global_offsets[x_internal_mapping[targets[i]] + x_offset])
                    offset_targets.append(global_offsets[x_internal_mapping[targets[i + 1]] + x_offset])
            else:
                offset_targets = [global_offsets[x_internal_mapping[t] + x_offset] for t in targets if t in x_internal_mapping]

            circuit.append(name, offset_targets)

        c = remove_from_first_layer(layered_z_circuit, z_internal_qubit)
        n = remove_from_first_layer(layered_x_circuit, x_internal_qubit)
        circuit.append("CX", [global_offsets[z_internal_mapping[c] + z_offset], global_offsets[x_internal_mapping[n] + x_offset]])

    for remaining_layers, offset, z_internal_mapping in zip(layered_z_circuits, z_spider_offsets, z_internal_mappings):
        for name, targets, _ in remaining_layers:
            targets = [stim.target_rec(t[1]) if isinstance(t, tuple) else global_offsets[z_internal_mapping[t] + offset] for t in targets]
            circuit.append(name, targets)

    for remaining_layers, offset, x_internal_mapping in zip(layered_x_circuits, x_spider_offsets, x_internal_mappings):
        for name, targets, _ in remaining_layers:
            targets = [stim.target_rec(t[1]) if isinstance(t, tuple) else global_offsets[x_internal_mapping[t] + offset] for t in targets]
            circuit.append(name, targets)

    return circuit


def calculate_offsets(N, non_pivots: list[int], x_circuits: list[stim.Circuit], z_circuits: list[stim.Circuit],
                      x_spiders, z_spiders) -> tuple[list[int], dict[int, int]]:
    o = 0
    offsets = []
    flag_qubit = N
    global_mapping = {}
    i, j = 0, 0
    while i + j < N:
        if i + j in non_pivots:
            flag_count = x_circuits[i].num_qubits - x_spiders[i]
            i += 1
        else:
            flag_count = z_circuits[j].num_qubits - z_spiders[j]
            j += 1
        global_mapping[o] = i + j - 1
        for f in range(flag_count):
            global_mapping[o + f + 1] = flag_qubit
            flag_qubit += 1
        offsets.append(o)
        o += flag_count + 1
    return offsets, global_mapping


def calculate_internal_qubit_mapping(num_qubits, num_flags) -> dict[int, int]:
    ret = {0: 0}
    for q, f in enumerate(range(num_flags), 1):
        ret[f + num_qubits] = q
    return ret



# def calculate_internal_qubit_mapping(layered_z_circuit, output_qubit):
#     mapping = {output_qubit: 0}
#     for i, f in enumerate(flags, 1):
#         mapping[f] = i


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
    print(flag_by_construction(H_x, 1))
