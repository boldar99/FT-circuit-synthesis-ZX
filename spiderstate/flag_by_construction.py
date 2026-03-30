from functools import lru_cache

import numpy as np
import stim

from spidercat.circuit_extraction import expand_graph_and_forest, build_traversal_digraph, \
    resolve_dag_by_removing_missing_link, CatStateExtractor, StimBuilder
from spidercat.mdsf import constrained_mdsf_generation
from spidercat.simulate import _layer_cnot_circuit
from spidercat.spanning_tree import find_min_height_degree_3_roots
from spidercat.utils import load_solution_triplet
from spiderstate.utils import layer_stim_circuit, flatten, find_pivots_in_matrix, cat_state_circuit_in_dual_basis, \
    well_ordered_ft_cat_state


class CatAtOrigin:
    def __init__(self, H: np.ndarray, distance: int):
        self.d = distance
        self.t = (distance - 1) // 2
        self.H = H
        self.N = H.shape[1]
        self.circuit = stim.Circuit()


    def get_output_ordering(self):
        flat_cnots = []
        cnots = []
        for op, ixs, _ in self.circuit.flattened_operations():
            if op == "CX":
                for i in range(0, len(ixs), 2):
                    flat_cnots.append(ixs[i])
                    flat_cnots.append(ixs[i + 1])
                    cnots.append((ixs[i], ixs[i + 1]))
        cnot_layers = _layer_cnot_circuit(cnots)
        flat_layers = [flatten(layer) for layer in cnot_layers]

        qubit_to_layer = {}
        for q in range(1, self.N):
            last_occurrence = 0
            for i, layer in enumerate(flat_layers):
                if q in layer:
                    last_occurrence = i
            qubit_to_layer[q] = last_occurrence

        return qubit_to_layer


    def get_connectivity_ordering(self, non_pivots: list[int], H: np.ndarray,
                                  x_circuits: list[tuple[stim.Circuit, int]],
                                  z_circuits: list[tuple[stim.Circuit, int]]) -> list[
        tuple[tuple[int, int], tuple[int, int]]]:
        z_output_availability = [get_output_ordering(c, n) for c, n in z_circuits]
        x_output_availability = [get_output_ordering(c, n) for c, n in x_circuits]

        edge_list = [
            (i, j)
            for i, r in enumerate(self.H)
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


def filter_targets(name, offset_targets):
    # Define which gates require 2 targets per operation
    two_qubit_gates = {"CX", "CZ", "CY", "SWAP", "ISWAP"}

    filtered_targets = []

    if name in two_qubit_gates:
        # 1. Chunk into operational pairs
        pairs = [tuple(offset_targets[i:i + 2]) for i in range(0, len(offset_targets), 2)]
        filtered_pairs = []

        # 2. Cancel out duplicate gate applications (mod 2 cancellation)
        for p in pairs:
            if p in filtered_pairs:
                filtered_pairs.remove(p)  # Annihilate the redundant gate
            else:
                filtered_pairs.append(p)

        # 3. Flatten back for Stim
        filtered_targets = [t for p in filtered_pairs for t in p]

    else:
        # Safe to do 1-qubit flat cancellation
        for t in offset_targets:
            if t in filtered_targets:
                filtered_targets.remove(t)  # Annihilate the redundant operation
            else:
                filtered_targets.append(t)


def flag_by_construction(parity_matrix, t):
    circuit = stim.Circuit()

    parity_matrix = np.asarray(parity_matrix)
    N = parity_matrix.shape[1]
    pivots, rows_without_pivots = find_pivots_in_matrix(parity_matrix)
    non_pivots = [p for p in range(N) if p not in pivots.values()]
    assert len(rows_without_pivots) == 0

    z_spiders = np.sum(parity_matrix, axis=1)
    x_spiders = np.sum(parity_matrix[:, non_pivots], axis=0) + 1

    z_circuits, z_primaries = list(zip(*[well_ordered_ft_cat_state(zs, t) for zs in z_spiders]))
    x_circuits, x_primaries = list(zip(*[well_ordered_ft_cat_state(xs, t) for xs in x_spiders]))
    x_circuits = [cat_state_circuit_in_dual_basis(c) for c in x_circuits]

    layered_z_circuits = [layer_stim_circuit(z_circ, z) for z_circ, z in zip(z_circuits, z_spiders)]
    layered_x_circuits = [layer_stim_circuit(x_circ, x) for x_circ, x in zip(x_circuits, x_spiders)]

    offsets, global_offsets = calculate_offsets(N, non_pivots, x_circuits, z_circuits, x_spiders, z_spiders)
    global_offsets = {i:i for i in global_offsets}
    x_spider_offsets = [o for i, o in enumerate(offsets) if i in non_pivots]
    z_spider_offsets = [o for i, o in enumerate(offsets) if i not in non_pivots]

    z_internal_mappings = [
        calculate_internal_qubit_mapping(z_circuits[i], zs, z_primaries[i])
        for i, zs in enumerate(z_spiders)
    ]
    x_internal_mappings = [
        calculate_internal_qubit_mapping(x_circuits[i], xs, x_primaries[i])
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
        print(z_spider_index, z_internal_qubit)
        print(x_spider_index, x_internal_qubit)

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
            filtered_targets = filter_targets(name, offset_targets)
            if filtered_targets:
                circuit.append(name, filtered_targets)
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
            filtered_targets = filter_targets(name, offset_targets)
            if filtered_targets:
                circuit.append(name, filtered_targets)

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


def calculate_internal_qubit_mapping(circuit: stim.Circuit, num_data: int, primary_qubit: int) -> dict[int, int]:
    ret = {primary_qubit: 0}
    num_flags = circuit.num_qubits - num_data

    # 1. Map the flag qubits (assuming they occupy the indices after the data qubits)
    for q, f in enumerate(range(num_data, circuit.num_qubits), 1):
        ret[f] = q

    # 2. Track the number of CX interactions and the last interacting partner for each data qubit
    cx_counts = {i: 0 for i in range(num_data)}
    last_cx_partner = {i: None for i in range(num_data)}

    # We use .flattened() to unpack any REPEAT blocks in the Stim circuit cleanly
    for inst in circuit.flattened():
        if inst.name == "CX":
            targets = inst.targets_copy()
            # CX targets come in pairs: (control, target)
            for i in range(0, len(targets), 2):
                c = targets[i].value
                t = targets[i + 1].value

                # If control is a data qubit, record the interaction
                if c < num_data:
                    cx_counts[c] += 1
                    last_cx_partner[c] = t

                # If target is a data qubit, record the interaction
                if t < num_data:
                    cx_counts[t] += 1
                    last_cx_partner[t] = c

    # 3. Apply the specific mapping rules for the rest of the data qubits
    for dq in range(num_data):
        if dq == primary_qubit:
            continue

        if cx_counts[dq] > 1:
            partner = last_cx_partner[dq]

            # Verify the last partner is a flag qubit (meaning its index is >= num_data)
            if partner is None or partner < num_data:
                raise ValueError(
                    f"Data qubit {dq} interacts with {cx_counts[dq]} CX gates, but its last "
                    f"CX partner is {partner} (a data qubit). It MUST be connected to a flag qubit."
                )

            # Assign the data qubit the same mapping as its flag qubit partner
            ret[dq] = ret[partner]

    return ret


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

