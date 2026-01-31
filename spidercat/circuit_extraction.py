from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np
import pyzx as zx
import stim

from spidercat.utils import ed


class CircuitBuilder(ABC):
    @abstractmethod
    def add_h(self, qubit): pass

    @abstractmethod
    def add_cnot(self, control, target): pass

    @abstractmethod
    def init_ancilla(self, qubit):
        """Inits ancilla and applies H for your specific extraction logic."""
        pass

    @abstractmethod
    def post_select(self, qubit):
        """Applies H and post-selects (or measures) for your logic."""
        pass

    @abstractmethod
    def add_feedback_x(self, meas_idx, target_qubit):
        """
        Adds an X gate on target_qubit controlled by the measurement at absolute index meas_idx.
        Stim uses relative indexing (rec[-k]), so we calculate offset.
        """
        pass

    @abstractmethod
    def add_detector(self, m_idx1, m_idx2):
        """
        Adds a detector that fires if measurement[m_idx1] != measurement[m_idx2].
        (i.e., parity is 1).
        """
        pass

    @abstractmethod
    def get_circuit(self): pass


class PyZXBuilder(CircuitBuilder):
    def __init__(self):
        self.circ = zx.Circuit(0)

    def add_h(self, q): self.circ.add_gate("H", q)

    def add_cnot(self, c, t): self.circ.add_gate("CNOT", c, t)

    def init_ancilla(self, q):
        self.circ.add_gate("InitAncilla", q)
        self.add_h(q)

    def post_select(self, q):
        self.add_h(q)
        self.circ.add_gate("PostSelect", q)

    def get_circuit(self): return self.circ


class StimBuilder(CircuitBuilder):
    def __init__(self):
        self.circ = stim.Circuit()
        self.meas_count = 0

    def add_h(self, q):
        self.circ.append("H", [q])

    def add_cnot(self, c, t):
        self.circ.append("CNOT", [c, t])

    def init_ancilla(self, q):
        # self.circ.append("R", [q])
        pass

    def post_select(self, q):
        """Performs MR and returns the absolute index of this measurement."""
        self.circ.append("MR", [q])
        idx = self.meas_count
        self.meas_count += 1
        return idx

    def add_feedback_x(self, meas_idx, target_qubit):
        offset = meas_idx - self.meas_count
        self.circ.append("CX", [stim.target_rec(offset), target_qubit])

    def add_detector(self, *meas_indices):
        """
        Adds a detector on the parity of the provided measurement indices.
        - If 1 index is provided: Checks that measurement == 0.
        - If 2 indices are provided: Checks that m1 == m2.
        """
        targets = []
        for m_idx in meas_indices:
            offset = m_idx - self.meas_count
            targets.append(stim.target_rec(offset))
        self.circ.append("DETECTOR", targets)

    def get_circuit(self):
        return self.circ


def extract_circuit(G, path_cover, marks, matching, builder: CircuitBuilder, verbose=False) -> stim.Circuit:
    if verbose:
        print("=== Extracting Circuit ===")

    # --- Setup Mappings ---
    node_to_path_idx = {node: p_idx for p_idx, path in enumerate(path_cover) for node in path}
    marks_map = {ed(v1, v2): int(v) for (v1, v2), v in
                 (marks.items() if isinstance(marks, dict) else [(e, 1) for e in marks])}

    flag_map: dict[tuple[int, int], int] = {}
    link_info: dict[tuple[int, int], dict[str, int]] = {}
    path_to_marks = defaultdict(list)
    path_qubits = {}

    # Count flags to reserve qubits
    cover_edges = {ed(u, v) for path in path_cover for u, v in zip(path, path[1:])}
    num_flags = len([e for e in G.edges() if ed(*e) not in cover_edges])
    next_cat = num_flags + len(path_cover)

    # 3. Initial Setup
    for qidx in range(next_cat):
        builder.init_ancilla(qidx)

    # --- Helper Logic ---
    def handle_link(path_qubit, link, current_p_id, decrement=False):
        nonlocal next_cat
        if verbose:
            print(f"    Handling link {link} (path_qubit={path_qubit}, path_id={current_p_id})")

        # 1. Create or retrieve the flag qubit
        if link not in link_info:
            # First visit
            fq = len(link_info)
            link_info[link] = {'q': fq, 'owner': current_p_id}

            if verbose:
                print(f"      New Flag: {fq} for link {link} (Owner: Path {current_p_id})")
                print(f"      CNOT {path_qubit} -> {fq}")
            builder.add_cnot(path_qubit, fq)
        else:
            # Second visit
            info = link_info[link]
            fq = info['q']
            owner_id = info['owner']

            if verbose:
                print(f"      Existing Flag: {fq} for link {link} (Belongs to Path {owner_id})")

            # Add Marks on Link
            count = marks_map.get(link, 0) - (1 if decrement else 0)
            for _ in range(count):
                if verbose:
                    print(f"      Init Ancilla {next_cat}")
                builder.init_ancilla(next_cat)

                path_to_marks[owner_id].append(next_cat)

                if verbose:
                    print(f"      CNOT {fq} -> {next_cat} (Mark assigned to Path {owner_id})")
                builder.add_cnot(fq, next_cat)
                next_cat += 1

            if verbose:
                print(f"      CNOT {path_qubit} -> {fq}")
            builder.add_cnot(path_qubit, fq)

            # Post-selection / Measurement
            m_idx = builder.post_select(fq)
            flag_map[link] = m_idx
            if verbose:
                print(f"      PostSelect {fq} -> Meas Idx {m_idx}")

    # 4. Main Loop
    if verbose:
        print("Starting Main Loop...")

    for p_id, path in enumerate(path_cover):
        if verbose:
            print(f"Path {p_id}: {path}")
        path_q = num_flags + p_id
        path_qubits[p_id] = path_q

        if verbose:
            print(f"  Unfusing path start {path_q} (H gate)")
        builder.add_h(path_q)  # Unfuse path start

        # Neighbors of v0
        v0, v1 = path[0], path[1]
        if verbose:
            print(f"  Neighbors of start {v0} (excluding {v1})...")
        for n in set(G.neighbors(v0)) - {v1}:
            decrement = (matching.get(v0) == n) or (matching.get(n) == v0)
            handle_link(path_q, ed(v0, n), p_id, decrement=decrement)

        # Path segments and internal nodes
        if verbose:
            print(f"  Internal segments...")
        for i, v_curr in enumerate(path[1:], 1):
            v_prev = path[i - 1]
            # Markings on the path itself
            marks_count = marks_map.get(ed(v_prev, v_curr), 0)
            if marks_count > 0 and verbose:
                print(f"    Processing {marks_count} marks on edge {(v_prev, v_curr)}")
            for _ in range(marks_count):
                if verbose:
                    print(f"      Init Ancilla {next_cat}")
                builder.init_ancilla(next_cat)
                path_to_marks[p_id].append(next_cat)
                if verbose:
                    print(f"      CNOT Path {path_q} -> {next_cat}")
                builder.add_cnot(path_q, next_cat)
                next_cat += 1

            if i + 1 < len(path):
                v_next = path[i + 1]
                # Internal non-cover neighbor
                for n in set(G.neighbors(v_curr)) - {v_prev, v_next}:
                    if verbose:
                        print(f"    Internal neighbor {v_curr}-{n}")
                    decrement = (matching.get(v_curr) == n) or (matching.get(n) == v_curr)
                    handle_link(path_q, ed(v_curr, n), p_id, decrement=decrement)

        # End of path logic
        if len(path) >= 2:
            if verbose:
                print(f"  End of path logic...")
            v_last, v_pen = path[-1], path[-2]
            ends = list(set(G.neighbors(v_last)) - {v_pen})
            if ends and matching.get(v_last) == ends[0]: ends.reverse()
            for end_v in ends:
                if verbose:
                    print(f"    End neighbor {v_last}-{end_v}")
                decrement = (matching.get(v_last) == end_v) or (matching.get(end_v) == v_last)
                handle_link(path_q, ed(v_last, end_v), p_id, decrement=decrement)

    # --- 5. DETECTORS ---
    if verbose: print("Adding Detectors...")
    consistency_groups: dict[tuple[int, int], list[int]] = defaultdict(list)

    # "Meta-graph" for finding cycles: Nodes are paths, Edges are connections
    meta_graph = nx.Graph()

    for (u, v), m_idx in flag_map.items():
        p1 = node_to_path_idx[u]
        p2 = node_to_path_idx[v]

        if p1 == p2:
            # Intra-path: Must be 0
            if verbose: print(f"  Intra-path detector on link {u}-{v} (meas {m_idx})")
            builder.add_detector(m_idx)
        else:
            key = ed(p1, p2)
            consistency_groups[key].append(m_idx)
            # Add to meta-graph for cycle detection later
            # We store the *first* measurement index as the representative for this edge
            if not meta_graph.has_edge(p1, p2):
                meta_graph.add_edge(p1, p2, representative_meas=m_idx)

    # A. Local Consistency (Parallel Edges)
    # If there are multiple measurements between p1 and p2, they must match.
    for pair, indices in consistency_groups.items():
        if len(indices) > 1:
            for k in range(len(indices) - 1):
                builder.add_detector(indices[k], indices[k + 1])

    # B. Global Consistency (Cycles)
    # Use NetworkX to find the cycle basis of the path connectivity graph.
    # For every cycle, the sum of representative measurements must be even (0).
    cycle_basis = nx.cycle_basis(meta_graph)
    if verbose and cycle_basis:
        print(f"  Found {len(cycle_basis)} cycles in path graph.")

    for cycle in cycle_basis:
        # cycle is a list of nodes [0, 1, 2] meaning 0-1-2-0
        detectors_indices = []

        # Walk through the cycle edges
        # Edge (0,1), (1,2), (2,0)
        cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1]))

        for p_u, p_v in cycle_edges:
            # Retrieve the representative measurement for this pair
            edge_data = meta_graph.get_edge_data(p_u, p_v)
            detectors_indices.append(edge_data['representative_meas'])

        if verbose:
            print(f"  Adding cycle detector for paths {cycle} using measurements {detectors_indices}")

        builder.add_detector(*detectors_indices)

    # --- 6. FEEDBACK/CORRECTION LOGIC ---
    if verbose: print("Generating Feedback...")
    path_graph = nx.Graph()
    path_graph.add_nodes_from(range(len(path_cover)))

    for (p1, p2), m_indices in consistency_groups.items():
        if not path_graph.has_edge(p1, p2):
            path_graph.add_edge(p1, p2, meas_idx=m_indices[0])

    try:
        bfs_tree = dict(nx.bfs_predecessors(path_graph, 0))
    except Exception:
        bfs_tree = {}

    for target_path_idx in range(1, len(path_cover)):
        if target_path_idx not in bfs_tree:
            continue

        # 1. Determine correction chain (XOR sum of measurements)
        current = target_path_idx
        correction_measurements = []
        while current != 0:
            parent = bfs_tree[current]
            edge_data = path_graph.get_edge_data(parent, current)
            correction_measurements.append(edge_data['meas_idx'])
            current = parent

        # 2. Identify all qubits that need this correction
        #    (The main path qubit + all mark ancillas attached to this path)
        qubits_to_correct = [path_qubits[target_path_idx]] + path_to_marks[target_path_idx]

        if verbose:
            print(f"Correcting Path {target_path_idx}")
            print(f"  Targets: PathQ {path_qubits[target_path_idx]} + Marks {path_to_marks[target_path_idx]}")
            print(f"  Controlled by measurements: {correction_measurements}")

        # 3. Apply Feedback
        for m_idx in correction_measurements:
            for q in qubits_to_correct:
                builder.add_feedback_x(m_idx, q)

    return builder.get_circuit()


def make_stim_circ_noisy(circ: stim.Circuit, p_1=0, p_2=0, p_mem=0, p_meas=0, p_init=0) -> stim.Circuit:
    noisy_circ = stim.Circuit()
    num_qubits = circ.num_qubits

    if p_init > 0:
        noisy_circ.append("DEPOLARIZE1", range(num_qubits), p_init)

    # --- 1. Scheduling (ASAP with Causality) ---
    moments = defaultdict(list)
    qubit_free_time = defaultdict(int)

    # We must track when the *last measurement* occurred to prevent
    # feedback operations from travelling back in time before their data exists.
    last_meas_time = 0

    # Annotations that must be preserved in time relative to measurements
    annotations = {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}

    # Gates that produce records
    measurement_gates = {"M", "MR", "MZ", "R", "RX", "RY"}  # R/RX/RY only produce if they are MR, handled below

    for instruction in circ:
        name = instruction.name

        # SKIP TICKS (we handle time manually)
        if name == "TICK":
            continue

        raw_targets = instruction.targets_copy()

        # --- Handle Annotations ---
        if name in annotations:
            # Detectors rely on records, so they must be scheduled
            # at least as late as the last measurement.
            # We also respect the wavefront of busy qubits to keep context.
            context_time = max(qubit_free_time.values()) if qubit_free_time else 0
            start_time = max(last_meas_time, context_time)
            moments[start_time].append((name, raw_targets))
            continue

        # --- Determine Arity ---
        if name in ("CNOT", "CX", "SWAP", "CZ"):
            arity = 2
        elif name in ("H", "X", "Y", "Z", "I", "M", "MR", "R", "RX", "RY", "MZ"):
            arity = 1
        else:
            arity = 1

        # Check if this instruction produces a measurement record
        # (Standard M/MR/MZ, or Reset if it's actually MR)
        is_measurement = name in ("M", "MR", "MZ")

        # Check if this instruction USES a record (Feedback)
        # e.g. CX rec[-1] 5
        uses_feedback = any(t.is_measurement_record_target for t in raw_targets)

        for i in range(0, len(raw_targets), arity):
            gate_targets = raw_targets[i:i + arity]

            # Identify which targets are physical qubits (for resource tracking)
            # and which are records (for causality tracking)
            physical_qubit_indices = [t.value for t in gate_targets if not t.is_measurement_record_target]

            # 1. Base Start Time: When are the physical qubits free?
            start_time = 0
            for q_idx in physical_qubit_indices:
                start_time = max(start_time, qubit_free_time[q_idx])

            # 2. Causality: Measurement Order
            # If this IS a measurement, it cannot happen before the previous measurement group
            # (Otherwise rec[-1] indices would point to the wrong data)
            if is_measurement:
                start_time = max(start_time, last_meas_time)

            # 3. Causality: Feedback
            # If this gate uses feedback, it must happen AFTER the measurements it relies on.
            # We pin it to last_meas_time. Since we append sequentially,
            # it will appear in the moment list *after* the M gate if they share the same time.
            if uses_feedback:
                start_time = max(start_time, last_meas_time)

            # Schedule
            moments[start_time].append((name, gate_targets))

            # Update State
            finish_time = start_time + 1

            if is_measurement:
                last_meas_time = start_time

            for q_idx in physical_qubit_indices:
                qubit_free_time[q_idx] = finish_time

    if not moments:
        max_time = 0
    else:
        max_time = max(moments.keys()) + 1

    # --- 2. Reconstruction with Noise ---

    for t in sorted(moments.keys()):
        ops_in_moment = moments[t]
        active_qubits = set()

        for gate_name, targets in ops_in_moment:
            # Handle Annotations
            if gate_name in annotations:
                noisy_circ.append(gate_name, targets)
                continue

            # Identify qubits for idle noise (exclude records)
            gate_qubit_indices = [t.value for t in targets if not t.is_measurement_record_target]
            active_qubits.update(gate_qubit_indices)

            has_record_target = any(t.is_measurement_record_target for t in targets)

            # Apply Gates & Noise
            if gate_name in ("CNOT", "CX", "CZ", "SWAP"):
                noisy_circ.append(gate_name, targets)

                # NO NOISE on feedback gates (CX rec 5)
                # (Applying noise to a record target is invalid in Stim)
                if has_record_target:
                    continue

                if p_2 > 0:
                    noisy_circ.append("DEPOLARIZE2", targets, p_2)

            # 1-Qubit Gates
            elif gate_name in ("H", "X", "Y", "Z", "I"):
                noisy_circ.append(gate_name, targets)
                if p_1 > 0:
                    noisy_circ.append("DEPOLARIZE1", targets, p_1)

            # Measurement / Reset (SPAM)
            elif gate_name in ("M", "MZ", "MR", "R", "RX", "RY"):

                # Pre-gate noise (Measurement Readout Error)
                if gate_name in ("M", "MZ", "MR") and p_meas > 0:
                    noisy_circ.append("DEPOLARIZE1", targets, p_meas)

                noisy_circ.append(gate_name, targets)

                # Post-reset error
                if gate_name in ("R", "RX", "RY", "MR") and p_init > 0:
                    noisy_circ.append("DEPOLARIZE1", targets, p_init)

            else:
                noisy_circ.append(gate_name, targets)

        # Apply Idle Noise
        # Any qubit NOT in active_qubits gets decoherence
        if p_mem > 0:
            # Filter valid qubits that are not in active_qubits
            idle_qubits = [q for q in range(num_qubits) if q not in active_qubits]
            if idle_qubits:
                noisy_circ.append("DEPOLARIZE1", idle_qubits, p_mem)

        # End of Moment
        noisy_circ.append("TICK")

    return noisy_circ


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
    for i in range(2, n + 1):
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


if __name__ == "__main__":
    data = {"G.edges": [[0, 7], [0, 1], [0, 2], [1, 2], [1, 4], [2, 3], [3, 4], [3, 6], [4, 5], [5, 6], [5, 7], [6, 7]],
            "M_inv": {"1": [[0, 7], [0, 1], [0, 2], [1, 2]],
                      "2": [[1, 4], [2, 3], [3, 4], [3, 6], [4, 5], [5, 6], [5, 7], [6, 7]]},
            "H": [[1, 0, 2], [3, 4, 5], [6, 7]], "matching": {"2": 1, "5": 6, "7": 0}, "t": 2, "n": 20, "p": 3}

    G = nx.from_edgelist(data["G.edges"])
    H = data["H"]
    M_inv = data["M_inv"]
    M = {}
    for k, v in M_inv.items():
        for (a, b) in v:
            M[(a, b)] = int(k)
    matching = {int(k): v for k, v in data["matching"].items()}

    circ = extract_circuit(G, H, M, matching, StimBuilder(), verbose=False)
    print(circ)
    # circ.append("M", range(8, 18))
    sampler = circ.compile_sampler()
    raw_samples = sampler.sample(shots=10)
    print(raw_samples)

    converter = circ.compile_m2d_converter()
    dets = converter.convert(measurements=raw_samples, append_observables=False)
    print(dets)

    is_good_shot = ~np.any(dets, axis=1)
    valid_samples = raw_samples[is_good_shot]
