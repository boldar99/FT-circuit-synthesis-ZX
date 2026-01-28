from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np
import pyzx as zx
import stim


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
        self.circ.append("R", [q])

    def post_select(self, q):
        """Performs MR and returns the absolute index of this measurement."""
        self.circ.append("MR", [q])
        idx = self.meas_count
        self.meas_count += 1
        return idx

    def add_feedback_x(self, meas_idx, target_qubit):
        offset = meas_idx - self.meas_count
        self.circ.append("CX", [stim.target_rec(offset), target_qubit])

    def add_detector(self, m_idx1, m_idx2):
        """
        Adds a detector that fires if measurement[m_idx1] != measurement[m_idx2].
        (i.e., parity is 1).
        """
        offset1 = m_idx1 - self.meas_count
        offset2 = m_idx2 - self.meas_count
        self.circ.append("DETECTOR", [stim.target_rec(offset1), stim.target_rec(offset2)])

    def get_circuit(self):
        return self.circ


def ed(v1: int, v2: int) -> tuple[int, int]:
    return tuple(sorted((v1, v2)))


def extract_circuit(G, path_cover, marks, matching, builder: CircuitBuilder, verbose=False) -> stim.Circuit:
    if verbose:
        print("=== Extracting Circuit ===")

    # --- Setup Mappings ---
    node_to_path_idx = {node: p_idx for p_idx, path in enumerate(path_cover) for node in path}
    marks_map = {ed(v1, v2): int(v) for (v1, v2), v in
                 (marks.items() if isinstance(marks, dict) else [(e, 1) for e in marks])}

    # 2. Setup Indexing
    cover_edges = {ed(u, v) for path in path_cover for u, v in zip(path, path[1:])}
    num_flags = len([e for e in G.edges() if ed(*e) not in cover_edges])

    flag_map: dict[tuple[int, int], int] = {}
    link_info: dict[tuple[int, int], dict[str, int]] = {}
    path_to_marks = defaultdict(list)

    if verbose:
        print("Number of flags:", num_flags)

    next_cat = num_flags + len(path_cover)

    # Map path_index -> qubit_index of the path's logical representative
    path_qubits = {}

    # 3. Initial Setup
    for qidx in range(next_cat):
        builder.init_ancilla(qidx)

    def handle_link(path_qubit, link, current_p_id, decrement=False):
        nonlocal next_cat
        if verbose:
            print(f"    Handling link {link} (path_qubit={path_qubit}, path_id={current_p_id})")

        # 1. Create or retrieve the flag qubit
        if link not in link_info:
            fq = len(link_info)
            link_info[link] = {'q': fq, 'owner': current_p_id}

            if verbose:
                print(f"      New Flag: {fq} for link {link} (Owner: Path {current_p_id})")
                print(f"      CNOT {path_qubit} -> {fq}")
            builder.add_cnot(path_qubit, fq)
        else:
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

    # --- 5. ORGANIZE MEASUREMENTS FOR CONSISTENCY CHECKS ---
    # Group measurements by the pair of paths they connect
    consistency_groups: dict[tuple[int, int], list[int]] = defaultdict(list)

    for (u, v), m_idx in flag_map.items():
        p1 = node_to_path_idx[u]
        p2 = node_to_path_idx[v]

        # We only care about flags between DIFFERENT paths
        if p1 != p2:
            consistency_groups[ed(p1, p2)].append(m_idx)

    # Add Detectors: Trigger if M[i] != M[i+1]
    # This means the file itself knows what a "consistent" shot is.
    if verbose: print("Adding Consistency Detectors...")
    for pair, indices in consistency_groups.items():
        if len(indices) > 1:
            for k in range(len(indices) - 1):
                builder.add_detector(indices[k], indices[k + 1])

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

    # --- 7. FINAL DATA READOUT ---
    # We explicitly measure the logical qubits (GHZ state) at the end.
    if verbose: print("Adding Final Data Measurements...")

    # Collect all data qubits in a deterministic order
    all_data_qubits = []
    for p_id in range(len(path_cover)):
        # Path qubit first, then its marks
        all_data_qubits.append(path_qubits[p_id])
        all_data_qubits.extend(path_to_marks[p_id])

    return builder.get_circuit()


def make_stim_circ_noisy(circ: stim.Circuit, p_1=0, p_2=0, p_mem=0, p_meas=0, p_init=0) -> stim.Circuit:
    noisy_circ = stim.Circuit()
    num_qubits = circ.num_qubits
    if p_init > 0:
        noisy_circ.append("DEPOLARIZE1", range(num_qubits), p_init)

    # --- 1. Scheduling (ASAP) ---
    # moments[t] = list of (gate_name, [qubits]) scheduled for time t
    moments = defaultdict(list)

    # qubit_free_time[q] = the earliest time index q is free
    qubit_free_time = defaultdict(int)

    for instruction in circ:
        name = instruction.name

        # Pass non-gate instructions (annotations) directly to the output?
        # Note: In a reconstruction approach, we usually append them at the end
        # or try to preserve context. For simplicity, we attach them to the
        # time of their first qubit, or 0 if global.
        if name in ["QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK"]:
            # These are complex to schedule perfectly in a reconstructed circuit.
            # Often, noisy simulations ignore TICKs (we make our own) and
            # append detectors at the end or track them.
            # For this snippet, let's assume we only process gates.
            # (If you need to preserve Detectors, they must be re-inserted
            # relative to the measurement targets).
            if name == "TICK": continue
            noisy_circ.append(instruction)
            continue

        # Determine Arity (number of qubits per atomic gate)
        # This handles the "CX 0 1 0 2" issue by breaking it down.
        if name in ("CNOT", "CX", "SWAP", "CZ"):
            arity = 2
        elif name in ("H", "X", "Y", "Z", "I", "M", "MR", "R", "RX", "RY", "MZ"):
            arity = 1
        else:
            # Fallback for 1-qubit gates or unknown; adjust as needed
            arity = 1

        # Extract all targets from the instruction
        raw_targets = instruction.targets_copy()

        # Iterate over the instruction in chunks of 'arity'
        # e.g. CX 0 1 2 3 -> (0,1), (2,3)
        for i in range(0, len(raw_targets), arity):
            # Extract qubit indices for this specific gate operation
            # Note: .value handles the integer index.
            # We assume standard Pauli targets (no sweep bits/combiners for noise models)
            gate_qubits = [t.value for t in raw_targets[i:i + arity]]

            # Find when these qubits are available
            # The gate can start only when ALL its qubits are free
            start_time = 0
            for q in gate_qubits:
                start_time = max(start_time, qubit_free_time[q])

            # Schedule the gate
            moments[start_time].append((name, gate_qubits))

            # Update availability
            # These qubits are now busy until start_time + 1
            for q in gate_qubits:
                qubit_free_time[q] = start_time + 1

    # Total duration of the circuit
    if not moments:
        max_time = 0
    else:
        max_time = max(moments.keys()) + 1

    # --- 2. Reconstruction with Noise ---

    for t in range(max_time):
        ops_in_moment = moments[t]

        # Set of qubits active in this moment
        active_qubits = set()

        # Apply Gates & Gate Noise
        for gate_name, targets in ops_in_moment:
            active_qubits.update(targets)

            # CNOT / 2-Qubit
            if gate_name in ("CNOT", "CX", "CZ", "SWAP"):
                noisy_circ.append(gate_name, targets)
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

                # Post-gate noise (Reset Preparation Error)
                if gate_name in ("R", "RX", "RY", "MR") and p_init > 0:
                    noisy_circ.append("DEPOLARIZE1", targets, p_init)

            else:
                noisy_circ.append(gate_name, targets)

        # Apply Idle Noise
        # Any qubit NOT in active_qubits gets decoherence
        if p_mem > 0:
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
    import networkx as nx

    G = nx.from_edgelist(
        [(0, 1), (0, 4), (0, 5), (1, 2), (1, 6), (2, 3), (2, 7), (3, 4), (3, 8), (4, 9), (5, 7), (5, 8), (6, 8), (6, 9),
         (7, 9)])
    H = [[0, 4, 9], [2, 1, 6], [3, 8, 5, 7]]
    M = {(0, 1): 1, (0, 4): 1, (0, 5): 1, (1, 2): 0, (1, 6): 0, (2, 3): 1, (2, 7): 1, (3, 4): 1, (3, 8): 0, (4, 9): 0,
         (5, 7): 1, (5, 8): 0, (6, 8): 1, (6, 9): 1, (7, 9): 1}
    matching = {9: 6, 6: 8, 7: 2}

    circ = extract_circuit(G, H, M, matching, StimBuilder(), verbose=True)
    # circ.append("M", range(8, 18))
    sampler = circ.compile_sampler()
    raw_samples = sampler.sample(shots=10)

    converter = circ.compile_m2d_converter()
    dets = converter.convert(measurements=raw_samples, append_observables=False)

    is_good_shot = ~np.any(dets, axis=1)
    valid_samples = raw_samples[is_good_shot]
