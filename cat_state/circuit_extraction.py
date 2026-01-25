from collections import defaultdict

import stim

from abc import ABC, abstractmethod
import pyzx as zx
import stim
import networkx as nx


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

    def add_h(self, q): self.circ.append("H", q)

    def add_cnot(self, c, t): self.circ.append("CNOT", [c, t])

    def init_ancilla(self, q): pass

    def post_select(self, q):
        self.circ.append("MR", q)  # Stim treats post-selection usually via measurement/detectors

    def get_circuit(self): return self.circ


def ed(v1, v2):
    return tuple(sorted((v1, v2)))


def extract_circuit(G, path_cover, marks, matching, builder: CircuitBuilder):
    # 1. Setup Markings
    marks_map = {ed(v1, v2): int(v) for (v1, v2), v in
                 (marks.items() if isinstance(marks, dict) else [(e, 1) for e in marks])}

    # 2. Setup Indexing
    cover_edges = {ed(u, v) for path in path_cover for u, v in zip(path, path[1:])}
    num_flags = len([e for e in G.edges() if ed(*e) not in cover_edges])
    flag_dict = {}
    next_cat = num_flags + len(path_cover)

    # 3. Initial Setup
    for qidx in range(next_cat):
        builder.init_ancilla(qidx)

    def handle_link(path_qubit, link, decrement=False):
        nonlocal next_cat
        if link not in flag_dict:
            flag_dict[link] = len(flag_dict)
            builder.add_cnot(path_qubit, flag_dict[link])
        else:
            fq = flag_dict[link]
            for _ in range(marks_map.get(link, 0) - (1 if decrement else 0)):
                builder.init_ancilla(next_cat)
                builder.add_cnot(fq, next_cat)
                next_cat += 1
            builder.add_cnot(path_qubit, fq)
            builder.post_select(fq)

    # 4. Main Loop
    for p_id, path in enumerate(path_cover):
        path_q = num_flags + p_id
        builder.add_h(path_q)  # Unfuse path start

        # Neighbors of v0
        v0, v1 = path[0], path[1]
        for n in set(G.neighbors(v0)) - {v1}:
            handle_link(path_q, ed(v0, n))

        # Path segments and internal nodes
        for i, v_curr in enumerate(path[1:-1], 1):
            v_prev, v_next = path[i - 1], path[i + 1]
            # Internal non-cover neighbor
            for n in set(G.neighbors(v_curr)) - {v_prev, v_next}:
                handle_link(path_q, ed(v_curr, n))
            # Markings on the path itself
            for _ in range(marks_map.get(ed(v_prev, v_curr), 0)):
                builder.init_ancilla(next_cat)
                builder.add_cnot(path_q, next_cat)
                next_cat += 1

        # End of path logic
        if len(path) > 2:
            v_last, v_pen = path[-1], path[-2]
            ends = list(set(G.neighbors(v_last)) - {v_pen})
            if matching.get(v_last) == ends[0]: ends.reverse()
            for end_v in ends:
                handle_link(path_q, ed(v_last, end_v), decrement=(matching.get(v_last) == end_v))

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

    G = nx.petersen_graph()
    print(len(list(find_all_hamiltonian_paths(G))))
