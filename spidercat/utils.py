import itertools

import stim
from qiskit import QuantumCircuit


def qasm_to_stim(qasm_str: str) -> stim.Circuit:
    """
    Parses QASM 2.0 directly to a Stim circuit, bypassing Cirq to avoid
    deprecated import issues.
    """
    try:
        import qiskit.qasm2
        qc = qiskit.qasm2.loads(qasm_str)
    except (ImportError, AttributeError):
        qc = QuantumCircuit.from_qasm_str(qasm_str)

    qubit_map = {q: i for i, q in enumerate(qc.qubits)}
    stim_circuit = stim.Circuit()

    gate_translation = {
        'id': 'I', 'x': 'X', 'y': 'Y', 'z': 'Z',
        'h': 'H', 's': 'S', 'sdg': 'S_DAG',
        'sx': 'SQRT_X', 'sxdg': 'SQRT_X_DAG',  # Square-root X
        'cx': 'CNOT', 'cy': 'CY', 'cz': 'CZ', 'swap': 'SWAP',
        'reset': 'R', 'measure': 'M', 'barrier': 'TICK'
    }

    for instruction in qc.data:
        op = instruction.operation
        name = op.name
        indices = [qubit_map[q] for q in instruction.qubits]
        if name in gate_translation:
            stim_circuit.append(gate_translation[name], indices)
        else:
            raise ValueError(f"Gate '{name}' is not supported in Stim (Non-Clifford or Unknown).")

    return stim_circuit


def ed(v1: int, v2: int) -> tuple[int, int]:
    return (v1, v2) if v1 < v2 else (v2, v1)


def check_k_fault_tolerance(clean_circuit: stim.Circuit, t: int, num_flags: int, n_data: int):
    # 1. Identify all single-qubit fault "slots"
    fault_slots = {(k, 0, 0) for k in range(clean_circuit.num_qubits)}
    for cmd_idx, cmd in enumerate(clean_circuit):
        targets = [t.value for t in cmd.targets_copy()]

        # Handle Multi-qubit gates (CX, CNOT, CZ, etc.)
        if cmd.name in ["CX", "CNOT", "CZ"]:
            # stim flattens targets; [control1, target1, control2, target2, ...]
            # Each pair is a separate gate operation
            for i in range(0, len(targets), 2):
                q_control = targets[i]
                q_target = targets[i + 1]
                # A fault can happen on either qubit involved in the pair
                fault_slots.add((cmd_idx + 1, q_control, i))  # i helps track the specific pair
                fault_slots.add((cmd_idx + 1, q_target, i + 1))

        # Handle Single-qubit gates
        elif cmd.name in ["R", "RX", "RY", "RZ", "H", "X", "Y", "Z", "S", "T"]:
            for i, q_idx in enumerate(targets):
                fault_slots.add((cmd_idx + 1, q_idx, i))

        # Handle Measurements (Fault must be injected BEFORE/DURING measurement)
        elif cmd.name.startswith("M"):
            for i, q_idx in enumerate(targets):
                fault_slots.add((cmd_idx, q_idx, i))

    # 2. Iterate through fault weights k
    for k in range(1, t + 1):
        print(f"Checking k={k} fault combinations...")
        for combo in itertools.combinations(list(fault_slots), k):
            for error_types in itertools.product(['X', 'Z'], repeat=k):

                test_circuit = inject_specific_faults(clean_circuit, combo, error_types)
                sampler = test_circuit.compile_sampler()
                sample = sampler.sample(shots=1)[0]

                flags = sample[:num_flags]
                data_bits = sample[num_flags:]

                if not any(flags):
                    raw_weight = sum(data_bits)
                    logical_weight = min(raw_weight, n_data - raw_weight)

                    if logical_weight > k:
                        print(f"❌ VIOLATION: k={k}, Faults: {list(zip(combo, error_types))}")
                        return False
    return True


def inject_specific_faults(circuit, locations, types):
    new_circuit = stim.Circuit()
    # Map cmd_idx -> list of (target_index_in_cmd, error_type)
    error_lookup = {}
    for (c_idx, q_idx, t_idx), etype in zip(locations, types):
        if c_idx not in error_lookup:
            error_lookup[c_idx] = []
        error_lookup[c_idx].append((q_idx, etype))

    for i, cmd in enumerate(circuit):
        is_meas = cmd.name.startswith("M")

        # If it's a measurement, inject error BEFORE the gate
        if is_meas and i in error_lookup:
            for q_idx, etype in error_lookup[i]:
                new_circuit.append(f"{etype}_ERROR", [q_idx], 1.0)

        new_circuit.append(cmd)

        # If it's a gate, inject error AFTER the gate
        if not is_meas and i in error_lookup:
            for q_idx, etype in error_lookup[i]:
                new_circuit.append(f"{etype}_ERROR", [q_idx], 1.0)

    return new_circuit


if __name__ == "__main__":
    circ = stim.Circuit("""
H 0
CX 0 1 1 2 1 10 1 3 1 11 1 12 0 4 4 13 4 14 0 12
MR 12
DETECTOR rec[-1]
CX 0 5 0 6 6 7 6 8 8 13
MR 13
DETECTOR rec[-1]
CX 6 11
MR 11
DETECTOR rec[-1]
CX 0 9 0 14
MR 14
DETECTOR rec[-1]
CX 0 10
MR 10
DETECTOR rec[-1]
    """)
    circ.append("M", range(10))
    check_k_fault_tolerance(circ, 3, num_flags=5, n_data=10)
