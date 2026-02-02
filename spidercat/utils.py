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


def check_fault_tolerance(circuit: stim.Circuit, t: int, num_flags: int):
    """
    Checks if a circuit tolerates up to t faults.

    Args:
        circuit: The stim.Circuit to analyze.
        t: The maximum number of faults to test (combinations of k=1 to t).
        num_flags: The number of measurements that act as detectors/flags.

    Returns:
        A dictionary mapping fault weight k to a boolean (True if safe).
    """
    # 1. Identify all possible error sources in the circuit
    # This includes any instruction that can fail (X_ERROR, Y_ERROR, Z_ERROR, DEPOLARIZE1, etc.)
    # For a deterministic check, we find all 'error' type instructions.
    error_instructions = []
    for i, inst in enumerate(circuit):
        if "ERROR" in inst.name or "DEPOLARIZE" in inst.name:
            error_instructions.append(i)

    # We use a sampler to see how specific errors propagate
    # To do this systematically, we'll strip the errors and inject them manually
    # or use stim's 'TableauSimulator' for exact tracking.
    sim = stim.TableauSimulator()

    results = {}

    for k in range(1, t + 1):
        is_safe = True
        for combo in itertools.combinations(error_instructions, k):
            sample = simulate_specific_errors(circuit, combo)

            flags = sample[:num_flags]
            data_out = sample[num_flags:]

            error_detected = any(flags)
            weight_of_data_error = sum(data_out)  # Assuming data_out is the error syndrome/logical flip

            # If it wasn't caught, the logical error weight must be <= k
            if not error_detected and weight_of_data_error > k:
                print(combo)
                is_safe = False
                break

        results[k] = is_safe
        if not is_safe:
            print(f"❌ Failed tolerance check at {k} faults.")
        else:
            print(f"✅ Passed tolerance check for {k} faults.")

    return results


def simulate_specific_errors(circuit, error_indices):
    """
    Simulates the circuit with 100% error probability at the specified indices.
    """
    sampler = circuit.compile_sampler()
    # Note: For exact k-fault analysis, one often uses stim.Circuit.flipped_pauli_errors
    return sampler.sample(shots=1)[0]


if __name__ == "__main__":
    circ = stim.Circuit("""
H 0
CX 0 6 0 1 1 2 1 7 1 3 1 8 1 6
MR 6
DETECTOR rec[-1]
CX 0 4 0 5 5 8
MR 8
DETECTOR rec[-1]
CX 0 7
MR 7
DETECTOR rec[-1]
    """)
    circ.append("M", range(10))
    from spidercat.circuit_extraction import make_stim_circ_noisy
    noisy_circ = make_stim_circ_noisy(circ, p_meas=0.001, p_init=0.001)
    print(noisy_circ)
    print(check_fault_tolerance(noisy_circ, 3, num_flags=5))
