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


if __name__ == "__main__":
    # --- Example Usage ---
    qasm_code = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg f[1];
        qreg q[8];
        h q[0];
        cx q[0],f[0];
        cx q[0],q[7];
        cx q[0],q[6];
        cx q[0],q[5];
        cx q[0],q[4];
        cx q[0],q[3];
        cx q[0],q[2];
        cx q[0],q[1];
        cx q[0],f[0];
    """

    circuit = qasm_to_stim(qasm_code)
    circuit.append("M", range(0, circuit.num_qubits - 8))
    print(circuit)


def ed(v1: int, v2: int) -> tuple[int, int]:
    return (v1, v2) if v1 < v2 else (v2, v1)
