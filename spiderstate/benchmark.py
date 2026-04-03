import json

import numpy as np

from spidercat.circuit_extraction import make_stim_circ_noisy
from spidercat.simulate import _layer_cnot_circuit
from spiderstate.cat_at_origin import row_optimized_cat_at_origin
from spiderstate.utils import load_qecc, FAO_QECCS


def benchmark_CAO_state_prep(code: str, method: str, p=1e-3, num_samples=10_000_000):
    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(code, method)
    print("State: |0>")
    circ = row_optimized_cat_at_origin(H_z, d, max_basis_tries=5_000)

    circuit = make_stim_circ_noisy(circ, p_init=p, p_2=p, p_meas=p, p_mem=p/100)
    circuit.append("M", range(H_x.shape[1]))
    samples = circuit.compile_sampler().sample(num_samples)
    is_flagged = np.any(samples[:, :-H_x.shape[1]], axis=1)
    filtered_measurements = samples[~is_flagged, -H_x.shape[1]:]
    # TODO: we need decoding to get LER
    syndromes = filtered_measurements @ H_x.T % 2
    correction = ...
    fixed_measurements = filtered_measurements
    logical_measurements = filtered_measurements @ L_x.T % 2
    LER = np.average(np.all(logical_measurements == 0, axis=1))
    AR = 1 - np.average(is_flagged)

    raw_cnots = [l for (name, l, _) in circ.flattened_operations() if name == "CX"]
    cnots = [(ops[i], ops[i + 1]) for ops in raw_cnots for i in range(0, len(ops), 2)]
    num_cx = len(cnots)
    num_flags = circ.num_qubits - H_x.shape[1]
    num_qubits = circ.num_qubits
    depth = len(_layer_cnot_circuit(cnots))

    return LER, AR, num_cx, num_flags, num_qubits, depth


if __name__ == "__main__":
    # methods = ["FAO", "MQT"]
    methods = {"FAO": FAO_QECCS}
    for method_name, code_iterator in methods.items():
        for code in code_iterator():
            print(method_name, code)
            LER, AR, num_cx, num_flags, num_qubits, depth = benchmark_CAO_state_prep(code, method_name)
            # print(f"Logical Error Rate = {LER:.2%}")
            print(f"Acceptance Rate = {AR:.2%}", end=" \t ")
            print(f"Number of CNOT gates = {num_cx}", end=" \t ")
            print(f"Number of flags = {num_flags}", end=" \t ")
            print(f"Number of qubits = {num_qubits}", end=" \t ")
            print(f"Depth = {depth}")
            print()




