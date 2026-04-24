import concurrent.futures
import json

import numpy as np
import stim
import tesseract_decoder
from tesseract_decoder import tesseract

from spidercat.simulate import _layer_cnot_circuit
from spiderstate.cat_at_origin import row_optimized_cat_at_origin
from spiderstate.utils import load_qecc, FAO_QECCS, _expand_stim_operation_list, _layer_circuit_ops, \
    layered_ops_to_noisy_stim_circuit, apply_qubit_reuse


def benchmark_CAO_state_prep(code: str, method: str, p=0.001, num_samples=10_000_000):
    is_self_dual, H_x, H_z, L_x, L_z, d = load_qecc(code, method)
    if code in ("49_1_5", "95_1_7"):
        print("State: |+>")
        H_x, H_z = H_z, H_x
        L_x, L_z = L_z, L_x
    else:
        print("State: |0>")
    circ = row_optimized_cat_at_origin(H_z, d, max_basis_tries=25_000)
    operations = [(op, targets) for (op, targets, params) in circ.flattened_operations() if op != "DETECTOR"]
    detectors = [(op, [stim.target_rec(targets[0][1])]) for (op, targets, params) in circ.flattened_operations() if op == "DETECTOR"]
    operations = _expand_stim_operation_list(operations)
    layered_ops = _layer_circuit_ops(operations, circ.num_qubits)
    # final_ops, num_sim_qubits = apply_qubit_reuse(layered_ops)
    noisy_circ = layered_ops_to_noisy_stim_circuit(layered_ops + [detectors], circ.num_qubits, 0, p, 2/3*p, 2/3*p, p/100)


    noisy_circ.append("M", range(H_x.shape[1]))

    for i, H in enumerate(H_x):
        qubit_indices = np.where(H == 1)[0]
        record_targets = [stim.target_rec(i - H_x.shape[1]) for i in qubit_indices]
        noisy_circ.append("DETECTOR", record_targets)
    for i, L in enumerate(L_x):
        qubit_indices = np.where(L == 1)[0]
        record_targets = [stim.target_rec(i - H_x.shape[1]) for i in qubit_indices]
        noisy_circ.append("OBSERVABLE_INCLUDE", record_targets, i)

    # print(noisy_circ)

    # 1. Compile the Detector Error Model (DEM) from the full noisy circuit
    # The DEM acts as the blueprint for Tesseract's search graph.
    dem = noisy_circ.detector_error_model()

    # 2. Initialize the Tesseract decoder
    tesseract_config = tesseract.TesseractConfig(dem=dem, beam_climbing=True, no_revisit_dets=True)
    decoder = tesseract.TesseractDecoder(tesseract_config)

    # 3. Sample detectors and logicals
    detectors, logicals = noisy_circ.compile_detector_sampler().sample(num_samples, separate_observables=True)

    # 4. Post-selection: Identify flagged shots
    is_flagged = np.any(detectors[:, :-H_x.shape[0]], axis=1)
    AR = 1.0 - np.average(is_flagged)
    detectors = detectors[~is_flagged]
    logicals = logicals[~is_flagged]
    # AR = 1.0

    # 4. Multithreaded Decoding
    corrections = decoder.decode_batch(detectors)

    # 6. Decode
    predicted_logicals = logicals ^ corrections

    # If any logical observable failed to be corrected in a shot, that shot is a logical error
    incorrect_predictions = np.any(predicted_logicals, axis=1)
    LER = np.average(incorrect_predictions)

    raw_cnots = [l for (name, l, _) in circ.flattened_operations() if name == "CX"]
    cnots = [(ops[i], ops[i + 1]) for ops in raw_cnots for i in range(0, len(ops), 2)]
    num_cx = len(cnots)
    num_flags = circ.num_qubits - H_x.shape[1]
    num_qubits = circ.num_qubits
    depth = len(_layer_cnot_circuit(cnots))

    return LER, AR, num_cx, num_flags, num_qubits, noisy_circ.num_qubits, depth


if __name__ == "__main__":
    # methods = ["FAO", "MQT"]
    MQT_codes = [
        "17_1_5",
        "19_1_5",
        "25_1_5",
        "20_2_6",
        "31_1_7",
        "39_1_7"
    ]

    # LER, AR, num_cx, num_flags, num_qubits, depth = benchmark_CAO_state_prep("95_1_7", "FAO")
    # methods = {"FAO": FAO_QECCS}
    # for method_name, code_iterator in methods.items():
    for code in MQT_codes:
        print("MQT", code)
        LER, AR, num_cx, num_flags, num_qubits, num_sim_qubits, depth = benchmark_CAO_state_prep(code, "MQT")
        print(f"Logical Error Rate = {LER:.4e}", end=";\t ")
        print(f"Acceptance Rate = {AR:.4f}", end=";\t ")
        print(f"CXs = {num_cx}", end=";\t ")
        print(f"Sim. Qubits = {num_sim_qubits}", end=";\t ")
        print(f"Flags = {num_flags}", end=";\t ")
        print(f"Depth = {depth}", end=";\t ")
        print(f"Expected Circuit Volume = {int(depth * num_sim_qubits / AR)}")
        print()




