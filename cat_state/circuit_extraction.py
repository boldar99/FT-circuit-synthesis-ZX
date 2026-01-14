from collections import defaultdict

import stim


def find_all_hamiltonian_paths(graph):
    """
    Optimized for 3-regular graphs.
    Uses bitmasks and static adjacency tuples for maximum speed.
    """
    nodes = list(graph.nodes)
    n = len(nodes)

    # 1. Map nodes to integers 0..N-1
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # 2. Create a static Adjacency Tuple (faster than lists)
    # Since it's 3-regular, every row has exactly 3 entries.
    adj = [None] * n
    for node in nodes:
        u = node_to_idx[node]
        # Convert neighbors to mapped integers
        neighbors = tuple(node_to_idx[v] for v in graph.neighbors(node))
        adj[u] = neighbors

    # Convert list of tuples to tuple of tuples for fastest read access
    adj = tuple(adj)

    # Pre-allocate path array to avoid list creation overhead
    path = [0] * n

    # Pre-compute bit powers to avoid bit-shifting in the tight loop
    powers = [1 << i for i in range(n)]

    def solve(u, pos, mask):
        path[pos] = u

        # Base Case: Path complete
        if pos == n - 1:
            # Yield the recovered node objects
            yield [nodes[i] for i in path]
            return

        # Recursive Step: Unrolled for performance
        # We iterate over the fixed tuple of neighbors
        for v in adj[u]:
            # Bitwise check: if (mask & 2^v) == 0
            if not (mask & powers[v]):
                yield from solve(v, pos + 1, mask | powers[v])

    # 3. Execution Strategy
    # Iterate through all start nodes
    for i in range(n):
        yield from solve(i, 0, powers[i])


def sorted_pair(v1, v2):
    return (v1, v2) if v1 < v2 else (v2, v1)


def extract_circuit(G, ham_path, marks: dict | list, noise_model: dict | None = None):
    circ = stim.Circuit()
    if isinstance(marks, dict):
        marks = {sorted_pair(v1, v2): int(v) for (v1, v2), v in marks.items()}
    else:
        marks = {sorted_pair(v1, v2): 1 for v1, v2 in marks}

    num_flags = G.number_of_edges() - len(ham_path)
    flag_dict = dict()

    v0, v1 = ham_path[0]
    neighbors_0 = tuple(set(G.neighbors(v0)) - {v1})
    flag_dict[sorted_pair(v0, neighbors_0[0])] = 0
    flag_dict[sorted_pair(v0, neighbors_0[1])] = 1

    circ.append("H", num_flags)
    circ.append("CNOT", [num_flags, 0])
    circ.append("CNOT", [num_flags, 1])

    next_free_flag = 2
    next_free_cat = num_flags + 1

    for _ in range(marks.get(sorted_pair(v0, v1), 0)):
        circ.append("CNOT", [num_flags, next_free_cat])
        next_free_cat += 1

    v_prev = v0
    v_current, v_next = None, None
    for v_current, v_next in ham_path[1:]:
        if len(set(G.neighbors(v_current)) - {v_prev, v_next}) != 1:
            pass
        [v_neighbor] = set(G.neighbors(v_current)) - {v_prev, v_next}
        link = sorted_pair(v_current, v_neighbor)

        if link not in flag_dict:
            circ.append("CNOT", [num_flags, next_free_flag])
            flag_dict[link] = next_free_flag
            next_free_flag += 1
        else:
            flag_qubit = flag_dict[link]

            for _ in range(marks.get(link, 0)):
                circ.append("CNOT", [flag_qubit, next_free_cat])
                next_free_cat += 1

            circ.append("CNOT", [num_flags, flag_qubit])
            circ.append("MR", flag_qubit)

        for _ in range(marks.get(sorted_pair(v_current, v_next), 0)):
            circ.append("CNOT", [num_flags, next_free_cat])
            next_free_cat += 1

        v_prev = v_current

    if len(ham_path) > 1:
        neighbors_last = tuple(set(G.neighbors(v_next)) - {v_current})
        link_penultimate = sorted_pair(v_next, neighbors_last[0])
        link_last = sorted_pair(v_next, neighbors_last[1])
        num_cat_legs = marks.get(link_penultimate, 0) + marks.get(link_last, 0)
        i = 0

        for _ in range(marks.get(link_penultimate, 0)):
            i += 1
            if i != num_cat_legs:
                circ.append("CNOT", [flag_dict[link_penultimate], next_free_cat])
                next_free_cat += 1
        for _ in range(marks.get(link_last, 0)):
            i += 1
            if i != num_cat_legs:
                circ.append("CNOT", [flag_dict[link_last], next_free_cat])
                next_free_cat += 1

        circ.append("CNOT", [num_flags, flag_dict[link_penultimate]])
        circ.append("CNOT", [num_flags, flag_dict[link_last]])
        circ.append("MR", flag_dict[link_penultimate])
        circ.append("MR", flag_dict[link_last])

    return circ


def make_stim_circ_noisy(circ, p_1=0, p_2=0, p_mem=0, p_meas=0, p_sp=0):
    noisy_circ = stim.Circuit()
    num_qubits = circ.num_qubits

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
                    noisy_circ.append(gate_name, targets, p_meas)
                else:
                    noisy_circ.append(gate_name, targets)

                # Post-gate noise (Reset Preparation Error)
                if gate_name in ("R", "RX", "RY", "MR") and p_sp > 0:
                    noisy_circ.append("X_ERROR", targets, p_sp)

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
