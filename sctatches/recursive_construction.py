import warnings
import stim

def recursive_construction(n: int, k: int, t: int, use_flags=False):
    """
    Construct the cat state using the recursive construction.
    
    Args:
        n: The number of qubits in each block.
        k: The number of recursive steps.
        t: The level of fault-tolerance.
        use_flags: Whether to use flags for post-selection.
    
    Returns:
        The cat state of 2^k * n qubits.
    """

    if t >= n - 1:
        warnings.warn("Using fully fault tolerant construction.")
        t = n - 1
    n_flags = ((2 ** k) - 1) * (t + 1) if use_flags else 0
    cat_size = 2 ** k * n
    flag_idx = 0
    cat_offset = n_flags
    print('num flags', n_flags)

    depths = [0] * cat_size
    circ = stim.Circuit()

    def add_zz_measurement(q1, q2):
        nonlocal flag_idx
        if use_flags:
            circ.append("CNOT", [cat_offset + q1, flag_idx])
            circ.append("CNOT", [cat_offset + q2, flag_idx])
            circ.append("MR", [flag_idx])
            flag_idx += 1
        else:
            circ.append("CZ", [cat_offset + q1, cat_offset + q2])
        assert depths[q1] == depths[q2]
        depths[q1] = depths[q2] = max(depths[q1], depths[q2]) + 1

    for lvl in range(k):
        block_size = 2 ** lvl * n
        n_blocks = 2 ** (k - lvl)
        for i in range(t + 1):
            for block_idx in range(n_blocks // 2):
                block_depths = depths[block_idx * 2 * block_size : (block_idx + 1) * 2 * block_size]
                min_depth = min(block_depths)
                offset = block_depths.index(min_depth) + block_idx * 2 * block_size
                add_zz_measurement(offset, offset + block_size)
    
    print('final depth', max(depths))
    print(circ.diagram("timeline-text"))
    return circ