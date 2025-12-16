import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_circular_cubic_graph(G: nx.Graph) -> None:
    """
    Draws the graph using a circular layout.
    """
    # Ensure nodes are sorted so the layout follows the index order 0, 1, ... N-1
    # This respects the circle logic [(i, (i+1) % N)...] visually.
    pos = nx.circular_layout(G)

    plt.figure(figsize=(6, 6))
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_color='lightblue',
        edge_color='gray',
    )
    plt.show()


def random_circular_cubic_graph(N: int) -> nx.Graph:
    """
    Constructs a random circular cubic graph.

    The circle is always [(i, (i+1) % N) for i in range(N)].
    Adds a random perfect matching (chords) such that no chord overlaps with the circle.
    """
    if N % 2 != 0:
        raise ValueError("N must be even to create a cubic graph.")
    if N < 4:
        raise ValueError("N must be at least 4.")

    while True:
        # 1. Initialize Graph and add the Cycle
        G = nx.Graph()
        G.add_nodes_from(range(N))

        cycle_edges = [(i, (i + 1) % N) for i in range(N)]
        G.add_edges_from(cycle_edges)

        # 2. Try to find a valid random perfect matching for the chords
        # (Every node needs exactly 1 more edge to become Degree 3)
        nodes = list(range(N))
        random.shuffle(nodes)

        chords = []
        valid_matching = True

        # Iterate through the shuffled nodes in pairs: (0,1), (2,3), etc.
        for i in range(0, N, 2):
            u, v = nodes[i], nodes[i + 1]

            # Constraint: The chord cannot be an edge that already exists in the circle.
            # Neighbors of u in the circle are (u+1)%N and (u-1)%N.
            if v == (u + 1) % N or v == (u - 1) % N:
                valid_matching = False
                break

            chords.append((u, v))

        # If we found a matching where no edge overlaps the circle, we are done.
        if valid_matching:
            G.add_edges_from(chords)
            return G


def random_circular_graph_with_girth(N: int, min_girth: int, max_steps: int = 100_000) -> nx.Graph | None:
    """
    Constructs a random circular cubic graph with girth >= min_girth using
    randomized backtracking.

    Args:
        N: Number of nodes (must be even).
        min_girth: Minimum girth required.
        max_steps: Maximum number of recursive steps (search tree nodes) to visit.
                   Prevents infinite loops or excessive runtimes on hard constraints.

    Returns:
        nx.Graph if successful, None if no solution found within max_steps.
    """
    if N % 2 != 0:
        raise ValueError("N must be even.")

    # 1. Setup
    adj = {i: [((i - 1) % N), ((i + 1) % N)] for i in range(N)}

    # Track nodes that still need a chord (nodes with degree 2)
    nodes_needing_chord = list(range(N))
    random.shuffle(nodes_needing_chord)

    min_chord_span = min_girth - 1

    # Counter for the recursion limit
    steps_taken = 0

    def is_valid_edge(u, v, current_adj):
        """
        Checks if adding edge (u, v) violates the girth constraint.
        """
        # Constraint A: Ring Distance
        dist = abs(u - v)
        ring_dist = min(dist, N - dist)
        if ring_dist < min_chord_span:
            return False

        # Constraint B: Path Distance (BFS)
        q = [(u, 0)]
        visited = {u}
        while q:
            curr, depth = q.pop(0)
            if depth >= min_chord_span:
                continue
            if curr == v:
                return False
            for neighbor in current_adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, depth + 1))
        return True

    def solve(nodes_to_pair):
        nonlocal steps_taken
        steps_taken += 1

        # STOP CONDITION: Exceeded computation budget
        if steps_taken > max_steps:
            return False

        # Base Case: All nodes paired
        if not nodes_to_pair:
            return True

        u = nodes_to_pair[0]
        candidates = nodes_to_pair[1:]
        random.shuffle(candidates)

        for v in candidates:
            if is_valid_edge(u, v, adj):
                # DO: Add edge
                adj[u].append(v)
                adj[v].append(u)

                # Recursive Step: Remove u and v from the list and continue
                remaining = [n for n in nodes_to_pair if n != u and n != v]
                if solve(remaining):
                    return True

                # Check again after returning from recursion:
                # If we hit the limit deep down, we want to stop trying neighbors here too.
                if steps_taken > max_steps:
                    return False

                # UNDO
                adj[u].pop()
                adj[v].pop()

        return False

    # 2. Run the Solver
    # Ensure recursion depth handles N (each step removes 2 nodes, depth is N/2)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), N + 100))

    success = solve(nodes_needing_chord)

    if success:
        return nx.Graph(adj)
    else:
        return None


def circular_double_edge_swap(G: nx.Graph, nswap: int = 1, max_tries: int = 100) -> None:
    """
    Performs double edge swaps on the 'chords' of a circular cubic graph.

    Assumptions:
    1. G is 3-regular.
    2. G contains the cycle edges [(i, (i+1)%N) for i in range(N)].

    This function preserves the 3-regularity and the outer circle structure.
    It modifies G in place.
    """
    if len(G) < 4:
        raise ValueError("Graph must have at least 4 nodes.")

    N = len(G)

    # Helper: Check if an edge (u, v) is part of the fixed circle
    def is_ring_edge(u, v):
        diff = abs(u - v)
        return diff == 1 or diff == N - 1

    for _ in range(nswap):
        # 1. Identify all chords
        # Since the graph changes, we must re-scan for chords (or track them)
        # Filter out the ring edges.
        chords = [e for e in G.edges() if not is_ring_edge(*e)]

        if len(chords) < 2:
            raise ValueError("Not enough chords to perform a swap.")

        swap_successful = False

        for attempt in range(max_tries):
            (u, v), (x, y) = random.sample(chords, 2)

            if random.random() < 0.5:
                new_1, new_2 = (u, x), (v, y)
            else:
                new_1, new_2 = (u, y), (v, x)

            # 3. Validation: Ring Collision
            # We must ensure the new "chords" are not actually existing ring edges.
            # We don't need to check if they are existing *chords*, because u only
            # has one chord (to v), so u cannot already have a chord to x.
            if G.has_edge(*new_1) or G.has_edge(*new_2):
                continue

            # 4. Apply Swap
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            G.add_edge(*new_1)
            G.add_edge(*new_2)

            swap_successful = True
            break

        if not swap_successful:
            pass


def girth_non_decreasing_circular_double_edge_swap(G: nx.Graph, nswap: int = 1, max_tries: int = 100) -> int:
    """
    Performs `nswap` double edge swaps on the chords of a circular cubic graph,
    ensuring that the girth never decreases.

    Args:
        G: The graph (modified in place).
        nswap: Number of successful swaps to perform.
        max_tries: Maximum attempts per swap to find a valid move.

    Returns:
        int: The number of successful swaps performed (may be less than nswap if max_tries is hit).
    """
    N = len(G)

    # Helper: Identify ring edges to protect them
    def is_ring_edge(u, v):
        diff = abs(u - v)
        return diff == 1 or diff == N - 1

    swaps_done = 0

    for _ in range(nswap):
        # 1. Calculate current baseline girth at the start of this step
        # We must re-calculate this every time because a previous swap might have
        # increased the girth, raising the bar for future swaps.
        current_girth = nx.girth(G)
        chords = [e for e in G.edges() if not is_ring_edge(*e)]
        if len(chords) < 2:
            raise ValueError("Not enough chords to perform a swap.")

        success = False
        for attempt in range(max_tries):
            # 2. Pick two random chords
            (u, v), (x, y) = random.sample(chords, 2)

            # Determine swap configuration
            if random.random() < 0.5:
                target_1, target_2 = (u, x), (v, y)
            else:
                target_1, target_2 = (u, y), (v, x)

            # 3. Check for Ring Collisions
            # The new edges must not duplicate existing ring edges
            if G.has_edge(*target_1) or G.has_edge(*target_2):
                continue

            # 4. Girth Safety Check
            # We must verify that adding target_1 and target_2 doesn't create
            # a cycle smaller than current_girth.
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            G.add_edge(*target_1)
            G.add_edge(*target_2)

            # Check shortest path for first new edge
            # d1 = nx.shortest_path_length(G, target_1[0], target_1[1])
            # d2 = nx.shortest_path_length(G, target_2[0], target_2[1])
            new_girth = nx.girth(G)
            if new_girth < current_girth:
                # Revert and fail attempt
                G.add_edge(u, v)
                G.add_edge(x, y)
                G.remove_edge(*target_1)
                G.remove_edge(*target_2)
                continue
            else:
                pass

            # 5. Apply Swap (Commit)
            # If we are here, the new local cycles are >= current_girth.
            # (Removing edges cannot decrease girth, so we are safe).
            success = True
            swaps_done += 1
            break

        if not success:
            # If we fail max_tries times for a single swap, we likely can't
            # improve/maintain the graph further or it's too constrained.
            break

    return swaps_done


def find_t_non_local_cut(G: nx.Graph, T: int) -> list[int] | None:
    """
    Finds a T-non-local cut in a circular 3-regular graph by scanning all
    contiguous segments of the circle.

    Args:
        G: A 3-regular graph with nodes 0..N-1 forming a Hamiltonian cycle.
        T: The boundary threshold.

    Returns:
        list[int]: The list of nodes in the cut if found, otherwise None.
    """
    N = G.number_of_nodes()

    # 1. Precompute Chords
    # We identify which edge is the 'chord' for every node (vs. ring edges)
    # This allows O(1) lookups during the scan.
    chord_map = {}
    for u in range(N):
        # Ring neighbors are strictly defined by the circular geometry
        prev_node = (u - 1) % N
        next_node = (u + 1) % N

        for v in G.neighbors(u):
            if v != prev_node and v != next_node:
                chord_map[u] = v
                break

    # 2. Scan All Segments
    # Outer loop: Try every possible start position for the segment
    for start_node in range(N):

        # We maintain the state of the current segment incrementally
        # to avoid re-calculating edges from scratch every time.
        current_boundary_chords = 0
        current_internal_chords = 0

        # Track which nodes are currently in the segment for O(1) checks
        in_segment = [False] * N
        segment_nodes = []

        # Inner loop: Expand the segment length from 1 up to N/2
        for length in range(1, (N // 2) + 1):

            # The new node we are adding to the segment
            new_node = (start_node + length - 1) % N

            segment_nodes.append(new_node)
            in_segment[new_node] = True

            # Check the chord of the new node
            partner = chord_map[new_node]

            if in_segment[partner]:
                # CRITICAL LOGIC:
                # The partner is ALREADY in the segment.
                # Previously, this chord counted as +1 boundary (leaving from partner).
                # Now, it connects two internal nodes.
                # So: Boundary decreases by 1, Internal Chords increases by 1.
                current_boundary_chords -= 1
                current_internal_chords += 1
            else:
                # The partner is outside the segment.
                # This chord adds to the boundary.
                current_boundary_chords += 1

            # --- Check Conditions ---

            # 1. Internal Complexity (Edges >= Nodes)
            # A segment of length L has (L-1) ring edges inside it.
            # Total Edges = (L-1) + internal_chords.
            # Condition: (L-1) + internal_chords >= L  ==>  internal_chords >= 1.
            has_cycle = (current_internal_chords >= 1)

            # 2. Small Boundary (<= T)
            # The boundary consists of:
            # - The 2 ring edges (one at the start, one at the end of the segment)
            # - The chords crossing out of the segment
            total_boundary = 2 + current_boundary_chords

            if has_cycle and total_boundary <= T:
                return list(segment_nodes)

    return None


def random_circular_cubic_graph_with_no_T_nonlocal_cut(N: int, T: int, max_iter: int = 100) -> nx.Graph | None:
    G = random_circular_graph_with_girth(N, T + 1)
    if G is None:
        return None
    for _ in range(max_iter):
        cut = find_t_non_local_cut(G, T)
        if cut is None:
            return G
        else:
            girth_non_decreasing_circular_double_edge_swap(G, T - nx.girth(G) - 2)
    return None


if __name__ == "__main__":
    try:
        G = random_circular_cubic_graph_with_no_T_nonlocal_cut(16, 5)
        if G is not None:
            draw_circular_cubic_graph(G)
    except ValueError as e:
        print(e)
