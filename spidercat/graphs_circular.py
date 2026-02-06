import random
import sys
from collections import deque

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from spidercat.graphs_random import has_small_nonlocal_cut
from spidercat.utils import graph_exists_with_girth


def draw_circular_cubic_graph(G: nx.Graph) -> None:
    """
    Draws the graph using a circular layout.
    """
    # Ensure nodes are sorted so the layout follows the index order 0, 1, ... N-1
    # This respects the circle logic [(i, (i+1) % N)...] visually.
    pos = nx.circular_layout(G)

    plt.figure(figsize=(4, 4))
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

    Returns:
        nx.Graph if successful, None if no solution found within max_steps.
    """
    if not graph_exists_with_girth(N, min_girth):
        return None

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
        q = deque([(u, 0)])
        visited = np.zeros(N, dtype=bool)
        visited[u] = True
        while q:
            curr, depth = q.popleft()
            if curr == v:
                return False
            for neighbor in current_adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    if depth + 1 < min_chord_span:
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


def generate_disjoint_arcs(N, K):
    def _search(k_left, min_start, first_start):
        if k_left == 0:
            yield ()
            return

        # Search for the next arc
        # We limit 's' to N to prevent generating rotated duplicates of the same set
        for s in range(min_start, N):
            for l in range(2, N // 2):
                e = s + l

                # Constraint: Check if arc wraps around and hits the first arc
                # logic: (End of this arc) + 2 <= (Start of first arc + N)
                if e + 2 > N + first_start:
                    break

                for tail in _search(k_left - 1, e + 2, first_start):
                    yield ((s, e % N),) + tail

    # Main Loop: Select the first arc
    for s in range(N):
        for l in range(2, N // 2):
            e = s + l
            # Check if first arc is already too long for the cycle wrap
            if e + 2 > N + s:
                break

            for tail in _search(K - 1, e + 2, s):
                yield ((s, e % N),) + tail


# TODO: This can be sped up by only looking at slices, and outgoing edges from the slices.
def _find_t_non_local_cut(G: nx.Graph, T: int) -> list[int] | None:
    """
    Finds a T-non-local cut in a circular 3-regular graph.

    Instead of checking only contiguous segments, this checks ALL valid
    unions of disjoint arcs that satisfy the boundary condition 2*K <= T.

    Args:
        G: A 3-regular graph with nodes 0..N-1 forming a Hamiltonian cycle.
        T: The boundary threshold.

    Returns:
        list[int]: The list of nodes in the cut if found, otherwise None.
    """
    N = G.number_of_nodes()

    # 1. Precompute Chord Map for O(1) lookups
    chord_map = np.zeros(N, dtype=int)
    for u in range(N):
        prev_and_next_nodes = (u - 1) % N, (u + 1) % N
        [last_neighbour] = [n for n in G.neighbors(u) if n not in prev_and_next_nodes]
        chord_map[u] = last_neighbour

    # --- MAIN SOLVER ---

    # We only check K up to T // 2.
    # For T=6, we check K=1, K=2, K=3.
    max_k = T // 2

    for k in range(1, max_k + 1):
        for arcs in generate_disjoint_arcs(N, k):
            # 'arcs' is a tuple of (start, length) tuples

            # 1. Construct the Vertex Set S
            # We build the set to check internal chords efficiently
            S_array = np.zeros(N, dtype=bool)
            S_list = []
            the_other_side = set()

            for start, end in arcs:
                node = start
                stop_at_node = (end + 1) % N
                while node != stop_at_node:
                    S_array[node] = True
                    S_list.append(node)
                    the_other_side.add(chord_map[node])
                    node = (node + 1) % N

            # Optimization: Symmetry Check
            if len(S_list) > N // 2:
                continue

            # 2. Calculate Metrics
            # Ring Cuts = 2 * k (Each arc cuts 2 ring edges)
            ring_cuts = 2 * k
            the_other_side -= set(S_list)
            chord_cuts = len(the_other_side)
            internal_chords = len(S_list) - chord_cuts

            # Correct double counting
            internal_chords //= 2

            # 3. Verify Conditions
            # A. Small Boundary
            total_cut = ring_cuts + chord_cuts
            if total_cut > T:
                continue

            # B. Non-Local (Cycle) Condition
            # Topology check: Induced Edges >= Nodes?
            # Induced Edges = (Nodes - k) + Internal Chords  [Each arc of len L has L-1 edges]
            # Condition: (Nodes - k + Internal Chords) >= Nodes
            # Simplifies to: Internal Chords >= k
            if internal_chords >= k:
                return S_list

    return None

def _has_small_nonlocal_cut(G, T):
    return _find_t_non_local_cut(G, T) is not None


def random_circular_cubic_graph_with_no_T_nonlocal_cut(N: int, T: int, max_iter: int = 100) -> nx.Graph | None:
    G = random_circular_graph_with_girth(N, min_girth=T + 1)
    if G is None:
        return None
    for _ in range(max_iter):
        cut = has_small_nonlocal_cut(G, T)
        if not cut:
            return G
        else:
            girth_non_decreasing_circular_double_edge_swap(G, T - nx.girth(G) - 2)
    return None


if __name__ == '__main__':
    G = construct_special_marked_graph(22)
    if G is not None:
        nx.draw(G, pos=nx.circular_layout(G), with_labels=True)
        plt.show()
    else:
        print("No graph")

