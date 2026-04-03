import networkx as nx
import numpy as np


def match_edges(H: np.ndarray, non_pivots: list[int],
                z_digraphs: list[nx.DiGraph], x_digraphs: list[nx.DiGraph],
                z_candidates: list[list[int]], x_candidates: list[list[int]]) -> list[
    tuple[tuple[int, int], tuple[int, int]]]:
    # 1. Identify all required logical connections dictated by the parity check matrix
    edge_list = [
        (i, j)
        for i, r in enumerate(H)
        for j, x in enumerate(r[non_pivots])
        if x == 1
    ]

    # 2. Initialize a global tracking digraph to strictly monitor transitive dependencies
    tracker = nx.DiGraph()

    # Prefix node names to avoid collisions between independent cat states
    for i, D in enumerate(z_digraphs):
        tracker.add_edges_from((f"Z_{i}_{u}", f"Z_{i}_{v}") for u, v in D.edges())
    for j, D in enumerate(x_digraphs):
        tracker.add_edges_from((f"X_{j}_{u}", f"X_{j}_{v}") for u, v in D.edges())

    # Deep copy candidate pools so we can mutate them during the search
    z_pools = [[c for c in pool] for pool in z_candidates]
    x_pools = [[c for c in pool] for pool in x_candidates]

    # 3. Backtracking Constraint Solver
    def backtrack(remaining_edges, current_matches):
        if not remaining_edges:
            return current_matches

        # Heuristic: Sort remaining edges by fewest available candidate combinations.
        # This dramatically prunes the search tree by attacking bottlenecks first.
        remaining_edges = sorted(remaining_edges, key=lambda e: len(z_pools[e[0]]) * len(x_pools[e[1]]))
        i, j = remaining_edges[0]

        for z_val in list(z_pools[i]):
            for x_val in list(x_pools[j]):

                # Determine provisional cross-edges dictated by the inter-cat CNOT logic
                new_edges = []
                for u, _ in z_digraphs[i].in_edges(z_val):
                    new_edges.append((f"Z_{i}_{u}", f"X_{j}_{x_val}"))
                for u, _ in x_digraphs[j].in_edges(x_val):
                    new_edges.append((f"X_{j}_{u}", f"Z_{i}_{z_val}"))

                # Cycle Verification: Ensure adding these edges preserves the DAG
                added_edges_this_step = []
                cycle_found = False

                for src, dst in new_edges:
                    # If src == dst or a path already exists from dst to src, adding src->dst creates a cycle
                    if src == dst or nx.has_path(tracker, dst, src):
                        cycle_found = True
                        break
                    tracker.add_edge(src, dst)
                    added_edges_this_step.append((src, dst))

                if not cycle_found:
                    # State transition: Commit provisional match and dive deeper
                    z_pools[i].remove(z_val)
                    x_pools[j].remove(x_val)

                    result = backtrack(remaining_edges[1:], current_matches + [((i, j), (z_val, x_val))])
                    if result is not None:
                        return result  # Valid global state found

                    # State rollback: The branch hit a dead end
                    z_pools[i].append(z_val)
                    x_pools[j].append(x_val)

                # Cleanup the tracker if a cycle was found or if we rolled back
                tracker.remove_edges_from(added_edges_this_step)

        return None  # Trigger upstream backtracking

    final_matching = backtrack(edge_list, [])

    if final_matching is None:
        raise ValueError("No valid cycle-free edge matching exists for this matrix topology.")

    return final_matching
