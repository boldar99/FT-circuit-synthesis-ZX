import warnings
from itertools import combinations

from find_cubic_local_mincut import generate_high_girth_cubic_graph

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
import stim


def density_lower_bound(t):
    with warnings.catch_warnings(action="ignore"):
        return np.where(t == 1, np.inf,
            (
                np.ceil((t + 3) / 2)
                * np.floor((t + 3) / 2)
            )
            /
            (
                np.ceil((t + 3) / 2)
                * np.floor((t + 3) / 2)
                +
                np.ceil((t - 3) / 2)
                * np.floor((t + 3) / 2)
                +
                np.floor((t - 3) / 2)
                * np.ceil((t + 3) / 2)
            )
        )


def minimum_E_and_V(n, t):
    density = density_lower_bound(t)
    E_nec = np.ceil(n / density).astype(int)
    remainder = E_nec % 3
    adjustment = (3 - remainder) % 3
    E_final = E_nec + adjustment
    V_final = (2 * E_final) // 3
    return E_final, V_final


def visualize_cat_state_base(G, ham_path, markings):
    marked_edges = [e for e, is_marked in markings.items() if is_marked]

    plt.figure(figsize=(5, 5))
    pos = nx.kamada_kawai_layout(G) # Kamada-Kawai usually looks best for regular graphs
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={e: "  |  " * num_marks for e, num_marks in markings.items()},
                                 font_size=18, font_weight='bold', bbox=dict(alpha=0))
    nx.draw_networkx_edges(
        G, pos=pos,
        edgelist=ham_path,
        edge_color='red', width=1.5
    )
    plt.show()


def find_all_hamiltonian_paths(graph):
    """
    Yields all Hamiltonian paths in a graph using backtracking.
    """
    n = len(graph.nodes)
    for start_node in graph.nodes:
        path = [start_node]
        visited = {start_node}

        def search(current_path):
            if len(current_path) == n:
                yield current_path
                return

            last_node = current_path[-1]
            for neighbor in graph.neighbors(last_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    yield from search(current_path + [neighbor])
                    visited.remove(neighbor) # Backtrack

        yield from search(path)


class GraphMarker:
    def __init__(self, G, ham_path, max_marks = None):
        self.G = G
        self.ham_path = ham_path
        self.n = max_marks
        self.L = nx.line_graph(G)

        # Create a mapping from edge (node in L) to SAT variable ID (1, 2, 3...)
        self.edge_to_id = {edge: i+1 for i, edge in enumerate(self.L.nodes())}
        self.id_to_edge = {i: edge for edge, i in self.edge_to_id.items()}

    def _get_id(self, u, v):
        """Helper to get edge ID safely handling (u,v) vs (v,u) ordering."""
        if (u, v) in self.edge_to_id:
            return self.edge_to_id[(u, v)]
        if (v, u) in self.edge_to_id:
            return self.edge_to_id[(v, u)]
        raise ValueError(f"Edge ({u}, {v}) not found in Line Graph mapping.")

    def _add_end_constraint(self, wcnf):
        """
        Constraint: There must be a marked edge incident to the END of the path
        (which is NOT part of the path itself).
        """
        prev_node, end_node = self.ham_path[-1]
        candidate_ids = []
        for neighbor in self.G.neighbors(end_node):
            if neighbor == prev_node:
                continue
            candidate_ids.append(self._get_id(end_node, neighbor))

        # Clause: [c1, c2, ...] -> (c1 OR c2 OR ...)
        wcnf.append(candidate_ids)

    def solve_t_2(self):
        ret = {e: 2 for e in self.G.edges()}
        sum = 2 * len(ret)
        i = 0
        keys = list(ret.keys())
        while sum > self.n and i < len(keys):
            ret[keys[i]] -= 1
            sum -= 1
            i = (i + 1) % len(keys)
        return ret

    def solve_t_3(self):
        if self.n is None:
            return {e: 1 for e in self.G.edges()}
        edges = list(self.G.edges())[:self.n]
        return {e: 1 for e in edges}

    def _solve_wcnf(self, wcnf):
        self._add_end_constraint(wcnf)
        with RC2(wcnf) as rc2:
            model = rc2.compute()
            if model is None:
                print("No solution found (UNSAT).")
                return None

            markings = {}
            model_set = set(model)
            sum = 0
            for edge in self.G.edges():
                # Normalize edge lookup
                key = edge if edge in self.edge_to_id else (edge[1], edge[0])
                markings[edge] = int(self.edge_to_id[key] in model_set and sum < self.n)
                sum += markings[edge]
            return markings

    def solve_t_4(self):
        """
        Problem 1: Marked edges must have an Unmarked neighbor.
        Constraint: x_e -> (NOT x_n1 OR NOT x_n2 ...)
        CNF Clause: [-x_e, -x_n1, -x_n2, ...]
        """
        wcnf = WCNF()

        for e in self.L.nodes():
            e_id = self.edge_to_id[e]
            neighbor_ids = [self.edge_to_id[n] for n in self.L.neighbors(e)]

            # HARD Constraint: If e is Marked, at least one neighbor must be Unmarked.
            # Logic: NOT(e) OR NOT(n1) OR NOT(n2)...
            # In SAT terms: [-e_id, -n1_id, -n2_id...]
            wcnf.append([-e_id] + [-nid for nid in neighbor_ids])

            # SOFT Constraint: We WANT to mark edge 'e' (Maximize True)
            # Weight 1 for setting e to True
            wcnf.append([e_id], weight=1)

        return self._solve_wcnf(wcnf)

    def solve_t_5(self):
        """
        Problem 2: EVERY edge (Marked or Unmarked) must have an Unmarked neighbor.
        Constraint: For any edge e, it implies (NOT x_n1 OR NOT x_n2 ...)
        CNF Clause: [-x_n1, -x_n2, ...] (The state of e itself doesn't relax the rule)
        """
        wcnf = WCNF()

        for e in self.L.nodes():
            neighbor_ids = [self.edge_to_id[n] for n in self.L.neighbors(e)]

            # HARD Constraint: Regardless of e's state, one neighbor must be Unmarked.
            # Logic: NOT(n1) OR NOT(n2) OR ...
            wcnf.append([-nid for nid in neighbor_ids])

            # SOFT Constraint: Maximize Marked edges
            e_id = self.edge_to_id[e]
            wcnf.append([e_id], weight=1)

        return self._solve_wcnf(wcnf)

    def solve_t_6(self):
        """
        3. Clustered Unmarked Edges.
        - Rule A: Marked edges must have an Unmarked neighbor.
        - Rule B: Unmarked edges must have at least 2 Unmarked neighbors.
        """
        wcnf = WCNF()
        for e in self.L.nodes():
            e_id = self.edge_to_id[e]
            neighbor_ids = [self.edge_to_id[n] for n in self.L.neighbors(e)]

            # Rule A: Prevent "All Marked" around a marked edge
            # Clause: [-e, -n1, -n2, -n3, -n4]
            wcnf.append([-e_id] + [-nid for nid in neighbor_ids])

            # Rule B: Force Clusters.
            # Logic: If 'e' is Unmarked (False), it cannot have >=3 Marked neighbors.
            # CNF: For every trio of neighbors (a,b,c): NOT(a & b & c & !e)
            # Clause: [-a, -b, -c, e]
            for combo in combinations(neighbor_ids, 3):
                clause = [-c for c in combo] + [e_id]
                wcnf.append(clause)

            # Soft: Maximize Marked
            wcnf.append([e_id], weight=1)

        return self._solve_wcnf(wcnf)

    def find_solution(self, T):
        t_to_function = {
            2: self.solve_t_2,
            3: self.solve_t_3,
            4: self.solve_t_4,
            5: self.solve_t_5,
            6: self.solve_t_6,
        }
        if T not in t_to_function:
            raise NotImplementedError
        return t_to_function[T]()


def sorted_pair(v1, v2):
    return (v1, v2) if v1 < v2 else (v2, v1)


def extract_circuit(G, ham_path, marks: dict | list):
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
    for i in range(2, n+1):
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



def cat_state_FT(n, t, max_tries=1_000_000) -> stim.Circuit | None:
    t_alt =  (np.floor(n / 2) - 1).astype(int)
    T = min(t, t_alt)

    if n < 1:
        raise ValueError
    if n <= 3:
        return unflagged_cat(n)
    if n <= 5 or T == 1:
        return one_flagged_cat(n)
    if n == 6:
        return cat_state_6()

    E, N = minimum_E_and_V(n, T)

    try:
        G = generate_high_girth_cubic_graph(N, T, max_tries=max_tries)
        if G is None:
            return None

        p = next(find_all_hamiltonian_paths(G))
        ham_path = list(zip(p, p[1:]))

        marker = GraphMarker(G, ham_path=ham_path, max_marks=n)
        marks = marker.find_solution(T)
    except:
        return None

    return extract_circuit(G, ham_path, marks)


if __name__ == "__main__":
    import time
    start_time = time.time()

    N = 44
    T = 7

    print("Theoretically optimal number of flags for given n and t (from actual circuit instances):")
    print()

    ns = range(2, N)
    print('t\\n |', end=' ')
    for f in ns:
        print(f if f > 9 else f' {f}', end=' ')
    print()
    print("-" * 3 * N)
    for t in range(1, T):
        print(f"t={t} |", end=' ')
        for n in ns:
            circ = cat_state_FT(n, t)
            if circ is None:
                flag = "-"
            else:
                flag = circ.num_qubits - n
            print(flag if len(str(flag)) == 2 else f' {flag}', end=' ')
        print()


    print("--- %s seconds ---" % (time.time() - start_time))
