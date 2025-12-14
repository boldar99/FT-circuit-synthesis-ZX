import warnings
from itertools import combinations

from find_cubic_local_mincut import generate_high_girth_cubic_graph, has_small_nonlocal_cut, \
    construct_cyclic_connected_graph

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from pysat.card import CardEnc, EncType
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


def minimum_number_of_flags(n, t):
    t_alt =  np.floor(n / 2) - 1
    t = np.where(t < t_alt, t, t_alt)
    E, N = minimum_E_and_V(n, t)
    return (np.ceil(E - N + 2).astype(int) - 1).tolist()


def visualize_cat_state_base(G, ham_path, markings):
    plt.figure(figsize=(5, 5))
    pos = nx.spring_layout(G, method="energy") # Kamada-Kawai usually looks best for regular graphs
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
    def __init__(self, G, ham_path = None, max_marks = None):
        self.G = G
        self.ham_path = ham_path
        self.n = max_marks
        self.L = nx.line_graph(G)

        # Create a mapping from edge (node in L) to SAT variable ID (1, 2, 3...)
        self.edge_to_id = {edge: i + 1 for i, edge in enumerate(self.L.nodes())}
        self.id_to_edge = {i: edge for edge, i in self.edge_to_id.items()}

        # Track the highest ID used so far (needed for generating new aux variables)
        self.top_id = len(self.edge_to_id)

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
        if self.ham_path is None:
            return
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
        if self.n is None:
            return ret
        sum = 2 * len(ret)
        i = 0
        keys = list(ret.keys())
        while sum > self.n and i < len(keys):
            ret[keys[i]] -= 1
            sum -= 1
            i = (i + 1) % len(keys)
        return ret

    def solve_t_3(self):
        ret = {e: 1 for e in self.G.edges()}
        if self.n is None:
            return ret
        sum = len(ret)
        non_ham_path = list(set(self.G.edges()).difference(self.ham_path))
        for e in self.ham_path + non_ham_path:
            if sum > self.n:
                if e in ret:
                    ret[e] -= 1
                    sum -= 1
            else:
                break
        return ret


    def _solve_wcnf(self, wcnf):
        self._add_end_constraint(wcnf)
        with RC2(wcnf) as rc2:
            model = rc2.compute()
            if model is None:
                return None

            markings = {}
            model_set = set(model)
            sum = 0
            for edge in self.G.edges():
                key = edge if edge in self.edge_to_id else (edge[1], edge[0])
                markings[edge] = int(self.edge_to_id[key] in model_set)
                sum += markings[edge]
            i = 0
            while sum > self.n:
                if self.ham_path[i] in markings and markings[self.ham_path[i]] > 0:
                    markings[self.ham_path[i]] -= 1
                    sum -= 1
                i += 1
            return markings

    def _find_short_cycles(self, limit):
        """
        Finds ALL simple cycles in G with length <= self.cycle_limit.
        Uses a DFS strategy where we only extend paths to nodes with ID > start_node
        to strictly enforce uniqueness (finding each cycle only once).
        """
        cycles = []

        # Iterate through every node as a potential 'start' of a cycle
        for start_node in self.G.nodes():
            # Stack stores: (current_node, path_list)
            stack = [(start_node, [start_node])]

            while stack:
                curr, path = stack.pop()

                # Check neighbors
                for neighbor in self.G[curr]:
                    # Case 1: Cycle closed (back to start)
                    if neighbor == start_node:
                        if len(path) > 2:
                            cycles.append(list(path))

                    # Case 2: Extend path
                    # We only visit neighbors > start_node to prevent duplicate cycle detection
                    # (e.g. finding 1-2-3-1 and 2-3-1-2 separately)
                    elif neighbor not in path:
                        if len(path) < limit and neighbor > start_node:
                            stack.append((neighbor, path + [neighbor]))

        return cycles

    def _wcnf_necessary(self, wcnf, T):
        """
        Constraint: For each cycle in the basis, the sum of marks on the cycle
        AND its outgoing edges must be <= 5.
        """

        for e in self.L.nodes():
            e_id = self.edge_to_id[e]
            wcnf.append([e_id], weight=1)

        basis = self._find_short_cycles(T)

        for cycle_nodes in basis:
            # Identify all edges involved (Cycle edges + Outgoing edges)
            relevant_edge_ids = set()

            # Use a set for cycle nodes for fast lookup
            cycle_node_set = set(cycle_nodes)

            # Iterate edges in the cycle (u -> v)
            for i in range(len(cycle_nodes)):
                u = cycle_nodes[i]
                v = cycle_nodes[(i + 1) % len(cycle_nodes)]

                # Add the cycle edge itself
                relevant_edge_ids.add(self._get_id(u, v))

                # Check neighbors of u to find outgoing edges
                for neighbor in self.G.neighbors(u):
                    if neighbor not in cycle_node_set:
                        # This is a spoke/outgoing edge
                        relevant_edge_ids.add(self._get_id(u, neighbor))

            # Create "At Most 5" constraint using CardEnc
            # enc returns a CNF formula object (clauses + new aux variables)
            cnf = CardEnc.atmost(lits=list(relevant_edge_ids), bound=len(cycle_nodes),
                                 top_id=self.top_id, encoding=EncType.seqcounter)

            wcnf.extend(cnf.clauses)

            # Update top_id so the next constraint doesn't reuse variables
            self.top_id = cnf.nv

        return wcnf

    def wcnf_t_4(self):
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

        return wcnf

    def wcnf_t_5(self):
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

        return wcnf

    def wcnf_t_6(self):
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

        return wcnf

    def find_solution(self, T):
        if T in (2, 3):
            return self.solve_t_2() if T == 2 else self.solve_t_3()
        t_to_function = {
            4: self.wcnf_t_4,
            5: self.wcnf_t_5,
            6: self.wcnf_t_6,
        }
        if T not in t_to_function:
            raise NotImplementedError
        return self._solve_wcnf(t_to_function[T]())

    def find_necessary_solution(self, T):
        t_to_function = {
            5: self.wcnf_t_5,
        }
        if T not in t_to_function:
            raise NotImplementedError
        return self._solve_wcnf(self._wcnf_necessary(WCNF(), T))


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



def cat_state_FT(n, t, max_iter_graph=100_000, max_new_graphs=100) -> stim.Circuit | None:
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
        for _ in range(max_new_graphs):
            G = construct_cyclic_connected_graph(N, T, max_iter=max_iter_graph)
            if G is None or has_small_nonlocal_cut(G, T):
                return None

            p = next(find_all_hamiltonian_paths(G))
            ham_path = list(zip(p, p[1:]))

            marker = GraphMarker(G, ham_path=ham_path, max_marks=n)
            marks = marker.find_solution(T)

            if sum(marks.values()) == n:
                # print(nx.algebraic_connectivity(G, tol=1e-3))
                # print(G.edges())
                break
        else:
            return None
    except:
        return None

    circ = extract_circuit(G, ham_path, marks)
    flag = circ.num_qubits - n
    if flag != minimum_number_of_flags(n,t):
        print()
        print(G.edges())
        print(ham_path)
        print(marks)
        visualize_cat_state_base(G, ham_path, marks)
    return circ


if __name__ == "__main__":
    import time
    start_time = time.time()

    N = 51
    T = 6

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
                if flag != minimum_number_of_flags(n, t):
                    flag = "?"
            print(flag if len(str(flag)) == 2 else f' {flag}', end=' ')
        print()

    circ = cat_state_FT(48, 4)
    print(circ)
    print(circ.num_qubits - 23)
    circ = cat_state_FT(24, 4)
    print(circ.num_qubits - 24)

    print("--- %s seconds ---" % (time.time() - start_time))
