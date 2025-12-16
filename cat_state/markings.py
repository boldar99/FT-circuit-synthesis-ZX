import itertools

import networkx as nx
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from pysat.card import CardEnc, EncType


def verify_marking_property(G, markings, T):
    """
    Verifies that for every cut of size <= T, the markings satisfy the condition:
    Even if we distribute the cut-marks to maximize the smaller side (balance the sides),
    that smaller side is still <= the cut size.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    # 1. Calculate Total Marks in the entire graph
    total_marks = sum(markings.values())

    def get_mark(u, v):
        # Checks (u,v) or (v,u)
        return markings.get((u, v)) or markings.get((v, u), 0)

    # 2. Iterate all valid subsets (Cuts)
    for k in range(1, n // 2 + 1):
        for S in itertools.combinations(nodes, k):
            S_set = set(S)

            cut_size = 0
            marks_S_doubled = 0  # Counts internal edges twice
            marks_on_cut = 0

            possible_small_cut = True

            # Calculate Cut Size and Internal Marks for S
            for u in S:
                for v in G[u]:
                    if v in S_set:
                        # Internal edge (we will encounter this again from v's side)
                        marks_S_doubled += get_mark(u, v)
                    else:
                        # Boundary edge
                        cut_size += 1
                        marks_on_cut += get_mark(u, v)

                    # Optimization: Stop if cut exceeds T
                    if cut_size > T:
                        possible_small_cut = False
                        break
                if not possible_small_cut:
                    break

            if possible_small_cut:
                # Derived Marks
                M_A = marks_S_doubled // 2
                M_B = total_marks - M_A - marks_on_cut
                M_cut = marks_on_cut

                # Option 1: Dump all cut marks on Side A
                max_A = M_A + M_cut
                # Option 2: Dump all cut marks on Side B
                max_B = M_B + M_cut

                check_value = min(max_A, max_B)

                # Verification
                if check_value > cut_size:
                    return False

    return True



def generate_markings(G, N):
    """
    Yields all valid markings of graph G with exactly N total marks,
    where each edge can have 0, 1, or 2 marks.
    """
    edges = list(G.edges())
    num_edges = len(edges)

    # k = number of edges with 2 marks
    # m = number of edges with 1 mark
    # Constraint: 2*k + m = N
    for k in range(N // 2):
        print(k, end=' ')
        m = N - 2 * k

        # We cannot mark more edges than exist in the graph
        if k + m > num_edges:
            continue

        # 1. Choose which edges get 2 marks
        for edges_2 in itertools.combinations(edges, k):
            s2 = set(edges_2)
            remaining_edges = [e for e in edges if e not in s2]

            # 2. Choose which of the remaining edges get 1 mark
            for edges_1 in itertools.combinations(remaining_edges, m):
                s1 = set(edges_1)

                # 3. Construct the dictionary
                # Edges in s2 -> 2, Edges in s1 -> 1, Others -> 0
                yield {e: (2 if e in s2 else (1 if e in s1 else 0)) for e in edges}
    print()


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
        if not self.ham_path:
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
            non_ham_path = list(set(self.G.edges()).difference(self.ham_path))
            edge_reduce_ordered = self.ham_path + non_ham_path
            while sum > self.n:
                if edge_reduce_ordered[i] in markings and markings[edge_reduce_ordered[i]] > 0:
                    markings[edge_reduce_ordered[i]] -= 1
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
            for combo in itertools.combinations(neighbor_ids, 3):
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
