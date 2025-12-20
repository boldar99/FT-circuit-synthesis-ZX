import itertools

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from pysat.card import CardEnc, EncType


def find_marking_property_violation(G: nx.Graph, markings: dict[tuple[int, int], int], T: int) -> set[int] | None:
    """
    Verifies that for every cut of size <= T, the markings satisfy the condition:
    Even if we distribute the cut-marks to maximize the smaller side (balance the sides),
    that smaller side is still <= the cut size.
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()
    nodes = list(G.nodes())

    # 1. Calculate Total Marks in the entire graph
    total_marks = sum(markings.values())

    # 1.2 Store markings in fast access array.
    marks_adj = np.zeros((e, e), dtype=np.int16)
    for (v, w), m in markings.items():
        marks_adj[v, w] = m
        marks_adj[w, v] = m

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
                        marks_S_doubled += marks_adj[u, v]
                    else:
                        # Boundary edge
                        cut_size += 1
                        marks_on_cut += marks_adj[u, v]

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
                    return S_set

    return None


def verify_marking_property(G: nx.Graph, markings: dict[tuple[int, int], int], T: int) -> bool:
    return find_marking_property_violation(G, markings, T) is None


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


def merge_wcnf(target_wcnf, source_wcnf):
    """
    Appends all hard and soft clauses from source_wcnf to target_wcnf.
    """
    # 1. Copy Hard Clauses
    for clause in source_wcnf.hard:
        target_wcnf.append(clause)

    # 2. Copy Soft Clauses with their weights
    # source_wcnf.soft is a list of clauses
    # source_wcnf.wght is a list of corresponding weights
    for clause, weight in zip(source_wcnf.soft, source_wcnf.wght):
        target_wcnf.append(clause, weight=weight)


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
            if self.ham_path is not None:
                non_ham_path = list(set(self.G.edges()).difference(self.ham_path))
                edge_reduce_ordered = self.ham_path + non_ham_path
                while sum > self.n:
                    if edge_reduce_ordered[i] in markings and markings[edge_reduce_ordered[i]] > 0:
                        markings[edge_reduce_ordered[i]] -= 1
                        sum -= 1
                    i += 1
            return markings

    def _add_wcnf_t_4(self, wcnf):
        """
        Constraint: Marked edges must have an Unmarked neighbor.
        """
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

    def _add_wcnf_t_5(self, wcnf):
        """
        Constraint: EVERY node must have an Unmarked neighbor.
        """
        for v0 in self.G.nodes():
            neighs = self.G.neighbors(v0)
            edges = [self.edge_to_id.get((v0, n), self.edge_to_id.get((n, v0))) for n in neighs]
            wcnf.append([-nid for nid in edges])

        for e in self.L.nodes():
            wcnf.append([self.edge_to_id[e]], weight=1)

        return wcnf

    def _add_wcnf_t_7(self, wcnf):
        """
        Constraint: EVERY edge must have an Unmarked neighbor.
        (This forces Unmarked edges to form a Total Dominating Set).
        """
        for e in self.L.nodes():
            neighbor_ids = [self.edge_to_id[n] for n in self.L.neighbors(e)]

            # Clause: [-n1, -n2, -n3, -n4]
            # Meaning: At least one neighbor must be False (Unmarked).
            wcnf.append([-nid for nid in neighbor_ids])

        # Soft Constraint: Maximize Marked edges
        for e in self.L.nodes():
            wcnf.append([self.edge_to_id[e]], weight=1)

        return wcnf

    def _find_subtrees_of_size(self, size):
        """
        Generates all connected subgraphs (sets of nodes) of a specific size.
        Returns a list of node-sets (tuples).
        """
        if size < 1:
            return []

        # Start with all single nodes
        subgraphs = set(tuple([n]) for n in self.G.nodes())

        # Iteratively expand
        for _ in range(size - 1):
            new_subgraphs = set()
            for sg in subgraphs:
                sg_set = set(sg)
                # Find all neighbors of the current subgraph
                neighbors = set()
                for node in sg:
                    for neighbor in self.G.neighbors(node):
                        if neighbor not in sg_set:
                            neighbors.add(neighbor)

                # Create new larger subgraphs by adding one neighbor
                for neighbor in neighbors:
                    new_sg = tuple(sorted(list(sg) + [neighbor]))
                    new_subgraphs.add(new_sg)
            subgraphs = new_subgraphs

        return list(subgraphs)

    def _add_wcnf_subtree_condition(self, wcnf, T):
        """
        Constraint: For every subtree of size T-2:
        The sum of marks on (Internal Edges + Outgoing Edges) must be <= T.
        """

        # Generate all node sets of size T-2
        candidate_node_sets = self._find_subtrees_of_size(T - 2)

        for nodes in candidate_node_sets:
            # Verify it is a tree (as per specification)
            subG = self.G.subgraph(nodes)
            if not nx.is_tree(subG):
                nx.draw(subG)
                plt.show()
                nx.draw(self.G)
                plt.show()
                raise AssertionError

            relevant_edge_ids = set()

            # Identify Edges
            for u in nodes:
                for v in self.G.neighbors(u):
                    relevant_edge_ids.add(self._get_id(u, v))

            # Create "At Most T" constraint
            cnf = CardEnc.atmost(lits=list(relevant_edge_ids), bound=T,
                                 top_id=self.top_id, encoding=EncType.seqcounter)

            wcnf.extend(cnf.clauses)
            self.top_id = cnf.nv

        return wcnf

    def general_solver(self, T):
        wcnf = WCNF()

        # 1. Soft Constraint: Maximize Marks
        for e in self.L.nodes():
            wcnf.append([self.edge_to_id[e]], weight=1)

        # 2. Hard Constraint: Subtree Limits
        for target_size in range(1, T-1):
            wcnf = self._add_wcnf_subtree_condition(wcnf, target_size)

        return self._solve_wcnf(wcnf)

    def find_solution(self, T):
        if T in (2, 3):
            return self.solve_t_2() if T == 2 else self.solve_t_3()
        if T == 6:
            return self.solve_t_6()
        t_to_function = {
            4: self._add_wcnf_t_4,
            5: self._add_wcnf_t_5,
            7: self._add_wcnf_t_7,
        }
        if T not in t_to_function:
            raise NotImplementedError
        wcnf = WCNF()
        return self._solve_wcnf(t_to_function[T](wcnf))

    def solve_t_6(self):
        wcnf = WCNF()
        wcnf = self._add_wcnf_t_5(wcnf)
        wcnf = self._add_wcnf_subtree_condition(wcnf, 6)
        return self._solve_wcnf(wcnf)

    def solve_t_8(self):
        wcnf = WCNF()
        wcnf = self._add_wcnf_t_7(wcnf)
        wcnf = self._add_wcnf_subtree_condition(wcnf, 8)
        return self._solve_wcnf(wcnf)


    def find_general_solution(self, T):
        # TODO: Could be done so that if actually does the necessary solution.
        wcnf = WCNF()

        # 1. Soft Constraint: Maximize Marks
        for e in self.L.nodes():
            wcnf.append([self.edge_to_id[e]], weight=1)

        # 2. Hard Constraint: Subtree Limits
        for target_size in range(1, T-1):
            wcnf = self._add_wcnf_subtree_condition(wcnf, target_size)

        return self._solve_wcnf(wcnf)


if __name__ == '__main__':
    G = nx.from_edgelist(
        [(0, 25), (0, 1), (0, 11), (1, 2), (1, 20), (2, 3), (2, 15), (3, 4), (3, 23), (4, 5), (4, 10), (5, 6), (5, 19), (6, 7), (6, 14), (7, 8), (7, 25), (8, 9), (8, 21), (9, 10), (9, 16), (10, 11), (11, 12), (12, 13), (12, 18), (13, 14), (13, 22), (14, 15), (15, 16), (16, 17), (17, 18), (17, 24), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25)]
    )
    ham_path = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25)]
    marker = GraphMarker(G, ham_path, 23)
    marks = {(0, 25): 0, (0, 1): 0, (0, 11): 0, (1, 2): 0, (1, 20): 1, (2, 3): 0, (2, 15): 1, (3, 4): 1, (3, 23): 1, (4, 5): 1, (4, 10): 0, (5, 6): 1, (5, 19): 0, (6, 7): 1, (6, 14): 0, (7, 8): 0, (7, 25): 1, (8, 9): 1, (8, 21): 1, (9, 10): 1, (9, 16): 0, (10, 11): 1, (11, 12): 1, (12, 13): 0, (12, 18): 1, (13, 14): 1, (13, 22): 1, (14, 15): 0, (15, 16): 1, (16, 17): 0, (17, 18): 1, (17, 24): 1, (18, 19): 0, (19, 20): 1, (20, 21): 0, (21, 22): 0, (22, 23): 1, (23, 24): 0, (24, 25): 1}
    print(sum(marks.values()))
    marks = marker.solve_t_6()
    print(sum(marks.values()))
    print(marks)
    if marker is not None:
        from cat_state_generation import visualize_cat_state_base
        pass
    visualize_cat_state_base(G, ham_path, marks)
    print(find_marking_property_violation(G, marks, 5))
    print(sum(marks.values()))