import itertools

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from pysat.card import CardEnc, EncType
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

from spidercat.utils import ed


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
    Yields all markings of graph G with exactly N total marks,
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
    def __init__(self, G, path_cover=None, max_marks=None):
        self.G = G
        self.path_cover = path_cover
        self.n = max_marks
        self.L = nx.line_graph(G)

        # Create a mapping from edge (node in L) to SAT variable ID (1, 2, 3...)
        self.edge_to_id = {edge: i + 1 for i, edge in enumerate(self.L.nodes())}
        self.id_to_edge = {i: edge for edge, i in self.edge_to_id.items()}

        # Track the highest ID used so far (needed for generating new aux variables)
        self.top_id = len(self.edge_to_id)

        # Precompute cover and non-cover edges for quick reuse
        if self.path_cover is not None:
            cover_edge_sorted_set = set()
            for path in self.path_cover:
                for u, v in zip(path, path[1:]):
                    cover_edge_sorted_set.add(ed(u, v))

            # Preserve ordering consistent with G.edges()
            self.cover_edges = [e for e in self.G.edges() if tuple(sorted(e)) in cover_edge_sorted_set]
            self.non_cover_edges = [e for e in self.G.edges() if tuple(sorted(e)) not in cover_edge_sorted_set]
        else:
            self.cover_edges = []
            self.non_cover_edges = list(self.G.edges())

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

        end_nodes = [] if self.path_cover is None else [path[-1] for path in self.path_cover]

        # For each end node, there must be a marked edge incident to it
        for end_node in end_nodes:
            candidate_ids = []
            for neighbor in self.G.neighbors(end_node):
                if (neighbor, end_node) in self.cover_edges or (end_node, neighbor) in self.cover_edges:
                    continue
                candidate_ids.append(self._get_id(end_node, neighbor))
            wcnf.append(candidate_ids)

        # if an edge is adjacent to two end nodes then at least 2 out of 3 of
        # those non-path edges must be marked
        # ===o--|--o===
        #    |     |
        #    -     -
        #    |     |
        for u, v in self.G.edges():
            if u in end_nodes and v in end_nodes:
                edge_ids = set()
                for neigh in self.G.neighbors(u):
                    if (neigh, u) not in self.cover_edges and (u, neigh) not in self.cover_edges:
                        edge_ids.add(self._get_id(u, neigh))
                for neigh in self.G.neighbors(v):
                    if (neigh, v) not in self.cover_edges and (v, neigh) not in self.cover_edges:
                        edge_ids.add(self._get_id(v, neigh))
                assert len(edge_ids) == 3, edge_ids
                edge_ids = list(edge_ids)
                wcnf.append([edge_ids[0], edge_ids[1]])
                wcnf.append([edge_ids[1], edge_ids[2]])
                wcnf.append([edge_ids[0], edge_ids[2]])

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
        # order matters here.
        # Q: what happens if max_marks < number of paths?
        # A: need ancilla.
        for e in self.cover_edges + self.non_cover_edges:
            if sum > self.n:
                if e in ret:
                    ret[e] -= 1
                    sum -= 1
            else:
                break
        return ret

    def _solve_wcnf(self, wcnf):
        self._add_end_constraint(wcnf)

        # Enforce maximum marks as a Hard Constraint
        if self.n is not None:
            all_edge_ids = list(self.edge_to_id.values())
            # AtMost self.n
            cnf = CardEnc.atmost(lits=all_edge_ids, bound=self.n,
                                 top_id=self.top_id, encoding=EncType.seqcounter)

            wcnf.extend(cnf.clauses)
            self.top_id = cnf.nv

        with RC2(wcnf) as rc2:
            model = rc2.compute()
            if model is None:
                return None

            markings = {}
            model_set = set(model)
            sum_ = 0
            for edge in self.G.edges():
                key = edge if edge in self.edge_to_id else (edge[1], edge[0])
                markings[edge] = int(self.edge_to_id[key] in model_set)
                sum_ += markings[edge]

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
            # Prioritize Non-Cover edges (weight 2) over Cover edges (weight 1)
            weight = 1
            if self.path_cover is not None:
                # Check if checks are correct. self.cover_edges are edges in G.
                # e is a node in L, which is an edge in G.
                # self.L.nodes() elements are tuples (u,v).
                # self.cover_edges elements are also tuples.
                # Need to handle ordering (u,v) vs (v,u).
                # Simpler: check if e in self.cover_edges_set (I should make one?)
                # Or just use the O(N) lookup since list is small?
                # Actually self.cover_edges is a list.
                # Let's perform a safer check.
                u, v = e
                if (u, v) in self.cover_edges or (v, u) in self.cover_edges:
                    weight = 1
                else:
                    weight = 2

            wcnf.append([e_id], weight=weight)

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
            weight = 1
            if self.path_cover is not None:
                u, v = e
                if (u, v) in self.cover_edges or (v, u) in self.cover_edges:
                    weight = 1
                else:
                    weight = 2
            wcnf.append([self.edge_to_id[e]], weight=weight)

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
            weight = 1
            if self.path_cover is not None:
                u, v = e
                if (u, v) in self.cover_edges or (v, u) in self.cover_edges:
                    weight = 1
                else:
                    weight = 2
            wcnf.append([self.edge_to_id[e]], weight=weight)

        return wcnf

    def _find_subtrees_of_size(self, size):
        """
        Generates all connected subgraphs (sets of nodes) of a specific size.
        Returns a list of node-sets (tuples).
        """
        if size < 1:
            return []
        # Start with all single nodes
        subgraphs = set((n,) for n in self.G.nodes())

        # If requested size is 1, return the singletons that are trivially trees
        if size == 1:
            return list(subgraphs)

        # Iteratively expand, but only keep node-sets whose induced subgraph is a tree
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
                    new_nodes = tuple(sorted(list(sg) + [neighbor]))
                    # Only keep the set if the induced subgraph is a tree
                    subG = self.G.subgraph(new_nodes)
                    if nx.is_tree(subG):
                        new_subgraphs.add(new_nodes)

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
                nx.draw(subG, with_labels=True)
                plt.show()
                nx.draw(self.G, with_labels=True)
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
        for target_size in range(1, T - 1):
            wcnf = self._add_wcnf_subtree_condition(wcnf, target_size)

        return self._solve_wcnf(wcnf)

    def find_solution(self, T):
        t_to_solver_function = {
            4: self._add_wcnf_t_4,
            5: self._add_wcnf_t_5,
            7: self._add_wcnf_t_7,
        }
        t_to_solution_function = {
            2: self.solve_t_2,
            3: self.solve_t_3,
            6: self.solve_t_6,
            8: self.solve_t_8,
        }
        if T in t_to_solver_function:
            wcnf = WCNF()
            return self._solve_wcnf(t_to_solver_function[T](wcnf))
        if T in t_to_solution_function:
            return t_to_solution_function[T]()
        else:
            raise NotImplementedError

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
        wcnf = WCNF()

        # 1. Soft Constraint: Maximize Marks
        for e in self.L.nodes():
            wcnf.append([self.edge_to_id[e]], weight=1)

        # 2. Hard Constraint: Subtree Limits
        for target_size in range(1, T - 1):
            wcnf = self._add_wcnf_subtree_condition(wcnf, target_size)

        return self._solve_wcnf(wcnf)


if __name__ == '__main__':
    G = nx.from_edgelist(
        [(0, 17), (0, 1), (0, 12), (1, 2), (1, 6), (2, 3), (2, 14), (3, 4), (3, 9), (4, 5), (4, 16), (5, 6), (5, 11),
         (6, 7), (7, 8), (7, 13), (8, 9), (8, 17), (9, 10), (10, 11), (10, 15), (11, 12), (12, 13), (13, 14), (14, 15),
         (15, 16), (16, 17)]
    )
    ham_path = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
                (12, 13), (13, 14), (14, 15), (15, 16), (16, 17)]
    marker = GraphMarker(G, ham_path, 16)
    pos = {0: np.array([1.00000000e+00, 2.80364046e-08]), 1: np.array([0.93969262, 0.34202015]),
           2: np.array([0.76604444, 0.64278759]), 3: np.array([0.49999998, 0.86602546]),
           4: np.array([0.17364824, 0.98480774]), 5: np.array([-0.17364818, 0.98480774]),
           6: np.array([-0.50000004, 0.8660254]), 7: np.array([-0.76604441, 0.64278765]),
           8: np.array([-0.93969259, 0.34202024]), 9: np.array([-9.99999970e-01, -5.93863707e-08]),
           10: np.array([-0.93969259, -0.34202012]), 11: np.array([-0.76604447, -0.64278754]),
           12: np.array([-0.49999989, -0.86602541]), 13: np.array([-0.17364812, -0.98480775]),
           14: np.array([0.17364818, -0.98480769]), 15: np.array([0.49999992, -0.86602541]),
           16: np.array([0.76604432, -0.64278772]), 17: np.array([0.93969256, -0.34202033])}
    marks = marker.solve_t_6()
    print(marks)
    if marker is not None:
        from draw import visualize_cat_state_base

    visualize_cat_state_base(G, ham_path, marks, pos=pos)
    print(find_marking_property_violation(G, marks, 6))
    print(sum(marks.values()))
