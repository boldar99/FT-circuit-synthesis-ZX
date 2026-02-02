from abc import ABC, abstractmethod
from collections import defaultdict

import networkx as nx
import numpy as np
import pyzx as zx
import stim

from spidercat.utils import ed


class CircuitBuilder(ABC):
    @abstractmethod
    def add_h(self, qubit): pass

    @abstractmethod
    def add_cnot(self, control, target): pass

    @abstractmethod
    def init_ancilla(self, qubit):
        """Inits ancilla and applies H for your specific extraction logic."""
        pass

    @abstractmethod
    def post_select(self, qubit):
        """Applies H and post-selects (or measures) for your logic."""
        pass

    @abstractmethod
    def add_feedback_x(self, meas_idx, target_qubit):
        """
        Adds an X gate on target_qubit controlled by the measurement at absolute index meas_idx.
        Stim uses relative indexing (rec[-k]), so we calculate offset.
        """
        pass

    @abstractmethod
    def add_detector(self, m_idx1, m_idx2):
        """
        Adds a detector that fires if measurement[m_idx1] != measurement[m_idx2].
        (i.e., parity is 1).
        """
        pass

    @abstractmethod
    def get_circuit(self): pass


class PyZXBuilder(CircuitBuilder):
    def __init__(self):
        self.circ = zx.Circuit(0)

    def add_h(self, q): self.circ.add_gate("H", q)

    def add_cnot(self, c, t): self.circ.add_gate("CNOT", c, t)

    def init_ancilla(self, q):
        self.circ.add_gate("InitAncilla", q)
        self.add_h(q)

    def post_select(self, q):
        self.add_h(q)
        self.circ.add_gate("PostSelect", q)

    def get_circuit(self): return self.circ


class StimBuilder(CircuitBuilder):
    def __init__(self):
        self.circ = stim.Circuit()
        self.meas_count = 0

    def add_h(self, q):
        self.circ.append("H", [q])

    def add_cnot(self, c, t):
        self.circ.append("CNOT", [c, t])

    def init_ancilla(self, q):
        # self.circ.append("R", [q])
        pass

    def post_select(self, q):
        """Performs MR and returns the absolute index of this measurement."""
        self.circ.append("M", [q])
        idx = self.meas_count
        self.meas_count += 1
        return idx

    def add_feedback_x(self, meas_idx, target_qubit):
        offset = meas_idx - self.meas_count
        self.circ.append("CX", [stim.target_rec(offset), target_qubit])

    def add_detector(self, *meas_indices):
        """
        Adds a detector on the parity of the provided measurement indices.
        - If 1 index is provided: Checks that measurement == 0.
        - If 2 indices are provided: Checks that m1 == m2.
        """
        targets = []
        for m_idx in meas_indices:
            offset = m_idx - self.meas_count
            targets.append(stim.target_rec(offset))
        self.circ.append("DETECTOR", targets)

    def get_circuit(self):
        return self.circ


def extract_circuit(G, path_cover, marks, matching, builder: CircuitBuilder, verbose=False) -> stim.Circuit:
    if verbose:
        print("=== Extracting Circuit ===")

    # --- Setup Mappings ---
    node_to_path_idx = {node: p_idx for p_idx, path in enumerate(path_cover) for node in path}
    marks_map = {ed(v1, v2): int(v) for (v1, v2), v in
                 (marks.items() if isinstance(marks, dict) else [(e, 1) for e in marks])}

    flag_map: dict[tuple[int, int], int] = {}
    link_info: dict[tuple[int, int], dict[str, int]] = {}
    path_to_marks = defaultdict(list)
    path_qubits = {}

    # Count flags to reserve qubits
    cover_edges = {ed(u, v) for path in path_cover for u, v in zip(path, path[1:])}
    num_flags = len([e for e in G.edges() if ed(*e) not in cover_edges])
    next_cat = num_flags + len(path_cover)

    # 3. Initial Setup
    for qidx in range(next_cat):
        builder.init_ancilla(qidx)

    # --- Helper Logic ---
    def handle_link(path_qubit, link, current_p_id, decrement=False):
        nonlocal next_cat
        if verbose:
            print(f"    Handling link {link} (path_qubit={path_qubit}, path_id={current_p_id})")

        # 1. Create or retrieve the flag qubit
        if link not in link_info:
            # First visit
            fq = len(link_info)
            link_info[link] = {'q': fq, 'owner': current_p_id}

            if verbose:
                print(f"      New Flag: {fq} for link {link} (Owner: Path {current_p_id})")
                print(f"      CNOT {path_qubit} -> {fq}")
            builder.add_cnot(path_qubit, fq)
        else:
            # Second visit
            info = link_info[link]
            fq = info['q']
            owner_id = info['owner']

            if verbose:
                print(f"      Existing Flag: {fq} for link {link} (Belongs to Path {owner_id})")

            # Add Marks on Link
            count = marks_map.get(link, 0) - (1 if decrement else 0)
            for _ in range(count):
                if verbose:
                    print(f"      Init Ancilla {next_cat}")
                builder.init_ancilla(next_cat)

                path_to_marks[owner_id].append(next_cat)

                if verbose:
                    print(f"      CNOT {fq} -> {next_cat} (Mark assigned to Path {owner_id})")
                builder.add_cnot(fq, next_cat)
                next_cat += 1

            if verbose:
                print(f"      CNOT {path_qubit} -> {fq}")
            builder.add_cnot(path_qubit, fq)

            # Post-selection / Measurement
            m_idx = builder.post_select(fq)
            flag_map[link] = m_idx
            if verbose:
                print(f"      PostSelect {fq} -> Meas Idx {m_idx}")

    # 4. Main Loop
    if verbose:
        print("Starting Main Loop...")

    for p_id, path in enumerate(path_cover):
        if verbose:
            print(f"Path {p_id}: {path}")
        path_q = num_flags + p_id
        path_qubits[p_id] = path_q

        if verbose:
            print(f"  Unfusing path start {path_q} (H gate)")
        builder.add_h(path_q)  # Unfuse path start

        # Neighbors of v0
        v0, v1 = path[0], path[1]
        if verbose:
            print(f"  Neighbors of start {v0} (excluding {v1})...")
        for n in set(G.neighbors(v0)) - {v1}:
            decrement = (matching.get(v0) == n) or (matching.get(n) == v0)
            handle_link(path_q, ed(v0, n), p_id, decrement=decrement)

        # Path segments and internal nodes
        if verbose:
            print(f"  Internal segments...")
        for i, v_curr in enumerate(path[1:], 1):
            v_prev = path[i - 1]
            # Markings on the path itself
            marks_count = marks_map.get(ed(v_prev, v_curr), 0)
            if marks_count > 0 and verbose:
                print(f"    Processing {marks_count} marks on edge {(v_prev, v_curr)}")
            for _ in range(marks_count):
                if verbose:
                    print(f"      Init Ancilla {next_cat}")
                builder.init_ancilla(next_cat)
                path_to_marks[p_id].append(next_cat)
                if verbose:
                    print(f"      CNOT Path {path_q} -> {next_cat}")
                builder.add_cnot(path_q, next_cat)
                next_cat += 1

            if i + 1 < len(path):
                v_next = path[i + 1]
                # Internal non-cover neighbor
                for n in set(G.neighbors(v_curr)) - {v_prev, v_next}:
                    if verbose:
                        print(f"    Internal neighbor {v_curr}-{n}")
                    decrement = (matching.get(v_curr) == n) or (matching.get(n) == v_curr)
                    handle_link(path_q, ed(v_curr, n), p_id, decrement=decrement)

        # End of path logic
        if len(path) >= 2:
            if verbose:
                print(f"  End of path logic...")
            v_last, v_pen = path[-1], path[-2]
            ends = list(set(G.neighbors(v_last)) - {v_pen})
            if ends and matching.get(v_last) == ends[0]: ends.reverse()
            for end_v in ends:
                if verbose:
                    print(f"    End neighbor {v_last}-{end_v}")
                decrement = (matching.get(v_last) == end_v) or (matching.get(end_v) == v_last)
                handle_link(path_q, ed(v_last, end_v), p_id, decrement=decrement)

    # --- 5. DETECTORS ---
    if verbose: print("Adding Detectors...")
    consistency_groups: dict[tuple[int, int], list[int]] = defaultdict(list)

    # "Meta-graph" for finding cycles: Nodes are paths, Edges are connections
    meta_graph = nx.Graph()

    for (u, v), m_idx in flag_map.items():
        p1 = node_to_path_idx[u]
        p2 = node_to_path_idx[v]

        if p1 == p2:
            # Intra-path: Must be 0
            if verbose: print(f"  Intra-path detector on link {u}-{v} (meas {m_idx})")
            builder.add_detector(m_idx)
        else:
            key = ed(p1, p2)
            consistency_groups[key].append(m_idx)
            # Add to meta-graph for cycle detection later
            # We store the *first* measurement index as the representative for this edge
            if not meta_graph.has_edge(p1, p2):
                meta_graph.add_edge(p1, p2, representative_meas=m_idx)

    # A. Local Consistency (Parallel Edges)
    # If there are multiple measurements between p1 and p2, they must match.
    for pair, indices in consistency_groups.items():
        if len(indices) > 1:
            for k in range(len(indices) - 1):
                builder.add_detector(indices[k], indices[k + 1])

    # B. Global Consistency (Cycles)
    # Use NetworkX to find the cycle basis of the path connectivity graph.
    # For every cycle, the sum of representative measurements must be even (0).
    cycle_basis = nx.cycle_basis(meta_graph)
    if verbose and cycle_basis:
        print(f"  Found {len(cycle_basis)} cycles in path graph.")

    for cycle in cycle_basis:
        # cycle is a list of nodes [0, 1, 2] meaning 0-1-2-0
        detectors_indices = []

        # Walk through the cycle edges
        # Edge (0,1), (1,2), (2,0)
        cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1]))

        for p_u, p_v in cycle_edges:
            # Retrieve the representative measurement for this pair
            edge_data = meta_graph.get_edge_data(p_u, p_v)
            detectors_indices.append(edge_data['representative_meas'])

        if verbose:
            print(f"  Adding cycle detector for paths {cycle} using measurements {detectors_indices}")

        builder.add_detector(*detectors_indices)

    # --- 6. FEEDBACK/CORRECTION LOGIC ---
    if verbose: print("Generating Feedback...")
    path_graph = nx.Graph()
    path_graph.add_nodes_from(range(len(path_cover)))

    for (p1, p2), m_indices in consistency_groups.items():
        if not path_graph.has_edge(p1, p2):
            path_graph.add_edge(p1, p2, meas_idx=m_indices[0])

    try:
        bfs_tree = dict(nx.bfs_predecessors(path_graph, 0))
    except Exception:
        bfs_tree = {}

    for target_path_idx in range(1, len(path_cover)):
        if target_path_idx not in bfs_tree:
            continue

        # 1. Determine correction chain (XOR sum of measurements)
        current = target_path_idx
        correction_measurements = []
        while current != 0:
            parent = bfs_tree[current]
            edge_data = path_graph.get_edge_data(parent, current)
            correction_measurements.append(edge_data['meas_idx'])
            current = parent

        # 2. Identify all qubits that need this correction
        #    (The main path qubit + all mark ancillas attached to this path)
        qubits_to_correct = [path_qubits[target_path_idx]] + path_to_marks[target_path_idx]

        if verbose:
            print(f"Correcting Path {target_path_idx}")
            print(f"  Targets: PathQ {path_qubits[target_path_idx]} + Marks {path_to_marks[target_path_idx]}")
            print(f"  Controlled by measurements: {correction_measurements}")

        # 3. Apply Feedback
        for m_idx in correction_measurements:
            for q in qubits_to_correct:
                builder.add_feedback_x(m_idx, q)

    return builder.get_circuit()

# --- 2. The Main Extractor Class ---
class CatStateExtractor:
    def __init__(self, builder: CircuitBuilder, verbose=False):
        self.builder = builder
        self.verbose = verbose

        self.node_to_qubit = {}
        self.tree_to_qubits = defaultdict(list)
        self.tree_of_node = {}
        self.link_info = {}
        self.link_measurements = {}
        self.next_data_idx = 0
        self.next_flag_idx = 0
        self.markings = {}
        self.matches = defaultdict(list)

    def extract(self, G, forest, roots, markings, matches):
        if self.verbose: print("=== Starting Clean Extraction ===")


        # A. Normalize Inputs
        self._normalize_inputs(markings, matches)

        # B. Reserve Flag Indices (Estimate)
        total_marks = sum(markings.values())
        estimated_data_qubits = total_marks
        self.next_flag_idx = estimated_data_qubits

        if self.verbose:
            print(f"  Estimated Data Qubits: {estimated_data_qubits}")
            print(f"  Flags start at: {self.next_flag_idx}")

        # C. Main Traversal (Grow Trees & Fuse Links)
        for tree_id, root_node in roots.items():
            self._grow_tree_recursive(root_node, None, tree_id, G, forest)

        # D. Classical Logic
        self._generate_detectors()
        self._generate_feedback()

        return self.builder.get_circuit()

    def _normalize_inputs(self, markings, matches):
        # Flatten markings to sorted tuples
        for (u, v), count in markings.items():
            self.markings[ed(u, v)] = count

        # Flatten matches to sorted tuples
        for node, edge_list in matches.items():
            for u, v in edge_list:
                self.matches[node].append(ed(u, v))

    # --- Core Logic: Tree Growth ---
    def _grow_tree_recursive(self, node, parent_qubit, tree_id, G, forest):
        # 1. Allocate / Inherit Qubit
        if parent_qubit is None:
            # Root
            current_qubit = self._get_new_data_qubit()
            self.builder.init_ancilla(current_qubit)
            self.builder.add_h(current_qubit)
            if self.verbose: print(f"Init Root {node} (Tree {tree_id}) -> Q{current_qubit}")
        else:
            # Branch (Logic handled by caller)
            current_qubit = parent_qubit

            # Register
        self.node_to_qubit[node] = current_qubit
        self.tree_to_qubits[tree_id].append(current_qubit)
        self.tree_of_node[node] = tree_id

        # 2. Check Children (To determine if Leaf)
        children = sorted([n for n in forest.neighbors(node)
                           if n not in self.node_to_qubit])
        is_leaf = (len(children) == 0)

        # 3. RESOURCE ALLOCATION
        # Only LEAVES can absorb matches. Internal nodes must spawn new qubits.
        link_resource_map = self._allocate_link_resources(node, current_qubit, tree_id, is_leaf)

        # 4. FUSE LINKS (Sorted by Priority)
        self._process_node_links(node, G, forest, current_qubit, link_resource_map, tree_id)

        if not children: return

        # Primary Child (Inherits wire)
        primary = children[0]
        secondary_children = children[1:]

        # Handle Secondary Branches
        for child in secondary_children:
            new_q = self._get_new_data_qubit()
            self.builder.init_ancilla(new_q)
            self.builder.add_cnot(current_qubit, new_q)

            if self.verbose: print(f"  Node {node} -> Branch {child} (New Q{new_q})")

            # Internal Marks for Branch
            self._apply_internal_marks(node, child, new_q, tree_id)

            # Recurse Secondary
            self._grow_tree_recursive(child, new_q, tree_id, G, forest)

        # 5. Handle Primary Child (The wire flows into this node)
        if self.verbose: print(f"  Node {node} -> Primary {primary} (Inherits Q{current_qubit})")

        # Internal Marks for Primary
        self._apply_internal_marks(node, primary, current_qubit, tree_id)

        # Recurse Primary
        self._grow_tree_recursive(primary, current_qubit, tree_id, G, forest)

    def _allocate_link_resources(self, node, node_qubit, tree_id, is_leaf):
        """
        Assigns physical qubits to matches.
        - If is_leaf=True: 1st Match absorbs node_qubit. Rest are new.
        - If is_leaf=False: ALL matches spawn new qubits (Main wire must continue).
        """
        resource_map = {}
        matched_edges = self.matches.get(node, [])
        if not matched_edges: return resource_map

        # Only absorb if leaf. If internal, we force absorbed_first=True so logic skips to 'else'
        absorbed_first = not is_leaf

        for u, v in matched_edges:
            edge = tuple(sorted((u, v)))

            if not absorbed_first:
                # First match = Node Absorbed (Uses Main Qubit)
                resource_map[edge] = node_qubit
                absorbed_first = True
                if self.verbose:
                    print(f"    Match Allocation: Node {node} absorbs {edge}")
            else:
                # Subsequent matches need explicit new qubits
                new_res_q = self._get_new_data_qubit()
                self.tree_to_qubits[tree_id].append(new_res_q)

                self.builder.init_ancilla(new_res_q)
                self.builder.add_cnot(node_qubit, new_res_q)

                resource_map[edge] = new_res_q
                if self.verbose:
                    role = "Internal Node" if not is_leaf else "Secondary Match"
                    print(f"    Match Allocation: Node {node} ({role}) spawns Q{new_res_q} for {edge}")

        return resource_map

    def _process_node_links(self, node, G, forest, node_qubit, link_resource_map, tree_id):
        # Identify all links
        links = []
        for neighbor in G.neighbors(node):
            if not forest.has_edge(node, neighbor):
                links.append(tuple(sorted((node, neighbor))))

        # Sort Links:
        # Priority 0: Unmatched links OR Matched-but-new-leaf links
        # Priority 1: Matched links that use node_qubit (Absorbed)

        def get_priority(edge):
            # If edge is matched AND mapped to the main node_qubit, it is the Absorbed link (Last)
            if edge in link_resource_map and link_resource_map[edge] == node_qubit:
                return 1  # Last
            return 0

        sorted_links = sorted(links, key=get_priority)

        for edge in sorted_links:
            # We need to recover the 'neighbor' from the sorted edge tuple to call fuse_link properly
            # edge is (u, v) sorted. One is node.
            u, v = edge
            neighbor = v if u == node else u
            self._fuse_link(node, neighbor, link_resource_map, tree_id)

    def _fuse_link(self, u, v, link_resource_map, tree_u):
        edge = tuple(sorted((u, v)))
        u_qubit = link_resource_map.get(edge, self.node_to_qubit[u])

        if edge not in self.link_info:
            # First Visit: Create Flag & Wait
            flag_q = self._get_new_flag_qubit()
            self.builder.init_ancilla(flag_q)
            self.builder.add_cnot(u_qubit, flag_q)

            self.link_info[edge] = {'flag': flag_q, 'tree_u': tree_u}
            if self.verbose:
                print(f"    Link {edge} (1st visit): Created Flag {flag_q}, CNOT {u_qubit}->{flag_q}")
        else:
            # Second Visit: Complete Fusion
            info = self.link_info[edge]
            flag_q = info['flag']
            tree_v = info['tree_u']

            if self.verbose:
                print(f"    Link {edge} (2nd visit): Retrieved Flag {flag_q}, CNOT Q{u_qubit}->{flag_q}")

            # Calculate REMAINING marks needed on Flag
            # We subtract 1 for EACH endpoint that matched this link
            matched_u = edge in self.matches.get(u, [])
            matched_v = edge in self.matches.get(v, [])
            total_absorbed = (1 if matched_u else 0) + (1 if matched_v else 0)

            raw_marks = self.markings.get(edge, 0)
            needed_on_flag = max(0, raw_marks - total_absorbed)

            for _ in range(needed_on_flag):
                mark_q = self._get_new_data_qubit()
                self.tree_to_qubits[tree_u].append(mark_q)

                self.builder.init_ancilla(mark_q)
                self.builder.add_cnot(flag_q, mark_q)
                if self.verbose: print(f"      Added Unmatched Mark Q{mark_q} to Flag")

            # Complete Fusion
            self.builder.add_cnot(u_qubit, flag_q)
            m_idx = self.builder.post_select(flag_q)

            # Record
            self._record_meas(tree_u, tree_v, m_idx)

    def _apply_internal_marks(self, u, v, attachment_qubit, tree_id):
        edge = tuple(sorted((u, v)))
        raw_marks = self.markings.get(edge, 0)
        if raw_marks == 0: return

        # Internal Edge Optimization
        matched_u = edge in self.matches.get(u, [])
        matched_v = edge in self.matches.get(v, [])
        total_absorbed = (1 if matched_u else 0) + (1 if matched_v else 0)

        final_count = max(0, raw_marks - total_absorbed)

        if total_absorbed > 0 and self.verbose:
            print(f"    Internal Mark {edge}: {total_absorbed} absorbed by nodes.")

        for _ in range(final_count):
            mark_q = self._get_new_data_qubit()
            self.tree_to_qubits[tree_id].append(mark_q)
            self.builder.init_ancilla(mark_q)
            self.builder.add_cnot(attachment_qubit, mark_q)
            if self.verbose: print(f"    Internal Mark {edge}: Added Q{mark_q}")

    # --- HELPERS ---
    def _record_meas(self, t1, t2, m_idx):
        if t1 == t2: self.builder.add_detector(m_idx)
        else:
            k = tuple(sorted((t1, t2)))
            self.link_measurements.setdefault(k, []).append(m_idx)

    def _get_new_data_qubit(self):
        q = self.next_data_idx; self.next_data_idx += 1; return q

    def _get_new_flag_qubit(self):
        q = self.next_flag_idx; self.next_flag_idx += 1; return q

    def _generate_detectors(self):
        if self.verbose: print("Generating Detectors...")
        for indices in self.link_measurements.values():
            for i in range(len(indices)-1): self.builder.add_detector(indices[i], indices[i+1])
        meta = nx.Graph()
        for (t1,t2), idxs in self.link_measurements.items(): meta.add_edge(t1, t2, m=idxs[0])
        for cyc in nx.cycle_basis(meta):
            self.builder.add_detector(*[meta[u][v]['m'] for u,v in zip(cyc, cyc[1:]+cyc[:1])])
        self.meta_graph = meta
    def _generate_feedback(self):
        if not self.link_measurements: return
        root = min(list(self.meta_graph.nodes()))
        preds = dict(nx.bfs_predecessors(self.meta_graph, root))
        for t in self.meta_graph.nodes():
            if t == root or t not in preds: continue
            path_ms = []
            cur = t
            while cur != root:
                path_ms.append(self.meta_graph[preds[cur]][cur]['m'])
                cur = preds[cur]
            for m in path_ms:
                for q in self.tree_to_qubits[t]: self.builder.add_feedback_x(m, q)


def extract_circuit_rooted(G, forest, roots, markings, matches, verbose=False) -> stim.Circuit:
    extractor = CatStateExtractor(StimBuilder(), verbose)
    return extractor.extract(G, forest, roots, markings, matches)


def implement_CNOT_circuit(cnots, num_qubits, p_2, p_mem):
    circ = stim.Circuit()
    all_qubits = set(range(num_qubits + 1))
    free_qubits = all_qubits.copy()
    for c, n in cnots:
        if c in free_qubits and n in free_qubits:
            free_qubits -= {c, n}
        else:
            if p_mem > 0:
                circ.append("Z_ERROR", free_qubits, p_mem)
                circ.append("TICK")
                free_qubits = all_qubits.copy() - {c, n}
        circ.append("CNOT", [c, n])
        if p_2 > 0:
            circ.append("DEPOLARIZE2", [c, n], p_2)
    if p_mem > 0:
        circ.append("Z_ERROR", free_qubits, p_mem)
    return circ


def make_stim_circ_noisy(circ: stim.Circuit, p_1=0., p_2=0., p_mem=0., p_meas=0., p_init=0.) -> stim.Circuit:
    noisy_circ = stim.Circuit()
    num_qubits = circ.num_qubits

    if p_init > 0:
        noisy_circ.append("DEPOLARIZE1", range(num_qubits), p_init)

    for instruction in circ:
        gate_name = instruction.name
        targets = instruction.targets_copy()

        if gate_name in ("CNOT", "CX", "CZ", "SWAP"):
            split_targets = [
                (targets[i], targets[i+1])
                for i in range(0, len(targets), 2)
            ]
            noisy_circ += implement_CNOT_circuit(split_targets, num_qubits, p_2, p_mem)

        elif gate_name in ("H", "X", "Y", "Z", "I"):
            noisy_circ.append(gate_name, targets)
            if p_1 > 0:
                noisy_circ.append("DEPOLARIZE1", targets, p_1)

        elif gate_name in ("M", "MZ", "MR", "R", "RX", "RY"):
            if gate_name in ("M", "MZ", "MR") and p_meas > 0:
                noisy_circ.append("DEPOLARIZE1", targets, p_meas)

            noisy_circ.append(gate_name, targets)

            if gate_name in ("R", "RX", "RY", "MR") and p_init > 0:
                noisy_circ.append("DEPOLARIZE1", targets, p_init)

        else:
            noisy_circ.append(gate_name, targets)

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
    circ.append("M", 0)
    circ.append("DETECTOR", stim.target_rec(-1))
    return circ


def cat_state_6():
    return stim.Circuit("""
        H 2
        CNOT 2 3 2 1 2 4 2 0 2 5 2 6 2 1 2 7 2 0
        M 0 1 
    """)


if __name__ == "__main__":
    circ = stim.Circuit("""
    H 0
    CX 0 6 0 1 1 2 1 7 1 3 1 8 1 6
    M 6
    DETECTOR rec[-1]
    CX 0 4 0 5 5 8
    M 8
    DETECTOR rec[-1]
    CX 0 7
    M 7
    DETECTOR rec[-1]
        """)
    noisy_circ = make_stim_circ_noisy(circ, p_2=0.1)

    print(noisy_circ)
