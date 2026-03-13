import stim
import networkx as nx
from matplotlib import pyplot as plt

import stim
import networkx as nx

import stim
import networkx as nx


def get_cnot_dag(stim_str: str, num_legs: int) -> nx.DiGraph:
    """
    Returns a minimal, weighted DAG of only the CNOTs acting on data qubits.
    The 'weight' attribute on each edge represents the minimum number of
    circuit layers required between the two CNOTs.
    """
    circuit = stim.Circuit(stim_str)
    full_dag = nx.DiGraph()

    last_seen_on_qubit = {}
    cnot_id = 0
    data_nodes = []

    # 1. Build the Full DAG (all edges have an implicit weight/time-cost of 1)
    for instruction in circuit:
        if instruction.name == "CX":
            targets = [t.value for t in instruction.targets_copy()]
            for i in range(0, len(targets), 2):
                control = targets[i]
                target = targets[i + 1]

                full_dag.add_node(cnot_id, control=control, target=target)

                if control in last_seen_on_qubit:
                    full_dag.add_edge(last_seen_on_qubit[control], cnot_id)
                if target in last_seen_on_qubit:
                    full_dag.add_edge(last_seen_on_qubit[target], cnot_id)

                last_seen_on_qubit[control] = cnot_id
                last_seen_on_qubit[target] = cnot_id

                # Tag Data CNOTs
                if control < num_legs and target < num_legs:
                    data_nodes.append(cnot_id)

                cnot_id += 1

    # 2. Compute Longest Paths (to find exact delay between any two nodes)
    # Using dynamic programming over the topological order
    dist = {u: {v: -1 for v in full_dag.nodes()} for u in full_dag.nodes()}
    for u in full_dag.nodes():
        dist[u][u] = 0

    topo_order = list(nx.topological_sort(full_dag))
    for i, u in enumerate(topo_order):
        for v in topo_order[i:]:
            if dist[u][v] != -1:  # If v is reachable from u
                for neighbor in full_dag.successors(v):
                    # Longest path update
                    if dist[u][v] + 1 > dist[u][neighbor]:
                        dist[u][neighbor] = dist[u][v] + 1

    # 3. Build a fully connected DAG of JUST the Data CNOTs
    data_dag = nx.DiGraph()
    for n in data_nodes:
        data_dag.add_node(n, **full_dag.nodes[n])

    for u in data_nodes:
        for v in data_nodes:
            if u != v and dist[u][v] > 0:
                # Add edge with the longest path as the minimum delay weight
                data_dag.add_edge(u, v, weight=dist[u][v])

    # 4. Custom Weighted Transitive Reduction
    edges_to_remove = []
    for u, v, edge_data in data_dag.edges(data=True):
        direct_delay = edge_data['weight']

        # Check if an intermediate data node (w) fully covers this delay
        for w in data_nodes:
            if w != u and w != v:
                if data_dag.has_edge(u, w) and data_dag.has_edge(w, v):
                    path_delay = data_dag[u][w]['weight'] + data_dag[w][v]['weight']

                    # If the chained delay enforces an equal or greater time gap,
                    # the direct edge is redundant and safe to remove.
                    if path_delay >= direct_delay:
                        edges_to_remove.append((u, v))
                        break  # Found a valid intermediary, stop checking others

    data_dag.remove_edges_from(edges_to_remove)

    return data_dag


def draw_cnot_dag(dag: nx.DiGraph):
    """
    Draws the weighted CNOT DAG.
    Nodes are spaced horizontally based on their critical path timing,
    making the delays introduced by hidden flag qubits visually apparent.
    """
    if not nx.is_directed_acyclic_graph(dag):
        print("Warning: Graph is not a DAG. Cannot draw properly.")
        return

    # 1. Calculate the earliest start times to use as the horizontal 'layer'
    earliest_start = {node: 1 for node in dag.nodes()}
    for node in nx.topological_sort(dag):
        for successor in dag.successors(node):
            weight = dag[node][successor].get('weight', 1)
            if earliest_start[node] + weight > earliest_start[successor]:
                earliest_start[successor] = earliest_start[node] + weight

    # Attach these as node attributes for the layout engine
    for node, layer in earliest_start.items():
        dag.nodes[node]['layer'] = layer

    # 2. Generate node labels
    labels = {}
    for node, data in dag.nodes(data=True):
        control = data.get('control', '?')
        target = data.get('target', '?')
        labels[node] = f"{control}→{target}"

    # 3. Create layout and plot
    pos = nx.multipartite_layout(dag, subset_key="layer", align="horizontal")

    plt.figure(figsize=(14, 8))

    # Draw nodes and edges
    nx.draw(
        dag,
        pos,
        labels=labels,
        with_labels=True,
        node_size=1000,
        node_color="#98FB98",  # Pale green
        font_size=15,
        font_weight="bold",
        edge_color="gray",
        arrows=True,
        arrowsize=20
    )

    # 4. Draw edge weights in red
    edge_labels = {(u, v): f"{d.get('weight', 1)}" for u, v, d in dag.edges(data=True)}
    nx.draw_networkx_edge_labels(
        dag, pos, edge_labels=edge_labels, font_color='orange', font_weight='bold', font_size=15,
    )

    total_depth = max(earliest_start.values()) if earliest_start else 0
    plt.title(f"Weighted CNOT DAG (Critical Path Depth: {total_depth})", fontsize=16)
    plt.tight_layout()
    plt.show()


def get_weighted_dag_depth(dag: nx.DiGraph) -> int:
    """
    Calculates the depth (longest weighted path) of the CNOT DAG.
    Assumes each CNOT takes 1 layer to execute, and edge weights represent
    the total layer delay between the start of one CNOT and the start of the next.
    """
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Graph is not a DAG. Cannot compute depth.")
    if not dag.nodes():
        return 0

    # Initialize the earliest start layer for each node
    # Nodes with no dependencies can start at layer 1
    earliest_start = {node: 1 for node in dag.nodes()}

    # Traverse in topological order to propagate the delays
    for node in nx.topological_sort(dag):
        for successor in dag.successors(node):
            weight = dag[node][successor].get('weight', 1)
            # The successor must start at least 'weight' layers after the current node
            if earliest_start[node] + weight > earliest_start[successor]:
                earliest_start[successor] = earliest_start[node] + weight

    # The total depth is the maximum layer assigned to any node
    return max(earliest_start.values())


if __name__ == "__main__":
    import networkx as nx
    import stim

    from cat_to_dag import *

    # --- Example Usage ---
    stim_circuit_str = """
    H 0
    CX 0 1 1 2 1 10 1 11 0 3 3 4 3 12 3 5 3 11
    M 11
    DETECTOR rec[-1]
    CX 3 6 3 7 7 13 3 14 0 8 0 9 9 13
    M 13
    DETECTOR rec[-1]
    CX 9 12
    M 12
    DETECTOR rec[-1]
    CX 0 14
    M 14
    DETECTOR rec[-1]
    CX 0 10
    M 10
    DETECTOR rec[-1]
    """

    # Number of legs as defined in your prompt
    NUM_LEGS = 10

    cnot_dag = get_cnot_dag(stim_circuit_str, NUM_LEGS)
    draw_cnot_dag(cnot_dag)
