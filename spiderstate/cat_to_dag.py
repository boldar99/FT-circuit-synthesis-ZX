import stim
import networkx as nx
from matplotlib import pyplot as plt

import stim
import networkx as nx


def get_cnot_dag(stim_str: str, num_legs: int) -> nx.DiGraph:
    """
    Returns a minimal DAG of only the CNOTs acting on data qubits (legs),
    preserving any hidden causal dependencies mediated by flag qubits.
    """
    circuit = stim.Circuit(stim_str)
    full_dag = nx.DiGraph()

    last_seen_on_qubit = {}
    cnot_id = 0
    data_cnot_nodes = []

    # 1. Build the full DAG
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

                # Check if this is a "Data CNOT"
                # (Assuming a data CNOT is one where BOTH qubits are data legs.
                # Change `and` to `or` if it just needs to touch one leg).
                if control < num_legs and target < num_legs:
                    data_cnot_nodes.append(cnot_id)

                cnot_id += 1

    # 2. Compute transitive closure to capture all indirect dependencies
    tc_dag = nx.transitive_closure(full_dag)

    # 3. Induce a subgraph containing ONLY the data CNOTs
    data_subgraph = tc_dag.subgraph(data_cnot_nodes)

    # 4. Transitive reduction to remove redundant edges (get the Hasse diagram)
    reduced_dag = nx.transitive_reduction(data_subgraph)

    # nx.transitive_reduction strips node attributes, so we copy them back
    final_dag = nx.DiGraph()
    for node in reduced_dag.nodes():
        final_dag.add_node(node, **full_dag.nodes[node])
    for u, v in reduced_dag.edges():
        final_dag.add_edge(u, v)

    return final_dag


def draw_cnot_dag(dag: nx.DiGraph):
    """
    Draws the CNOT dependency DAG.
    Nodes are layered left-to-right based on their circuit depth.
    """
    if not nx.is_directed_acyclic_graph(dag):
        print("Warning: Graph is not a DAG. Cannot compute layers properly.")
        return

    # 1. Calculate the 'depth' (layer) for each node to arrange them left-to-right
    for node in nx.topological_sort(dag):
        predecessors = list(dag.predecessors(node))
        if not predecessors:
            dag.nodes[node]['layer'] = 0
        else:
            # A node's layer is 1 + the maximum layer of its dependencies
            dag.nodes[node]['layer'] = max(dag.nodes[p]['layer'] for p in predecessors) + 1

    # 2. Generate descriptive labels for the nodes (e.g., "CX 0->1\n(ID: 4)")
    labels = {}
    for node, data in dag.nodes(data=True):
        control = data.get('control', '?')
        target = data.get('target', '?')
        labels[node] = f"{control}→{target}\n({node})"

    # 3. Set up the layout and plot
    # multipartite_layout uses the 'layer' attribute we just created
    pos = nx.multipartite_layout(dag, subset_key="layer", align="horizontal")

    plt.figure(figsize=(14, 8))

    nx.draw(
        dag,
        pos,
        labels=labels,
        with_labels=True,
        node_size=2500,
        node_color="#87CEFA",  # Light sky blue
        font_size=12,
        font_weight="bold",
        edge_color="gray",
        arrows=True,
        arrowsize=20
    )

    plt.title("CNOT Partial Ordering (Time flows Left to Right)", fontsize=16)
    plt.tight_layout()
    plt.show()


# --- To use it with the previous code ---
# draw_cnot_dag(cnot_dag)


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
