import networkx as nx
import numpy as np
import pyzx as zx

from spidercat.circuit_extraction import expand_graph_and_forest
from spidercat.utils import load_solution_triplet


def expanded_graph_to_zx_diagram(
        G: nx.Graph,
        layout: str = "kamada_kawai",
        scale: float | None = None,
        invert_y: bool = True,
) -> zx.Graph:
    """
    Converts an expanded graph (output of expand_graph_and_forest) into a PyZX ZX-diagram.

    Each node in G is represented as a spider (default Z-spider, or X-spider if node['spider_type'] == 'X').
    Nodes tagged with is_mark=True (which have arity 2 in G) are 3-legged Z-spiders:
    two legs are connected to its neighbors in G, and the third leg connects to a new
    boundary output vertex (VertexType.BOUNDARY) added to the diagram's outputs.

    Positions are automatically computed from G using an appropriate NetworkX layout (defaulting to
    Kamada-Kawai). Output nodes are added to the layout graph so that NetworkX places the output
    ports neatly around the perimeter with optimal spacing.

    Args:
        G: The expanded NetworkX graph with node attributes ('is_mark', 'spider_type', etc.).
        layout: Layout algorithm to use when computing positions ('kamada_kawai' or 'spring').
                Defaults to 'kamada_kawai', which produces clean, radially symmetric layouts for regular graphs.
        scale: Optional scaling factor applied to coordinates. If None, coordinates are auto-scaled
               so that median edge length is at least 1.5 (preventing node collisions in PyZX viewers).
        invert_y: If True (default), sets qubit = -y so that PyZX's matplotlib/d3 renderers
                  preserve the standard Cartesian upright orientation of NetworkX.

    Returns:
        zx.Graph: The constructed PyZX diagram with marked nodes connected to boundary outputs.
    """
    mark_nodes = sorted([n for n, d in G.nodes(data=True) if d.get("is_mark", False)])

    # Construct layout graph with output nodes added so NetworkX optimizes their placement
    G_layout = G.copy()
    mark_to_out: dict[int, int] = {}
    next_id = max(G_layout.nodes()) + 1 if len(G_layout) > 0 else 0
    for m in mark_nodes:
        out_id = next_id
        next_id += 1
        G_layout.add_node(out_id, is_output=True, mark_node=m)
        G_layout.add_edge(m, out_id)
        mark_to_out[m] = out_id

    # Compute layout on G_layout directly (positioning both G nodes and output nodes)
    if layout == "kamada_kawai":
        try:
            pos_all = nx.kamada_kawai_layout(G_layout)
        except Exception:
            pos_all = nx.spring_layout(G_layout, seed=42)
    else:
        pos_all = nx.spring_layout(G_layout, seed=42)

    pos_all = {k: np.array(v, dtype=float) for k, v in pos_all.items()}

    # Scale coordinates so median edge length is suitable for PyZX rendering (node radius is 0.2)
    edge_lens = [
        np.linalg.norm(pos_all[u] - pos_all[v])
        for u, v in G_layout.edges()
        if np.linalg.norm(pos_all[u] - pos_all[v]) > 1e-6
    ]
    median_edge_len = float(np.median(edge_lens)) if edge_lens else 1.0

    if scale is not None:
        pos_all = {k: v * scale for k, v in pos_all.items()}
    elif median_edge_len < 1.0:
        target_len = 1.5
        s = target_len / median_edge_len
        pos_all = {k: v * s for k, v in pos_all.items()}

    # Build the PyZX diagram
    zx_graph = zx.Graph()
    node_to_v: dict[int, int] = {}

    # 1. Add spiders for all nodes in G
    for n, data in G.nodes(data=True):
        spider_type = data.get("spider_type", "Z")
        ty = zx.VertexType.X if spider_type == "X" else zx.VertexType.Z
        phase = data.get("phase", 0)
        p = pos_all[n]
        row = float(p[0])
        qubit = -float(p[1]) if invert_y else float(p[1])

        v = zx_graph.add_vertex(ty=ty, qubit=qubit, row=row, phase=phase)
        zx_graph.set_vdata(v, "nx_node", n)
        node_to_v[n] = v

    # 2. Add edges from G
    for u, v, data in G.edges(data=True):
        et = data.get("edge_type", 1)
        edgetype = zx.EdgeType.HADAMARD if (et == "hadamard" or et == 2) else zx.EdgeType.SIMPLE
        zx_graph.add_edge((node_to_v[u], node_to_v[v]), edgetype=edgetype)

    # 3. Add boundary output vertices for mark nodes using positions from G_layout
    outputs = []
    for m in mark_nodes:
        v_mark = node_to_v[m]
        out_id = mark_to_out[m]
        out_p = pos_all[out_id]
        out_row = float(out_p[0])
        out_qubit = -float(out_p[1]) if invert_y else float(out_p[1])

        out_v = zx_graph.add_vertex(ty=zx.VertexType.BOUNDARY, qubit=out_qubit, row=out_row)
        zx_graph.set_vdata(out_v, "mark_node", m)
        zx_graph.add_edge((v_mark, out_v))
        outputs.append(out_v)

    zx_graph.set_inputs([])
    zx_graph.set_outputs(outputs)

    return zx_graph


expanded_graph_to_zx = expanded_graph_to_zx_diagram


def solution_to_zx_diagram(n: int, t: int) -> zx.Graph:
    """
    Loads a solution triplet for (n, t), expands the graph and forest,
    and converts it to a PyZX ZX-diagram with output ports on marks.

    Args:
        n: Number of marks (output qubits of the cat state).
        t: Fault-tolerance distance parameter.

    Returns:
        zx.Graph: The PyZX diagram representing the cat state preparation graph.
    """
    loaded = load_solution_triplet(n, t, 1)
    if loaded is None:
        raise ValueError(f"No solution triplet found for n={n}, t={t}.")
    grf, tree, M, matchings = loaded
    G_exp, _ = expand_graph_and_forest(grf, tree, M, matchings, expand_flags=False)
    return expanded_graph_to_zx_diagram(G_exp)


load_solution_zx_diagram = solution_to_zx_diagram
cat_state_to_zx_diagram = solution_to_zx_diagram


if __name__ == "__main__":
    import pyzx as zx

    d = solution_to_zx_diagram(12, 3)
    zx.draw(d)

