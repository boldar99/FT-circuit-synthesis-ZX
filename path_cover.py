from collections import Counter, defaultdict
from pytket import Circuit, OpType, Qubit, Bit
import networkx as nx
import random
import matplotlib.pyplot as plt


def _gen_graph(n_lines, n_more_edges):
    line_sizes = [random.randint(5, 8) for _ in range(n_lines)]
    g = nx.disjoint_union_all([nx.path_graph(size) for size in line_sizes])

    # Add additional random edges between different lines
    for _ in range(n_more_edges):
        while True:
            node1, node2 = random.sample(list(g.nodes()), 2)
            d1, d2 = g.degree(node1), g.degree(node2)
            if d1 == d2 == 2:
                break
        g.add_edge(node1, node2)

    # Remove degree 2 vertices and combine their neighbors
    changed = True
    while changed:
        changed = False
        for node in g.nodes():
            if g.degree(node) == 2:
                n1, n2 = list(g.neighbors(node))
                d1, d2 = g.degree(n1), g.degree(n2)
                if d1 <= 2 or d2 <= 2:
                    g.add_edge(n1, n2)
                    g.remove_node(node)
                    changed = True
                    break

    to_remove = set()
    for node in g.nodes():
        if g.degree(node) == 1:
            n = list(g.neighbors(node))[0]
            if g.degree(n) == 1:
                to_remove.add(n)
    for node in to_remove:
        g.remove_node(node)
    return g


def relabel_and_shuffle(g):
    n = len(g.nodes())
    indices = list(range(n))
    random.shuffle(indices)
    mapping = {node: indices[i] for i, node in enumerate(g.nodes())}
    return nx.relabel_nodes(g, mapping)


def gen_graph(n_lines, n_more_edges):
    g = _gen_graph(n_lines, n_more_edges)
    while len(g.nodes()) == 0:
        g = _gen_graph(n_lines, n_more_edges)
    return relabel_and_shuffle(g)


def extract_paths(path_cover):
    return [[edge for edge in path_cover
             if edge[0] in component and edge[1] in component]
            for component in nx.connected_components(nx.Graph(path_cover))
            if len(component) > 1]


def seq_from_path(path):
    cnt = Counter([v for edge in path for v in edge])
    adj = defaultdict(list)
    for v1, v2 in path:
        adj[v1].append(v2)
        adj[v2].append(v1)
    start = sorted(cnt.keys(), key=lambda x: (cnt[x], x))[0]
    seq = [start]
    while len(seq) < len(path) + 1:
        v = seq[-1]
        next_v = adj[v].pop()
        seq.append(next_v)
        adj[next_v].remove(v)
    return seq


def extract_circuit(g, path_cover, verbose=False):
    n_lines = len(path_cover)
    seen_edges = []
    circ = Circuit(n_lines)
    for i in range(len(path_cover)):
        circ.add_gate(OpType.H, [i])
        seq = seq_from_path(path_cover[i])
        not verbose and print(f'seq {i}: {seq}')
        for j, v in enumerate(seq):
            if j == 0:
                continue
            if g.degree(v) == 3:
                flag = (set(g.neighbors(v)) - set(seq[j-1:j+2])).pop()
                edge = tuple(sorted((v, flag)))
                not verbose and print(f'{(edge in seen_edges)=}')
                if edge not in seen_edges:
                    qidx = n_lines + len(seen_edges)
                    bidx = len(seen_edges)
                    circ.add_qubit(Qubit(qidx))
                    circ.add_bit(Bit(bidx))
                seen_edges.append(edge)
                flag_idx = seen_edges.index(edge)
                not verbose and print(f'CX[{i}, {n_lines + flag_idx}] on flag {flag_idx}')
                circ.add_gate(OpType.CX, [i, n_lines + flag_idx])
    # for i, edge in enumerate(new_edges):
    #     circ.add_gate(OpType.Measure, n_lines + i, i)
    return circ


def visualize_graph(g, path_cover=None, pos=None):
    plt.figure(figsize=(30, 25))
    pos = nx.spring_layout(g, k=3, iterations=50) if pos is None else pos

    if path_cover:
        path_edges_set = set(tuple(sorted(edge)) for edge in path_cover)
        other_edges = [edge for edge in g.edges()
                       if tuple(sorted(edge)) not in path_edges_set]
        if other_edges:
            nx.draw_networkx_edges(g, pos, edgelist=other_edges,
                                   edge_color='lightgray', width=2, alpha=0.5)

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'olive']
        for i, path_edges in enumerate(path_cover):
            nx.draw_networkx_edges(g, pos, edgelist=path_edges,
                                   edge_color=colors[i % len(colors)], width=4, alpha=0.8)
    else:
        nx.draw_networkx_edges(g, pos, edge_color='gray', width=2)

    nx.draw_networkx_nodes(g, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(g, pos, font_size=10, font_weight='bold')

    title = "Graph with path cover" if path_cover else "Graph"
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def remove_boundary(g):
    internal_nodes = [
        node for node in g.nodes() if g.degree(node) > 1
    ]
    return g.subgraph(internal_nodes)


def find_path_cover_n(g, n_paths, max_attempts=10000):
    g = remove_boundary(g)
    n = len(g.nodes()) - n_paths
    H = nx.empty_graph(len(g.nodes()))
    for _ in range(max_attempts):
        edge = random.choice(list(g.edges()))
        H.add_edge(*edge)
        d1, d2 = H.degree(edge[0]), H.degree(edge[1])
        if d1 > 2 and d2 > 2:
            H.remove_edge(*edge)
        elif d1 > 2:
            assert d1 == 3
            # remove one of the edges incident to edge[0]
            n1, n2, n3 = list(H.neighbors(edge[0]))
            H.remove_edge(edge[0], random.choice([n1, n2, n3]))
        elif d2 > 2:
            assert d2 == 3
            n1, n2, n3 = list(H.neighbors(edge[1]))
            H.remove_edge(edge[1], random.choice([n1, n2, n3]))
    return list(H.edges())


def find_path_cover(g):
    for n_paths in range(1, 11):
        try:
            path_cover = find_path_cover_n(g, n_paths)
            print(f"Path cover with {n_paths} paths found ({len(path_cover)} edges)")
            separate_paths = extract_paths(path_cover)
            return separate_paths
        except ValueError:
            print(f"No path cover found with {n_paths} paths")
            continue
    return None


def main():
    g = gen_graph(n_lines=4, n_more_edges=5)

    print(f"Generated graph with {len(g.nodes())} nodes and "
          f"{len(g.edges())} edges")
    print("Nodes:", list(g.nodes()))
    print("Edges:", list(g.edges()))

    path_cover = find_path_cover(g)

    if path_cover:
        for i, path_edges in enumerate(path_cover):
            print(f"  Path {i+1}: {path_edges}")

    # extract_circuit(g, path_cover)
    visualize_graph(g, path_cover)


if __name__ == "__main__":
    main()
