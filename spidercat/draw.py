import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


def draw_qubit_lines_state(G, path_cover, markings, matching):
    """
    Draws the graph with path cover, markings, and matchings.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 5. Visualization
    # pos = nx.shell_layout(G, nlist=[range(5), range(5,10)])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))

    # Draw all edges faint
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1)

    # Draw Marked Edges
    marked_edge_list = [(u, v) for (u, v), m in markings.items() if m > 0]
    nx.draw_networkx_edges(G, pos, edgelist=marked_edge_list, edge_color='black', style='solid', width=1.5)

    # Draw Edge Labels (Mark counts)
    edge_labels = {e: "|" * m for e, m in markings.items() if m > 0}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_weight='bold', font_size=20,
        bbox=dict(facecolor='none', edgecolor='none', alpha=0)
    )

    # Draw Nodes & Labels
    nx.draw_networkx_labels(G, pos)

    # Draw Path Edges
    colors = plt.cm.tab10.colors
    for i, path in enumerate(path_cover):
        color = colors[i % len(colors)]
        if len(path) > 1:
            edges = [(path[j], path[j + 1]) for j in range(len(path) - 1)]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=[color], width=3, label=f'Path {i}')
        # Draw path nodes
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=[color] * len(path), node_size=500)

    # Highlight Matching
    for end_node, neighbor in matching.items():
        # Reconstruct edge for clarification if needed, but we mostly need neighbor position

        # Determine path color for this end node
        path_color = 'green'  # fallback
        for i, path in enumerate(path_cover):
            if end_node in path:
                path_color = colors[i % len(colors)]
                break

        # Calculate Midpoint of the edge (end_node, neighbor)
        # Note: 'neighbor' here is the node across the marked edge
        pos_u = np.array(pos[end_node])
        pos_v = np.array(pos[neighbor])
        midpoint = (pos_u + pos_v) / 2
        pos_end = np.array(pos[end_node])

        # Draw line from End Node to Midpoint
        plt.plot([pos_end[0], midpoint[0]], [pos_end[1], midpoint[1]],
                 color=path_color, linewidth=4, linestyle='-')

    plt.axis('off')
    plt.savefig("qubit_lines_state.png")
    plt.show()
    plt.close()


def draw_path_cover(ax, G_base, pos, cover_paths, markings=None, matching=None, node_size=200, label_font_size=8):
    # Base faint background
    edges = list(G_base.edges())
    nx.draw_networkx_nodes(G_base, pos, node_color="#ecf0f1", node_size=node_size, ax=ax)
    nx.draw_networkx_labels(G_base, pos, font_size=label_font_size, ax=ax)
    nx.draw_networkx_edges(G_base, pos, edgelist=edges, edge_color="gray", alpha=0.2, ax=ax)

    # Draw each path with a distinct color
    colors = plt.cm.tab10.colors
    for idx, path in enumerate(cover_paths):
        color = colors[idx % len(colors)]
        if len(path) >= 2:
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(G_base, pos, edgelist=path_edges, edge_color=[color], width=3, ax=ax)
        nx.draw_networkx_nodes(G_base, pos, nodelist=path, node_color=[color], node_size=int(node_size * 1.2), ax=ax)

    # Overlay tick marks from `markings` (if provided)
    if markings:
        edge_labels = {}
        for u, v in G_base.edges():
            num_marks = markings.get((u, v), markings.get((v, u), 0))
            if num_marks:
                edge_labels[(u, v)] = "  |  " * num_marks
        if edge_labels:
            nx.draw_networkx_edge_labels(G_base, pos, edge_labels=edge_labels, font_size=14, font_weight="bold",
                                         bbox=dict(alpha=0), ax=ax)

    # Draw Matching Lines (Half-edges to midpoints)
    if matching:
        for end_node, neighbor in matching.items():
            # Determine path color for this end node
            path_color = 'black'  # fallback
            for i, path in enumerate(cover_paths):
                if end_node in path:
                    path_color = colors[i % len(colors)]
                    break

            # Calculate Midpoint
            pos_u = np.array(pos[end_node])
            pos_v = np.array(pos[neighbor])
            midpoint = (pos_u + pos_v) / 2
            pos_end = np.array(pos[end_node])

            # Draw solid line from end_node to midpoint
            ax.plot([pos_end[0], midpoint[0]], [pos_end[1], midpoint[1]],
                    color=path_color, linewidth=4, linestyle='-')


def visualize_cat_state_base(G, ham_path, markings, pos=None):
    plt.figure(figsize=(5, 5))
    pos = pos or nx.spring_layout(G)  # Kamada-Kawai usually looks best for regular graphs
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={e: "  |  " * num_marks for e, num_marks in markings.items()},
                                 font_size=18, font_weight='bold', bbox=dict(alpha=0))
    nx.draw_networkx_edges(
        G, pos=pos,
        edgelist=ham_path,
        edge_color='red', width=1.5
    )
    plt.show()
