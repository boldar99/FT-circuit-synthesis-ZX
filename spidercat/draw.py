import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_qubit_lines_state(G, path_cover, markings, matching, pos = None):
    """
    Draws the graph with path cover, markings, and matchings.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 5. Visualization
    # pos = nx.shell_layout(G, nlist=[range(5), range(5,10)])
    pos = pos or nx.spring_layout(G)
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


def draw_spanning_forest_solution(
        G: nx.Graph,
        forest: nx.Graph,
        markings: dict[tuple[int, int], int],
        matches: dict[int, list[tuple[int, int]]] | None = None,
        roots: dict[int, int] | None = None,
        figsize=(14, 10),
        pos = None,
):
    """
    Visualizes the graph state, highlighting the spanning forest,
    drawing connections to marked edges, and circling tree roots
    with the color corresponding to their tree.
    """
    # 1. Setup Layout
    pos = pos or nx.spring_layout(G, seed=42)
    plt.figure(figsize=figsize)

    # 2. Draw Background (All edges faint)
    nx.draw_networkx_edges(G, pos, edge_color='black', width=1, alpha=0.5)

    # 3. Draw Markings (Black, dashed lines)
    marked_edge_list = [(u, v) for (u, v), m in markings.items() if m > 0]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=marked_edge_list,
        edge_color='black',
        style='dashed',
        width=1.5,
        alpha=0.6
    )

    edge_labels = {e: "|" * m for e, m in markings.items() if m > 0}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_weight='bold',
        font_size=15,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.2)
    )

    # 4. Draw Forest Trees
    cmap = plt.cm.tab10
    colors = cmap.colors
    node_color_map = {}

    for i, component in enumerate(nx.connected_components(forest)):
        color = colors[i % len(colors)]
        tree = forest.subgraph(component)

        # Draw Tree Edges
        nx.draw_networkx_edges(
            G, pos,
            edgelist=tree.edges(),
            edge_color=[color],
            width=3.5,
            alpha=0.6
        )

        # Draw Tree Nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=list(component),
            node_color=[color] * len(component),
            node_size=600,
            edgecolors='black'
        )

        # Store color for later lookups (roots & matches)
        for node in component:
            node_color_map[node] = color

    # --- NEW: Draw Root Highlights (Color Matched) ---
    if roots:
        for tree_id, root_node in roots.items():
            # Lookup the color of this specific root
            root_color = node_color_map.get(root_node, 'black')

            # Draw distinct circle for this root
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[root_node],
                node_size=1200,
                node_color='none',  # Transparent inside
                edgecolors=[root_color],  # Border matches tree color
                linewidths=3
            )

    # Draw Node Labels
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')

    # 5. Draw Matches
    matches = matches if matches is not None else {}
    for node, assigned_edges in matches.items():
        node_color = node_color_map.get(node, 'gray')
        start_pos = np.array(pos[node])

        for edge_tuple in assigned_edges:
            pos_u = np.array(pos[edge_tuple[0]])
            pos_v = np.array(pos[edge_tuple[1]])
            target = (pos_u + pos_v) / 2

            plt.plot(
                [start_pos[0], target[0]],
                [start_pos[1], target[1]],
                color=node_color,
                linewidth=3.5,
                alpha=0.6,
                linestyle='-'
            )

    plt.title("Spanning Forest with Marked Assignments & Roots")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()
