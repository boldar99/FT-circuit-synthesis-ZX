import networkx as nx

def match_path_ends_to_marked_edges(
    G: nx.Graph,
    path_cover: list[list[int]],
    markings: dict[tuple[int, int], int]
) -> dict[int, int]:
    """
     matches path endpoints to incident marked edges using a maximum bipartite matching.
    
    Args:
        G: The original graph.
        path_cover: List of paths (each path is a list of node IDs).
        markings: Dictionary mapping edges (tuples) to number of marks.
        
    Returns:
        A dictionary mapping path end nodes to their assigned marked edge.
        {end_node: (u, v)}
    """
    
    # 1. Identify all end nodes
    end_nodes = []
    path_edges = set()
    
    for path in path_cover:
        if not path:
            continue
        # Both ends are endpoints
        # Note: In a path cover, a path could be a single isolated node (len=1)
        if len(path) == 1:
            end_nodes.append(path[0])
        else:
            # ONLY add the "end" node (path[-1])
            end_nodes.append(path[-1])
            
            # Collect path edges to exclude them
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                path_edges.add(tuple(sorted((u, v))))

    end_nodes_set = set(end_nodes)
    
    # 2. Identify relevant marked edges
    # Filter markings to those > 0 and NOT in path_edges?
    # Usually markings on path edges are for SWAPs or similar, markings OFF path are interactions.
    # The requirement is "dictionary of end nodes to edges with markings".
    # We'll assume we want the off-path edges.
    
    relevant_marked_edges = []
    for edge, marks in markings.items():
        if marks > 0:
            u, v = sorted(edge)
            if (u, v) not in path_edges:
                relevant_marked_edges.append((u, v))
                
    # 3. Build Bipartite Graph
    B = nx.Graph()
    
    # Left Nodes: End Nodes (integers)
    # Right Nodes: Marked Edges (tuples)
    # To avoid ID collision if edges are somehow integers (unlikely), we keeps types distinct.
    # But networkx nodes can be anything hashable.
    
    B.add_nodes_from(end_nodes, bipartite=0)
    B.add_nodes_from(relevant_marked_edges, bipartite=1)
    
    for edge in relevant_marked_edges:
        u, v = edge
        # If u is an end node, add edge in B
        if u in end_nodes_set:
             B.add_edge(u, edge)
        # If v is an end node, add edge in B
        if v in end_nodes_set:
             B.add_edge(v, edge)
             
    # 4. Compute Maximum Matching
    # top_nodes = end_nodes
    try:
        matching = nx.bipartite.maximum_matching(B, top_nodes=end_nodes)
    except:
        # Fallback for empty graph or other issues
        matching = {}
        
    # 5. Extract result
    # matching contains both u->v and v->u. We only want end_node -> neighbor
    result = {}
    for node in end_nodes:
        if node in matching:
            # edge is (u, v) where one is node
            edge = matching[node]
            neighbor = edge[0] if edge[1] == node else edge[1]
            result[node] = neighbor
            
    return result

def draw_qubit_lines_state(G, path_cover, markings, matching):
    """
    Draws the graph with path cover, markings, and matchings.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 5. Visualization
    #pos = nx.shell_layout(G, nlist=[range(5), range(5,10)])
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
            edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=[color], width=3, label=f'Path {i}')
        # Draw path nodes
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=[color] * len(path), node_size=500)
    
    # Highlight Matching
    for end_node, neighbor in matching.items():
        # Reconstruct edge for clarification if needed, but we mostly need neighbor position
        
        # Determine path color for this end node
        path_color = 'green' # fallback
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
