import networkx as nx
import matplotlib.pyplot as plt
from pysat.solvers import Glucose3
from itertools import combinations
from collections import defaultdict, Counter

def find_all_path_covers(num_vertices, edges):
    undirected_edges = sorted(list(set([tuple(sorted(e)) for e in edges])))
    edge_to_id = {e: i + 1 for i, e in enumerate(undirected_edges)}
    id_to_edge = {i + 1: e for i, e in enumerate(undirected_edges)}
    
    solver = Glucose3()

    # 1. Degree Constraints: Every vertex degree 1 or 2
    for i in range(num_vertices):
        incident_vars = [edge_to_id[e] for e in undirected_edges if i in e]
        if incident_vars:
            solver.add_clause(incident_vars)
        if len(incident_vars) >= 3:
            for combo in combinations(incident_vars, 3):
                solver.add_clause([-v for v in combo])

    all_path_covers = []
    
    # 2. Solving Loop
    while solver.solve():
        model = solver.get_model()
        selected_edges = [id_to_edge[idx] for idx in model if idx > 0]
        
        G_temp = nx.Graph(selected_edges)
        try:
            cycle = nx.find_cycle(G_temp)
            cycle_edge_ids = [edge_to_id[tuple(sorted((u, v)))] for u, v in cycle]
            solver.add_clause([-eid for eid in cycle_edge_ids])
            continue
        except nx.NetworkXNoCycle:
            G_full = nx.Graph(selected_edges)
            G_full.add_nodes_from(range(num_vertices))
            num_paths = nx.number_connected_components(G_full)
            
            all_path_covers.append((num_paths, selected_edges))
            solver.add_clause([-idx for idx in model])
            
            if len(all_path_covers) >= 10000: break

    return all_path_covers, undirected_edges

def run_and_visualize(num_vertices, edges):
    all_solutions, base_edges = find_all_path_covers(num_vertices, edges)
    
    # 1. Print Statistics
    counts = Counter([s[0] for s in all_solutions])
    print("\n" + "="*45)
    print(f"{'PATH COVER STATISTICS (N=' + str(num_vertices) + ')':^45}")
    print("="*45)
    print(f"Total Unique Solutions Found: {len(all_solutions)}")
    for n_paths in sorted(counts.keys()):
        label = "★ HAMILTONIAN ★" if n_paths == 1 else f"{n_paths} separate paths"
        print(f" - {label:20} : {counts[n_paths]} solutions")
    print("="*45 + "\n")

    # 2. Grouping
    grouped = defaultdict(list)
    for num_paths, path_edges in all_solutions:
        grouped[num_paths].append(path_edges)

    G_base = nx.Graph()
    G_base.add_nodes_from(range(num_vertices))
    G_base.add_edges_from(base_edges)
    pos = nx.shell_layout(G_base, nlist=[range(5, 10), range(5)])
    
    # Path Colors (Qualitative colormap)
    colors = plt.cm.tab10.colors 

    for path_count in sorted(grouped.keys()):
        covers = grouped[path_count][:6]
        num_to_plot = len(covers)
        
        plt.figure(figsize=(15, 7))
        title = "HAMILTONIAN PATHS" if path_count == 1 else f"COVERS WITH {path_count} PATHS"
        plt.suptitle(f"{title}\nEach path is highlighted in a different color", fontsize=16, fontweight='bold')

        for i, cover_edges in enumerate(covers):
            ax = plt.subplot(2, 3, i + 1)
            
            # Draw base background
            nx.draw_networkx_nodes(G_base, pos, node_color='#ecf0f1', node_size=300, ax=ax)
            nx.draw_networkx_labels(G_base, pos, font_size=7, ax=ax)
            nx.draw_networkx_edges(G_base, pos, edgelist=base_edges, edge_color='gray', alpha=0.05, ax=ax)
            
            # Identify individual paths to color them differently
            G_cover = nx.Graph(cover_edges)
            G_cover.add_nodes_from(range(num_vertices))
            path_components = list(nx.connected_components(G_cover))
            
            for path_idx, nodes in enumerate(path_components):
                path_subgraph_edges = G_cover.edges(nodes)
                color = colors[path_idx % len(colors)]
                
                # Draw edges for this specific path
                nx.draw_networkx_edges(G_cover, pos, edgelist=path_subgraph_edges, 
                                       edge_color=[color], width=4, ax=ax)
                # Draw nodes for this specific path to match
                nx.draw_networkx_nodes(G_cover, pos, nodelist=list(nodes), 
                                       node_color=[color], node_size=350, ax=ax)

            plt.title(f"Solution {i+1}", fontsize=10)
            plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        print(f"[*] Showing: {title}. Close window to proceed.")
        plt.show()

# --- Peterson Graph Setup ---
V = 10
peterson_edges = [
    (0,1), (1,2), (2,3), (3,4), (4,0),  # Outer
    (5,7), (7,9), (9,6), (6,8), (8,5),  # Inner
    (0,5), (1,6), (2,7), (3,8), (4,9)   # Spokes
]

run_and_visualize(V, peterson_edges)