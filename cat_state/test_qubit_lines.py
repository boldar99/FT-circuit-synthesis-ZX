import networkx as nx
import unittest
import matplotlib.pyplot as plt
import sys
import numpy as np
from cat_state.qubit_lines import match_path_ends_to_marked_edges, draw_qubit_lines_state
from cat_state.sat_path_cover import find_all_path_covers
from cat_state.markings import GraphMarker

class TestQubitLines(unittest.TestCase):
    def test_simple_matching(self):
        # Graph: 0-1-2 (Path), 0-3 (Marked), 2-4 (Marked)
        G = nx.Graph()
        G.add_edges_from([(0,1), (1,2), (0,3), (2,4)])
        
        path_cover = [[0, 1, 2]]
        markings = {
            (0, 3): 1,
            (2, 4): 1
        }
        
        matching = match_path_ends_to_marked_edges(G, path_cover, markings)
        
        # Expect 2 -> (2,4) only (since 2 is the end, 0 is start)
        self.assertNotIn(0, matching)
        self.assertIn(2, matching)
        self.assertEqual(matching[2], 4)
        
    def test_shared_edge_conflict(self):
        # Two paths ending at 0 and 1. Edge (0,1) is marked.
        # This shouldn't normally happen if (0,1) is interaction, but let's see.
        # If 0 and 1 are end nodes of DIFFERENT paths.
        # Path 1: [2, 0], Path 2: [3, 1]. Marked edge (0,1).
        # Only one can claim (0,1).
        
        G = nx.Graph()
        G.add_edges_from([(2,0), (0,1), (1,3)])
        path_cover = [[2, 0], [3, 1]] 
        markings = {(0, 1): 1} # Marked edge between endpoints
        
        matching = match_path_ends_to_marked_edges(G, path_cover, markings)
        
        # Only one of them should get it
        self.assertTrue(0 in matching or 1 in matching)
        self.assertFalse(0 in matching and 1 in matching)
        
    def test_path_edge_exclusion(self):
        # Path [0, 1, 2]. Edge (0,1) marked (maybe swap). Edge (0,3) marked.
        # Ensure (0,1) is NOT picked as the "interaction" edge for 0 or 1.
        G = nx.Graph()
        G.add_edges_from([(0,1), (1,2), (0,3)])
        path_cover = [[0, 1, 2]]
        markings = {
            (0, 1): 1, # On path
            (0, 3): 1  # Off path interaction
        }
        
        matching = match_path_ends_to_marked_edges(G, path_cover, markings)
        
        # Path [0, 1, 2]. End is 2. 
        # Markings: (0, 1), (0, 3).
        # End node 2 has neighbors 1 (on path).
        # Wait, if we only look at path[-1] which is 2.
        # Node 2 has NO marked interaction edges in this setup (0,3 is incident to 0).
        # So matching should be empty for optimal behavior or just checking exclusions.
        
        # New test:
        # Path [0, 1, 2]. End 2.
        # Markings: (2, 3) -> valid interaction.
        # Markings: (2, 1) -> invalid (on path).
        
        G2 = nx.Graph()
        G2.add_edges_from([(0,1), (1,2), (2,3)])
        path_cover2 = [[0, 1, 2]] # End is 2
        markings2 = {
            (1, 2): 1, # On path
            (2, 3): 1  # Valid interaction
        }
        matching2 = match_path_ends_to_marked_edges(G2, path_cover2, markings2)
        
        self.assertIn(2, matching2)
        self.assertEqual(matching2[2], 3)
        self.assertNotIn(2, matching)

    
def visualize_demo():
    print("Running Visualization Demo with Petersen Graph...")
    
    # 1. Setup Petersen Graph (10 nodes, 15 edges)
    G = nx.petersen_graph()
    
    # 2. Find a Path Cover
    # We want a cover that is interesting (e.g. 2 paths)
    # Petersen graph is not Hamiltonian, so we expect at least 2 paths.
    print("Finding path cover...")
    covers = list(find_all_path_covers(G))
    if not covers:
        print("No path cover found!")
        return
        
    # Sort by number of paths
    covers.sort(key=len)
    
    # Try to find a cover with multiple paths for the demo
    path_cover = covers[0] # Default
    for cover in covers:
        if len(cover) > 1:
            path_cover = cover
            break
            
    print(f"Using path cover with {len(path_cover)} paths: {path_cover}")

    # 3. Generate Markings
    # using GraphMarker to find valid markings (T=5 or 6)
    print("Generating markings...")
    marker = GraphMarker(G, path_cover=path_cover, max_marks=20)
    # T=6 is usually good for cubic graphs / Petersen
    markings = marker.find_solution(6)
    
    if not markings:
        print("Could not find valid markings for T=6. Trying T=8...")
        markings = marker.find_solution(8)
        
    if not markings:
         print("No markings found. Using dummy markings for demo.")
         # Fallback dummy markings just to show something
         markings = {}
         edges = list(G.edges())
         for i in range(0, len(edges), 2):
             markings[edges[i]] = 1

    # 4. Compute Matching
    matching = match_path_ends_to_marked_edges(G, path_cover, markings)
    
    print("\nCalculated Matching:")
    for end_node, neighbor in matching.items():
        print(f"  End Node {end_node} -> Neighbor {neighbor}")

        
    # 5. Visualization
    draw_qubit_lines_state(G, path_cover, markings, matching)

if __name__ == '__main__':
    # If arguments are passed (like 'python -m ...'), unittest usually parses them.
    # To support "just running" as a demo while keeping tests:
    # Check if we are running standard unittest discovery or direct execution.
    # We'll just run the demo AND the tests.
    
    # Run demo
    visualize_demo()
    
    # Then run tests
    unittest.main()
