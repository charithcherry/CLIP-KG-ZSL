# graph/manual_edges.py

import json



def add_manual_edges(graph_path: str, manual_edges: list):
    # Load existing graph
    with open(graph_path, 'r') as f:
        existing_graph = json.load(f)

    # Append new edges
    existing_graph.extend(manual_edges)

    # Save updated graph
    with open(graph_path, 'w') as f:
        json.dump(existing_graph, f, indent=2)
    print(f"Successfully added {len(manual_edges)} edges to {graph_path}.")


