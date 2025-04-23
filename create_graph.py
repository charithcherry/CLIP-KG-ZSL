# Graph creation/main.py

import os
from graph import graph_generator
from graph.manual_edge_addition import add_manual_edges
from graph.edges import manual_edges
from pathlib import Path


def main():
    datadir = "data\Animals_with_Attributes2"

    print("Initializing Graph Generator...")
    graph_gen = graph_generator.GraphGenerator(
        classes_path=os.path.join(datadir, "classes.txt"),
        predicates_path=os.path.join(datadir, "predicates.txt"),
        annotation_continuous_path=os.path.join(datadir, "predicate-matrix-continuous.txt")
    )

    print("Selecting classes and predicates...")
    selected_classes = [
        "horse", "cow", "deer", "gorilla", "blue+whale",
        "zebra", "buffalo", "antelope", "chimpanzee", "killer+whale"
    ]

    selected_predicates = [
        'black', 'white', 'blue', 'brown', 'gray',
        'patches', 'spots', 'stripes', 'furry', 'hairless',
        'flippers', 'hands', 'hooves', 'pads', 'paws',
        'longleg', 'longneck', 'tail', 'horns', 'claws', 'tusks'
    ]
    graph_gen.set_selection(selected_classes, selected_predicates)

    print("Generating edges from attribute differences...")
    edges = graph_gen.generate_edges(
        MAX_DIFF=10,
        MIN_DIFF=45,
        ALPHA=1.2,
        BETA=0.75,
        weight_threshold=0.5
    )

    print("Filtering edges by predicate threshold...")
    new_edges = graph_gen.filter_by_predicate_threshold(edges)

    print("Matching similar class pairs manually...")
    matches = [
        ('zebra', 'horse'), ('cow', 'buffalo'),
        ('deer', 'antelope'), ('blue+whale', 'killer+whale'),
        ('gorilla', 'chimpanzee')
    ]
    matched_edges = graph_gen.match(new_edges, matches)

    output_path = "output/filtered_class_pairwise_weighted_graph.json"
    print(f"Saving matched edges to {output_path}...")
    graph_gen.save_edges(matched_edges, output_path)

    print("Adding manual edges to the saved graph...")
    graph_file_path = Path("output/filtered_class_pairwise_weighted_graph.json")
    add_manual_edges(graph_file_path, manual_edges)

    print("Graph generation and edge refinement complete.")


if __name__ == "__main__":
    main()
