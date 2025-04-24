import os
import torch
import argparse
from graph.graph_generator import GraphGenerator
from models.rgcn.rgcn import KnowledgeGraphRGCN
from models.rgcn.loss import PrototypeRefinementLoss
from graph.manual_edge_addition import add_manual_edges
from graph.edges import manual_edges
from torch_geometric.data import Data
import json

from test_requirements import clip_requirement

clip_requirement.test_clip_installation()

def load_graph(graph_path):
    """Load the pre-generated graph from the specified file."""
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    return graph_data


def get_target_similarity_matrix_from_graph(edges, class_to_idx, num_classes, epsilon=1e-6):
    """
    edges: list of edge dicts from the KG
    class_to_idx: dict mapping class names to indices
    num_classes: total number of classes
    epsilon: small value to avoid division by zero
    """
    sim_matrix = torch.eye(num_classes)  

    for edge in edges:
        src = class_to_idx[edge['source']]
        tgt = class_to_idx[edge['target']]
        num_sim = edge.get('num_similar', 0)
        num_con = edge.get('num_contrasts', 0)

        total_sum = num_sim + num_con

        # Default similarity computation (without reward or penalty)
        sim_value = (num_sim + epsilon) / (total_sum + epsilon)

        # Apply reward or penalty based on num_sim / total_sum
        if num_sim > 0.75 * total_sum:
            sim_value *= 1.0  # Reward: Increase similarity
        elif num_sim < 0.25 * total_sum:
            sim_value = 0  # Penalty: Set similarity to 0

        # Assign similarity value to the matrix
        sim_matrix[src, tgt] = sim_value
        sim_matrix[tgt, src] = sim_value  # Assuming undirected similarity

    return sim_matrix


def create_triples_from_graph(graph_data, class_to_idx):
    """Create the triples from the loaded graph."""
    triples = []
    for edge in graph_data:
        src = edge["source"].replace("+", "_")
        tgt = edge["target"].replace("+", "_")
        for rel in edge.get("relationship", []):
            predicate = rel["predicate"]
            triples.append((src, tgt, predicate))
            triples.append((tgt, src, predicate))  # undirected

        for contrast in edge.get("contrasts", []):
            predicate = "contrast_" + contrast["predicate"]
            triples.append((src, tgt, predicate))
            triples.append((tgt, src, predicate))  # undirected
    return triples

def train_model(triples, target_sim_matrix, num_classes, epochs=300):
    """Train the KnowledgeGraphRGCN model."""
    model_gcn = KnowledgeGraphRGCN(triples)
    optimizer = torch.optim.Adam(model_gcn.parameters(), lr=1e-4)
    criterion = PrototypeRefinementLoss(alpha=1.0, beta=0.8, target_sim_matrix=target_sim_matrix)
    
    for epoch in range(epochs):
        loss = model_gcn.train_step(optimizer, criterion)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss: {loss:.4f}")

    model_gcn.eval()
    model_gcn.gc_embeddings = model_gcn.forward().detach()

    return model_gcn

def main(args):
    # Load graph
    print("Loading graph data...")
    graph_data = load_graph(args.graph_path)
    
    # Create triples from the graph
    class_to_idx = {
        'antelope': 0, 'blue+whale': 1, 'buffalo': 2, 'chimpanzee': 3,
        'cow': 4, 'deer': 5, 'gorilla': 6, 'horse': 7, 'killer+whale': 8, 'zebra': 9
    }
    
    triples = create_triples_from_graph(graph_data, class_to_idx)
    num_classes = len(class_to_idx)
    
    # Create the target similarity matrix
    target_sim_matrix = get_target_similarity_matrix_from_graph(graph_data, class_to_idx, num_classes)

    # Train the model
    print("Training the model...")
    model_gcn = train_model(triples, target_sim_matrix, num_classes, epochs=300)

    # Save model and results
    torch.save(model_gcn.state_dict(), os.path.join(args.output_dir, 'trained_model.pth'))
    print("Model training complete. Saved model to:", os.path.join(args.output_dir, 'trained_model.pth'))

    print("Extracting and saving prototype vectors...")

    model_gcn.eval()
    with torch.no_grad():
        prototype_vectors = model_gcn.gc_embeddings

    # Define orders
    original_order = ['antelope', 'blue_whale', 'buffalo', 'chimpanzee', 'cow', 'deer', 'gorilla', 'horse', 'killer_whale', 'zebra']
    desired_order = ["horse", "cow", "deer", "gorilla", "blue+whale", "zebra", "buffalo", "antelope", "chimpanzee", "killer+whale"]

    name_to_index = {name: idx for idx, name in enumerate(original_order)}
    standardized_desired_order = [name.replace('+', '_') for name in desired_order]

    # Reorder prototypes
    reordered_prototypes = {
        i: prototype_vectors[name_to_index[class_name]].cpu()
        for i, class_name in enumerate(standardized_desired_order)
    }

    # Save as torch file
    torch.save(reordered_prototypes, os.path.join(args.output_dir, 'reordered_prototypes.pt'))
    print("Saved reordered prototype vectors to:", os.path.join(args.output_dir, 'reordered_prototypes.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the KnowledgeGraphRGCN model.")
    parser.add_argument('data_dir', type=str, help="Path to the dataset directory")
    parser.add_argument('graph_path', type=str, help="Path to the graph JSON file")
    parser.add_argument('output_dir', type=str, help="Directory to save the output")
    

    args = parser.parse_args()
    main(args)
