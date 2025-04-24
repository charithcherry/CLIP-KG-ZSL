import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.rgcn.rgcn import KnowledgeGraphRGCN
from test_requirements import clip_requirement
from graph.graph_generator import GraphGenerator
from train_rgcn import create_triples_from_graph, get_target_similarity_matrix_from_graph, load_graph  
from utils.image_embedding_utils import get_image_embedding  

import argparse

# Check CLIP installation
clip_requirement.test_clip_installation()


graph_path = 'output/filtered_class_pairwise_weighted_graph.json'
model_path = 'output/trained_model.pth'


# Parse command line args
parser = argparse.ArgumentParser(description="Test RGCN model with an image.")
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
args = parser.parse_args()

image_path = args.image_path

print("image path", image_path)

class_to_idx = {
    'antelope': 0, 'blue+whale': 1, 'buffalo': 2, 'chimpanzee': 3,
    'cow': 4, 'deer': 5, 'gorilla': 6, 'horse': 7, 'killer+whale': 8, 'zebra': 9
}
num_classes = len(class_to_idx)

print("Loading graph...")
graph_data = load_graph(graph_path)
triples = create_triples_from_graph(graph_data, class_to_idx)
target_sim_matrix = get_target_similarity_matrix_from_graph(graph_data, class_to_idx, num_classes)

print("Loading model...")
model_gcn = KnowledgeGraphRGCN(triples)
model_gcn.load_state_dict(torch.load(model_path))
model_gcn.eval()
model_gcn.gc_embeddings = model_gcn.forward().detach()

print("Generating image embedding...")
image_embedding = get_image_embedding(image_path)

prediction = model_gcn.classify_image(image_embedding)
print("Predicted class:", prediction)

similarities = model_gcn.get_class_similarities(image_embedding)
print("Similarity scores:", similarities)
