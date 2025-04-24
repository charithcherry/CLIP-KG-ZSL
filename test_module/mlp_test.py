import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import torch
import torch.nn.functional as F
import argparse
from models.mlp.mlp import EmbedProjector
from utils.image_embedding_utils import get_image_embedding  
from test_requirements.clip_requirement import test_clip_installation


test_clip_installation()


def predict_class_from_image(image_path, model, prototype_matrix):
    model.eval()
    image_embedding = get_image_embedding(image_path).unsqueeze(0)  # shape: (1, 512)

    with torch.no_grad():
        output = model(image_embedding)
        output = F.normalize(output, dim=1)

        proto_norm = F.normalize(prototype_matrix, dim=1)
        sim = output @ proto_norm.T  # shape: (1, 10)
        print("Similarity scores:", sim.squeeze().tolist())
        pred = sim.argmax(dim=1).item()

    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, help="Path to final_model.pt or any epoch checkpoint")
    parser.add_argument("prototype_path", type=str, help="Path to reordered_prototypes.pt")
    parser.add_argument("image_path", type=str, help="Path to the test image")
    args = parser.parse_args()

    # === Load model ===
    model = EmbedProjector()
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded model from checkpoint")

    # === Load prototypes ===
    prototypes = torch.load(args.prototype_path)
    prototype_matrix = torch.stack([prototypes[i] for i in range(10)])
    print("Loaded prototype matrix")

    # === Predict class ===
    class_names = ["horse", "cow", "deer", "gorilla", "blue+whale",
                   "zebra", "buffalo", "antelope", "chimpanzee", "killer+whale"]
    pred_class_id = predict_class_from_image(args.image_path, model, prototype_matrix)
    print("Predicted Class ID:", pred_class_id)
    print("Predicted Class Name:", class_names[pred_class_id])
