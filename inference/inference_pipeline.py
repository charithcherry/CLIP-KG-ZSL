import os
import torch
import torch.nn.functional as F
import argparse
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp.mlp import EmbedProjector
from utils.image_embedding_utils import get_image_embedding  # You define this
from test_requirements.clip_requirement import test_clip_installation  # Optional sanity check


# Label mappings
class_map = {
    0: 'horse', 1: 'cow', 2: 'deer', 3: 'gorilla', 4: 'blue+whale',
    5: 'zebra', 6: 'buffalo', 7: 'antelope', 8: 'chimpanzee', 9: 'killer+whale'
}
label_to_id = {v: k for k, v in class_map.items()}

semantic_groups = [
    {"zebra", "horse"}, {"buffalo", "cow"},
    {"antelope", "deer"}, {"chimpanzee", "gorilla"},
    {"killer+whale", "blue+whale"}
]

def is_semantically_equivalent(true_cls, predicted_cls):
    for group in semantic_groups:
        if true_cls in group and predicted_cls in group:
            return True
    return False

def predict_from_embedding(image_embedding, model, prototype_matrix):
    model.eval()
    with torch.no_grad():
        output = model(image_embedding.unsqueeze(0))
        output = F.normalize(output, dim=1)
        proto_norm = F.normalize(prototype_matrix, dim=1)
        sim = output @ proto_norm.T
        top2_sim, top2_idx = torch.topk(sim.squeeze(), k=2)
        return (
            top2_idx[0].item(),
            class_map[top2_idx[1].item()],
            top2_sim[1].item(),
            sim.squeeze()
        )

def load_test_embeddings(embedding_dir):
    loaders = defaultdict(list)
    filename_to_class = {
        "zebra": "zebra", "antelope": "antelope",
        "buffalo": "buffalo", "chimpanzee": "chimpanzee",
        "killer_whale": "killer+whale",
    }

    for file in os.listdir(embedding_dir):
        if file.endswith(".pt"):
            key = file.replace("_embeddings.pt", "")
            if key not in filename_to_class:
                continue
            path = os.path.join(embedding_dir, file)
            data = torch.load(path)
            embeddings = data["embeddings"]
            labels = data["labels"]
            for emb, label in zip(embeddings, labels):
                loaders[filename_to_class[key]].append((emb, label))
    return loaders

def evaluate_model(model, prototype_matrix, loaders):
    total_count = 0
    top1_correct = 0
    adjusted_correct = 0

    y_true, y_pred_top1 = [], []
    per_class_stats = defaultdict(lambda: {"correct_top1": 0, "adjusted": 0, "total": 0})

    for cls, loader in loaders.items():
        print(f"\nClass: {cls}\n" + "-"*80)
        for i, (embedding, label_str) in enumerate(loader):
            true_cls = label_str
            true_id = label_to_id[true_cls]

            pred_id, second_best_label, second_best_score, sim = predict_from_embedding(
                embedding, model, prototype_matrix
            )
            pred_cls = class_map[pred_id]

            is_top1_correct = pred_cls == true_cls
            is_adjusted_correct = (
                is_top1_correct or
                second_best_label == true_cls or
                is_semantically_equivalent(true_cls, pred_cls) or
                is_semantically_equivalent(true_cls, second_best_label)
            )

            print(f"Image {i+1:02d}: True = {true_cls}, Predicted = {pred_cls}")
            print(f"           Second best: {second_best_label} (score={second_best_score:.4f})")
            print(f"           Correct (Top-2/Semantic Match): {is_adjusted_correct}")
            print("-"*60)

            y_true.append(true_id)
            y_pred_top1.append(pred_id)

            total_count += 1
            top1_correct += is_top1_correct
            adjusted_correct += is_adjusted_correct

            per_class_stats[true_cls]["total"] += 1
            per_class_stats[true_cls]["correct_top1"] += is_top1_correct
            per_class_stats[true_cls]["adjusted"] += is_adjusted_correct

    # Print per-class stats
    print("\n" + "#"*30 + " PER-CLASS ACCURACY " + "#"*30)
    for cls, stats in per_class_stats.items():
        acc = stats["correct_top1"] / stats["total"]
        adj_acc = stats["adjusted"] / stats["total"]
        print(f"{cls:<15} Top-1 Acc: {acc:.4f} | Adjusted Acc: {adj_acc:.4f}")

    # Print overall metrics
    top1_accuracy = top1_correct / total_count
    adjusted_accuracy = adjusted_correct / total_count
    precision = precision_score(y_true, y_pred_top1, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred_top1, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred_top1, average='macro', zero_division=0)

    print("\n" + "#"*30 + " OVERALL METRICS " + "#"*30)
    print(f"Top-1 Accuracy:            {top1_accuracy:.4f}")
    print(f"Adjusted Accuracy (Top-2 + Semantic): {adjusted_accuracy:.4f}")
    print(f"Macro Precision (Top-1):   {precision:.4f}")
    print(f"Macro Recall (Top-1):      {recall:.4f}")
    print(f"Macro F1 Score (Top-1):    {f1:.4f}")
    print("="*80)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to saved model checkpoint (.pt)")
    parser.add_argument("--prototype_path", type=str, required=True, help="Path to prototype matrix (.pt)")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Path to test embeddings directory")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Loading model checkpoint...")

    checkpoint = torch.load(args.checkpoint_path)
    model = EmbedProjector()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded.")

    prototypes = torch.load(args.prototype_path)
    prototype_matrix = torch.stack([prototypes[i] for i in range(10)])
    print("Prototype matrix loaded.")

    loaders = load_test_embeddings(args.embedding_dir)
    print("Test embeddings loaded.")

    evaluate_model(model, prototype_matrix, loaders)
