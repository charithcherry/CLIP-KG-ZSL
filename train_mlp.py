import torch
import os
import argparse
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from models.mlp.mlp import EmbedProjector

# === Argparse for dynamic paths ===
parser = argparse.ArgumentParser()
parser.add_argument('save_dir', type=str, help='Path to directory containing embedding .pt files')
parser.add_argument('output_path', type=str, help='Path to directory containing prototype_matrix.pt')
args = parser.parse_args()

save_dir = args.save_dir
output_path = args.output_path
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

seen_classes = ["horse", "cow", "deer", "gorilla", "blue+whale"]

# === Load prototype matrix ===
print("Loading Prototypes")
prototype_file = os.path.join(output_path, "reordered_prototypes.pt")
prototypes = torch.load(prototype_file)
prototype_matrix = torch.stack([prototypes[i] for i in range(10)])  # shape: [10, D]

# === Load Embeddings ===
all_embeddings = []
all_labels = []
label_to_id = {name: idx for idx, name in enumerate(seen_classes)}

for class_name in seen_classes:
    file_path = os.path.join(save_dir, f"{class_name.replace('+', '_')}_embeddings.pt")
    data = torch.load(file_path, weights_only=True)

    all_embeddings.append(data["embeddings"])
    all_labels.extend([label_to_id[label] for label in data["labels"]])

X_seen = torch.cat(all_embeddings, dim=0)
y_seen = torch.tensor(all_labels)

print("Loaded embeddings and labels")
print("X_seen shape:", X_seen.shape)

# === Generate Soft Similarity Labels ===
with torch.no_grad():
    X_seen_norm = F.normalize(X_seen, dim=1)
    proto_norm = F.normalize(prototype_matrix, dim=1)
    full_similarity_targets = X_seen_norm @ proto_norm.T

dataset = TensorDataset(X_seen, full_similarity_targets)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# === Model + Training ===
model = EmbedProjector()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0.0

    for x_batch, y_sim_batch in loader:
        output = model(x_batch)
        output = F.normalize(output, dim=1)
        sims = output @ proto_norm.T

        loss = F.mse_loss(sims, y_sim_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss
    }, checkpoint_path)

    print(f"Saved checkpoint to {checkpoint_path}")

# === Save final model after training ===
final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}, final_model_path)
print(f"Final model saved to {final_model_path}")


    
