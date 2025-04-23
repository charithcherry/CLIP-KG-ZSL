import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import gc
from torchvision import transforms
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import argparse



import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from dataloaders.train_loader import SingleClassDataset
from test_requirements.clip_requirement import test_clip_installation

test_clip_installation()

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_image_embedding(image, clip_model_name="openai/clip-vit-base-patch32"):
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    model.eval()

    # If it's a path, load the image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)

    return F.normalize(image_embedding, dim=-1).squeeze(0).cpu()

def pil_collate_fn(batch):
    return batch[0]  # batch size is 1



def main(root_path):
    seen_classes = ["horse", "cow", "deer", "gorilla", "blue+whale"]
    save_dir = "./class_wise_embeddings"
    os.makedirs(save_dir, exist_ok=True)

    for class_name in seen_classes:
        print(f"Processing class: {class_name}")
        dataset = SingleClassDataset(root_path, class_name)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=pil_collate_fn)

        class_embeddings = []
        class_labels = []

        counter = 0
        checkpoint = 50

        for img, label in dataloader:
            image = img
            label = label
            if counter % checkpoint == 0:
                print("checkpoint ", counter)
            counter += 1
            try:
                embedding = get_image_embedding(image)
                class_embeddings.append(embedding)
                class_labels.append(label)
            except Exception as e:
                print(f"Skipping image due to error: {e}")
                continue

        class_embeddings = torch.stack(class_embeddings)
        torch.save({
            "embeddings": class_embeddings,
            "labels": class_labels
        }, os.path.join(save_dir, f"{class_name.replace('+', '_')}_embeddings.pt"))

        print(f"Saved embeddings for class: {class_name} ({len(class_labels)} samples)")

        del dataset, dataloader, class_embeddings, class_labels
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLIP image embeddings for animal classes.")
    parser.add_argument("root_path", type=str, help="Path to the root JPEGImages directory.")
    args = parser.parse_args()
    main(args.root_path)
