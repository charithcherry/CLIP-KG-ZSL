import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import os

from test_requirements import clip_requirement

clip_requirement()

def generate_clip_embeddings(classes, clip_model_name="openai/clip-vit-base-patch32", device=None):
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    prompts = [f"a photo of a {cls.replace('+', ' ')}" for cls in classes]
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    return F.normalize(text_features, dim=-1)

def save_reordered_prototypes(prototype_vectors, original_order, desired_order, output_path):
    name_to_index = {name: idx for idx, name in enumerate(original_order)}
    standardized_desired_order = [name.replace('+', '_') for name in desired_order]
    prototypes = {
        i: prototype_vectors[name_to_index[class_name]]
        for i, class_name in enumerate(standardized_desired_order)
    }

    torch.save(prototypes, output_path)
    print(f"Saved reordered prototypes to {output_path}")

clip_model_name = "openai/clip-vit-base-patch32"
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

original_order = ['antelope', 'blue_whale', 'buffalo', 'chimpanzee', 'cow', 'deer', 'gorilla', 'horse', 'killer_whale', 'zebra']
desired_order = ["horse", "cow", "deer", "gorilla", "blue+whale", "zebra", "buffalo", "antelope", "chimpanzee", "killer+whale"]

prototype_vectors = generate_clip_embeddings(original_order)
save_reordered_prototypes(
    prototype_vectors,
    original_order=[name.replace('_', ' ') for name in original_order],
    desired_order=desired_order,
    output_path=os.path.join(output_dir, 'clip_prototypes.pt')
)
