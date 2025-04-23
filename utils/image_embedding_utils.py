
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import torch.nn.functional as F


def get_image_embedding(image_path, clip_model_name="openai/clip-vit-base-patch32"):
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name).to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
    return F.normalize(image_embedding, dim=-1).squeeze(0).cpu()