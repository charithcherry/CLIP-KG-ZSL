import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import gc
from torchvision import transforms
import torch.nn.functional as F

class SingleClassDataset(Dataset):
    def __init__(self, root_dir, class_name, max_samples=250):
        self.samples = []
        class_dir = os.path.join(root_dir, class_name)
        images = [
            os.path.join(class_dir, img_name)
            for img_name in os.listdir(class_dir)
            if img_name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.originals = images[:max_samples]
        self.augment_needed = max(0, max_samples - len(self.originals))
        self.class_name = class_name

    def __len__(self):
        return 250

    def __getitem__(self, idx):
        if idx < len(self.originals):
            img_path = self.originals[idx]
            image = Image.open(img_path).convert("RGB")
            image = base_transform(image)
        else:
            aug_idx = idx % len(self.originals)
            img_path = self.originals[aug_idx]
            image = Image.open(img_path).convert("RGB")
            image = augment_transform(image)

        return image, self.class_name

# Augmentations and resizing
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
])
base_transform = transforms.Compose([
    transforms.Resize((224, 224))
])