import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from torch_geometric.nn import GCNConv
from PIL import Image


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return F.normalize(x, dim=-1)



class KnowledgeGraphGCN(nn.Module):
    def __init__(self, kg, clip_model_name="openai/clip-vit-base-patch32", hidden_channels=256, out_channels=512):
        super().__init__()
        self.kg = kg
        self.clip_model_name = clip_model_name
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.classes, self.edge_index, self.edge_weight = self._create_graph()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)

        self.node_features = self._generate_clip_embeddings()
        self.gcn = GCNEncoder(in_channels=512, hidden_channels=self.hidden_channels, out_channels=self.out_channels)
        self.dropout = nn.Dropout(p=0.5)

    def _create_graph(self):
        all_classes = set()
        edge_list = []
        edge_weights = []

        for edge in self.kg:
            src, tgt = edge["source"], edge["target"]
            all_classes.update([src, tgt])

        all_classes = sorted(list(all_classes))
        class_to_idx = {cls: i for i, cls in enumerate(all_classes)}

        for edge in self.kg:
            src_idx = class_to_idx[edge["source"]]
            tgt_idx = class_to_idx[edge["target"]]
            weight = edge["weight"]

            # Add both directions since GCN assumes undirected graphs
            edge_list.append((src_idx, tgt_idx))
            edge_list.append((tgt_idx, src_idx))
            edge_weights.append(weight)
            edge_weights.append(weight)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)

        return all_classes, edge_index, edge_weight

    def _generate_clip_embeddings(self):
        prompts = [f"a close-up photo of a {cls.replace('+', ' ')}" for cls in self.classes]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        return F.normalize(text_features, dim=-1)

    def classify_image(self, image_embedding):
        similarities = F.cosine_similarity(image_embedding.unsqueeze(0), self.gc_embeddings)
        best_idx = similarities.argmax().item()
        return self.classes[best_idx]

    def forward(self):
        x = self.gcn(self.node_features, self.edge_index, self.edge_weight)
        x = self.dropout(x)
        return x

    def train_step(self, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        out = self.forward()
        loss = criterion(out, self.node_features)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_class_similarities(self, image_embedding):
        similarities = F.cosine_similarity(image_embedding.unsqueeze(0), self.gc_embeddings)
        return {cls: sim.item() for cls, sim in zip(self.classes, similarities)}