import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class RelationalGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.rgcn1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.rgcn2 = RGCNConv(hidden_channels, out_channels, num_relations)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_type):
        x = self.rgcn1(x, edge_index, edge_type)
        x = self.relu(x)
        x = self.rgcn2(x, edge_index, edge_type)
        return F.normalize(x, dim=-1)
    

class KnowledgeGraphRGCN(nn.Module):
    def __init__(self, triples, clip_model_name="openai/clip-vit-base-patch32", hidden_channels=256, out_channels=512):
        super().__init__()
        self.triples = triples
        self.clip_model_name = clip_model_name
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.classes, self.edge_index, self.edge_type, self.rel_to_idx = self._create_graph()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)

        self.node_features = self._generate_clip_embeddings()
        self.rgcn = RelationalGCN(in_channels=512, hidden_channels=hidden_channels,
                                  out_channels=out_channels, num_relations=len(self.rel_to_idx))
        self.dropout = nn.Dropout(p=0.5)

    def _create_graph(self):
        classes = set()
        rel_types = set()
        edges = []
        edge_type_ids = []

        for h, t, r in self.triples:
            classes.update([h, t])
            rel_types.add(r)

        classes = sorted(list(classes))
        rel_to_idx = {rel: i for i, rel in enumerate(sorted(rel_types))}
        class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for h, t, r in self.triples:
            h_idx = class_to_idx[h]
            t_idx = class_to_idx[t]
            r_idx = rel_to_idx[r]
            
            # undirected edges (both directions)
            edges.append([h_idx, t_idx])
            edges.append([t_idx, h_idx])
            edge_type_ids.append(r_idx)
            edge_type_ids.append(r_idx)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_type_ids, dtype=torch.long)

        return classes, edge_index, edge_type, rel_to_idx

    def _generate_clip_embeddings(self):
        prompts = [f"a photo of a {cls.replace('+', ' ')}" for cls in self.classes]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        return F.normalize(text_features, dim=-1)

    def classify_image(self, image_embedding):
        similarities = F.cosine_similarity(image_embedding.unsqueeze(0), self.gc_embeddings)
        best_idx = similarities.argmax().item()
        return self.classes[best_idx]

    def get_class_similarities(self, image_embedding):
        similarities = F.cosine_similarity(image_embedding.unsqueeze(0), self.gc_embeddings)
        return {cls: sim.item() for cls, sim in zip(self.classes, similarities)}, self.classes

    def forward(self):
        x = self.rgcn(self.node_features, self.edge_index, self.edge_type)
        x = self.dropout(x)
        return x

    def train_step(self, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        
        refined = self.forward()                # RGCN output (refined prototypes)
        original = self.node_features.detach()  # original CLIP text embeddings
        
        loss = criterion(refined, original)
        loss.backward()
        optimizer.step()
    
        return loss.item()