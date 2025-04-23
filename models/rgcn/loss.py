import torch.nn as nn
import torch.nn.functional as F


class PrototypeRefinementLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.75,target_sim_matrix = None):
        super().__init__()
        self.alpha = alpha  # alignment
        self.beta = beta    # separation
        self.target_sim_matrix = target_sim_matrix

    def forward(self, refined, original):
        """
        refined: [num_classes, 512] refined prototypes
        original: [num_classes, 512] original prototypes before GCN
        edges: list of edge dicts representing relationships between classes
        class_to_idx: dictionary mapping class names to indices
        num_classes: total number of classes
        """
        # 1. Alignment loss: cosine similarity between refined and original prototypes
        alignment_loss = 1 - F.cosine_similarity(refined, original).mean()

        # 2. Separation loss: encourage dissimilar classes to be far apart
        sim_matrix_refined = F.cosine_similarity(refined.unsqueeze(1), refined.unsqueeze(0), dim=-1)
        separation_loss = F.mse_loss(sim_matrix_refined, self.target_sim_matrix)

        # Final loss: encourage alignment, discourage inter-prototype similarity
        total_loss = self.alpha * alignment_loss + self.beta * separation_loss
        return total_loss


