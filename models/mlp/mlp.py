import torch.nn as nn

class EmbedProjector(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
