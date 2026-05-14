import torch
import torch.nn as nn


class STESelectorMLP(nn.Module):
    """Lightweight tool router that outputs per-tool logits."""

    def __init__(self, embed_dim: int = 4096, hidden_dim: int = 1024, num_tools: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_tools),
        )

    def forward(self, state_embeddings: torch.Tensor) -> torch.Tensor:
        if state_embeddings.dim() == 1:
            pooled = state_embeddings
        else:
            pooled = state_embeddings.mean(dim=0)
        return self.net(pooled.float())


class SampleBetaGater(nn.Module):
    """Sample-level gater that predicts fusion beta in [0, 1]."""

    def __init__(self, embed_dim: int = 4096, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, state_embeddings: torch.Tensor) -> torch.Tensor:
        if state_embeddings.dim() == 1:
            pooled = state_embeddings
        else:
            pooled = state_embeddings.mean(dim=0)
        return self.net(pooled.float()).view(1)

