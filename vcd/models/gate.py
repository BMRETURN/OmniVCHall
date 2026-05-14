import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryVisualFusionGater(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256):
        super().__init__()
        self.dtype = torch.float32
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),       
            nn.Sigmoid()               
        ).to(self.dtype)

    def forward(self, video_query_embedding):
        global_embeds = video_query_embedding.mean(dim=0, keepdim=True) # [1, embed_dim]
        beta = self.gate_network(global_embeds) 
        return beta