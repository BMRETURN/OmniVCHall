import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


def pad_to_max_len(seqs: List[torch.Tensor], pad_value: float = 0.0, Lmax: int = 2000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    把一组变长序列 pad 到同一长度，并返回 key_padding_mask。
    输入:
        seqs: List of [Li, D]
    返回:
        padded: [B, Lmax, D]
        key_padding_mask: [B, Lmax] (bool)  True=padding(忽略), False=valid
    注意: 序列长度超过Lmax的会被截断
    """
    assert len(seqs) > 0
    device = seqs[0].device
    dtype = seqs[0].dtype
    D = seqs[0].shape[-1]
    if Lmax is None:
        Lmax = max(x.shape[0] for x in seqs)

    padded = torch.full((len(seqs), Lmax, D), pad_value, device=device, dtype=dtype)
    key_padding_mask = torch.ones((len(seqs), Lmax), device=device, dtype=torch.bool)  # True=padding

    for i, x in enumerate(seqs):
        Li = x.shape[0]
        # 如果序列长度超过Lmax，则截断；否则按原长度处理
        actual_len = min(Li, Lmax)
        padded[i, :actual_len, :] = x[:actual_len, :]
        key_padding_mask[i, :actual_len] = False  # valid

    return padded, key_padding_mask


class QFormerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)

        hidden_ffn = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ffn, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,                  # [B, M, D]
        memory: torch.Tensor,                 # [B, S, D]
        memory_key_padding_mask: torch.Tensor  # [B, S] True=padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        query = self.ln1(query + attn_out)

        ffn_out = self.ffn(query)
        query = self.ln2(query + ffn_out)

        return query, attn_weights


class QFormerToolRouter(nn.Module):
    """
    Stage A: queries -> (video+query) memory 进行条件化
    Stage B: conditioned queries -> tool memory 做路由与打分
    """
    def __init__(
        self,
        num_tools: int = 8,
        d_in: int = 4096,
        d_model: int = 1024,
        n_heads: int = 8,
        n_query_tokens: int = 16,
        n_cond_blocks: int = 2,
        n_tool_blocks: int = 2,
        dropout: float = 0.1,
        ffn_ratio: float = 4.0,
        use_l2norm: bool = True,
        threshold: float = 0.4,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        super().__init__()
        self.use_l2norm = use_l2norm
        self.num_tools = num_tools
        self.d_in = d_in
        self.d_model = d_model
        self.n_query_tokens = n_query_tokens
        self.device = device
        self.threshold = threshold
        self.dtype = torch.float32

        self.proj_in = nn.Linear(d_in, d_model).to(self.device).to(self.dtype)

        self.query_tokens = nn.Parameter(torch.randn(1, n_query_tokens, d_model, dtype=self.dtype) * 0.02).to(self.device)

        self.type_videoquery = nn.Parameter(torch.zeros(1, 1, d_model, dtype=self.dtype)).to(self.device)
        self.type_tool = nn.Parameter(torch.zeros(1, 1, d_model, dtype=self.dtype)).to(self.device)
        self.tool_id_emb = nn.Embedding(num_tools, d_model, dtype=self.dtype).to(self.device)

        self.cond_blocks = nn.ModuleList([
            QFormerBlock(d_model, n_heads, ffn_ratio=ffn_ratio, dropout=dropout).to(self.device).to(self.dtype)
            for _ in range(n_cond_blocks)
        ])

        self.tool_blocks = nn.ModuleList([
            QFormerBlock(d_model, n_heads, ffn_ratio=ffn_ratio, dropout=dropout).to(self.device).to(self.dtype)
            for _ in range(n_tool_blocks)
        ])

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        ).to(self.device).to(self.dtype)

    def _maybe_l2norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1) if self.use_l2norm else x
    
    def select_tools_by_threshold(self, tool_names: List[str], probs: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """
        根据阈值选择工具，按概率从高到低排序，选择累积概率超过阈值的工具
        Args:
            tool_names: 工具名称列表
            probs: 工具概率张量 [num_tools]
            threshold: 选择工具的累积概率阈值
        Returns:
            selected_names: 选中的工具名称列表
            selected_probs: 选中的工具概率张量
        """
        # 获取排序后的索引（按概率从高到低）
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_probs = probs[sorted_indices]
        
        # 计算累积概率并找到需要选择的工具数量
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)
        selected_indices = torch.nonzero(cumsum_probs <= self.threshold, as_tuple=True)[0]
        
        # 至少选择一个工具，即使最高概率的工具概率值已经超过阈值
        if len(selected_indices) == 0:
            selected_indices = torch.tensor([0], dtype=torch.long, device=probs.device)
        elif cumsum_probs[selected_indices[-1]] < self.threshold and len(selected_indices) < len(sorted_indices):
            # 如果当前选中的工具累积概率未达到阈值，则添加下一个工具
            next_idx = selected_indices[-1] + 1
            if next_idx < len(sorted_indices):
                selected_indices = torch.cat([selected_indices, next_idx.unsqueeze(0)])

        # 获取选中的工具索引（在原始顺序中）
        original_selected_indices = sorted_indices[selected_indices]
        
        # 获取选中的工具名称和概率
        selected_names = [tool_names[i] for i in original_selected_indices.cpu().numpy()]
        selected_probs = probs[original_selected_indices]
        
        return selected_names, selected_probs

    def forward(
        self,
        video_query_embedding: torch.Tensor,       # [Lvq, 4096]
        tool_desc_dict: Dict[str, torch.Tensor],   # each: [Li, 4096]
    ):
        device = video_query_embedding.device

        tool_names = list(tool_desc_dict.keys())
        if len(tool_names) != self.num_tools:
            raise ValueError(f"Expected num_tools={self.num_tools}, but got {len(tool_names)} tool embeddings.")

        vq, vq_kpm = pad_to_max_len([video_query_embedding], pad_value=0.0, Lmax=1000)
        vq = self.proj_in(vq)  # [1, Lmax, Dm]
        vq = vq + self.type_videoquery
        vq = self._maybe_l2norm(vq)

        tool_seqs = [tool_desc_dict[name].to(device) for name in tool_names]  # list of [Li, 4096]
        tool_padded, tool_kpm = pad_to_max_len(tool_seqs, pad_value=0.0, Lmax=800)      # [8, Lmax, 4096], [8, Lmax]
        tool_padded = self.proj_in(tool_padded)
        tool_padded = tool_padded + self.type_tool                                # [8, Lmax, Dm]

        tool_ids = torch.arange(self.num_tools, device=device, dtype=torch.long)  # [8]
        tool_id_vec = self.tool_id_emb(tool_ids).unsqueeze(1)                    # [8, 1, Dm]
        tool_padded = tool_padded + tool_id_vec
        tool_padded = self._maybe_l2norm(tool_padded)

        q = self._maybe_l2norm(self.query_tokens)

        for blk in self.cond_blocks:
            q, _ = blk(q, vq, vq_kpm)

        router_q = q  # [1, M, Dm]
        router_q_8 = router_q.expand(self.num_tools, self.n_query_tokens, self.d_model)  # [8, M, Dm]

        for blk in self.tool_blocks:
            router_q_8, _ = blk(router_q_8, tool_padded, tool_kpm)

        tool_repr = router_q_8.mean(dim=1)        # [8, Dm]
        tool_repr = self._maybe_l2norm(tool_repr)

        logits = self.score_head(tool_repr).squeeze(-1)  # [8]
        return logits
    
        # probs = torch.softmax(logits, dim=-1)

        # selected_tools, selected_probs = self.select_tools_by_threshold(tool_names, probs)
        # return selected_tools



# # ------------------------ 用法示例 ------------------------
# if __name__ == "__main__":
#     torch.manual_seed(2025)

#     device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     video_query_embedding = torch.randn(137, 4096).to(device)

#     tool_desc_dict = {
#         f"tool_{i}": torch.randn(torch.randint(low=20, high=80, size=(1,)).item(), 4096).to(device)
#         for i in range(8)
#     }

#     router = QFormerToolRouter(
#         num_tools=8, d_in=4096, d_model=1024,
#         n_heads=8, n_query_tokens=16,
#         n_cond_blocks=2, n_tool_blocks=2,
#         use_l2norm=True,
#         device=device
#     )

#     selected_tools = router(video_query_embedding, tool_desc_dict)
#     print(selected_tools)