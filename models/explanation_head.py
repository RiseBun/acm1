"""解释头: 2层 Transformer-based 解释头，在规范模板空间预测缺陷描述."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerExplanationLayer(nn.Module):
    """单层 Transformer decoder: Self-Attn -> Cross-Attn -> FFN."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_sa = nn.LayerNorm(dim)
        
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ca = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(dim)
    
    def forward(
        self, queries: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, L, D) 查询 tokens
            memory: (B, K, D) evidence tokens 作为 memory
        Returns:
            out: (B, L, D)
        """
        # Self-attention
        sa_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm_sa(queries + sa_out)
        
        # Cross-attention (queries attend to evidence tokens)
        ca_out, _ = self.cross_attn(queries, memory, memory)
        queries = self.norm_ca(queries + ca_out)
        
        # FFN
        ff_out = self.ffn(queries)
        queries = self.norm_ff(queries + ff_out)
        
        return queries


class StructuredExplanationHead(nn.Module):
    """2层 Transformer-based 解释头.
    
    使用 evidence tokens 作为 memory，通过轻量级 Transformer 
    预测结构化的缺陷描述 (presence, defect_type, location)。
    """

    def __init__(
        self,
        dim: int,
        num_presence: int = 2,
        num_defect_type: int = 6,
        num_location: int = 4,
        hidden: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        
        # 可学习的查询 token (单个 token 用于池化 evidence)
        self.query_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # 投影层: dim -> hidden
        self.input_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
        )
        
        # 2层 Transformer decoder
        self.transformer_layers = nn.ModuleList([
            TransformerExplanationLayer(hidden, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden)
        
        # 三个并行分类头
        self.presence_head = nn.Linear(hidden, num_presence)
        self.defect_type_head = nn.Linear(hidden, num_defect_type)
        self.location_head = nn.Linear(hidden, num_location)

    def forward(self, evidence_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """基于 evidence tokens 预测结构化解释.
        
        Args:
            evidence_tokens: (B, K, D) evidence tokens from bottleneck
        Returns:
            presence_logits: (B, num_presence)
            defect_type_logits: (B, num_defect_type)
            location_logits: (B, num_location)
        """
        b = evidence_tokens.shape[0]
        
        # 扩展查询 token
        query = self.query_token.expand(b, -1, -1)  # (B, 1, D)
        
        # 投影到 hidden dim
        query = self.input_proj(query)  # (B, 1, hidden)
        memory = self.input_proj(evidence_tokens)  # (B, K, hidden)
        
        # 通过 Transformer layers
        for layer in self.transformer_layers:
            query = layer(query, memory)
        
        # 最终归一化
        query = self.final_norm(query)  # (B, 1, hidden)
        
        # 池化 (只有一个 token，直接取)
        pooled = query.squeeze(1)  # (B, hidden)
        
        # 预测
        presence_logits = self.presence_head(pooled)
        defect_type_logits = self.defect_type_head(pooled)
        location_logits = self.location_head(pooled)
        
        return presence_logits, defect_type_logits, location_logits
