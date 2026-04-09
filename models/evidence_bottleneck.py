"""增强的证据瓶颈模块: K=16 learned queries, 2-layer cross-attention, 可学习位置编码."""

from __future__ import annotations

import torch
import torch.nn as nn


class EvidenceBottleneckLayer(nn.Module):
    """单层: Self-Attn -> Cross-Attn -> FFN，带残差连接和 LayerNorm."""

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
        self, queries: torch.Tensor, kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: (B, K, D)
            kv: (B, N, D) patch tokens
        Returns:
            out: (B, K, D)
            attn_weights: (B, K, N) 本层 cross-attention 权重
        """
        # Self-attention (queries 之间交互)
        sa_out, _ = self.self_attn(queries, queries, queries)
        queries = self.norm_sa(queries + sa_out)

        # Cross-attention (queries attend to patches)
        ca_out, attn = self.cross_attn(
            queries, kv, kv, need_weights=True, average_attn_weights=True
        )
        queries = self.norm_ca(queries + ca_out)

        # FFN
        ff_out = self.ffn(queries)
        queries = self.norm_ff(queries + ff_out)

        return queries, attn


class EvidenceBottleneck(nn.Module):
    """K 个可学习 query tokens 通过 2 层 cross-attention 压缩 N 个 patch tokens.

    输出 evidence tokens 和最终层的 cross-attention 权重.
    """

    def __init__(
        self,
        dim: int,
        K: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.dim = dim

        # 可学习 query tokens + 位置编码
        self.queries = nn.Parameter(torch.randn(1, K, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, K, dim) * 0.02)

        # 多层 Transformer decoder
        self.layers = nn.ModuleList(
            [EvidenceBottleneckLayer(dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, N, D)
        Returns:
            bottleneck: (B, K, D) 压缩后的 evidence tokens
            attn_weights: (B, K, N) 最终层 cross-attention 权重
        """
        b = patch_tokens.shape[0]
        q = self.queries.expand(b, -1, -1) + self.pos_embed.expand(b, -1, -1)

        attn = None
        for layer in self.layers:
            q, attn = layer(q, patch_tokens)

        q = self.final_norm(q)
        return q, attn
