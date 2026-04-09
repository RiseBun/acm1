"""定位头: Evidence-Gated Dense Localization + 渐进式解码器.

本实现采用 evidence-gated 机制:
1. Dense pathway: 直接从 patch tokens 计算异常分数，保持空间分辨率
2. Evidence gate: 从 bottleneck 的 evidence tokens 生成门控信号，调制异常分数
3. Progressive decoder: 逐级 2x 上采样，细化边界

这种设计确保定位头既利用 evidence bottleneck 的紧凑表示，
又保持足够的空间细节用于像素级定位。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceGateGenerator(nn.Module):
    """从 evidence tokens 和 cross-attention 权重生成 per-patch 门控."""

    def __init__(self, dim: int):
        super().__init__()
        self.token_weight = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(
        self, evidence_tokens: torch.Tensor, cross_attn_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            evidence_tokens: (B, K, D)
            cross_attn_weights: (B, K, N)
        Returns:
            gate: (B, N) 值域 [0, 1]
            token_contrib: (B, K) 每个 evidence token 的贡献度
        """
        w = self.token_weight(evidence_tokens).squeeze(-1)  # (B, K)
        weighted_attn = w.unsqueeze(-1) * cross_attn_weights  # (B, K, N)
        gate = torch.sigmoid(weighted_attn.sum(dim=1))  # (B, N)
        token_contrib = (w.unsqueeze(-1) * cross_attn_weights).abs().sum(dim=-1)  # (B, K)
        return gate, token_contrib


class DensePathway(nn.Module):
    """直接从 patch tokens 计算 per-patch 异常分数."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """patch_tokens: (B, N, D) -> dense_scores: (B, N)"""
        return self.scorer(patch_tokens).squeeze(-1)


class ProgressiveDecoder(nn.Module):
    """渐进式上采样解码器: 结合 patch features + 异常分数, 逐级 2x 上采样.

    解决 14x14 -> 224x224 一次性上采样导致边界模糊、AUPRO 低的问题.
    每级: Conv(降通道) -> bilinear 2x 上采样 -> Conv(细化边界)
    """

    def __init__(
        self,
        dim: int,
        grid_h: int,
        grid_w: int,
        out_size: int = 224,
        base_ch: int = 64,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.out_size = out_size

        # 将 patch tokens 降维到 base_ch
        self.token_proj = nn.Sequential(
            nn.Linear(dim, base_ch),
            nn.GELU(),
        )

        # 计算上采样级数: 14 -> 28 -> 56 -> 112 -> 224 = 4 级
        num_stages = 0
        s = min(grid_h, grid_w)
        while s < out_size:
            s *= 2
            num_stages += 1
        self._num_stages = num_stages

        # 通道规划: 逐级减半, 最低 16
        in_ch = base_ch + 1  # patch features + score map
        chs = [in_ch]
        for i in range(num_stages):
            chs.append(max(base_ch // (2 ** i), 16))

        # 各级上采样模块
        self.ups = nn.ModuleList()
        self.refines = nn.ModuleList()
        for i in range(num_stages):
            # 上采样前的通道变换
            self.ups.append(nn.Sequential(
                nn.Conv2d(chs[i], chs[i + 1], 3, padding=1),
                nn.BatchNorm2d(chs[i + 1]),
                nn.GELU(),
            ))
            # 上采样后的边界细化
            self.refines.append(nn.Sequential(
                nn.Conv2d(chs[i + 1], chs[i + 1], 3, padding=1),
                nn.BatchNorm2d(chs[i + 1]),
                nn.GELU(),
            ))

        self.head = nn.Conv2d(chs[-1], 1, 1)

    def forward(self, scores: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (B, N) per-patch 异常分数 (gated 或 raw)
            patch_tokens: (B, N, D) 原始 patch features (来自冻结 backbone)
        Returns:
            mask_logits: (B, 1, out_size, out_size)
        """
        b = scores.shape[0]

        # 投影 patch tokens: (B, N, D) -> (B, base_ch, H, W)
        feat = self.token_proj(patch_tokens)
        feat = feat.transpose(1, 2).view(b, -1, self.grid_h, self.grid_w)

        # 异常分数图: (B, 1, H, W)
        smap = scores.view(b, 1, self.grid_h, self.grid_w)

        # 拼接 patch features + 异常分数
        x = torch.cat([feat, smap], dim=1)

        # 渐进式上采样: 每级 Conv -> bilinear 2x -> Conv
        for up, refine in zip(self.ups, self.refines):
            x = up(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = refine(x)

        logits = self.head(x)

        # 确保精确输出尺寸
        if logits.shape[-2] != self.out_size or logits.shape[-1] != self.out_size:
            logits = F.interpolate(
                logits, (self.out_size, self.out_size),
                mode="bilinear", align_corners=False,
            )
        return logits


class EvidenceGatedLocalizationHead(nn.Module):
    """Evidence-Gated Dense Localization + 渐进式解码器.

    核心: dense pathway (全部 patches) 提供空间分辨率,
    evidence gate (来自 bottleneck) 控制异常判定,
    渐进式解码器逐级 2x 上采样细化边界.
    """

    def __init__(
        self,
        dim: int,
        grid_h: int,
        grid_w: int,
        out_size: int = 224,
        hidden: int = 256,
        refine_ch: int = 64,
    ):
        super().__init__()
        self.dense_pathway = DensePathway(dim, hidden)
        self.gate_generator = EvidenceGateGenerator(dim)
        self.decoder = ProgressiveDecoder(dim, grid_h, grid_w, out_size, refine_ch)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        evidence_tokens: torch.Tensor,
        cross_attn_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: (B, N, D)
            evidence_tokens: (B, K, D)
            cross_attn_weights: (B, K, N)
        Returns:
            mask_logits: (B, 1, out_size, out_size)
            token_contrib: (B, K)
        """
        dense_scores = self.dense_pathway(patch_tokens)  # (B, N)
        gate, token_contrib = self.gate_generator(evidence_tokens, cross_attn_weights)
        gated_scores = dense_scores * gate  # (B, N)
        logits = self.decoder(gated_scores, patch_tokens)
        return logits, token_contrib


class DenseLocalizationHead(nn.Module):
    """MTL-Naive / loc_only: 纯 dense 定位头 + 渐进式解码器 (无 bottleneck gate)."""

    def __init__(
        self,
        dim: int,
        grid_h: int,
        grid_w: int,
        out_size: int = 224,
        hidden: int = 256,
        refine_ch: int = 64,
    ):
        super().__init__()
        self.dense_pathway = DensePathway(dim, hidden)
        self.decoder = ProgressiveDecoder(dim, grid_h, grid_w, out_size, refine_ch)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """patch_tokens: (B, N, D) -> mask_logits: (B, 1, H, W)"""
        dense_scores = self.dense_pathway(patch_tokens)
        return self.decoder(dense_scores, patch_tokens)


# 旧名称兼容
LocalizationHeadFromBottleneck = EvidenceGatedLocalizationHead
LocalizationHeadFromPatches = DenseLocalizationHead
