"""完整模型: 集成 CLIP backbone, Evidence Bottleneck, Gated Localization, Explanation Head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.clip_backbone import FrozenCLIPPatchEncoder
from models.evidence_bottleneck import EvidenceBottleneck
from models.explanation_head import StructuredExplanationHead
from models.localization_head import DenseLocalizationHead, EvidenceGatedLocalizationHead


@dataclass
class ModelOutput:
    mask_logits: torch.Tensor
    presence_logits: torch.Tensor | None
    defect_type_logits: torch.Tensor | None
    location_logits: torch.Tensor | None
    mask_logits_perturbed: torch.Tensor | None
    presence_logits_p: torch.Tensor | None
    defect_type_logits_p: torch.Tensor | None
    location_logits_p: torch.Tensor | None
    token_contrib: torch.Tensor | None


class GroundedDefectModel(nn.Module):
    """多模态一致性学习的 Grounded 缺陷解释模型.

    变体支持:
    - "ours": Evidence Bottleneck + Gated Dense Localization + Explanation Head + Consistency
    - "mtl_naive": Dense Localization + Explanation Head (无 bottleneck, 无 gate)
    - "loc_only": Evidence Bottleneck + Gated Dense Localization (无 explanation)
    - "w/o_shared_bottleneck": 两个独立的 bottleneck (消融)
    - "w/o_consistency": 共享 bottleneck 但无一致性损失 (消融)
    - "random_masking": 共享 bottleneck + 随机 token 掩码 (消融)
    - "separate_features": 无 bottleneck, 直接使用独立投影的 patch features (消融)
    """

    def __init__(
        self,
        variant: str,
        clip_model: str,
        clip_pretrained: str,
        K: int,
        r_suppress: int,
        image_size: int,
        loc_hidden: int = 256,
        loc_refine_ch: int = 64,
        num_presence: int = 2,
        num_defect_type: int = 6,
        num_location: int = 4,
        suppress_alpha: float = 0.1,
    ):
        super().__init__()
        # 扩展变体支持
        ablation_variants = (
            "ours", "mtl_naive", "loc_only",
            "w/o_shared_bottleneck", "w/o_consistency",
            "random_masking", "separate_features"
        )
        assert variant in ablation_variants, f"Unknown variant: {variant}. Must be one of {ablation_variants}"
        self.variant = variant
        self.r_suppress = r_suppress
        self.image_size = image_size

        # 冻结 CLIP backbone
        self.backbone = FrozenCLIPPatchEncoder(clip_model, clip_pretrained)
        d = self.backbone.patch_dim
        gh, gw = self.backbone.grid_h, self.backbone.grid_w

        # 软扰动缩放因子 (可学习)
        self.suppress_alpha = nn.Parameter(torch.tensor(suppress_alpha))

        # 根据变体构建不同的架构
        if variant == "separate_features":
            # 消融: 无 bottleneck, 两个分支使用独立的投影 patch features
            self.bottleneck = None
            self.bottleneck_loc = None
            self.bottleneck_txt = None
            self.loc_head = DenseLocalizationHead(
                d, gh, gw,
                out_size=image_size,
                hidden=loc_hidden,
                refine_ch=loc_refine_ch,
            )
            # 为文本分支添加独立的投影层
            self.txt_feature_proj = nn.Sequential(
                nn.Linear(d, d),
                nn.LayerNorm(d),
                nn.GELU(),
            )
        elif variant == "w/o_shared_bottleneck":
            # 消融: 两个独立的 bottleneck
            self.bottleneck = None  # 不使用共享 bottleneck
            self.bottleneck_loc = EvidenceBottleneck(d, K=K)
            self.bottleneck_txt = EvidenceBottleneck(d, K=K)
            self.loc_head = EvidenceGatedLocalizationHead(
                d, gh, gw,
                out_size=image_size,
                hidden=loc_hidden,
                refine_ch=loc_refine_ch,
            )
        elif variant in ("ours", "w/o_consistency", "random_masking"):
            # 共享 bottleneck
            self.bottleneck = EvidenceBottleneck(d, K=K)
            self.bottleneck_loc = None
            self.bottleneck_txt = None
            self.loc_head = EvidenceGatedLocalizationHead(
                d, gh, gw,
                out_size=image_size,
                hidden=loc_hidden,
                refine_ch=loc_refine_ch,
            )
        else:
            # mtl_naive, loc_only: 无 bottleneck
            self.bottleneck = None
            self.bottleneck_loc = None
            self.bottleneck_txt = None
            self.loc_head = DenseLocalizationHead(
                d, gh, gw,
                out_size=image_size,
                hidden=loc_hidden,
                refine_ch=loc_refine_ch,
            )

        if variant != "loc_only":
            self.exp_head = StructuredExplanationHead(
                d, num_presence, num_defect_type, num_location, hidden=loc_hidden
            )
        else:
            self.exp_head = None

    def forward(self, images: torch.Tensor, apply_perturbation: bool = False) -> ModelOutput:
        """前向传播.

        Args:
            images: (B, 3, H, W)
            apply_perturbation: 是否执行扰动 (用于一致性损失)
        """
        patches, _ = self.backbone(images)  # (B, N, D)

        if self.variant == "ours":
            return self._forward_ours(patches, apply_perturbation)
        elif self.variant == "w/o_consistency":
            # 与 ours 相同架构，但训练时不使用一致性损失
            return self._forward_ours(patches, apply_perturbation=False)
        elif self.variant == "random_masking":
            # 与 ours 相同，但扰动时使用随机掩码
            return self._forward_ours(patches, apply_perturbation, use_random_masking=True)
        elif self.variant == "w/o_shared_bottleneck":
            return self._forward_separate_bottlenecks(patches, apply_perturbation)
        elif self.variant == "separate_features":
            return self._forward_separate_features(patches, apply_perturbation)
        else:
            # mtl_naive, loc_only
            return self._forward_mtl_naive(patches, apply_perturbation)

    def _forward_ours(
        self,
        patches: torch.Tensor,
        apply_perturbation: bool,
        use_random_masking: bool = False,
    ) -> ModelOutput:
        """ours / w/o_consistency / random_masking 的前向传播."""
        # Evidence Bottleneck
        evidence_tokens, attn_weights = self.bottleneck(patches)

        # Evidence-Gated Dense Localization
        mask_logits, token_contrib = self.loc_head(patches, evidence_tokens, attn_weights)

        if self.exp_head is None:
            # loc_only 模式: 无解释头
            return ModelOutput(
                mask_logits=mask_logits,
                presence_logits=None,
                defect_type_logits=None,
                location_logits=None,
                mask_logits_perturbed=None,
                presence_logits_p=None,
                defect_type_logits_p=None,
                location_logits_p=None,
                token_contrib=token_contrib,
            )

        # Explanation Head (基于 evidence tokens)
        pl, tl, ll = self.exp_head(evidence_tokens)

        # 扰动路径 (用于一致性损失)
        mp, plp, tlp, llp = None, None, None, None
        if apply_perturbation and self.r_suppress > 0:
            if use_random_masking:
                # 消融: 随机 token 掩码而非基于重要性的抑制
                evidence_perturbed = self._random_mask(evidence_tokens)
            else:
                # 正常: 基于重要性抑制 top-r tokens
                evidence_perturbed = self._soft_suppress(evidence_tokens, token_contrib)
            # 用扰动后的 evidence tokens 重新计算定位和解释
            mp, _ = self.loc_head(patches, evidence_perturbed, attn_weights)
            plp, tlp, llp = self.exp_head(evidence_perturbed)

        return ModelOutput(
            mask_logits=mask_logits,
            presence_logits=pl,
            defect_type_logits=tl,
            location_logits=ll,
            mask_logits_perturbed=mp,
            presence_logits_p=plp,
            defect_type_logits_p=tlp,
            location_logits_p=llp,
            token_contrib=token_contrib,
        )

    def _forward_mtl_naive(
        self,
        patches: torch.Tensor,
        apply_perturbation: bool,
    ) -> ModelOutput:
        """mtl_naive 的前向传播: 无 bottleneck, 无 gate."""
        mask_logits = self.loc_head(patches)
        pl, tl, ll = self.exp_head(patches)

        mp, plp, tlp, llp = None, None, None, None
        if apply_perturbation and self.r_suppress > 0:
            # 基于 patch importance 做扰动
            contrib = self._patch_importance(mask_logits, patches)
            patches_p = self._soft_suppress_seq(patches, contrib)
            mp = self.loc_head(patches_p)
            plp, tlp, llp = self.exp_head(patches_p)

        return ModelOutput(
            mask_logits=mask_logits,
            presence_logits=pl,
            defect_type_logits=tl,
            location_logits=ll,
            mask_logits_perturbed=mp,
            presence_logits_p=plp,
            defect_type_logits_p=tlp,
            location_logits_p=llp,
            token_contrib=None,
        )

    def _random_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """消融实验: 随机掩码 token 而非基于重要性."""
        b, k, _d = tokens.shape
        r = min(self.r_suppress, k)
        alpha = torch.clamp(self.suppress_alpha, 0.0, 1.0)

        out = tokens.clone()
        # 随机选择 r 个 token
        indices = torch.randperm(k, device=tokens.device)[:r]
        mask = torch.ones(b, k, 1, device=tokens.device)
        mask[:, indices, :] = alpha
        return out * mask

    def _forward_separate_bottlenecks(
        self,
        patches: torch.Tensor,
        apply_perturbation: bool,
    ) -> ModelOutput:
        """消融: w/o shared bottleneck - 两个独立的 bottleneck."""
        # 两个独立的 bottleneck
        evidence_tokens_loc, attn_weights_loc = self.bottleneck_loc(patches)
        evidence_tokens_txt, _ = self.bottleneck_txt(patches)

        # 定位头使用自己的 bottleneck
        mask_logits, token_contrib = self.loc_head(patches, evidence_tokens_loc, attn_weights_loc)

        if self.exp_head is None:
            return ModelOutput(
                mask_logits=mask_logits,
                presence_logits=None,
                defect_type_logits=None,
                location_logits=None,
                mask_logits_perturbed=None,
                presence_logits_p=None,
                defect_type_logits_p=None,
                location_logits_p=None,
                token_contrib=token_contrib,
            )

        # 解释头使用自己的 bottleneck
        pl, tl, ll = self.exp_head(evidence_tokens_txt)

        # 扰动路径 (扰动定位分支的 bottleneck)
        mp, plp, tlp, llp = None, None, None, None
        if apply_perturbation and self.r_suppress > 0:
            evidence_loc_perturbed = self._soft_suppress(evidence_tokens_loc, token_contrib)
            mp, _ = self.loc_head(patches, evidence_loc_perturbed, attn_weights_loc)
            # 注意: 解释头使用未扰动的 bottleneck (因为它们不共享)
            plp, tlp, llp = self.exp_head(evidence_tokens_txt)

        return ModelOutput(
            mask_logits=mask_logits,
            presence_logits=pl,
            defect_type_logits=tl,
            location_logits=ll,
            mask_logits_perturbed=mp,
            presence_logits_p=plp,
            defect_type_logits_p=tlp,
            location_logits_p=llp,
            token_contrib=token_contrib,
        )

    def _forward_separate_features(
        self,
        patches: torch.Tensor,
        apply_perturbation: bool,
    ) -> ModelOutput:
        """消融: separate features - 无 bottleneck, 直接使用独立投影的 patch features."""
        # 定位头直接使用 patch features
        mask_logits = self.loc_head(patches)

        if self.exp_head is None:
            return ModelOutput(
                mask_logits=mask_logits,
                presence_logits=None,
                defect_type_logits=None,
                location_logits=None,
                mask_logits_perturbed=None,
                presence_logits_p=None,
                defect_type_logits_p=None,
                location_logits_p=None,
                token_contrib=None,
            )

        # 解释头使用独立投影的 patch features
        txt_features = self.txt_feature_proj(patches)
        pl, tl, ll = self.exp_head(txt_features)

        # 扰动路径 (基于 patch importance)
        mp, plp, tlp, llp = None, None, None, None
        if apply_perturbation and self.r_suppress > 0:
            contrib = self._patch_importance(mask_logits, patches)
            patches_p = self._soft_suppress_seq(patches, contrib)
            mp = self.loc_head(patches_p)
            txt_features_p = self.txt_feature_proj(patches_p)
            plp, tlp, llp = self.exp_head(txt_features_p)

        return ModelOutput(
            mask_logits=mask_logits,
            presence_logits=pl,
            defect_type_logits=tl,
            location_logits=ll,
            mask_logits_perturbed=mp,
            presence_logits_p=plp,
            defect_type_logits_p=tlp,
            location_logits_p=llp,
            token_contrib=None,
        )

    def _soft_suppress(self, tokens: torch.Tensor, token_contrib: torch.Tensor) -> torch.Tensor:
        """软扰动: 将 top-r evidence tokens 缩放至 suppress_alpha 而非清零.

        Args:
            tokens: (B, K, D) evidence tokens
            token_contrib: (B, K) 每个 token 的贡献度
        Returns:
            perturbed tokens: (B, K, D)
        """
        b, k, _d = tokens.shape
        r = min(self.r_suppress, k)
        alpha = torch.clamp(self.suppress_alpha, 0.0, 1.0)

        out = tokens.clone()
        topi = token_contrib.topk(r, dim=1).indices  # (B, r)
        mask = torch.ones(b, k, 1, device=tokens.device)
        for i in range(b):
            mask[i, topi[i], :] = alpha
        return out * mask

    def _patch_importance(self, mask_logits: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        """MTL-Naive 的 patch 重要性估计."""
        b, n, _d = patch_tokens.shape
        gh, gw = self.backbone.grid_h, self.backbone.grid_w
        m = torch.sigmoid(mask_logits)
        pooled = nn.functional.adaptive_avg_pool2d(m, (gh, gw)).view(b, gh * gw)
        tok_norm = patch_tokens.norm(dim=-1)
        return pooled * tok_norm

    def _soft_suppress_seq(self, seq: torch.Tensor, contrib: torch.Tensor) -> torch.Tensor:
        """对 patch 序列做软扰动."""
        b, n, _d = seq.shape
        r = min(self.r_suppress, n)
        alpha = torch.clamp(self.suppress_alpha, 0.0, 1.0)
        out = seq.clone()
        topi = contrib.topk(r, dim=1).indices
        mask = torch.ones(b, n, 1, device=seq.device)
        for i in range(b):
            mask[i, topi[i], :] = alpha
        return out * mask
