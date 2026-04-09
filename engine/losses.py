"""损失函数: Focal Loss + Dice Loss + Detached Consistency Loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from engine.template_embed import soft_template_embedding


def focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Focal Loss 处理正负样本不平衡.

    Args:
        logits: (B, 1, H, W) 预测 logits
        target: (B, 1, H, W) GT mask
        gamma: 聚焦参数
        alpha: 正样本权重
    """
    bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target.float() + (1 - prob) * (1 - target.float())
    alpha_t = alpha * target.float() + (1 - alpha) * (1 - target.float())
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    t = target.float()
    inter = (prob * t).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def loc_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """定位损失: BCE Loss + Dice Loss (与论文一致)."""
    bce = F.binary_cross_entropy_with_logits(logits, mask.float())
    dice = dice_loss_with_logits(logits, mask)
    return bce + dice


def explanation_loss(
    presence_logits: torch.Tensor,
    type_logits: torch.Tensor,
    loc_logits: torch.Tensor,
    presence: torch.Tensor,
    defect_type: torch.Tensor,
    location: torch.Tensor,
) -> torch.Tensor:
    """解释损失: 三个分类任务的交叉熵之和."""
    lp = F.cross_entropy(presence_logits, presence)
    lt = F.cross_entropy(type_logits, defect_type)
    ll = F.cross_entropy(loc_logits, location)
    return lp + lt + ll


def _norm01(x: torch.Tensor) -> torch.Tensor:
    """归一化到 [0, 1]."""
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-8:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn)


def consistency_loss(
    mask_logits: torch.Tensor,
    mask_logits_p: torch.Tensor,
    presence_logits: torch.Tensor,
    type_logits: torch.Tensor,
    loc_logits: torch.Tensor,
    presence_logits_p: torch.Tensor,
    type_logits_p: torch.Tensor,
    loc_logits_p: torch.Tensor,
    template_emb: torch.Tensor,
) -> torch.Tensor:
    """一致性损失: 使用 L1 距离 (与论文公式一致).

    核心实现:
    1. delta_loc detached: 一致性损失只训练解释头和 bottleneck，不干扰定位
    2. 使用 L1 距离: |norm(Δ_loc) - norm(Δ_txt)| (与论文公式5一致)
    3. 更稳健的归一化
    """
    # 文本嵌入变化 (可微分 → 梯度流向解释头和 bottleneck)
    e = soft_template_embedding(presence_logits, type_logits, loc_logits, template_emb)
    ep = soft_template_embedding(presence_logits_p, type_logits_p, loc_logits_p, template_emb)
    delta_txt = 1.0 - (e * ep).sum(dim=-1).clamp(-1.0, 1.0)

    # 定位变化 (detached → 不影响定位头训练)
    m = torch.sigmoid(mask_logits)
    mp = torch.sigmoid(mask_logits_p)
    mass = m.flatten(1).sum(dim=1)
    mass_p = mp.flatten(1).sum(dim=1)
    eps = 1e-6
    delta_loc = ((mass - mass_p) / (mass + eps)).abs()
    delta_loc = delta_loc.detach()  # 关键: 阻断梯度

    # 归一化后对齐 (使用 L1 距离)
    nd = _norm01(delta_loc)
    nt = _norm01(delta_txt)
    return F.l1_loss(nt, nd)  # L1 距离，与论文一致


def training_losses(
    out,
    batch: dict,
    variant: str,
    stage: int,
    lambda_txt: float,
    lambda_cons: float,
    template_emb: torch.Tensor | None,
) -> tuple[torch.Tensor, dict]:
    """计算训练损失，根据阶段组合不同的损失项."""
    mask = batch["mask"]
    l_loc = loc_loss(out.mask_logits, mask)
    logs = {"l_loc": float(l_loc.detach())}
    total = l_loc

    if variant == "loc_only" or stage < 2 or out.presence_logits is None:
        return total, logs

    l_txt = explanation_loss(
        out.presence_logits,
        out.defect_type_logits,
        out.location_logits,
        batch["presence"].to(mask.device),
        batch["defect_type"].to(mask.device),
        batch["location"].to(mask.device),
    )
    logs["l_txt"] = float(l_txt.detach())
    total = total + lambda_txt * l_txt

    if variant != "ours" or stage < 3 or out.mask_logits_perturbed is None or template_emb is None:
        return total, logs

    l_c = consistency_loss(
        out.mask_logits,
        out.mask_logits_perturbed,
        out.presence_logits,
        out.defect_type_logits,
        out.location_logits,
        out.presence_logits_p,
        out.defect_type_logits_p,
        out.location_logits_p,
        template_emb,
    )
    logs["l_cons"] = float(l_c.detach())
    total = total + lambda_cons * l_c
    return total, logs
