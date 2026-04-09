"""模板文本嵌入: 编码所有有效 (presence, type, location) 配置的 CLIP 文本嵌入."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from datasets.text_templates import DEFECT_TYPE_PHRASE, render_explanation


def build_valid_template_meta(num_defect_type: int = 6) -> list[tuple[int, int, int]]:
    """枚举所有有效的 (presence, defect_type, location) 组合.

    规则:
    - presence=0 → defect_type=0, location=3 (唯一)
    - presence=1 → defect_type in [1..num_defect_type-1], location in [0..3]
    """
    meta: list[tuple[int, int, int]] = []
    # 正常样本: 只有一种配置
    meta.append((0, 0, 3))
    # 异常样本: 所有 type×location 组合
    for t in range(1, num_defect_type):
        for loc in range(4):
            meta.append((1, t, loc))
    return meta


def encode_templates(
    clip_model: torch.nn.Module,
    device: torch.device,
    num_defect_type: int = 6,
) -> torch.Tensor:
    """编码所有有效模板文本为 CLIP 嵌入.

    Returns:
        (V, D) L2-归一化的嵌入矩阵
    """
    import open_clip

    meta = build_valid_template_meta(num_defect_type)
    texts = [render_explanation(p, t, loc) for p, t, loc in meta]
    try:
        tokens = open_clip.tokenize(texts, truncate=True).to(device)
    except TypeError:
        tokens = open_clip.tokenize(texts).to(device)
    with torch.no_grad():
        e = clip_model.encode_text(tokens).float()
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
    return e


def soft_template_embedding(
    presence_logits: torch.Tensor,
    type_logits: torch.Tensor,
    loc_logits: torch.Tensor,
    template_emb: torch.Tensor,
    num_defect_type: int = 6,
) -> torch.Tensor:
    """可微分的软模板嵌入: 基于分解 softmax 的期望 CLIP 文本嵌入.

    Args:
        presence_logits: (B, 2)
        type_logits: (B, num_defect_type)
        loc_logits: (B, 4)
        template_emb: (V, D) 预编码的模板嵌入
    Returns:
        (B, D) 加权平均的模板嵌入
    """
    meta = build_valid_template_meta(num_defect_type)
    lpp = F.log_softmax(presence_logits, dim=1)
    ltt = F.log_softmax(type_logits, dim=1)
    lll = F.log_softmax(loc_logits, dim=1)

    scores = []
    for p, t, loc in meta:
        s = lpp[:, p]
        if t < type_logits.shape[1]:
            s = s + ltt[:, t]
        if loc < loc_logits.shape[1]:
            s = s + lll[:, loc]
        scores.append(s)

    joint = torch.stack(scores, dim=1)  # (B, V)
    w = F.softmax(joint, dim=1)  # (B, V)
    e = w @ template_emb  # (B, D)
    return e / (e.norm(dim=-1, keepdim=True) + 1e-8)
