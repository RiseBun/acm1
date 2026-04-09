from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from datasets.text_templates import render_explanation
from engine.aupro import compute_aupro


def _norm01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def pixel_auroc(scores: np.ndarray, masks: np.ndarray) -> float:
    s = scores.reshape(-1)
    m = (masks.reshape(-1) > 0.5).astype(np.int32)
    if m.sum() == 0 or m.sum() == len(m):
        return float("nan")
    return float(roc_auc_score(m, s))


def image_auroc(max_scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int32)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, max_scores))


def grounding_score_batch(
    presence_pred: torch.Tensor,
    type_pred: torch.Tensor,
    loc_pred: torch.Tensor,
    batch: dict,
    device: torch.device,
) -> float:
    """Rule-based GS: 0.4 presence + 0.3 type + 0.3 location accuracy."""
    p = batch["presence"].to(device)
    t = batch["defect_type"].to(device)
    l = batch["location"].to(device)
    ap = (presence_pred.argmax(dim=1) == p).float()
    at = (type_pred.argmax(dim=1) == t).float()
    al = (loc_pred.argmax(dim=1) == l).float()
    gs = 0.4 * ap + 0.3 * at + 0.3 * al
    return float(gs.mean().item())


def pcs_from_outputs(
    mask_logits: torch.Tensor,
    mask_logits_p: torch.Tensor,
    presence_logits: torch.Tensor,
    type_logits: torch.Tensor,
    loc_logits: torch.Tensor,
    presence_logits_p: torch.Tensor,
    type_logits_p: torch.Tensor,
    loc_logits_p: torch.Tensor,
    clip_model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """计算 PCS，使用软模板嵌入 (与论文和训练一致).
    
    与训练时的一致性损失计算方式保持一致，使用可微分的软模板嵌入
    而非 argmax 后的离散文本。
    """
    from engine.template_embed import soft_template_embedding, encode_templates
    
    b = mask_logits.shape[0]
    
    # 使用软模板嵌入 (与训练时一致)
    template_emb = encode_templates(clip_model, device)
    
    e = soft_template_embedding(presence_logits, type_logits, loc_logits, template_emb)
    ep = soft_template_embedding(presence_logits_p, type_logits_p, loc_logits_p, template_emb)
    
    delta_txt = 1.0 - (e * ep).sum(dim=-1).clamp(-1.0, 1.0)
    
    m = torch.sigmoid(mask_logits)
    mp = torch.sigmoid(mask_logits_p)
    mass = m.flatten(1).sum(dim=1)
    mass_p = mp.flatten(1).sum(dim=1)
    eps = 1e-6
    delta_loc = (mass - mass_p) / (mass + eps)
    
    nd = _norm01(delta_loc.abs().cpu().numpy())
    nt = _norm01(delta_txt.detach().cpu().numpy())
    pcs = 1.0 - np.abs(nd - nt)
    return torch.tensor(pcs.mean(), device=device)
