"""模型评估: 计算多种指标."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.mvtec import MvtecADDataset, collate_fn
from engine.aupro import compute_aupro
from engine.metrics import grounding_score_batch, image_auroc, pcs_from_outputs, pixel_auroc
from models.full_model import GroundedDefectModel


@torch.no_grad()
def evaluate_model(
    model: GroundedDefectModel,
    dataset_root: str,
    image_size: int,
    categories: list[str] | None,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    clip_model: torch.nn.Module,
    max_fpr: float = 0.3,
    dataset_cfg: dict | None = None,
    dataset_type: str = "mvtec",
) -> dict[str, float]:
    dataset_cfg = dataset_cfg or {}
    model.eval()
    
    # 根据数据集类型导入不同的数据集类
    if dataset_type == "mvtec":
        from datasets.mvtec import MvtecADDataset as DatasetClass, collate_fn
    elif dataset_type == "visa":
        from datasets.visa import VisADataset as DatasetClass, collate_fn
    elif dataset_type == "loco":
        from datasets.mvtec_loco import MvtecLOCODataset as DatasetClass, collate_fn
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    ds = DatasetClass(
        dataset_root,
        "test",
        image_size=image_size,
        categories=categories,
        train=False,
        protocol=dataset_cfg.get("protocol", "supervised_test_split"),
        train_ratio=float(dataset_cfg.get("train_ratio", 0.6)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.2)),
        seed=int(dataset_cfg.get("seed", 42)),
        include_train_good_in_train=bool(dataset_cfg.get("include_train_good_in_train", True)),
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    all_scores = []
    all_masks = []
    all_labels = []
    max_img_scores = []
    gs_vals = []
    pcs_vals = []
    variant = model.variant

    for batch in tqdm(loader, desc="eval"):
        images = batch["image"].to(device)
        mask = batch["mask"].to(device)
        out = model(images, apply_perturbation=False)
        prob = torch.sigmoid(out.mask_logits)
        max_img_scores.append(prob.amax(dim=(1, 2, 3)).cpu().numpy())
        all_labels.append(batch["is_anomaly"].numpy())
        for i in range(images.shape[0]):
            all_scores.append(prob[i, 0].cpu().numpy())
            all_masks.append(mask[i, 0].cpu().numpy())

        if out.presence_logits is not None:
            gs_vals.append(
                grounding_score_batch(
                    out.presence_logits,
                    out.defect_type_logits,
                    out.location_logits,
                    batch,
                    device,
                )
            )

        if variant in ("ours", "mtl_naive", "w/o_shared_bottleneck", "random_masking", "separate_features") and out.presence_logits is not None:
            out_p = model(images, apply_perturbation=True)
            if out_p.mask_logits_perturbed is not None:
                pcs = pcs_from_outputs(
                    out.mask_logits,
                    out_p.mask_logits_perturbed,
                    out.presence_logits,
                    out.defect_type_logits,
                    out.location_logits,
                    out_p.presence_logits_p,
                    out_p.defect_type_logits_p,
                    out_p.location_logits_p,
                    clip_model,
                    device,
                )
                pcs_vals.append(float(pcs.item()))

    labels = np.concatenate(all_labels, axis=0)
    max_img_scores_arr = np.concatenate(max_img_scores, axis=0)
    img_auc = image_auroc(max_img_scores_arr, labels)
    pix_auc = pixel_auroc(
        np.stack([s.ravel() for s in all_scores]),
        np.stack([m.ravel() for m in all_masks]),
    )
    aupro = compute_aupro(all_scores, all_masks, max_fpr=max_fpr)

    outd: dict[str, float] = {
        "image_auroc": img_auc,
        "pixel_auroc": pix_auc,
        "aupro": aupro,
    }
    if gs_vals:
        outd["gs"] = float(np.mean(gs_vals))
    if pcs_vals:
        outd["pcs"] = float(np.mean(pcs_vals))
    return outd


def load_model_from_checkpoint(
    ckpt_path: Path, image_size: int | None, device: torch.device
) -> GroundedDefectModel:
    try:
        data = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        data = torch.load(ckpt_path, map_location=device)
    cfg = data["cfg"]["model"]
    image_size = int(data["cfg"].get("image_size", image_size or 224))
    model = GroundedDefectModel(
        variant=cfg["variant"],
        clip_model=cfg["clip_model"],
        clip_pretrained=cfg["clip_pretrained"],
        K=cfg["K"],
        r_suppress=cfg["r_suppress"],
        image_size=image_size,
        loc_hidden=cfg.get("loc_hidden_dim", 256),
        loc_refine_ch=cfg.get("loc_refine_channels", 64),
        num_presence=cfg.get("num_presence", 2),
        num_defect_type=cfg.get("num_defect_type", 6),
        num_location=cfg.get("num_location", 4),
        suppress_alpha=cfg.get("suppress_alpha", 0.1),
    ).to(device)
    model.load_state_dict(data["model"], strict=False)
    return model
