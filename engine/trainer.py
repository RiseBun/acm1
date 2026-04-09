"""训练器: 三阶段训练 + Cosine LR + 混合精度 + 改进的超参数."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.mvtec import MvtecADDataset, collate_fn
from engine.losses import training_losses
from engine.template_embed import encode_templates
from models.full_model import GroundedDefectModel


def _stage_for_epoch(epoch: int, s1: int, s2: int) -> int:
    if epoch < s1:
        return 1
    if epoch < s2:
        return 2
    return 3


def _cosine_lr(base_lr: float, epoch: int, total_epochs: int, min_lr: float = 1e-6) -> float:
    """Cosine annealing 学习率."""
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))


def _lambda_cons_warmup(epoch: int, stage2_epochs: int, total_epochs: int,
                         min_val: float = 0.1, max_val: float = 0.3) -> float:
    """一致性损失权重从 min_val 线性增长到 max_val."""
    if epoch < stage2_epochs:
        return 0.0
    progress = (epoch - stage2_epochs) / max(total_epochs - stage2_epochs, 1)
    return min_val + (max_val - min_val) * min(progress, 1.0)


def train_one_epoch(
    model: GroundedDefectModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    variant: str,
    epoch: int,
    stage1_epochs: int,
    stage2_epochs: int,
    lambda_txt: float,
    lambda_cons: float,
    template_emb: torch.Tensor | None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> dict:
    model.train()
    backbone = model.backbone
    backbone.eval()
    stage = _stage_for_epoch(epoch, stage1_epochs, stage2_epochs)
    if variant == "loc_only":
        stage = 1
    # w/o_consistency 变体只训练 stage 1-2，不使用一致性损失
    if variant == "w/o_consistency":
        stage = min(stage, 2)

    sums = {"loss": 0.0, "l_loc": 0.0, "l_txt": 0.0, "l_cons": 0.0}
    n = 0
    # mtl_naive 和消融变体也需要 perturbation 用于 PCS 评估
    # w/o_consistency 训练时不使用一致性损失，但仍需要扰动路径用于评估
    apply_p = variant in ("ours", "mtl_naive", "random_masking", "w/o_shared_bottleneck", "separate_features") and stage >= 3

    for batch in tqdm(loader, desc=f"train e{epoch+1} s{stage}", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        batch["mask"] = batch["mask"].to(device, non_blocking=True)
        batch["presence"] = batch["presence"].to(device, non_blocking=True)
        batch["defect_type"] = batch["defect_type"].to(device, non_blocking=True)
        batch["location"] = batch["location"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        # 混合精度前向
        with autocast(enabled=use_amp):
            out = model(images, apply_perturbation=apply_p)
            loss, logs = training_losses(
                out, batch, variant, stage, lambda_txt, lambda_cons, template_emb,
            )
        
        # 混合精度反向
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        bs = images.shape[0]
        sums["loss"] += float(loss.detach()) * bs
        for k in sums:
            if k == "loss":
                continue
            if k in logs:
                sums[k] += logs[k] * bs
        n += bs

    return {k: v / max(n, 1) for k, v in sums.items()}


def run_training(
    dataset_root: str,
    model_cfg: dict,
    train_cfg: dict,
    image_size: int,
    categories: list[str] | None,
    dataset_cfg: dict | None = None,
    dataset_type: str = "mvtec",
) -> None:
    dataset_cfg = dataset_cfg or {}
    device = torch.device(train_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    variant = model_cfg["variant"]

    # 根据数据集类型导入不同的数据集类
    if dataset_type == "mvtec":
        from datasets.mvtec import MvtecADDataset as DatasetClass, collate_fn
    elif dataset_type == "visa":
        from datasets.visa import VisADataset as DatasetClass, collate_fn
    elif dataset_type == "loco":
        from datasets.mvtec_loco import MvtecLOCODataset as DatasetClass, collate_fn
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    train_ds = DatasetClass(
        dataset_root, "train",
        image_size=image_size,
        categories=categories,
        train=True,
        protocol=dataset_cfg.get("protocol", "supervised_test_split"),
        train_ratio=float(dataset_cfg.get("train_ratio", 0.6)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.2)),
        seed=int(dataset_cfg.get("seed", train_cfg.get("seed", 42))),
        include_train_good_in_train=bool(dataset_cfg.get("include_train_good_in_train", True)),
    )
    loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = GroundedDefectModel(
        variant=variant,
        clip_model=model_cfg["clip_model"],
        clip_pretrained=model_cfg["clip_pretrained"],
        K=model_cfg["K"],
        r_suppress=model_cfg["r_suppress"],
        image_size=image_size,
        loc_hidden=model_cfg.get("loc_hidden_dim", 256),
        loc_refine_ch=model_cfg.get("loc_refine_channels", 64),
        num_presence=model_cfg.get("num_presence", 2),
        num_defect_type=model_cfg.get("num_defect_type", 6),
        num_location=model_cfg.get("num_location", 4),
        suppress_alpha=model_cfg.get("suppress_alpha", 0.1),
    ).to(device)

    template_emb = None
    if variant in ("ours", "mtl_naive"):
        template_emb = encode_templates(model.backbone.clip, device)

    # AdamW 优化器
    base_lr = train_cfg.get("lr", 5e-4)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # Cosine annealing LR scheduler
    epochs = train_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs, eta_min=1e-6
    )

    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stage1_epochs = train_cfg["stage1_epochs"]
    stage2_epochs = train_cfg["stage2_epochs"]
    lambda_txt = train_cfg["lambda_txt"]

    # 混合精度训练
    use_amp = train_cfg.get("use_amp", False) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print(f"启用混合精度训练 (AMP)")
    print(f"batch_size={train_cfg['batch_size']}, num_workers={train_cfg['num_workers']}")

    for epoch in range(epochs):
        # 一致性损失权重 warmup
        lambda_cons = _lambda_cons_warmup(
            epoch, stage2_epochs, epochs,
            min_val=train_cfg.get("lambda_cons_min", 0.1),
            max_val=train_cfg.get("lambda_cons_max", 0.3),
        )

        stats = train_one_epoch(
            model, loader, optim, device, variant,
            epoch, stage1_epochs, stage2_epochs,
            lambda_txt, lambda_cons, template_emb,
            scaler=scaler, use_amp=use_amp,
        )

        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]

        stage = _stage_for_epoch(epoch, stage1_epochs, stage2_epochs)
        print(
            f"Epoch {epoch+1}/{epochs} stage={stage} lr={cur_lr:.2e} λ_cons={lambda_cons:.3f} | "
            + " ".join(f"{k}={v:.4f}" for k, v in stats.items())
        )

        save_every = train_cfg.get("save_every", 10)
        if save_every and (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            payload = {
                "model": model.state_dict(),
                "cfg": {
                    "model": model_cfg,
                    "train": train_cfg,
                    "image_size": image_size,
                },
                "epoch": epoch + 1,
            }
            torch.save(payload, ckpt_dir / f"checkpoint_e{epoch+1}.pt")
            torch.save(payload, ckpt_dir / "latest.pt")
