#!/usr/bin/env python3
"""
Export qualitative PNGs: image, predicted map overlay, predicted text (and optional perturbation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.text_templates import render_explanation
from datasets.mvtec import MvtecADDataset, collate_fn
from engine.evaluator import load_model_from_checkpoint
from torch.utils.data import DataLoader


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def denormalize(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    x = t.cpu() * std + mean
    x = (x.clamp(0, 1) * 255).byte().numpy().transpose(1, 2, 0)
    return x


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "configs" / "dataset_mvtec.yaml")
    ap.add_argument("--eval", type=Path, default=ROOT / "configs" / "eval.yaml")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "figures")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()
    ds_cfg = load_yaml(args.dataset)
    ev_cfg = load_yaml(args.eval)
    ckpt = Path(args.checkpoint or ev_cfg["checkpoint"])
    device = torch.device(ev_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    image_size = int(ds_cfg.get("image_size", 224))
    model = load_model_from_checkpoint(ckpt, image_size, device)
    model.eval()
    ds = MvtecADDataset(
        ds_cfg["root"],
        "test",
        image_size=image_size,
        categories=ds_cfg.get("categories"),
        train=False,
        protocol=ds_cfg.get("protocol", "supervised_test_split"),
        train_ratio=float(ds_cfg.get("train_ratio", 0.6)),
        val_ratio=float(ds_cfg.get("val_ratio", 0.2)),
        seed=int(ds_cfg.get("seed", 42)),
        include_train_good_in_train=bool(ds_cfg.get("include_train_good_in_train", True)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for batch in loader:
        if n >= args.limit:
            break
        images = batch["image"].to(device)
        out = model(images, apply_perturbation=False)
        prob = torch.sigmoid(out.mask_logits)[0, 0].cpu().numpy()
        heat = (prob * 255).astype(np.uint8)
        heat_color = np.array(Image.fromarray(heat).convert("RGB"))
        img = denormalize(images[0])
        overlay = (0.55 * img + 0.45 * heat_color).astype(np.uint8)
        text = ""
        if out.presence_logits is not None:
            p = int(out.presence_logits[0].argmax().item())
            t = int(out.defect_type_logits[0].argmax().item())
            l = int(out.location_logits[0].argmax().item())
            if p == 0:
                t, l = 0, 3
            text = render_explanation(p, t, l)
        out_p = model(images, apply_perturbation=True)
        text_p = ""
        if out_p.presence_logits_p is not None:
            p = int(out_p.presence_logits_p[0].argmax().item())
            t = int(out_p.defect_type_logits_p[0].argmax().item())
            l = int(out_p.location_logits_p[0].argmax().item())
            if p == 0:
                t, l = 0, 3
            text_p = render_explanation(p, t, l)
        stem = Path(batch["path"][0]).stem
        Image.fromarray(img).save(args.out / f"{n:03d}_{stem}_input.png")
        Image.fromarray(overlay).save(args.out / f"{n:03d}_{stem}_overlay.png")
        with open(args.out / f"{n:03d}_{stem}_caption.txt", "w", encoding="utf-8") as f:
            f.write(f"pred: {text}\nperturbed: {text_p}\n")
        n += 1
    print(f"Wrote {n} cases to {args.out}")


if __name__ == "__main__":
    main()
