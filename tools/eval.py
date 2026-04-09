#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.evaluator import evaluate_model, load_model_from_checkpoint


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "configs" / "dataset_mvtec.yaml")
    ap.add_argument("--eval", type=Path, default=ROOT / "configs" / "eval.yaml")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--dataset_type", type=str, default="mvtec", choices=["mvtec", "visa", "loco"],
                    help="Dataset type: mvtec (MVTec AD), visa (VisA), loco (MVTec LOCO)")
    args = ap.parse_args()
    ds_cfg = load_yaml(args.dataset)
    ev_cfg = load_yaml(args.eval)
    ckpt = Path(args.checkpoint or ev_cfg["checkpoint"])
    device = torch.device(ev_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    image_size = int(ds_cfg.get("image_size", 224))
    model = load_model_from_checkpoint(ckpt, image_size, device)
    clip_model = model.backbone.clip
    metrics = evaluate_model(
        model,
        ds_cfg["root"],
        image_size,
        ds_cfg.get("categories"),
        ev_cfg["batch_size"],
        ev_cfg["num_workers"],
        device,
        clip_model,
        max_fpr=float(ev_cfg.get("max_fpr", 0.3)),
        dataset_cfg=ds_cfg,
        dataset_type=args.dataset_type,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
