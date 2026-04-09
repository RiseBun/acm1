#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.trainer import run_training


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "configs" / "dataset_mvtec.yaml")
    ap.add_argument("--model", type=Path, default=ROOT / "configs" / "model.yaml")
    ap.add_argument("--train", type=Path, default=ROOT / "configs" / "train.yaml")
    ap.add_argument("--dataset_type", type=str, default="mvtec", choices=["mvtec", "visa", "loco"],
                    help="Dataset type: mvtec (MVTec AD), visa (VisA), loco (MVTec LOCO)")
    args = ap.parse_args()
    ds_cfg = load_yaml(args.dataset)
    model_cfg = load_yaml(args.model)
    train_cfg = load_yaml(args.train)
    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    categories = ds_cfg.get("categories")
    
    # 根据数据集类型导入不同的数据集类
    if args.dataset_type == "mvtec":
        from datasets.mvtec import MvtecADDataset as DatasetClass
    elif args.dataset_type == "visa":
        from datasets.visa import VisADataset as DatasetClass
    elif args.dataset_type == "loco":
        from datasets.mvtec_loco import MvtecLOCODataset as DatasetClass
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")
    
    run_training(
        dataset_root=ds_cfg["root"],
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        image_size=int(ds_cfg.get("image_size", 224)),
        categories=categories,
        dataset_cfg=ds_cfg,
        dataset_type=args.dataset_type,
    )


if __name__ == "__main__":
    main()
