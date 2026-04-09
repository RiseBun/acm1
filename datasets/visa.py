"""VisA 数据集支持."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from datasets.text_templates import get_defect_type_id, render_explanation, VISA_CATEGORIES
from datasets.transforms import eval_image_mask_transform, train_image_mask_transform


class VisADataset(Dataset):
    """
    VisA 数据集: train only on good images; test on good + bad.
    每个样本: image, pixel mask (0/1), structured labels, rendered text.
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_size: int = 224,
        categories: Optional[list[str]] = None,
        train: bool = True,
        protocol: str = "supervised_test_split",
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        seed: int = 42,
        include_train_good_in_train: bool = True,
    ):
        assert split in ("train", "val", "test")
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.train = train
        self.protocol = protocol
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.include_train_good_in_train = include_train_good_in_train
        
        cats = categories if categories else VISA_CATEGORIES
        self.samples: list[dict] = []
        for cat in cats:
            self._collect_category(cat)
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No VisA samples found under {root}. Check `root` in configs/dataset_visa.yaml.")

    def _collect_category(self, category: str) -> None:
        base = self.root / category
        if not base.is_dir():
            return
        
        if self.split == "train":
            self._collect_train(category, base)
        elif self.split == "val":
            self._collect_val(category, base)
        else:  # test
            self._collect_test(category, base)

    def _collect_train(self, category: str, base: Path) -> None:
        if self.include_train_good_in_train:
            good_dir = base / "Images" / "Normal"
            if good_dir.is_dir():
                for p in sorted(good_dir.glob("*.JPG")) + sorted(good_dir.glob("*.jpg")):
                    self._append_sample(
                        image_path=p,
                        category=category,
                        mask_path=None,
                        presence=0,
                        defect_type=0,
                        location=3,
                        is_anomaly=False,
                    )

    def _collect_val(self, category: str, base: Path) -> None:
        # VisA 通常不单独设 val，从 train good 中划分
        import random
        good_dir = base / "Images" / "Normal"
        if not good_dir.is_dir():
            return
        
        all_good = sorted(good_dir.glob("*.JPG")) + sorted(good_dir.glob("*.jpg"))
        if not all_good:
            return
        
        rnd = random.Random(f"{self.seed}:{category}:val")
        items = list(all_good)
        rnd.shuffle(items)
        n = len(items)
        n_val = int(n * self.val_ratio)
        val_items = items[:n_val]
        
        for p in val_items:
            self._append_sample(
                image_path=p,
                category=category,
                mask_path=None,
                presence=0,
                defect_type=0,
                location=3,
                is_anomaly=False,
            )

    def _collect_test(self, category: str, base: Path) -> None:
        import random
        import numpy as np
        
        # 收集 good 样本
        good_dir = base / "Images" / "Normal"
        if good_dir.is_dir():
            all_good = sorted(good_dir.glob("*.JPG")) + sorted(good_dir.glob("*.jpg"))
            # 从 good 中划分一部分到 test
            rnd = random.Random(f"{self.seed}:{category}:test_good")
            items = list(all_good)
            rnd.shuffle(items)
            n = len(items)
            n_test = max(1, int(n * (1.0 - self.train_ratio - self.val_ratio)))
            test_good = items[:n_test]
            
            for p in test_good:
                self._append_sample(
                    image_path=p,
                    category=category,
                    mask_path=None,
                    presence=0,
                    defect_type=0,
                    location=3,
                    is_anomaly=False,
                )
        
        # 收集 anomaly 样本
        anomaly_dir = base / "Images" / "Anomaly"
        gt_dir = base / "Annotations" / "PixelLevel"
        if anomaly_dir.is_dir():
            for defect_path in sorted(anomaly_dir.glob("*.JPG")) + sorted(anomaly_dir.glob("*.jpg")):
                defect_name = "bad"
                # 查找对应的 mask
                mask_path = gt_dir / category / (defect_path.stem + ".png")
                if not mask_path.is_file():
                    mask_path = gt_dir / category / (defect_path.stem + "_mask.png")
                
                if mask_path.is_file():
                    m = Image.open(mask_path).copy()
                    loc = _coarse_location_from_mask(m)
                else:
                    mask_path = None
                    loc = 3
                
                # VisA 使用 category 特定的 defect type
                dtype = get_defect_type_id(defect_name, category=category)
                
                self._append_sample(
                    image_path=defect_path,
                    category=category,
                    mask_path=mask_path,
                    presence=1,
                    defect_type=dtype,
                    location=loc,
                    is_anomaly=True,
                )

    def _append_sample(
        self,
        image_path: Path,
        category: str,
        mask_path: Optional[Path],
        presence: int,
        defect_type: int,
        location: int,
        is_anomaly: bool,
    ) -> None:
        self.samples.append(
            {
                "image_path": image_path,
                "mask_path": mask_path,
                "category": category,
                "presence": presence,
                "defect_type": defect_type,
                "location": location,
                "is_anomaly": is_anomaly,
            }
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        import torch
        
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        h, w = image.size[1], image.size[0]
        
        if s["mask_path"] is None or not s["is_anomaly"]:
            mask = Image.new("L", (w, h), 0)
        else:
            mask = Image.open(s["mask_path"]).convert("L")
            if mask.size != (w, h):
                mask = mask.resize((w, h), Image.NEAREST)
        
        if self.train:
            img_t, mask_t = train_image_mask_transform(image, mask, self.image_size)
        else:
            img_t, mask_t = eval_image_mask_transform(image, mask, self.image_size)
        
        text = render_explanation(s["presence"], s["defect_type"] if s["is_anomaly"] else 0, s["location"])
        
        return {
            "image": img_t,
            "mask": mask_t,
            "presence": s["presence"],
            "defect_type": s["defect_type"],
            "location": s["location"],
            "text": text,
            "is_anomaly": int(s["is_anomaly"]),
            "category": s["category"],
            "path": str(s["image_path"]),
        }


def _coarse_location_from_mask(mask: Image.Image) -> int:
    """0=left, 1=center, 2=right based on anomaly centroid x; 3=global if empty."""
    import numpy as np

    m = np.array(mask.convert("L")) > 0
    if not m.any():
        return 3
    ys, xs = np.where(m)
    cx = xs.mean() / max(m.shape[1], 1)
    if cx < 1.0 / 3.0:
        return 0
    if cx < 2.0 / 3.0:
        return 1
    return 2


def collate_fn(batch: list[dict]) -> dict:
    import torch

    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "presence": torch.tensor([b["presence"] for b in batch], dtype=torch.long),
        "defect_type": torch.tensor([b["defect_type"] for b in batch], dtype=torch.long),
        "location": torch.tensor([b["location"] for b in batch], dtype=torch.long),
        "text": [b["text"] for b in batch],
        "is_anomaly": torch.tensor([b["is_anomaly"] for b in batch], dtype=torch.long),
        "category": [b["category"] for b in batch],
        "path": [b["path"] for b in batch],
    }
