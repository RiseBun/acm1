from __future__ import annotations

from pathlib import Path
from typing import Optional
import random

from PIL import Image
from torch.utils.data import Dataset

from datasets.text_templates import get_defect_type_id, render_explanation
from datasets.transforms import eval_image_mask_transform, train_image_mask_transform

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


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


class MvtecADDataset(Dataset):
    """
    MVTec AD: train only on good images; test on good + all defect types.
    Each sample: image, pixel mask (0/1), structured labels, rendered text.
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
        assert protocol in ("official_unsupervised", "supervised_test_split")
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.train = train
        self.protocol = protocol
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.include_train_good_in_train = include_train_good_in_train
        cats = categories if categories else MVTEC_CATEGORIES
        self.samples: list[dict] = []
        for cat in cats:
            self._collect_category(cat)
        if len(self.samples) == 0:
            raise RuntimeError(f"No MVTec samples found under {root}. Check `root` in configs/dataset_mvtec.yaml.")

    def _collect_category(self, category: str) -> None:
        base = self.root / category
        if not base.is_dir():
            return
        if self.protocol == "official_unsupervised":
            self._collect_official_unsupervised(category, base)
            return
        self._collect_supervised_split(category, base)

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

    def _collect_official_unsupervised(self, category: str, base: Path) -> None:
        if self.split == "val":
            return
        if self.split == "train":
            good_dir = base / "train" / "good"
            if not good_dir.is_dir():
                return
            for p in sorted(good_dir.glob("*.png")):
                self._append_sample(
                    image_path=p,
                    category=category,
                    mask_path=None,
                    presence=0,
                    defect_type=0,
                    location=3,
                    is_anomaly=False,
                )
            return
        test_root = base / "test"
        gt_root = base / "ground_truth"
        for sub in sorted(test_root.iterdir()):
            if not sub.is_dir():
                continue
            defect_name = sub.name
            for p in sorted(sub.glob("*.png")):
                is_good = defect_name == "good"
                if is_good:
                    mask_path = None
                    presence, dtype, loc = 0, 0, 3
                    is_anomaly = False
                else:
                    rel = p.relative_to(sub)
                    mask_path = gt_root / defect_name / rel.name
                    if not mask_path.is_file():
                        mask_path = gt_root / defect_name / (p.stem + "_mask.png")
                    m = Image.open(mask_path).copy() if mask_path.is_file() else Image.new("L", Image.open(p).size, 0)
                    loc = _coarse_location_from_mask(m)
                    dtype = get_defect_type_id(defect_name, category=category)
                    presence, is_anomaly = 1, True
                self._append_sample(
                    image_path=p,
                    category=category,
                    mask_path=mask_path if not is_good else None,
                    presence=presence,
                    defect_type=dtype if is_anomaly else 0,
                    location=loc,
                    is_anomaly=is_anomaly,
                )

    def _subset_for_split(self, paths: list[Path], key: str) -> list[Path]:
        if not paths:
            return []
        rnd = random.Random(f"{self.seed}:{key}")
        items = list(paths)
        rnd.shuffle(items)
        n = len(items)
        if n == 1:
            return items if self.split == "test" else []
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        n_train = max(n_train, 1)
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        if n_train >= n:
            n_train = n - 1
        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val :]
        if not test_items:
            test_items = items[-1:]
            if val_items:
                val_items = val_items[:-1]
            elif train_items:
                train_items = train_items[:-1]
        if self.split == "train":
            return train_items
        if self.split == "val":
            return val_items
        return test_items

    def _collect_supervised_split(self, category: str, base: Path) -> None:
        if self.split == "train" and self.include_train_good_in_train:
            good_dir = base / "train" / "good"
            if good_dir.is_dir():
                for p in sorted(good_dir.glob("*.png")):
                    self._append_sample(
                        image_path=p,
                        category=category,
                        mask_path=None,
                        presence=0,
                        defect_type=0,
                        location=3,
                        is_anomaly=False,
                    )
        test_root = base / "test"
        gt_root = base / "ground_truth"
        for sub in sorted(test_root.iterdir()):
            if not sub.is_dir():
                continue
            defect_name = sub.name
            subset = self._subset_for_split(sorted(sub.glob("*.png")), f"{category}:{defect_name}")
            for p in subset:
                is_good = defect_name == "good"
                if is_good:
                    self._append_sample(
                        image_path=p,
                        category=category,
                        mask_path=None,
                        presence=0,
                        defect_type=0,
                        location=3,
                        is_anomaly=False,
                    )
                    continue
                mask_path = gt_root / defect_name / (p.stem + "_mask.png")
                if not mask_path.is_file():
                    alt = gt_root / defect_name / p.name
                    mask_path = alt if alt.is_file() else mask_path
                m = Image.open(mask_path).copy() if mask_path.is_file() else Image.new("L", Image.open(p).size, 0)
                loc = _coarse_location_from_mask(m)
                self._append_sample(
                    image_path=p,
                    category=category,
                    mask_path=mask_path if mask_path.is_file() else None,
                    presence=1,
                    defect_type=get_defect_type_id(defect_name, category=category),
                    location=loc,
                    is_anomaly=True,
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = Image.open(s["image_path"]).copy()
        h, w = image.size[1], image.size[0]
        if s["mask_path"] is None or not s["is_anomaly"]:
            mask = Image.new("L", (w, h), 0)
        else:
            mask = Image.open(s["mask_path"]).copy()
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
