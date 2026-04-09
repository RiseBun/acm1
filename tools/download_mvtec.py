#!/usr/bin/env python3
"""下载 MVTec AD 完整数据集 (15 类) 到本地目录结构.

使用 HuggingFace datasets 下载, 然后转换为标准 MVTec 目录格式:
  {root}/{category}/train/good/
  {root}/{category}/test/{defect_type}/
  {root}/{category}/ground_truth/{defect_type}/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

TARGET_ROOT = Path("/path/to/mvtec/ad/dataset")


def download_and_convert() -> None:
    from datasets import load_dataset

    # 检查已有类别
    existing = []
    missing = []
    for cat in MVTEC_CATEGORIES:
        cat_dir = TARGET_ROOT / cat
        if cat_dir.is_dir() and (cat_dir / "test").is_dir():
            existing.append(cat)
        else:
            missing.append(cat)

    print(f"已有: {existing}")
    print(f"缺失: {missing}")

    if not missing:
        print("所有 15 类已存在, 无需下载.")
        return

    print(f"\n开始下载 {len(missing)} 个缺失类别...")

    # 下载 MVTec AD
    # 尝试不同的 HuggingFace 数据源
    dataset_names = [
        "alexriedel/MVTec-AD",
        "Bingsu/MVTecAD",
    ]

    ds = None
    for ds_name in dataset_names:
        try:
            print(f"尝试从 {ds_name} 下载...")
            ds = load_dataset(ds_name, trust_remote_code=True)
            print(f"  成功加载 {ds_name}")
            break
        except Exception as e:
            print(f"  {ds_name} 失败: {e}")
            continue

    if ds is None:
        print("无法从 HuggingFace 下载 MVTec AD.")
        print("请手动下载: https://www.mvtec.com/company/research/datasets/mvtec-ad/")
        print("或使用: wget https://www.mydrive.ch/shares/38536/...")
        _try_wget_download(missing)
        return

    _convert_hf_dataset(ds, missing)


def _convert_hf_dataset(ds, categories: list[str]) -> None:
    """将 HuggingFace 数据集转换为标准 MVTec 目录格式."""
    # HuggingFace MVTec AD 格式因数据源而异
    # 常见字段: image, label, mask, split, category, defect_type

    splits = list(ds.keys())
    print(f"可用 splits: {splits}")

    # 尝试理解数据格式
    sample = ds[splits[0]][0]
    print(f"样本字段: {list(sample.keys())}")
    print(f"样本值示例: { {k: type(v).__name__ for k, v in sample.items()} }")

    for split_name in splits:
        split_data = ds[split_name]
        print(f"\n处理 split: {split_name}, 样本数: {len(split_data)}")

        for i, sample in enumerate(split_data):
            # 提取信息 (不同数据源字段名可能不同)
            category = _get_field(sample, ["category", "object_name", "class_name"])
            if category is None or category not in categories:
                continue

            is_train = _get_field(sample, ["split"]) == "train" or split_name == "train"
            defect = _get_field(sample, ["defect_type", "anomaly_type", "label_name", "defect"])
            if defect is None:
                defect = "good" if _get_field(sample, ["label", "anomaly"]) == 0 else "unknown"

            image = _get_field(sample, ["image"])
            mask = _get_field(sample, ["mask", "anomaly_mask"])

            if image is None:
                continue

            # 构建路径
            cat_dir = TARGET_ROOT / category
            if is_train:
                img_dir = cat_dir / "train" / "good"
            else:
                img_dir = cat_dir / "test" / defect

            img_dir.mkdir(parents=True, exist_ok=True)

            # 保存图像
            idx = len(list(img_dir.glob("*.png")))
            img_path = img_dir / f"{idx:03d}.png"

            if isinstance(image, Image.Image):
                image.save(img_path)
            elif isinstance(image, np.ndarray):
                Image.fromarray(image).save(img_path)
            elif isinstance(image, dict) and "bytes" in image:
                with open(img_path, "wb") as f:
                    f.write(image["bytes"])

            # 保存 mask
            if mask is not None and defect != "good":
                mask_dir = cat_dir / "ground_truth" / defect
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_path = mask_dir / f"{idx:03d}_mask.png"

                if isinstance(mask, Image.Image):
                    mask.save(mask_path)
                elif isinstance(mask, np.ndarray):
                    Image.fromarray(mask).save(mask_path)
                elif isinstance(mask, dict) and "bytes" in mask:
                    with open(mask_path, "wb") as f:
                        f.write(mask["bytes"])

            if (i + 1) % 200 == 0:
                print(f"  已处理 {i + 1} 样本...")

    # 验证
    for cat in categories:
        cat_dir = TARGET_ROOT / cat
        n_train = len(list((cat_dir / "train").rglob("*.png"))) if (cat_dir / "train").exists() else 0
        n_test = len(list((cat_dir / "test").rglob("*.png"))) if (cat_dir / "test").exists() else 0
        n_gt = len(list((cat_dir / "ground_truth").rglob("*.png"))) if (cat_dir / "ground_truth").exists() else 0
        print(f"  {cat}: train={n_train}, test={n_test}, gt={n_gt}")


def _get_field(sample: dict, keys: list[str]):
    """尝试多个可能的字段名."""
    for k in keys:
        if k in sample:
            return sample[k]
    return None


def _try_wget_download(categories: list[str]) -> None:
    """尝试通过 wget 下载 MVTec AD 各类别的 tar.xz."""
    # MVTec AD 官方下载链接 (需要有效链接)
    base_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download"
    # 已知的文件 ID 映射 (可能会过期)
    file_ids = {
        "bottle": "420937370-1629951468",
        "cable": "420937413-1629951498",
        "capsule": "420937454-1629951595",
        "carpet": "420937484-1629951672",
        "grid": "420937487-1629951814",
        "hazelnut": "420937545-1629951845",
        "leather": "420937607-1629951964",
        "metal_nut": "420937637-1629952063",
        "pill": "420937700-1629952103",
        "screw": "420937823-1629952152",
        "tile": "420938133-1629952256",
        "toothbrush": "420938166-1629952320",
        "transistor": "420938383-1629952537",
        "wood": "420938569-1629952744",
        "zipper": "420938611-1629952813",
    }
    print("\n尝试通过 wget 下载...")
    for cat in categories:
        fid = file_ids.get(cat)
        if fid is None:
            print(f"  {cat}: 无下载链接, 跳过")
            continue
        url = f"{base_url}/{fid}/{cat}.tar.xz"
        tar_path = TARGET_ROOT / f"{cat}.tar.xz"
        if tar_path.exists():
            print(f"  {cat}: tar.xz 已存在, 解压中...")
        else:
            print(f"  {cat}: 下载中...")
            ret = os.system(f'wget -q -O "{tar_path}" "{url}"')
            if ret != 0:
                print(f"  {cat}: 下载失败 (wget exit={ret})")
                continue

        # 解压
        ret = os.system(f'tar -xf "{tar_path}" -C "{TARGET_ROOT}"')
        if ret == 0:
            print(f"  {cat}: 解压完成")
        else:
            print(f"  {cat}: 解压失败")


if __name__ == "__main__":
    download_and_convert()
