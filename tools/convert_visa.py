#!/usr/bin/env python3
"""将 VisA 数据集转换为 MVTec AD 格式.

VisA 原始结构:
    category/Data/Images/Normal/xxx.JPG
    category/Data/Images/Anomaly/xxx.JPG  
    category/Data/Masks/Anomaly/xxx.png

转换后结构 (MVTec-like):
    category/train/good/xxx.png
    category/test/good/xxx.png
    category/test/bad/xxx.png
    category/ground_truth/bad/xxx_mask.png

使用方法:
    python tools/convert_visa.py --src /path/to/VisA --dst /path/to/visa_mvtec
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def convert_category(src_cat_dir: Path, dst_cat_dir: Path) -> dict:
    """转换单个类别."""
    stats = {"normal": 0, "anomaly": 0}
    
    # 创建目录结构
    (dst_cat_dir / "train" / "good").mkdir(parents=True, exist_ok=True)
    (dst_cat_dir / "test" / "good").mkdir(parents=True, exist_ok=True)
    (dst_cat_dir / "test" / "bad").mkdir(parents=True, exist_ok=True)
    (dst_cat_dir / "ground_truth" / "bad").mkdir(parents=True, exist_ok=True)
    
    # 查找 Images 目录 (可能是 Data/Images 或直接 Images)
    images_dir = src_cat_dir / "Data" / "Images"
    if not images_dir.exists():
        images_dir = src_cat_dir / "Images"
    if not images_dir.exists():
        print(f"  WARNING: Images 目录不存在: {src_cat_dir}")
        return stats
    
    masks_dir = src_cat_dir / "Data" / "Masks"
    if not masks_dir.exists():
        masks_dir = src_cat_dir / "Masks"
    
    # 处理 Normal 图像 -> test/good (VisA 没有明确的 train 分割，全放 test 然后用我们的 supervised_split)
    normal_dir = images_dir / "Normal"
    if normal_dir.exists():
        for img_path in sorted(normal_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            # 转换为 PNG 并复制到 test/good
            dst_path = dst_cat_dir / "test" / "good" / (img_path.stem + ".png")
            img = Image.open(img_path).convert("RGB")
            img.save(dst_path)
            stats["normal"] += 1
    
    # 处理 Anomaly 图像 -> test/bad
    anomaly_dir = images_dir / "Anomaly"
    if anomaly_dir.exists():
        anomaly_masks_dir = masks_dir / "Anomaly" if masks_dir.exists() else None
        
        for img_path in sorted(anomaly_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            
            # 复制图像到 test/bad
            dst_img_path = dst_cat_dir / "test" / "bad" / (img_path.stem + ".png")
            img = Image.open(img_path).convert("RGB")
            img.save(dst_img_path)
            
            # 复制 mask 到 ground_truth/bad (MVTec 命名: xxx_mask.png)
            if anomaly_masks_dir:
                # 尝试多种可能的 mask 文件名
                possible_masks = [
                    anomaly_masks_dir / (img_path.stem + ".png"),
                    anomaly_masks_dir / img_path.name,
                    anomaly_masks_dir / (img_path.stem + "_mask.png"),
                ]
                mask_found = False
                for mask_path in possible_masks:
                    if mask_path.exists():
                        dst_mask_path = dst_cat_dir / "ground_truth" / "bad" / (img_path.stem + "_mask.png")
                        mask = Image.open(mask_path).convert("L")
                        mask.save(dst_mask_path)
                        mask_found = True
                        break
                if not mask_found:
                    # 创建空 mask 作为 fallback
                    dst_mask_path = dst_cat_dir / "ground_truth" / "bad" / (img_path.stem + "_mask.png")
                    mask = Image.new("L", img.size, 0)
                    mask.save(dst_mask_path)
            
            stats["anomaly"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="将 VisA 转换为 MVTec AD 格式")
    parser.add_argument("--src", type=str, required=True, help="VisA 源目录")
    parser.add_argument("--dst", type=str, required=True, help="输出目录")
    parser.add_argument("--categories", type=str, nargs="*", default=None, 
                        help="要转换的类别列表，默认全部")
    args = parser.parse_args()
    
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    categories = args.categories if args.categories else VISA_CATEGORIES
    
    print(f"VisA -> MVTec 格式转换")
    print(f"  源目录: {src_root}")
    print(f"  目标目录: {dst_root}")
    print(f"  类别数: {len(categories)}")
    print()
    
    dst_root.mkdir(parents=True, exist_ok=True)
    
    total_stats = {"normal": 0, "anomaly": 0, "categories": 0}
    
    for cat in tqdm(categories, desc="转换类别"):
        src_cat = src_root / cat
        if not src_cat.exists():
            # 尝试其他可能的目录结构
            alt_paths = [
                src_root / cat.capitalize(),
                src_root / cat.upper(),
            ]
            found = False
            for alt in alt_paths:
                if alt.exists():
                    src_cat = alt
                    found = True
                    break
            if not found:
                print(f"  跳过 {cat}: 目录不存在")
                continue
        
        dst_cat = dst_root / cat
        stats = convert_category(src_cat, dst_cat)
        
        if stats["normal"] > 0 or stats["anomaly"] > 0:
            total_stats["categories"] += 1
            total_stats["normal"] += stats["normal"]
            total_stats["anomaly"] += stats["anomaly"]
            print(f"  {cat}: normal={stats['normal']}, anomaly={stats['anomaly']}")
    
    print()
    print("转换完成!")
    print(f"  总类别数: {total_stats['categories']}")
    print(f"  Normal 图像: {total_stats['normal']}")
    print(f"  Anomaly 图像: {total_stats['anomaly']}")


if __name__ == "__main__":
    main()
