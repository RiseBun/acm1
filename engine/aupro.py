"""
Region-based PRO integration (approx. MVTec-style): integrate mean per-region overlap vs FPR up to max_fpr.
"""

from __future__ import annotations

import numpy as np


def compute_aupro(
    anomaly_maps: list[np.ndarray],
    gt_masks: list[np.ndarray],
    num_thresholds: int = 200,
    max_fpr: float = 0.3,
) -> float:
    from skimage.measure import label, regionprops

    if len(anomaly_maps) == 0:
        return float("nan")
    amaps = [np.asarray(a, dtype=np.float32) for a in anomaly_maps]
    gts = [np.asarray(g, dtype=np.float32) > 0.5 for g in gt_masks]
    normed = []
    for a in amaps:
        a = a - a.min()
        denom = a.max() + 1e-8
        normed.append(a / denom)
    thresholds = np.linspace(1.0, 0.0, num_thresholds)
    fprs = []
    pros = []
    normal_pixels = []
    for n, g in zip(normed, gts):
        inv = ~g
        normal_pixels.append(n[inv])
    n_flat = np.concatenate([x.ravel() for x in normal_pixels])
    total_normal = max(len(n_flat), 1)
    for t in thresholds:
        fp = float((n_flat >= t).sum()) / total_normal
        if fp > max_fpr + 1e-6:
            break
        per_image_pro = []
        for n, g in zip(normed, gts):
            if not g.any():
                continue
            pred = n >= t
            lab = label(g.astype(np.uint8))
            props = regionprops(lab)
            overlaps = []
            for p in props:
                coords = p.coords
                ov = pred[coords[:, 0], coords[:, 1]].mean()
                overlaps.append(float(ov))
            per_image_pro.append(float(np.mean(overlaps)) if overlaps else 0.0)
        pros.append(float(np.mean(per_image_pro)) if per_image_pro else 0.0)
        fprs.append(fp)
    if len(fprs) < 2:
        return float("nan")
    order = np.argsort(fprs)
    fprs = np.asarray(fprs)[order]
    pros = np.asarray(pros)[order]
    return float(np.trapz(pros, fprs) / max(max_fpr, 1e-8))
