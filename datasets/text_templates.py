"""结构化缺陷解释: 渲染和解析模板文本."""

from __future__ import annotations

# defect_type id -> 短语
DEFECT_TYPE_PHRASE = {
    0: "normal",          # 无缺陷
    1: "structural",      # 结构性缺陷 (broken, hole, poke, etc.)
    2: "contamination",   # 污染 (contamination, stain, print, color, etc.)
    3: "scratch",         # 划痕 (scratch, cut, thread, etc.)
    4: "crack",           # 裂纹 (crack, split, etc.)
    5: "texture",         # 纹理异常 (bent, fold, glue, etc.)
}

LOCATION_PHRASE = {
    0: "left",
    1: "center",
    2: "right",
    3: "global",
}

# MVTec AD 缺陷子类型到 defect_type_id 的映射
MVTEC_DEFECT_TYPE_MAP: dict[str, int] = {
    "good": 0,
    # bottle
    "broken_large": 1,
    "broken_small": 1,
    "contamination": 2,
    # cable
    "bent_wire": 5,
    "cable_swap": 1,
    "combined": 1,
    "cut_inner_insulation": 3,
    "cut_outer_insulation": 3,
    "missing_cable": 1,
    "missing_wire": 1,
    "poke_insulation": 1,
    # capsule
    "crack": 4,
    "faulty_imprint": 2,
    "poke": 1,
    "scratch": 3,
    "squeeze": 1,
    # carpet
    "color": 2,
    "cut": 3,
    "hole": 1,
    "metal_contamination": 2,
    "thread": 3,
    # grid
    "bent": 5,
    "broken": 1,
    "glue": 5,
    "metal_contamination_grid": 2,
    # hazelnut
    "print": 2,
    # leather
    "fold": 5,
    "glue_strip": 5,
    # metal_nut
    "flip": 1,
    "moved": 1,
    # pill
    "combined_pill": 1,
    "contamination_pill": 2,
    "crack_pill": 4,
    "faulty_imprint_pill": 2,
    "pill_type": 1,
    "scratch_neck": 3,
    "scratch_pill": 3,
    # screw
    "manipulated_front": 1,
    "scratch_head": 3,
    "scratch_neck_screw": 3,
    "thread_side": 3,
    "thread_top": 3,
    # tile
    "crack_tile": 4,
    "glue_strip_tile": 5,
    "gray_stroke": 2,
    "oil": 2,
    "rough": 5,
    # toothbrush
    "defective": 1,
    # transistor
    "bent_lead": 5,
    "cut_lead": 3,
    "damaged_case": 1,
    "misplaced": 1,
    # wood
    "combined_wood": 1,
    "liquid": 2,
    "scratch_wood": 3,
    # zipper
    "broken_teeth": 1,
    "fabric_border": 5,
    "fabric_interior": 5,
    "rough_zipper": 5,
    "split_teeth": 4,
    "squeezed_teeth": 1,
    # VisA 数据集 — 只有 "bad" 一种异常标签，按类别默认映射
    "bad": 1,  # 默认: structural
}

# VisA 按类别的缺陷类型映射 (可选覆盖)
VISA_CATEGORY_DEFECT_MAP: dict[str, int] = {
    "candle": 2,       # contamination (蜡滴/颜色)
    "capsules": 1,     # structural (裂纹/缺口)
    "cashew": 1,       # structural (碎裂)
    "chewinggum": 1,   # structural (变形)
    "fryum": 1,        # structural (断裂)
    "macaroni1": 1,    # structural (断裂)
    "macaroni2": 1,    # structural (断裂)
    "pcb1": 1,         # structural (缺失元件)
    "pcb2": 1,         # structural (缺失元件)
    "pcb3": 1,         # structural (缺失元件)
    "pcb4": 1,         # structural (缺失元件)
    "pipe_fryum": 1,   # structural (变形)
}

VISA_CATEGORIES = list(VISA_CATEGORY_DEFECT_MAP.keys())


def get_defect_type_id(defect_name: str, category: str | None = None) -> int:
    """根据缺陷子文件夹名获取 defect_type_id.
    
    对于 VisA 数据集 (defect_name="bad"), 根据 category 查找映射.
    """
    # VisA 特殊处理: 使用 category 进行映射
    if defect_name == "bad" and category is not None:
        return VISA_CATEGORY_DEFECT_MAP.get(category, 1)
    
    if defect_name in MVTEC_DEFECT_TYPE_MAP:
        return MVTEC_DEFECT_TYPE_MAP[defect_name]
    # 模糊匹配
    lower = defect_name.lower()
    for key, val in MVTEC_DEFECT_TYPE_MAP.items():
        if key in lower or lower in key:
            return val
    return 1  # 默认: structural


def render_explanation(presence: int, defect_type_id: int, location_id: int) -> str:
    """渲染结构化解释为模板文本.

    Args:
        presence: 0=normal, 1=anomaly
        defect_type_id: 0-5 对应不同缺陷类型
        location_id: 0=left, 1=center, 2=right, 3=global
    """
    if presence == 0:
        return "no visible defect"
    dt = DEFECT_TYPE_PHRASE.get(int(defect_type_id), "unknown")
    loc = LOCATION_PHRASE.get(int(location_id), "global")
    return f"{dt} anomaly on {loc} region"


def structured_from_logits(
    presence_logits,
    type_logits,
    loc_logits,
) -> tuple[int, int, int]:
    """从 logits 解码为结构化 (presence, defect_type, location)."""
    import torch

    p = int(torch.argmax(presence_logits, dim=-1).item())
    t = int(torch.argmax(type_logits, dim=-1).item())
    l_ = int(torch.argmax(loc_logits, dim=-1).item())
    if p == 0:
        return 0, 0, 3
    return p, t, l_
