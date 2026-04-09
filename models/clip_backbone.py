"""CLIP backbone: 冻结 ViT 编码器，返回 patch tokens."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FrozenCLIPPatchEncoder(nn.Module):
    """冻结 CLIP ViT 图像编码器; 返回 patch tokens (B, N, D) 和 (H, W) grid."""

    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "laion400m_e32"):
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.clip = model
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False
        self.visual = self.clip.visual
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            flat, (gh, gw), d = self._encode_to_flat(dummy)
        self.grid_h = gh
        self.grid_w = gw
        self.num_patches = gh * gw
        self.patch_dim = d

    def _trunk_features(self, images: torch.Tensor) -> torch.Tensor:
        v = self.visual
        if hasattr(v, "forward_intermediates"):
            feats = v.forward_intermediates(
                images,
                indices=[-1],
                intermediates_only=True,
                output_fmt="NLC",
            )
            x = feats["image_intermediates"][-1]
        elif hasattr(v, "trunk") and hasattr(v.trunk, "forward_features"):
            x = v.trunk.forward_features(images)
            if isinstance(x, dict):
                x = x.get("x", list(x.values())[-1])
        elif hasattr(v, "forward_features"):
            x = v.forward_features(images)
        else:
            raise RuntimeError("Unsupported open_clip visual; could not access patch features.")
        return x

    def _encode_to_flat(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], int]:
        x = self._trunk_features(images)
        if x.dim() == 4:
            b, c, h, w = x.shape
            flat = x.flatten(2).transpose(1, 2)
            return flat, (h, w), c
        if x.dim() != 3:
            raise RuntimeError(f"Unexpected feature shape {x.shape}")
        b, n, c = x.shape
        seq = x
        if n > 1 and (math.isqrt(n - 1) ** 2 == n - 1):
            seq = x[:, 1:, :]
            n = seq.shape[1]
        side = math.isqrt(n)
        if side * side != n:
            for h in range(int(math.sqrt(n)), 0, -1):
                if n % h == 0:
                    gh, gw = h, n // h
                    break
            else:
                gh, gw = 1, n
        else:
            gh = gw = side
        seq2 = seq.transpose(1, 2).reshape(b, c, gh, gw)
        flat = seq2.flatten(2).transpose(1, 2)
        return flat, (gh, gw), c

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        with torch.no_grad():
            flat, _, _ = self._encode_to_flat(images)
        return flat, (self.grid_h, self.grid_w)


def encode_text_normalized(model: nn.Module, texts: list[str], device: torch.device) -> torch.Tensor:
    """用 CLIP text encoder 编码文本并 L2 归一化."""
    import open_clip

    try:
        tokens = open_clip.tokenize(texts, truncate=True).to(device)
    except TypeError:
        tokens = open_clip.tokenize(texts).to(device)
    with torch.no_grad():
        t = model.encode_text(tokens).float()
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
    return t
