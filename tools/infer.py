#!/usr/bin/env python3
"""Single-image inference stub: expects same preprocessing as training (CLIP normalize, 224)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.text_templates import render_explanation
from engine.evaluator import load_model_from_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, None, device)
    model.eval()
    image_size = getattr(model, "image_size", 224)
    tfm = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    img = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img, apply_perturbation=False)
    print("mask logits", tuple(out.mask_logits.shape))
    if out.presence_logits is not None:
        p = int(out.presence_logits[0].argmax().item())
        t = int(out.defect_type_logits[0].argmax().item())
        l = int(out.location_logits[0].argmax().item())
        if p == 0:
            t, l = 0, 3
        print(render_explanation(p, t, l))


if __name__ == "__main__":
    main()
