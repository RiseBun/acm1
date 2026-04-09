# Grounded Defect Explanation with Multimodal Consistency Learning

This repository contains the implementation for **Grounded Defect Explanation with Multimodal Consistency Learning**, submitted to ACM Multimedia.

## Overview

This work proposes a novel framework for anomaly detection and explanation in industrial inspection. The model leverages a frozen CLIP vision backbone, a shared evidence bottleneck, and a structured explanation head to provide both pixel-level defect localization and human-readable defect descriptions.

### Key Components

- **Frozen CLIP Backbone**: ViT-B/16 extracts patch-level features without fine-tuning
- **Evidence Bottleneck**: K learnable query tokens compress N patch tokens via cross-attention (K=8 or 16)
- **Evidence-Gated Dense Localization**: Combines dense anomaly scores with evidence-based gating for precise pixel-level defect maps
- **Structured Explanation Head**: Transformer-based decoder predicts defect presence, type, and coarse location
- **Consistency Loss**: Perturbation-based regularization ensures robustness by suppressing top-contributing evidence tokens

### Model Variants

The codebase supports multiple variants for ablation studies:

- **`ours`**: Full model with shared bottleneck, gated localization, explanation head, and consistency loss
- **`mtl_naive`**: Multi-task baseline without bottleneck or gating mechanism
- **`loc_only`**: Localization-only variant without explanation head
- **`w/o_shared_bottleneck`**: Separate bottlenecks for localization and explanation branches
- **`w/o_consistency`**: Shared bottleneck without consistency regularization
- **`random_masking`**: Random token masking instead of importance-based suppression
- **`separate_features`**: Independent feature projections without bottleneck

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 2.0
- open-clip-torch
- PyYAML
- Pillow
- tqdm
- numpy

## Dataset Preparation

### MVTec AD

Download MVTec AD dataset and set the path in `configs/dataset_mvtec.yaml`:

```yaml
root: /path/to/mvtec/ad/dataset
```

Expected directory structure:
```
{root}/
├── bottle/
│   ├── train/good/
│   ├── test/{defect_type}/
│   └── ground_truth/{defect_type}/
├── cable/
└── ... (15 categories total)
```

### VisA (Optional)

Convert VisA dataset to MVTec-like format:

```bash
python tools/convert_visa.py --src /path/to/VisA --dst /path/to/visa_mvtec
```

Then update `configs/dataset_visa.yaml` with the converted path.

### MVTec LOCO (Optional)

Set the path in `configs/dataset_loco.yaml`:

```yaml
root: /path/to/mvtec/loco/dataset
```

## Usage

### Training

Train the model using YAML configuration files:

```bash
# Train on MVTec AD
python tools/train.py \
  --dataset configs/dataset_mvtec.yaml \
  --model configs/model.yaml \
  --train configs/train.yaml

# Train on VisA
python tools/train.py \
  --dataset configs/dataset_visa.yaml \
  --model configs/model.yaml \
  --train configs/train.yaml \
  --dataset_type visa

# Train on MVTec LOCO
python tools/train.py \
  --dataset configs/dataset_loco.yaml \
  --model configs/model.yaml \
  --train configs/train.yaml \
  --dataset_type loco
```

**Model Configuration**: Edit `configs/model.yaml` to change the variant:

```yaml
variant: ours  # Options: ours, mtl_naive, loc_only, w/o_shared_bottleneck, w/o_consistency, random_masking, separate_features
K: 8
clip_model: ViT-B-16
clip_pretrained: laion400m_e32
r_suppress: 2
```

**Training Configuration**: Key parameters in `configs/train.yaml`:

```yaml
batch_size: 64
epochs: 50
lr: 0.0005
stage1_epochs: 20    # Stage 1: Localization only
stage2_epochs: 30    # Stage 2: Add explanation head
lambda_txt: 0.5      # Text loss weight
lambda_cons_max: 0.3 # Consistency loss weight (warmup)
use_amp: true        # Mixed precision training
```

### Three-Stage Training Protocol

1. **Stage 1** (epochs 1-20): Train localization head with pixel-level BCE loss
2. **Stage 2** (epochs 21-30): Add explanation head with classification losses
3. **Stage 3** (epochs 31-50): Enable consistency loss with perturbation-based regularization

### Evaluation

Evaluate a trained checkpoint:

```bash
python tools/eval.py \
  --dataset configs/dataset_mvtec.yaml \
  --eval configs/eval.yaml \
  --checkpoint outputs/checkpoints/latest.pt \
  --dataset_type mvtec
```

Output metrics (JSON format):
- `image_auroc`: Image-level anomaly detection AUROC
- `pixel_auroc`: Pixel-level localization AUROC
- `aupro`: Approximate AUPRO (per-region overlap)
- `gs`: Grounding Score (alignment between localization and explanation)
- `pcs`: Perturbation Consistency Score (ours and mtl_naive only)

### Inference

Run inference on individual images:

```bash
python tools/infer.py \
  --checkpoint outputs/checkpoints/latest.pt \
  --image /path/to/test_image.png \
  --output /path/to/output.png
```

### Export Visualization Cases

Export qualitative results for paper figures:

```bash
python tools/export_cases.py \
  --checkpoint outputs/checkpoints/latest.pt \
  --out outputs/figures \
  --limit 12
```

## Code Structure

```
.
├── configs/
│   ├── dataset_mvtec.yaml    # MVTec AD dataset config
│   ├── dataset_visa.yaml     # VisA dataset config
│   ├── dataset_loco.yaml     # MVTec LOCO dataset config
│   ├── model.yaml            # Model architecture config
│   ├── train.yaml            # Training hyperparameters
│   └── eval.yaml             # Evaluation settings
├── models/
│   ├── full_model.py         # Main model: GroundedDefectModel
│   ├── clip_backbone.py      # Frozen CLIP ViT patch encoder
│   ├── evidence_bottleneck.py # Evidence bottleneck with cross-attention
│   ├── localization_head.py  # Evidence-gated dense localization
│   └── explanation_head.py   # Structured explanation head
├── datasets/
│   ├── mvtec.py              # MVTec AD dataset loader
│   ├── visa.py               # VisA dataset loader
│   ├── mvtec_loco.py         # MVTec LOCO dataset loader
│   ├── text_templates.py     # Defect type templates and rendering
│   └── transforms.py         # Image/mask augmentations
├── engine/
│   ├── trainer.py            # Three-stage training loop
│   ├── evaluator.py          # Metric computation
│   ├── losses.py             # Loss functions (BCE, CE, consistency)
│   ├── metrics.py            # AUROC, AUPRO, GS, PCS
│   ├── aupro.py              # AUPRO implementation
│   └── template_embed.py     # Template text encoding
├── tools/
│   ├── train.py              # Training entry point
│   ├── eval.py               # Evaluation entry point
│   ├── infer.py              # Inference script
│   ├── export_cases.py       # Visualization export
│   ├── convert_visa.py       # VisA format converter
│   └── download_mvtec.py     # MVTec AD downloader
└── requirements.txt
```

## Reproducing Results

### Main Results (MVTec AD)

```bash
# Train our model
python tools/train.py \
  --dataset configs/dataset_mvtec.yaml \
  --model configs/model.yaml \
  --train configs/train.yaml

# Evaluate
python tools/eval.py \
  --dataset configs/dataset_mvtec.yaml \
  --eval configs/eval.yaml \
  --checkpoint outputs/checkpoints/latest.pt
```

### Ablation Studies

Modify `configs/model.yaml` to switch variants:

```yaml
variant: mtl_naive  # or: loc_only, w/o_shared_bottleneck, w/o_consistency, random_masking, separate_features
```

## Notes

- **AUPRO**: The implementation in `engine/aupro.py` is an approximation based on connected components. Values may slightly differ from the official MVTec script but are suitable for trend analysis and ablation studies.
- **CLIP Features**: If upgrading `open-clip-torch` causes training errors, adjust the flattening logic in `models/clip_backbone.py` to match the actual `forward_features` output shape.
- **Reported Numbers**: Results in the paper are based on local training runs. This repository does not provide pretrained weights.
- **GPU Memory**: Training with `batch_size=64` requires ~16GB GPU memory. Reduce batch size if needed and adjust learning rate accordingly.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{your_acm_mm_paper,
  title={Grounded Defect Explanation with Multimodal Consistency Learning},
  author={Your Names},
  booktitle={ACM Multimedia},
  year={2024}
}
```

## License

This code is provided for research purposes only.
