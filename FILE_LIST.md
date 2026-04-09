# Supplementary Material File List

## Repository Structure

```
submit/
├── README.md                          # Main documentation (English)
├── requirements.txt                   # Python dependencies
├── configs/                           # Configuration files
│   ├── dataset_mvtec.yaml            # MVTec AD dataset configuration
│   ├── dataset_visa.yaml             # VisA dataset configuration
│   ├── dataset_loco.yaml             # MVTec LOCO dataset configuration
│   ├── model.yaml                    # Model architecture hyperparameters
│   ├── train.yaml                    # Training configuration
│   └── eval.yaml                     # Evaluation settings
├── models/                            # Core model implementations
│   ├── __init__.py
│   ├── full_model.py                 # Main model: GroundedDefectModel (394 lines)
│   ├── clip_backbone.py              # Frozen CLIP ViT-B/16 patch encoder
│   ├── evidence_bottleneck.py        # Evidence bottleneck with cross-attention
│   ├── localization_head.py          # Evidence-gated dense localization head
│   └── explanation_head.py           # Structured explanation head
├── datasets/                          # Dataset loaders
│   ├── __init__.py
│   ├── mvtec.py                      # MVTec AD dataset (291 lines)
│   ├── visa.py                       # VisA dataset loader
│   ├── mvtec_loco.py                 # MVTec LOCO dataset loader
│   ├── text_templates.py             # Defect type templates and text rendering
│   └── transforms.py                 # Image/mask augmentations
├── engine/                            # Training and evaluation engine
│   ├── __init__.py
│   ├── trainer.py                    # Three-stage training loop (239 lines)
│   ├── evaluator.py                  # Metric computation
│   ├── losses.py                     # Loss functions (BCE, CE, consistency)
│   ├── metrics.py                    # AUROC, AUPRO, GS, PCS metrics
│   ├── aupro.py                      # AUPRO approximation implementation
│   └── template_embed.py             # Template text encoding with CLIP
└── tools/                             # Entry-point scripts
    ├── train.py                      # Training entry point
    ├── eval.py                       # Evaluation entry point
    ├── infer.py                      # Single image inference
    ├── export_cases.py               # Visualization export for figures
    ├── convert_visa.py               # VisA format converter
    └── download_mvtec.py             # MVTec AD dataset downloader
```

## Key Implementation Details

### 1. Model Architecture (models/full_model.py)
- **Lines**: 394
- **Core Class**: `GroundedDefectModel`
- **Variants Supported**: 7 (ours, mtl_naive, loc_only, w/o_shared_bottleneck, w/o_consistency, random_masking, separate_features)
- **Key Methods**: 
  - `forward()`: Main forward pass with perturbation support
  - `_forward_ours()`: Our method with evidence bottleneck and consistency
  - `_soft_suppress()`: Perturbation mechanism for consistency loss

### 2. Evidence Bottleneck (models/evidence_bottleneck.py)
- **Lines**: 103
- **Core Class**: `EvidenceBottleneck`
- **Architecture**: 2-layer Transformer decoder with cross-attention
- **Parameters**: K=8 or 16 learnable query tokens

### 3. Localization Head (models/localization_head.py)
- **Lines**: 229
- **Core Classes**: 
  - `EvidenceGatedLocalizationHead`: Our gated mechanism
  - `DenseLocalizationHead`: Baseline without gating
  - `ProgressiveDecoder`: Multi-scale upsampling (14×14 → 224×224)

### 4. Explanation Head (models/explanation_head.py)
- **Lines**: 134
- **Core Class**: `StructuredExplanationHead`
- **Outputs**: Defect presence (2 classes), type (6 classes), location (4 classes)
- **Architecture**: 2-layer Transformer decoder

### 5. Training Loop (engine/trainer.py)
- **Lines**: 239
- **Protocol**: Three-stage training
  - Stage 1: Localization only (epochs 1-20)
  - Stage 2: Add explanation head (epochs 21-30)
  - Stage 3: Enable consistency loss (epochs 31-50)
- **Features**: Mixed precision (AMP), cosine LR scheduling, gradient clipping

### 6. Dataset Loader (datasets/mvtec.py)
- **Lines**: 291
- **Core Class**: `MvtecADDataset`
- **Protocols**: 
  - `supervised_test_split`: Our proposed split for learning from anomalies
  - `official_unsupervised`: Original MVTec protocol
- **Outputs**: Images, masks, structured labels (presence, type, location), text

## Configuration Files

### model.yaml
```yaml
K: 8
clip_model: ViT-B-16
clip_pretrained: laion400m_e32
variant: ours
r_suppress: 2
suppress_alpha: 0.1
```

### train.yaml
```yaml
batch_size: 64
epochs: 50
lr: 0.0005
stage1_epochs: 20
stage2_epochs: 30
lambda_txt: 0.5
lambda_cons_max: 0.3
use_amp: true
```

## Usage Examples

### Training
```bash
python tools/train.py \
  --dataset configs/dataset_mvtec.yaml \
  --model configs/model.yaml \
  --train configs/train.yaml
```

### Evaluation
```bash
python tools/eval.py \
  --dataset configs/dataset_mvtec.yaml \
  --eval configs/eval.yaml \
  --checkpoint outputs/checkpoints/latest.pt
```

## Notes for Reviewers

1. **All hardcoded paths have been removed**: Dataset paths use placeholder `/path/to/...` in configuration files
2. **No pretrained weights included**: This is a clean codebase for reproducibility
3. **Complete implementation**: All core components are included (model, training, evaluation, datasets)
4. **Self-contained**: Only requires standard PyTorch ecosystem dependencies
5. **Well-documented**: English README with detailed usage instructions and architecture overview

## Total Statistics

- **Python files**: 26
- **Total lines of code**: ~3,500 (excluding comments and blank lines)
- **Configuration files**: 6
- **Documentation**: 2 files (README.md, FILE_LIST.md)
