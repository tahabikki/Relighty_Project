# Relighty - Quick Start Guide

## Prerequisites
- Python 3.12+
- Install dependencies: `pip install -r requirements.txt`
- PyTorch installed (see setup.bat or setup.sh)

---

## Step 1: Prepare Dataset

Just put your images in the dataset folders:
```
dataset/
├── input/   # Images with shadows
│   ├── photo1.jpg
│   └── ...
└── target/  # Clean target images
    ├── photo1.jpg
    └── ...
```

---

## Step 2: Split Dataset

```bash
python -m utils.split
```

Creates `train.txt` and `val.txt` in `dataset/splits/`

---

## Step 3: Configure (Optional)

Edit `configs/config.yaml` to customize:

```yaml
# BiSeNet Masking - choose model and what to mask
masking:
  model_name: resnet34        # or resnet18
  include:
    skin: true       # face
    l_brow: true     # left eyebrow
    r_brow: true     # right eyebrow
    l_eye: true      # left eye
    r_eye: true      # right eye
    nose: true       # nose
    mouth: true      # mouth
    u_lip: true      # upper lip
    l_lip: true      # lower lip
    neck: true       # neck
    neck_l: true     # neck label
    cloth: false     # exclude clothes
    hair: false      # exclude hair
    hat: false       # exclude hat

# Training settings
training:
  batch_size: 4
  epochs: 200
  lr: 0.00001
```

---

## Step 4: Train

```bash
# Start training
python -m training.train

# Resume from checkpoint
python -m training.train --resume

# Train with custom epochs
python -m training.train --epochs 100
```

Checkpoints saved to `checkpoints/`

---

## Step 5: Inference

```bash
# Single image → PNG with transparent background
python -m evaluation.inference --input photo.jpg --output result.png

# Single image → keep original background
python -m evaluation.inference --input photo.jpg --output result.jpg --keep-bg

# Batch folder
python -m evaluation.inference --input Results/input --output Results/output

# Using pipeline
python -m deployment.pipeline --input photo.jpg --output result.png
```

---

## Quick Commands Summary

| Task | Command |
|------|---------|
| Split data | `python -m utils.split` |
| Train | `python -m training.train` |
| Resume | `python -m training.train --resume` |
| Inference | `python -m evaluation.inference --input photo.jpg --output result.png` |
| Evaluate | `python -m evaluation.evaluate` |

---

## Configuration

All settings in `configs/config.yaml`:
- Dataset paths
- Model selection (resnet18/34)
- Which face parts to mask (face, neck, mouth, etc.)
- Training parameters
- Inference options

Edit once, works for both training and inference!