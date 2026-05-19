# Relighty — Face and Neck Shadow Removal

A modular deep learning project for automatic shadow removal from face and neck regions using U-Net with ResNet34 backbone and BiSeNet face parsing.

## Features

- **BiSeNet Face Parsing** - Best-in-class face segmentation (19 classes: face, neck, mouth, eyes, nose, etc.)
- **Config-Driven** - Control model and mask parts in config.yaml
- **Transparent Background** - PNG output with alpha channel
- **Texture Preservation** - Keeps original skin texture while removing shadows
- **Cross-Platform** - Works on Windows, macOS, and Linux (CUDA, MPS, CPU)

---

## Quick Start

### Step 0: Install Dependencies

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Install PyTorch for your machine separately if needed:

```bash
# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

BiSeNet face parsing weights are downloaded automatically on first use if missing:

- `Masking/weights/resnet18.pt`
- `Masking/weights/resnet34.pt`

The `Masking/weights/.gitkeep` file keeps the weights folder in GitHub, but the large `.pt` files are intentionally not committed.

### Step 1: Configure Dataset Path

Edit `configs/config.yaml`:

```yaml
data:
  dataset_root: dataset  # or absolute path: /path/to/your/dataset
```

### Step 2: Prepare Dataset

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

### Step 3: Create Train/Val Split

```bash
python -m utils.split
```

### Step 4: Train the Model

```bash
python -m training.train
```

### Step 5: Run Inference (PNG with transparent background)

```bash
# Single image
python -m evaluation.inference --input photo.jpg --output result.png

# Batch folder
python -m evaluation.inference --input Results/input --output Results/output

# Or use deployment pipeline
python -m deployment.pipeline --input photo.jpg --output result.png
```

---

## BiSeNet Face Parsing Configuration

Control which face parts to mask in `configs/config.yaml`:

```yaml
masking:
  model_name: resnet34        # or resnet18
  include:
    skin: true       # face
    l_brow: true    # left eyebrow
    r_brow: true    # right eyebrow
    l_eye: true     # left eye
    r_eye: true     # right eye
    nose: true      # nose
    mouth: true     # mouth
    u_lip: true     # upper lip
    l_lip: true     # lower lip
    neck: true      # neck
    neck_l: true    # neck label
    cloth: false    # exclude clothes
    hair: false     # exclude hair
    hat: false      # exclude hat
```

---

## Complete Pipeline

### Training Flow
```
dataset/input/ + dataset/target/
         ↓
python -m utils.split
         ↓
dataset/splits/ (train/val files)
         ↓
python -m training.train
         ↓
checkpoints/ (model weights)
```

### Inference Flow
```
Input Image → BiSeNet (from Masking) → Face parsing mask (19 classes)
                                             ↓
                        ShadowRemovalNet → Shadow-free image
                                             ↓
                        fix_light.py → Fix lighting + preserve texture
                                             ↓
                        Output: PNG with transparent BG (no shadows, good texture)
```

---

## Required External Files

The repository contains the code and BiSeNet architecture. Large generated files are not committed.

For inference, you still need a trained Relighty checkpoint:

```text
checkpoints/shadow_removal_best.pth
```

If that file is missing, train first:

```bash
python -m training.train
```

or provide a checkpoint explicitly:

```bash
python -m evaluation.inference --input photo.jpg --output result.png --checkpoint path/to/model.pth
```

For training, you need your dataset outside Git:

```text
dataset/
├── input/
└── target/
```

BiSeNet weights do not need to be copied manually. If `Masking/weights/resnet34.pt` or `Masking/weights/resnet18.pt` is missing, Relighty downloads it from the public `yakhyo/face-parsing` release.

---

## Configuration

All settings in `configs/config.yaml`:

```yaml
# Dataset paths - EDIT THIS TO DEPLOY ANYWHERE
data:
  dataset_root: dataset         # or absolute path: /path/to/dataset
  input_subdir: input           # folder with input images
  target_subdir: target         # folder with target images
  splits_dir: dataset/splits

# Training
training:
  batch_size: 4
  epochs: 200
  device: auto                 # auto, cuda, cpu, mps

# Model
model:
  input_size: 256

# BiSeNet Masking
masking:
  model_name: resnet34
  include:
    skin: true
    neck: true
    mouth: true
    # ... etc
```

---

## Project Structure

```
Relighty/
├── configs/
│   └── config.yaml              # All configuration
├── data/
│   ├── input/                   # Images with shadows
│   ├── target/                  # Clean target images
│   └── splits/                 # Train/val split files
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
├── models/
│   └── shadow_remover.py       # U-Net + ResNet34
├── masking_bg/
│   └── bisenet_mask.py         # BiSeNet face parsing (from Masking project)
├── Masking/                     # BiSeNet model & weights
│   ├── models/
│   │   └── bisenet.py          # BiSeNet architecture
│   └── weights/
│       ├── resnet18.pt
│       └── resnet34.pt
├── training/
│   ├── train.py                # Training script
│   ├── dataset.py              # Uses PNG with alpha channel
│   └── losses.py               # L1 + SSIM loss
├── evaluation/
│   ├── evaluate.py             # Model evaluation
│   └── inference.py            # Inference (PNG output)
├── deployment/
│   └── pipeline.py            # End-to-end pipeline
└── postprocessing/
    └── fix_light.py           # Texture preservation + lighting fix
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Change dataset path | Edit `configs/config.yaml` - no code changes! |
| DataLoader issues on Windows | Set `num_workers: 0` in config |
| No GPU | Set `device: cpu` in config |
| Change face mask parts | Edit `masking.include` in config.yaml |

---

## License

[Your License]
