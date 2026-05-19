# Relighty - Face and Neck Shadow Removal System

A modular computer vision pipeline for removing shadows from face and neck regions using deep learning with BiSeNet face parsing.

## Project Structure

```
Relighty/
├── configs/                 # Configuration files
│   └── config.yaml         # Main configuration
├── data/                   # Dataset directory
│   ├── input/              # Images with shadows
│   ├── target/             # Clean target images
│   └── splits/            # Train/validation split files
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
├── models/                 # Neural network architectures
│   └── shadow_remover.py  # U-Net with ResNet-34 encoder
├── masking_bg/            # Mask generation (BiSeNet)
│   └── bisenet_mask.py   # BiSeNet face parsing wrapper
├── Masking/               # BiSeNet model & weights
│   ├── models/            # BiSeNet architecture
│   └── weights/           # Model weights (resnet18/34)
├── preprocessing/         # Data augmentation
│   └── augment.py        # Augmentation utilities
├── training/              # Model training
│   ├── train.py          # Training script
│   ├── dataset.py        # Dataset class (uses PNG with alpha)
│   └── losses.py         # Loss functions
├── evaluation/            # Model evaluation & inference
│   ├── evaluate.py        # Evaluation metrics
│   └── inference.py       # Inference (PNG output)
├── postprocessing/       # Post-processing utilities
│   ├── fix_light.py      # Shadow removal with texture preservation
│   └── bg_remove.py      # Background removal
├── deployment/           # End-to-end pipeline
│   └── pipeline.py       # Complete pipeline
└── utils/                 # Utility scripts
    └── split.py          # Dataset splitting
```

## Usage

### Training

```bash
# Step 1: Put images in dataset folders
# dataset/input/  - images with shadows
# dataset/target/ - clean images

# Step 2: Create train/val split
python -m utils.split

# Step 3: Train the model
python -m training.train

# Step 4: Resume training
python -m training.train --resume
```

### Inference

```bash
# Single image (outputs PNG with transparent background)
python -m evaluation.inference --input photo.jpg --output result.png

# Batch folder
python -m evaluation.inference --input Results/input --output Results/output

# Use deployment pipeline
python -m deployment.pipeline --input photo.jpg --output result.png
```

### Evaluation

```bash
# Evaluate on validation set
python -m evaluation.evaluate

# Use specific checkpoint
python -m evaluation.evaluate --checkpoint checkpoints/shadow_removal_best.pth
```

## Configuration

All paths are configured in `configs/config.yaml`. To deploy in any environment, simply edit this file.

### Dataset Path (Key for Deployment)

```yaml
data:
  # EDIT THIS: Point to your dataset location
  dataset_root: dataset  # relative path

  # OR use absolute path (works anywhere):
  # dataset_root: /path/to/your/dataset

  input_subdir: input      # folder with shadow images (PNG with transparency)
  target_subdir: target    # folder with clean images (PNG with transparency)
  splits_dir: dataset/splits
```

## Key Features

1. **BiSeNet Face Parsing**: Best-in-class face segmentation (19 classes: face, neck, mouth, eyes, nose, etc.)
2. **Config-Driven**: Control which face parts to mask in config.yaml
3. **BiSeNet Face Parsing**: Automatic face mask during inference
4. **Texture Preservation**: Keeps original skin texture while removing shadows
5. **Cross-platform**: Works on CUDA, MPS, and CPU

## Dependencies

- torch >= 2.0
- opencv-python
- numpy
- pyyaml
- tqdm

Install with: `pip install -r requirements.txt`

## Pipeline Flow

### Training:
```
dataset/input + target → train/val split → training → model weights
```

### Inference:
```
Input Image → BiSeNet (Masking project) → Face parsing mask → ShadowRemovalNet → fix_light → Output PNG
```

All inference uses BiSeNet for face parsing (configurable which parts to include)!