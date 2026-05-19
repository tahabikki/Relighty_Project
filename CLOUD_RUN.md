# Relighty - Cloud Training Guide

## Dataset Structure

Your dataset should have this structure:

```
dataset/
├── input/          # Images with shadows (PNG with transparent background)
│   ├── photo1.png
│   ├── photo2.png
│   └── ...
└── target/          # Clean target images (PNG with transparent background)
    ├── photo1.png
    ├── photo2.png
    └── ...
```

The filenames in `input/` and `target/` must match!

---

## Quick Start on Cloud

### 1. Upload Project + Dataset to Cloud

```bash
# Upload project folder (excluding large files)
git init
git add .
git commit -m "Relighty project"

# Push to GitHub or upload to GCS/S3
git push origin main
# OR
gsutil -m rsync -r . gs://your-bucket/Relighty/
```

### 2. On Cloud VM - Setup

```bash
# Clone project
git clone your-repo/Relighty
cd Relighty

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Edit config.yaml for Cloud Paths

```yaml
# configs/config.yaml - use absolute paths for cloud
data:
  dataset_root: /path/to/your/dataset  # absolute path
  input_subdir: input
  target_subdir: target
  splits_dir: /path/to/your/dataset/splits

checkpoint:
  dir: /path/to/checkpoints

logging:
  log_dir: /path/to/logs
```

### 4. Run Training Pipeline

```bash
# Step 1: Split dataset into train/val (creates relative paths in split files)
python -m utils.split

# Step 2: Train
python -m training.train

# Step 3: Inference
python -m evaluation.inference --input photo.jpg --output result.png
```

---

## Cloud Path Notes

- **Split files** (`train_input.txt`, etc.) use relative paths - works on any machine
- **Config** - change `dataset_root` to absolute path on cloud VM
- **All other paths** in config can be absolute or relative

---

## Common Cloud Commands

```bash
# Google Cloud
gcloud compute ssh your-vm --zone=us-central1-a
gsutil rsync gs://bucket/dataset/ ./dataset/

# AWS
aws s3 sync s3://bucket/dataset/ ./dataset/

# Resume training
python -m training.train --resume
```

---

## Quick Commands Summary

| Step | Command |
|------|---------|
| Split data | `python -m utils.split` |
| Train | `python -m training.train` |
| Resume | `python -m training.train --resume` |
| Inference | `python -m evaluation.inference --input img.jpg --output out.png` |