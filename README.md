# Relighty

Face and neck shadow removal using a PyTorch shadow-removal model plus BiSeNet face parsing.

## Setup

```bash
git clone https://github.com/tahabikki/Relighty_Project.git
cd Relighty_Project

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install PyTorch, then the project dependencies:

```bash
# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# or CPU only
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

## Dataset

Put your training images here, or edit `configs/config.yaml` and change `data.dataset_root`.

```text
dataset/
|-- input/
`-- target/
```

## Train

```bash
python -m utils.split
python -m training.train
```

Resume:

```bash
python -m training.train --resume
```

## Inference

Use the trained checkpoint:

```text
checkpoints/shadow_removal_best.pth
```

Single image:

```bash
python -m evaluation.inference --input photo.jpg --output result.png
```

Keep original background:

```bash
python -m evaluation.inference --input photo.jpg --output result.jpg --keep-bg
```

Batch folder:

```bash
python -m evaluation.inference --input Results/input --output Results/output
```

Custom checkpoint:

```bash
python -m evaluation.inference --input photo.jpg --output result.png --checkpoint path/to/model.pth
```

## Notes

- BiSeNet weights download automatically if missing:
  - `Masking/weights/resnet18.pt`
  - `Masking/weights/resnet34.pt`
- Do not commit datasets, checkpoints, logs, `.venv`, `.pt`, `.pth`, or `.onnx` files.
