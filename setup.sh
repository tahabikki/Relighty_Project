#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Relighty — macOS / Linux setup (Python 3.12)
#  Run once: bash setup.sh
#  Then activate: source .venv/bin/activate
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo ""
echo " Relighty — macOS / Linux Setup (Python 3.12)"
echo " ═════════════════════════════════════════════"
echo ""

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"

# Check Python 3.12+
if ! command -v python3.12 &>/dev/null; then
    echo " ERROR: python3.12 not found. Install Python 3.12+ first."
    echo " macOS (Homebrew): brew install python@3.12"
    echo " Ubuntu/Debian:    sudo apt install python3.12 python3.12-venv"
    exit 1
fi

# Remove old venv if it exists
if [ -d ".venv" ]; then
    echo " [0/4] Removing old .venv ..."
    rm -rf .venv
fi

# Create virtual environment
echo " [1/4] Creating .venv with Python 3.12 ..."
python3.12 -m venv .venv

# Activate
echo " [2/4] Activating .venv ..."
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Install PyTorch — platform-specific
echo " [3/4] Installing PyTorch ..."
if [[ "$OS" == "Darwin" ]]; then
    # macOS — use standard PyPI build (supports MPS on Apple Silicon)
    echo "       Detected macOS ($ARCH) — installing with MPS support"
    pip install torch torchvision --quiet
else
    # Linux — CUDA 12.4
    echo "       Detected Linux — installing with CUDA 12.4"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
fi

# Install remaining dependencies
echo " [4/4] Installing dependencies (MediaPipe Tasks, etc.) ..."
pip install -r requirements.txt --quiet

echo ""
echo " ════════════════════════════════"
echo " Setup complete!"
echo ""
echo " To activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo " Workflow:"
echo "   python utils/split.py           # create train/val split — once"
echo "   python train/train.py           # train"
echo "   python evaluation/inference.py --input Results/input"
echo "   python evaluation/evaluate.py   # metrics"
echo " ════════════════════════════════"
echo ""
