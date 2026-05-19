@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  Relighty — Windows setup (Python 3.12, CUDA 12.4)
REM  Run once: setup.bat
REM  Then activate: .venv\Scripts\activate.bat
REM ─────────────────────────────────────────────────────────────────────────────

echo.
echo  Relighty — Windows Setup (Python 3.12, CUDA 12.4)
echo  ════════════════════════════════════════════════
echo.

REM Check Python 3.12+
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Python not found. Install Python 3.12+ from https://python.org
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Python 3.12 or newer is required.
    python --version
    pause
    exit /b 1
)

REM Remove old venv if it exists
if exist .venv (
    echo  [0/4] Removing old .venv ...
    rmdir /s /q .venv
)

REM Create virtual environment
echo  [1/4] Creating .venv with Python 3.12 ...
python -m venv .venv
if %errorlevel% neq 0 (
    echo  ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate
echo  [2/4] Activating .venv ...
call .venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip --quiet

REM Install PyTorch with CUDA 12.4
echo  [3/4] Installing PyTorch (CUDA 12.4) ...
echo        This may take a few minutes...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
if %errorlevel% neq 0 (
    echo  ERROR: Failed to install PyTorch. Check your internet connection.
    pause
    exit /b 1
)

REM Install remaining dependencies
echo  [4/4] Installing dependencies (MediaPipe Tasks, etc.) ...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo  ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo  ════════════════════════════════════════════════
echo  Setup complete!
echo.
echo  To activate the environment:
echo    .venv\Scripts\activate.bat
echo.
echo  Workflow:
echo    python utils\split.py           (create train/val split — once)
echo    python train\train.py           (train)
echo    python evaluation\inference.py --input Results\input
echo    python evaluation\evaluate.py   (metrics)
echo  ════════════════════════════════════════════════
echo.
pause
