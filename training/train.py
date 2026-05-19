#!/usr/bin/env python
"""
Train the face shadow removal model.

Quick start
──────────
    # Step 1 – create train/val split (only once)
    python -m utils.split

    # Step 2 – train
    python -m training.train

    # Step 3 – resume after interruption
    python -m training.train --resume

Common overrides
────────────────
    python -m training.train --epochs 150 --batch 16
    python -m training.train --batch 4 --workers 0   # low-memory GPU / Windows
    python -m training.train --device cpu             # CPU-only
    python -m training.train --device mps             # Apple Silicon

All paths are configured in configs/config.yaml - edit it to deploy anywhere.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel",      "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import argparse
import platform
import time
import warnings
import csv
warnings.filterwarnings("ignore")

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import get_config, get_split_path, get_checkpoint_path, get_log_path
from models.shadow_remover import ShadowRemovalNet
from training.dataset import ShadowDataset
from training.losses import CombinedLoss

config = get_config()

W = 60  # console width


def parse_args():
    p = argparse.ArgumentParser(description="Shadow removal training")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch",      type=int,   default=8,
                   help="Batch size (reduce to 4 for low VRAM)")
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--enc_lr",     type=float, default=1e-5,
                   help="Encoder LR (smaller → preserve ImageNet init)")
    p.add_argument("--workers",    type=int,   default=4,
                   help="DataLoader workers (use 0 on Windows)")
    p.add_argument("--save_every", type=int,   default=10,
                   help="Save a numbered checkpoint every N epochs")
    p.add_argument("--img_size",   type=int,   default=256)
    p.add_argument("--no_amp",     action="store_true",
                   help="Disable mixed-precision training")
    p.add_argument("--device",     default="auto",
                   help="cuda | mps | cpu | auto")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from the latest checkpoint")
    return p.parse_args()


def _check_split_files():
    required = [
        get_split_path("train_input.txt"),
        get_split_path("train_target.txt"),
        get_split_path("val_input.txt"),
        get_split_path("val_target.txt"),
    ]
    missing = [f for f in required if not f.exists()]
    if missing:
        print("  ERROR: Split files not found. Run first:")
        print("    python -m utils.split")
        sys.exit(1)


def _bar(label: str, char: str = "-") -> str:
    pad = W - len(label) - 2
    return f"  {label} {char * pad}"


def _setup_logging(log_dir: Path):
    """Create logs directory and return path to CSV log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "best_loss",
            "lr", "grad_norm", "time_sec", "is_best"
        ])

    return log_file


def train():
    args = parse_args()
    _check_split_files()

    data_cfg = config.get("data") or {}
    train_cfg = config.get("training") or {}
    dl_cfg = config.get("dataloader") or {}

    if args.img_size == 256:
        args.img_size = data_cfg.get("image_size", 256)

    if args.batch == 8:
        args.batch = train_cfg.get("batch_size", args.batch)
    if args.epochs == 100:
        args.epochs = train_cfg.get("epochs", args.epochs)
    if abs(args.lr - 1e-4) < 1e-12:
        args.lr = train_cfg.get("lr", args.lr)
    if abs(args.enc_lr - 1e-5) < 1e-12:
        args.enc_lr = train_cfg.get("enc_lr", args.enc_lr)
    if args.workers == 4:
        args.workers = dl_cfg.get("num_workers", args.workers)
    if args.device == "auto":
        args.device = train_cfg.get("device", args.device)
    args.no_amp = train_cfg.get("no_amp", args.no_amp)

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    use_amp = (not args.no_amp) and (device.type == "cuda")

    if platform.system() == "Windows" and args.workers > 0:
        args.workers = min(args.workers, 4)
    use_persistent = args.workers > 0 and platform.system() != "Windows"

    print(f"\n  {'=' * W}")
    print(f"  {'Relighty - Face Shadow Removal':^{W}}")
    print(f"  {'=' * W}")
    gpu_name = (
        torch.cuda.get_device_name(0) if device.type == "cuda"
        else device.type.upper()
    )
    print(f"  Device   : {device}  ({gpu_name})")
    print(f"  AMP      : {'on' if use_amp else 'off'}")
    print(f"  Epochs   : {args.epochs}   Batch : {args.batch}   LR : {args.lr}")
    print(f"  {'-' * W}")

    # Checkpoints
    CKPT_DIR = get_checkpoint_path()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    latest_ckpt = CKPT_DIR / "shadow_removal_latest.pth"
    best_ckpt   = CKPT_DIR / "shadow_removal_best.pth"

    # Logging
    log_dir = get_log_path()
    log_file = _setup_logging(log_dir)
    print(f"  Logs     : {log_file}")

    # Early stopping
    early_stop_cfg = config.get("early_stopping") or {}
    early_stop_enabled = early_stop_cfg.get("enabled", True)
    early_stop_patience = early_stop_cfg.get("patience", 15)
    patience_counter = 0

    # Data
    print("  Loading dataset ...", end=" ", flush=True)
    train_ds = ShadowDataset(
        str(get_split_path("train_input.txt")),
        str(get_split_path("train_target.txt")),
        augment=True,
        image_size=args.img_size,
    )
    val_ds = ShadowDataset(
        str(get_split_path("val_input.txt")),
        str(get_split_path("val_target.txt")),
        augment=False,
        image_size=args.img_size,
    )
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
        persistent_workers=use_persistent,
    )

    # Model
    print("  Building model ...", end=" ", flush=True)
    model = ShadowRemovalNet().to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ShadowRemovalNet  {total_params:.1f}M params")

    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder_parameters()), "lr": args.enc_lr},
        {"params": list(model.decoder_parameters()), "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    scaler    = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    criterion = CombinedLoss(l1_weight=1.0, ssim_weight=0.5).to(device)

    # Resume
    start_epoch   = 1
    best_val_loss = float("inf")

    if args.resume and latest_ckpt.exists():
        print(f"  Resuming from {latest_ckpt.name} ...", end=" ", flush=True)
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"epoch {ckpt['epoch']}  best_val={best_val_loss:.4f}")

    # Training loop
    n_train = len(train_loader)
    n_val   = len(val_loader)

    print(f"  {'-' * W}")
    print(f"  Starting - {args.epochs - start_epoch + 1} epochs  "
          f"({n_train} train batches / {n_val} val batches)\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0.0
        grad_norm_avg = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"  [{epoch:>3}/{args.epochs}] train",
            unit="it",
            ncols=85,
            leave=False,
        )
        for inp, tgt, mask in pbar:
            inp  = inp.to(device,  non_blocking=True)
            tgt  = tgt.to(device,  non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(inp)
                loss = criterion(pred, tgt, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            has_nan_inf = any(
                torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                for p in model.parameters() if p.grad is not None
            )
            if has_nan_inf:
                scaler.update()
                gn = float('nan')
            else:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item()
                scaler.step(optimizer)
                scaler.update()
            scaler.update()

            bl = loss.item()
            train_loss   += bl
            grad_norm_avg += gn
            pbar.set_postfix(loss=f"{bl:.4f}", gn=f"{gn:.2f}", refresh=False)
        pbar.close()

        train_loss    /= n_train
        grad_norm_avg /= n_train

        # Validate
        model.eval()
        val_loss = 0.0

        pbar = tqdm(
            val_loader,
            desc=f"  [{epoch:>3}/{args.epochs}]  val ",
            unit="it",
            ncols=85,
            leave=False,
        )
        with torch.no_grad():
            for inp, tgt, mask in pbar:
                inp  = inp.to(device,  non_blocking=True)
                tgt  = tgt.to(device,  non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pred = model(inp)
                    loss = criterion(pred, tgt, mask)
                vl = loss.item()
                val_loss += vl
                pbar.set_postfix(loss=f"{vl:.4f}", refresh=False)
        pbar.close()

        val_loss /= n_val
        scheduler.step()

        elapsed = time.time() - t0
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        cur_lr = optimizer.param_groups[1]["lr"]

        star = "  ★" if is_best else ""
        print(
            f"  [{epoch:>3}/{args.epochs}]"
            f"  loss {train_loss:.4f} → {val_loss:.4f}"
            f"  best {best_val_loss:.4f}"
            f"  gn {grad_norm_avg:.2f}"
            f"  lr {cur_lr:.1e}"
            f"  {elapsed:.0f}s"
            f"{star}"
        )

        # Save checkpoints
        state = dict(
            epoch         = epoch,
            model         = model.state_dict(),
            optimizer     = optimizer.state_dict(),
            scheduler     = scheduler.state_dict(),
            scaler        = scaler.state_dict(),
            best_val_loss = best_val_loss,
            args          = vars(args),
        )
        torch.save(state, latest_ckpt)

        if is_best:
            torch.save(state, best_ckpt)
            print(f"           -> best  : {best_ckpt.name}")

        if epoch % args.save_every == 0:
            ep_path = CKPT_DIR / f"shadow_removal_epoch_{epoch:03d}.pth"
            torch.save(state, ep_path)
            print(f"           -> ckpt  : {ep_path.name}")

        # Early stopping
        if early_stop_enabled:
            if is_best:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"           ! Early stopping: no improvement for {patience_counter} epochs")
                    break

        # Log to CSV
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{best_val_loss:.6f}", f"{cur_lr:.2e}",
                f"{grad_norm_avg:.4f}", f"{elapsed:.1f}",
                int(is_best)
            ])

    print(f"\n  {'=' * W}")
    print(f"  Training complete!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Best model    : {best_ckpt.name}")
    print(f"  {'-' * W}")
    print(f"  Run inference :")
    print(f"    python -m evaluation.inference --input Results/input")
    print(f"  {'=' * W}\n")


if __name__ == "__main__":
    train()