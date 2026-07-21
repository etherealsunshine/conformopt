from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
from pathlib import Path, PosixPath

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import (
    DensityPairDataset,
    manifest_path,
    protein_train_validation_split,
    read_manifest,
)
from .model import ResidualDensityDenoiser, spatial_gradient


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _pearson(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    prediction = prediction - prediction.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    return (prediction * target).sum(dim=1) / (
        prediction.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
    ).clamp_min(1e-8)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    losses, correlations = [], []
    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        prediction = model(inputs)
        losses.extend((prediction - targets).square().flatten(1).mean(dim=1).cpu().tolist())
        correlations.extend(_pearson(prediction, targets).cpu().tolist())
    return {
        "validation_l2": float(np.mean(losses)),
        "validation_correlation": float(np.mean(correlations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the residual 3D density denoiser")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("QFIT_UNET_DATA", Path.home() / "qfit_unet_data")),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--min-delta", type=float, default=0.0,
        help="minimum validation-L2 decrease required to reset early stopping",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--frame", choices=("crystal", "residue"), default="crystal")
    parser.add_argument(
        "--rotation-augmentation", action=argparse.BooleanOptionalAction, default=None,
        help="default: enabled for crystal-frame data and disabled for residue-frame data",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    default_model = "model" if args.frame == "crystal" else "model_canonical"
    output = args.output or args.data_root / "density_denoiser" / default_model
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = read_manifest(manifest_path(args.data_root, "train", args.frame))
    train_records, validation_records = protein_train_validation_split(
        records, args.validation_fraction, args.seed
    )
    if not train_records or not validation_records:
        raise RuntimeError("training requires at least two proteins for a protein-level validation split")
    (output / "split.json").write_text(json.dumps({
        "train_proteins": sorted({item["pdb_id"] for item in train_records}),
        "validation_proteins": sorted({item["pdb_id"] for item in validation_records}),
        "frame": args.frame,
        "test_manifest": str(manifest_path(args.data_root, "test", args.frame)),
    }, indent=2))
    rotation_augmentation = (
        args.frame == "crystal"
        if args.rotation_augmentation is None
        else args.rotation_augmentation
    )
    train_loader = DataLoader(
        DensityPairDataset(
            train_records, augment=True,
            rotation_augmentation=rotation_augmentation,
        ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        DensityPairDataset(validation_records), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualDensityDenoiser(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch, best_loss, stale_epochs = 0, float("inf"), 0
    last_path = output / "denoiser_last.pt"
    if args.resume and last_path.exists():
        # The saved argparse namespace contains PosixPath values. Keep the
        # restricted tensor loader enabled and allowlist only that known type.
        with torch.serialization.safe_globals([PosixPath]):
            checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        stale_epochs = checkpoint["stale_epochs"]

    log_path = output / "training_log.csv"
    write_header = not log_path.exists() or start_epoch == 0
    mode = "w" if write_header else "a"
    with log_path.open(mode, newline="") as log_handle:
        writer = csv.DictWriter(log_handle, fieldnames=[
            "epoch", "training_loss", "validation_l2", "validation_correlation", "learning_rate"
        ])
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            model.train()
            training_losses = []
            for batch in train_loader:
                inputs = batch["input"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    prediction = model(inputs)
                    l2 = (prediction - targets).square().mean()
                    gradient = (
                        spatial_gradient(prediction) - spatial_gradient(targets)
                    ).square().mean()
                    loss = l2 + args.gradient_weight * gradient
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                training_losses.append(float(loss.detach().cpu()))
            metrics = validate(model, validation_loader, device)
            scheduler.step()
            row = {
                "epoch": epoch,
                "training_loss": float(np.mean(training_losses)),
                **metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            log_handle.flush()
            os.fsync(log_handle.fileno())
            improved = metrics["validation_l2"] < best_loss - args.min_delta
            if improved:
                best_loss, stale_epochs = metrics["validation_l2"], 0
            else:
                stale_epochs += 1
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
                "stale_epochs": stale_epochs,
                "base_channels": args.base_channels,
                "args": vars(args),
            }
            _atomic_checkpoint(last_path, checkpoint)
            if improved:
                _atomic_checkpoint(output / "denoiser_best.pt", checkpoint)
            print(json.dumps(row, sort_keys=True), flush=True)
            if stale_epochs >= args.patience:
                break


if __name__ == "__main__":
    main()
