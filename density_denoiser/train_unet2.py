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
from torch.utils.data import DataLoader, Subset

from .dataset import DensityPairDataset, manifest_path, read_manifest
from .landscape import (
    LandscapeDataset,
    landscape_distillation_loss,
    radial_mask,
    render_candidates,
)
from .model import ResidualDensityDenoiser, spatial_gradient
from .train import _pearson


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def validate_density(model: torch.nn.Module, loader: DataLoader,
                     device: torch.device) -> dict[str, float]:
    model.eval()
    losses, correlations = [], []
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(inputs)
        losses.extend((prediction.float() - targets).square().flatten(1).mean(dim=1).cpu().tolist())
        correlations.extend(_pearson(prediction.float(), targets).cpu().tolist())
    return {
        "validation_l2": _mean(losses),
        "validation_correlation": _mean(correlations),
    }


@torch.no_grad()
def validate_landscape(model: torch.nn.Module, loader: DataLoader,
                       device: torch.device, mask: torch.Tensor,
                       args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    collected: dict[str, list[float]] = {}
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(inputs)
        candidates = render_candidates(
            batch["positions"].to(device, non_blocking=True),
            batch["sigma2"].to(device, non_blocking=True),
            batch["weights"].to(device, non_blocking=True),
            batch["atom_mask"].to(device, non_blocking=True),
            size=args.patch_size, spacing=args.spacing,
        )
        _loss, metrics = landscape_distillation_loss(
            prediction, targets, candidates, mask,
            minimum_oracle_gap=args.minimum_oracle_gap,
            margin_fraction=args.margin_fraction,
            ranking_weight=args.ranking_weight,
        )
        batch_size = len(inputs)
        for key, value in metrics.items():
            collected.setdefault(key, []).extend([float(value.cpu())] * batch_size)
    return {f"validation_landscape_{key}": _mean(values) for key, values in collected.items()}


def _checkpoint_payload(epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        scheduler: torch.optim.lr_scheduler.LRScheduler,
                        base_channels: int, args: argparse.Namespace,
                        best_landscape: float, stale_epochs: int,
                        baseline_validation_l2: float,
                        best_native_top1: float) -> dict:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "base_channels": base_channels,
        "best_landscape": best_landscape,
        "best_native_top1": best_native_top1,
        "stale_epochs": stale_epochs,
        "baseline_validation_l2": baseline_validation_l2,
        "args": vars(args),
        "model_name": "unet2_landscape",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="U-Net 2.0: full-dataset denoising plus ARG/MET/ASP landscape supervision"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--landscape-cache", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--initialize", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--landscape-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    parser.add_argument("--landscape-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--margin-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-oracle-gap", type=float, default=1e-4)
    parser.add_argument("--max-native-render-mse", type=float, default=1e-5)
    parser.add_argument("--min-oracle-native-top1", type=float, default=0.999)
    parser.add_argument("--density-guardrail-fraction", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--min-top1-delta", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--landscape-radius", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-density-train", type=int, default=0)
    parser.add_argument("--max-density-validation", type=int, default=0)
    parser.add_argument("--max-landscape-train", type=int, default=0)
    parser.add_argument("--max-landscape-validation", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    split = json.loads(args.split_json.read_text())
    train_proteins = set(split["train_proteins"])
    validation_proteins = set(split["validation_proteins"])
    if train_proteins & validation_proteins:
        raise ValueError("protein leakage between train and validation split")
    records = read_manifest(manifest_path(args.data_root, "train", "crystal"))
    train_records = [row for row in records if row["pdb_id"] in train_proteins]
    validation_records = [row for row in records if row["pdb_id"] in validation_proteins]
    if args.max_density_train:
        train_records = train_records[:args.max_density_train]
    if args.max_density_validation:
        validation_records = validation_records[:args.max_density_validation]

    density_train = DensityPairDataset(
        train_records, augment=True, rotation_augmentation=True
    )
    density_validation = DensityPairDataset(validation_records)
    landscape_train: torch.utils.data.Dataset = LandscapeDataset(args.landscape_cache, "train")
    landscape_validation: torch.utils.data.Dataset = LandscapeDataset(
        args.landscape_cache, "validation"
    )
    if args.max_landscape_train:
        landscape_train = Subset(
            landscape_train, range(min(args.max_landscape_train, len(landscape_train)))
        )
    if args.max_landscape_validation:
        landscape_validation = Subset(
            landscape_validation,
            range(min(args.max_landscape_validation, len(landscape_validation))),
        )

    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    density_train_loader = DataLoader(
        density_train, batch_size=args.batch_size, shuffle=True, **loader_options
    )
    density_validation_loader = DataLoader(
        density_validation, batch_size=args.batch_size, shuffle=False, **loader_options
    )
    landscape_train_loader = DataLoader(
        landscape_train, batch_size=args.landscape_batch_size, shuffle=True, **loader_options
    )
    landscape_validation_loader = DataLoader(
        landscape_validation, batch_size=args.landscape_batch_size, shuffle=False,
        **loader_options,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.serialization.safe_globals([PosixPath]):
        initialization = torch.load(args.initialize, map_location="cpu", weights_only=True)
    base_channels = int(initialization["base_channels"])
    model = ResidualDensityDenoiser(base_channels).to(device)
    model.load_state_dict(initialization["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    mask = radial_mask(
        args.patch_size, args.spacing, args.landscape_radius, device=device
    )

    _atomic_json(args.output / "run_config.json", {
        **vars(args),
        "model_name": "unet2_landscape",
        "supported_landscape_residues": ["ARG", "MET", "ASP"],
        "density_train_sites": len(density_train),
        "density_validation_sites": len(density_validation),
        "landscape_train_sites": len(landscape_train),
        "landscape_validation_sites": len(landscape_validation),
        "train_proteins": len(train_proteins),
        "validation_proteins": len(validation_proteins),
        "device": str(device),
    })
    _atomic_json(args.output / "split.json", split)

    last_path = args.output / "denoiser_last.pt"
    start_epoch = 0
    stale_epochs = 0
    if args.resume and last_path.exists():
        with torch.serialization.safe_globals([PosixPath]):
            checkpoint = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        stale_epochs = int(checkpoint["stale_epochs"])

    baseline_density = validate_density(model, density_validation_loader, device)
    baseline_landscape = validate_landscape(
        model, landscape_validation_loader, device, mask, args
    )
    baseline_validation_l2 = float(baseline_density["validation_l2"])
    if baseline_landscape["validation_landscape_native_render_mse"] > args.max_native_render_mse:
        raise RuntimeError(
            "landscape cache failed native-render calibration: "
            f"MSE={baseline_landscape['validation_landscape_native_render_mse']:.6g}"
        )
    if baseline_landscape["validation_landscape_oracle_native_top1"] < args.min_oracle_native_top1:
        raise RuntimeError(
            "landscape cache failed oracle-ordering calibration: "
            f"native_top1={baseline_landscape['validation_landscape_oracle_native_top1']:.6g}"
        )
    guardrail = baseline_validation_l2 * (1.0 + args.density_guardrail_fraction)
    best_landscape = float(baseline_landscape["validation_landscape_loss"])
    best_native_top1 = float(baseline_landscape["validation_landscape_native_top1"])
    initial_row = {
        "epoch": -1,
        "training_density_loss": float("nan"),
        "training_landscape_loss": float("nan"),
        **baseline_density,
        **baseline_landscape,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "density_guardrail": guardrail,
        "selected": True,
    }
    initial_payload = _checkpoint_payload(
        -1, model, optimizer, scheduler, base_channels, args,
        best_landscape, stale_epochs, baseline_validation_l2, best_native_top1,
    )
    _atomic_checkpoint(args.output / "denoiser_best_landscape.pt", initial_payload)

    fieldnames = list(initial_row)
    log_path = args.output / "training_log.csv"
    write_header = not args.resume or not log_path.exists()
    with log_path.open("w" if write_header else "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            writer.writerow(initial_row)
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(initial_row, sort_keys=True), flush=True)

        density_steps = len(density_train_loader)
        landscape_steps = len(landscape_train_loader)
        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, args.epochs):
            last_epoch = epoch
            model.train()
            density_losses: list[float] = []
            landscape_losses: list[float] = []
            landscape_iterator = iter(landscape_train_loader)
            landscape_accumulator = 0
            for density_batch in density_train_loader:
                inputs = density_batch["input"].to(device, non_blocking=True)
                targets = density_batch["target"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    prediction = model(inputs)
                    l2 = (prediction - targets).square().mean()
                    gradient = (
                        spatial_gradient(prediction) - spatial_gradient(targets)
                    ).square().mean()
                    density_loss = l2 + args.gradient_weight * gradient
                landscape_accumulator += landscape_steps
                use_landscape = (
                    args.landscape_weight > 0
                    and landscape_accumulator >= density_steps
                )
                landscape_loss = None
                if use_landscape:
                    landscape_accumulator -= density_steps
                    try:
                        landscape_batch = next(landscape_iterator)
                    except StopIteration:
                        use_landscape = False
                if use_landscape:
                    landscape_inputs = landscape_batch["input"].to(
                        device, non_blocking=True
                    )
                    landscape_targets = landscape_batch["target"].to(
                        device, non_blocking=True
                    )
                    with torch.autocast(
                        device_type=device.type, enabled=device.type == "cuda"
                    ):
                        landscape_prediction = model(landscape_inputs)
                    with torch.no_grad():
                        candidates = render_candidates(
                            landscape_batch["positions"].to(device, non_blocking=True),
                            landscape_batch["sigma2"].to(device, non_blocking=True),
                            landscape_batch["weights"].to(device, non_blocking=True),
                            landscape_batch["atom_mask"].to(device, non_blocking=True),
                            size=args.patch_size, spacing=args.spacing,
                        )
                    landscape_loss, _metrics = landscape_distillation_loss(
                        landscape_prediction, landscape_targets, candidates, mask,
                        minimum_oracle_gap=args.minimum_oracle_gap,
                        margin_fraction=args.margin_fraction,
                        ranking_weight=args.ranking_weight,
                    )
                total_loss = density_loss + (
                    args.landscape_weight * landscape_loss
                    if landscape_loss is not None else 0.0
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                density_losses.append(float(density_loss.detach().cpu()))
                if landscape_loss is not None:
                    landscape_losses.append(float(landscape_loss.detach().cpu()))

            density_metrics = validate_density(model, density_validation_loader, device)
            landscape_metrics = validate_landscape(
                model, landscape_validation_loader, device, mask, args
            )
            scheduler.step()
            current_landscape = float(landscape_metrics["validation_landscape_loss"])
            current_native_top1 = float(
                landscape_metrics["validation_landscape_native_top1"]
            )
            better_top1 = current_native_top1 > best_native_top1 + args.min_top1_delta
            tied_top1_better_loss = (
                current_native_top1 >= best_native_top1 - args.min_top1_delta
                and current_landscape < best_landscape - args.min_delta
            )
            improved = (
                density_metrics["validation_l2"] <= guardrail
                and (better_top1 or tied_top1_better_loss)
            )
            if improved:
                best_landscape = current_landscape
                best_native_top1 = max(best_native_top1, current_native_top1)
                stale_epochs = 0
            else:
                stale_epochs += 1
            row = {
                "epoch": epoch,
                "training_density_loss": _mean(density_losses),
                "training_landscape_loss": _mean(landscape_losses),
                **density_metrics,
                **landscape_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "density_guardrail": guardrail,
                "selected": improved,
            }
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            payload = _checkpoint_payload(
                epoch, model, optimizer, scheduler, base_channels, args,
                best_landscape, stale_epochs, baseline_validation_l2, best_native_top1,
            )
            _atomic_checkpoint(last_path, payload)
            if improved:
                _atomic_checkpoint(args.output / "denoiser_best_landscape.pt", payload)
            _atomic_json(args.output / "progress.json", {
                "status": "running", "epoch": epoch, "best_landscape": best_landscape,
                "best_native_top1": best_native_top1,
                "stale_epochs": stale_epochs, "latest": row,
            })
            print(json.dumps(row, sort_keys=True), flush=True)
            if stale_epochs >= args.patience:
                break

    _atomic_json(args.output / "progress.json", {
        "status": "complete", "last_epoch": last_epoch,
        "best_landscape": best_landscape, "best_native_top1": best_native_top1,
        "stale_epochs": stale_epochs,
    })


if __name__ == "__main__":
    main()
