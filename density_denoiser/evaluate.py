from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path, PosixPath

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .dataset import DensityPairDataset, manifest_path, read_manifest
from .model import ResidualDensityDenoiser


def pearson(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        first, second = first[mask], second[mask]
    else:
        first, second = first.flatten(), second.flatten()
    first, second = first - first.mean(), second - second.mean()
    return float((first * second).sum() / (first.square().sum().sqrt() * second.square().sum().sqrt()).clamp_min(1e-8))


def gaussian_blur(volume: torch.Tensor) -> torch.Tensor:
    axis = torch.arange(-2, 3, dtype=volume.dtype, device=volume.device)
    kernel = torch.exp(-0.5 * axis.square())
    kernel = kernel / kernel.sum()
    kernel3 = torch.einsum("i,j,k->ijk", kernel, kernel, kernel)[None, None]
    return functional.conv3d(volume[None], kernel3, padding=2)[0]


def save_visual(path: Path, raw: np.ndarray, denoised: np.ndarray,
                target: np.ndarray, title: str) -> None:
    index = raw.shape[0] // 2
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    limit = max(np.percentile(np.abs(raw), 99), np.percentile(np.abs(target), 99))
    for axis, image, label in zip(
        axes, (raw[index], denoised[index], target[index]),
        ("Experimental", "Denoised", "Synthetic target"),
    ):
        axis.imshow(image.T, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a denoiser on held-out proteins")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("QFIT_UNET_DATA", Path.home() / "qfit_unet_data")),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-visuals", type=int, default=20)
    parser.add_argument("--frame", choices=("crystal", "residue"), default="crystal")
    args = parser.parse_args()
    model_dir = "model" if args.frame == "crystal" else "model_canonical"
    evaluation_dir = "evaluation" if args.frame == "crystal" else "evaluation_canonical"
    checkpoint_path = (
        args.checkpoint
        or args.data_root / "density_denoiser" / model_dir / "denoiser_best.pt"
    )
    output = args.output or args.data_root / "density_denoiser" / evaluation_dir
    output.mkdir(parents=True, exist_ok=True)
    records = read_manifest(manifest_path(args.data_root, "test", args.frame))
    dataset = DensityPairDataset(records)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Keep restricted loading enabled; only the locally saved PosixPath values
    # in the argparse dictionary need to be explicitly allowlisted.
    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    model = ResidualDensityDenoiser(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metrics_path = output / "reconstruction_metrics.csv"
    rows = []
    completed: set[str] = set()
    if metrics_path.exists() and metrics_path.stat().st_size:
        with metrics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        completed = {row["key"] for row in rows}
        for row in rows:
            row["is_altloc"] = str(row["is_altloc"]).lower() == "true"
            for key in list(row):
                if key.endswith("l2") or key.endswith("correlation"):
                    row[key] = float(row[key])
    fieldnames = [
        "key", "pdb_id", "is_altloc",
        "identity_l2", "denoised_l2", "blur_l2",
        "identity_correlation", "denoised_correlation", "blur_correlation",
        "identity_local_correlation", "denoised_local_correlation",
        "blur_local_correlation",
    ]
    mode = "a" if completed else "w"
    with metrics_path.open(mode, newline="") as handle, torch.no_grad():
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not completed:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())
        for index, sample in enumerate(dataset):
            if sample["key"] in completed:
                continue
            raw = sample["input"].to(device)
            target = sample["target"].to(device)
            mask = sample["local_mask"].to(device).bool()
            denoised = model(raw[None])[0]
            blurred = gaussian_blur(raw)
            row = {
                "key": sample["key"],
                "pdb_id": sample["pdb_id"],
                "is_altloc": sample["is_altloc"],
                "identity_l2": float((raw - target).square().mean()),
                "denoised_l2": float((denoised - target).square().mean()),
                "blur_l2": float((blurred - target).square().mean()),
                "identity_correlation": pearson(raw, target),
                "denoised_correlation": pearson(denoised, target),
                "blur_correlation": pearson(blurred, target),
                "identity_local_correlation": pearson(raw, target, mask),
                "denoised_local_correlation": pearson(denoised, target, mask),
                "blur_local_correlation": pearson(blurred, target, mask),
            }
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            if index < args.max_visuals:
                save_visual(
                    output / "visual_comparison" / f"{sample['key']}.png",
                    raw[0].cpu().numpy(), denoised[0].cpu().numpy(),
                    target[0].cpu().numpy(), sample["key"],
                )
    metric_keys = [
        key for key in rows[0]
        if key.endswith("l2") or key.endswith("correlation")
    ]
    summary = {}
    for label, selected in (
        ("all", rows),
        ("altloc", [row for row in rows if row["is_altloc"]]),
        ("negative", [row for row in rows if not row["is_altloc"]]),
    ):
        summary[label] = {
            "count": len(selected),
            **{
                key: float(np.mean([row[key] for row in selected]))
                for key in metric_keys
            },
        }
    (output / "reconstruction_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
