from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import DensityPairDataset, manifest_path, read_manifest
from .model import ResidualDensityDenoiser, spatial_gradient


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark denoiser training throughput")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("QFIT_UNET_DATA", Path.home() / "qfit_unet_data")),
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--frame", choices=("crystal", "residue"), default="crystal")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the H100 benchmark")

    records = read_manifest(manifest_path(args.data_root, "train", args.frame))
    device = torch.device("cuda")
    results = []
    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = ResidualDensityDenoiser(args.base_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = DataLoader(
            DensityPairDataset(
                records, augment=True,
                rotation_augmentation=args.frame == "crystal",
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        iterator = iter(loader)
        completed = 0
        started = None
        try:
            for step in range(args.warmup_steps + args.steps):
                batch = next(iterator)
                inputs = batch["input"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda"):
                    prediction = model(inputs)
                    l2 = (prediction - targets).square().mean()
                    gradient = (
                        spatial_gradient(prediction) - spatial_gradient(targets)
                    ).square().mean()
                    loss = l2 + 0.1 * gradient
                loss.backward()
                optimizer.step()
                if step + 1 == args.warmup_steps:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                elif step >= args.warmup_steps:
                    completed += 1
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            result = {
                "status": "passed",
                "base_channels": args.base_channels,
                "frame": args.frame,
                "batch_size": batch_size,
                "steps": completed,
                "seconds": elapsed,
                "batches_per_second": completed / elapsed,
                "samples_per_second": completed * batch_size / elapsed,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
            }
        except torch.OutOfMemoryError as error:
            result = {
                "status": "out_of_memory",
                "base_channels": args.base_channels,
                "frame": args.frame,
                "batch_size": batch_size,
                "error": str(error),
            }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        del iterator, loader, optimizer, model
        torch.cuda.empty_cache()

    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
