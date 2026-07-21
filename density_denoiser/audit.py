from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import manifest_path, read_manifest


def _sample_records(records: list[dict], count: int, seed: int) -> list[dict]:
    if count >= len(records):
        return records
    return random.Random(seed).sample(records, count)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def _audit_split(records: list[dict], samples: int, seed: int) -> tuple[dict, list[str]]:
    errors: list[str] = []
    keys = [record["key"] for record in records]
    duplicate_keys = sum(count - 1 for count in Counter(keys).values() if count > 1)
    if duplicate_keys:
        errors.append(f"manifest contains {duplicate_keys} duplicate keys")

    sampled = _sample_records(records, samples, seed)
    shapes: Counter[tuple[int, ...]] = Counter()
    input_means: list[float] = []
    input_stds: list[float] = []
    target_means: list[float] = []
    target_stds: list[float] = []
    correlations: list[float] = []
    missing = 0
    nonfinite = 0
    metadata_mismatches = 0

    for record in sampled:
        path = Path(record["pair_path"])
        if not path.is_file():
            missing += 1
            continue
        with np.load(path) as archive:
            input_patch = archive["input"]
            target_patch = archive["target"]
            local_mask = archive["local_mask"]
            metadata = json.loads(str(archive["metadata"].item()))
        shapes[(input_patch.shape, target_patch.shape, local_mask.shape)] += 1
        if not (
            np.isfinite(input_patch).all()
            and np.isfinite(target_patch).all()
            and np.isfinite(local_mask).all()
        ):
            nonfinite += 1
            continue
        if metadata.get("key") != record["key"]:
            metadata_mismatches += 1
        input_means.append(float(input_patch.mean()))
        input_stds.append(float(input_patch.std()))
        target_means.append(float(target_patch.mean()))
        target_stds.append(float(target_patch.std()))
        correlations.append(_correlation(input_patch, target_patch))

    expected_shapes = {((1, 32, 32, 32), (1, 32, 32, 32), (1, 32, 32, 32))}
    observed_shapes = set(shapes)
    if observed_shapes != expected_shapes:
        errors.append(f"unexpected sampled tensor shapes: {sorted(map(str, observed_shapes))}")
    if missing:
        errors.append(f"{missing} sampled pair files are missing")
    if nonfinite:
        errors.append(f"{nonfinite} sampled pairs contain non-finite values")
    if metadata_mismatches:
        errors.append(f"{metadata_mismatches} sampled metadata keys do not match the manifest")

    def summary(values: list[float]) -> dict:
        if not values:
            return {"mean": math.nan, "min": math.nan, "max": math.nan}
        array = np.asarray(values)
        return {
            "mean": float(array.mean()),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    return {
        "records": len(records),
        "proteins": len({record["pdb_id"] for record in records}),
        "altloc_records": sum(bool(record["is_altloc"]) for record in records),
        "negative_records": sum(not bool(record["is_altloc"]) for record in records),
        "duplicate_keys": duplicate_keys,
        "sampled": len(sampled),
        "sampled_missing": missing,
        "sampled_nonfinite": nonfinite,
        "sampled_metadata_mismatches": metadata_mismatches,
        "sampled_shapes": {str(key): value for key, value in shapes.items()},
        "input_mean": summary(input_means),
        "input_std": summary(input_stds),
        "target_mean": summary(target_means),
        "target_std": summary(target_stds),
        "input_target_correlation": summary(correlations),
    }, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit density-denoiser manifests and patches")
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("QFIT_UNET_DATA", Path.home() / "qfit_unet_data")),
    )
    parser.add_argument("--samples-per-split", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--frame", choices=("crystal", "residue"), default="crystal")
    args = parser.parse_args()

    train = read_manifest(manifest_path(args.data_root, "train", args.frame))
    test = read_manifest(manifest_path(args.data_root, "test", args.frame))
    train_summary, train_errors = _audit_split(
        train, args.samples_per_split, args.seed
    )
    test_summary, test_errors = _audit_split(
        test, args.samples_per_split, args.seed + 1
    )
    overlap = sorted(
        {record["pdb_id"] for record in train}
        & {record["pdb_id"] for record in test}
    )
    errors = [
        *(f"train: {error}" for error in train_errors),
        *(f"test: {error}" for error in test_errors),
    ]
    if overlap:
        errors.append(f"train/test protein leakage: {overlap}")
    result = {
        "status": "passed" if not errors else "failed",
        "frame": args.frame,
        "train": train_summary,
        "test": test_summary,
        "train_test_protein_overlap": overlap,
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
