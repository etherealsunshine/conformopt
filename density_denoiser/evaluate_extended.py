from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path, PosixPath

import gemmi
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import DensityPairDataset, manifest_path, read_manifest
from .landscape import candidate_energies, radial_mask, render_candidates
from .model import ResidualDensityDenoiser
from .prepare_landscape_cache import CANDIDATE_LABELS, _build_site


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _masked_values(volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return volume.flatten()[mask.flatten().bool()]


def _masked_mse(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    return float((_masked_values(first, mask) - _masked_values(second, mask)).square().mean())


def _masked_pearson(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> float:
    first_values = _masked_values(first, mask).float()
    second_values = _masked_values(second, mask).float()
    first_values = first_values - first_values.mean()
    second_values = second_values - second_values.mean()
    denominator = first_values.square().sum().sqrt() * second_values.square().sum().sqrt()
    return float((first_values * second_values).sum() / denominator.clamp_min(1e-8))


def _top_fraction_overlap(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    fraction: float = 0.10,
) -> float:
    prediction_values = _masked_values(prediction, mask)
    target_values = _masked_values(target, mask)
    count = max(1, math.ceil(len(target_values) * fraction))
    prediction_top = set(torch.topk(prediction_values, count).indices.cpu().tolist())
    target_top = set(torch.topk(target_values, count).indices.cpu().tolist())
    return len(prediction_top & target_top) / count


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_ranks, second_ranks = _rankdata(first), _rankdata(second)
    if np.std(first_ranks) == 0 or np.std(second_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def _native_metrics(energies: np.ndarray) -> dict[str, float | int | bool]:
    energies = np.asarray(energies, dtype=np.float64)
    native_rank = int(np.argsort(energies, kind="mergesort").tolist().index(0) + 1)
    native_margin = float(np.min(energies[1:]) - energies[0])
    return {
        "native_rank": native_rank,
        "native_top1": native_rank == 1,
        "native_top3": native_rank <= 3,
        "native_margin": native_margin,
        "ab_beats_a_and_b_only": bool(
            energies[0] < energies[CANDIDATE_LABELS.index("A_only")]
            and energies[0] < energies[CANDIDATE_LABELS.index("B_only")]
        ),
    }


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def _summarize_reconstruction(rows: list[dict]) -> dict:
    metric_keys = [
        "raw_global_mse", "denoised_global_mse",
        "raw_global_pearson", "denoised_global_pearson",
        "raw_local_mse", "denoised_local_mse",
        "raw_local_pearson", "denoised_local_pearson",
        "raw_top10_overlap", "denoised_top10_overlap",
        "local_mse_fractional_improvement",
    ]
    summary = {}
    for label, selected in (
        ("all", rows),
        ("altloc", [row for row in rows if row["is_altloc"]]),
        ("negative", [row for row in rows if not row["is_altloc"]]),
    ):
        summary[label] = {
            "sites": len(selected),
            "proteins": len({row["pdb_id"] for row in selected}),
            **{key: _mean(selected, key) for key in metric_keys},
            "denoised_beats_raw_local_mse_fraction": float(np.mean([
                row["denoised_local_mse"] < row["raw_local_mse"]
                for row in selected
            ])),
        }
    return summary


def _summarize_landscape(rows: list[dict]) -> dict:
    summary: dict[str, object] = {
        "sites": len(rows),
        "candidate_labels": list(CANDIDATE_LABELS),
        "oracle_resolved_sites": sum(row["target_native_top1"] for row in rows),
        "mean_raw_vs_target_spearman": _mean(rows, "raw_vs_target_spearman"),
        "mean_denoised_vs_target_spearman": _mean(rows, "denoised_vs_target_spearman"),
    }
    for source in ("raw", "denoised", "target"):
        summary[source] = {
            "native_top1": sum(row[f"{source}_native_top1"] for row in rows),
            "native_top3": sum(row[f"{source}_native_top3"] for row in rows),
            "ab_beats_a_and_b_only": sum(
                row[f"{source}_ab_beats_a_and_b_only"] for row in rows
            ),
            "mean_native_margin": _mean(rows, f"{source}_native_margin"),
        }
    oracle_rows = [row for row in rows if row["target_native_top1"]]
    summary["oracle_resolved"] = {
        "sites": len(oracle_rows),
        "raw_native_top1": sum(row["raw_native_top1"] for row in oracle_rows),
        "denoised_native_top1": sum(
            row["denoised_native_top1"] for row in oracle_rows
        ),
    }
    return summary


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ResidualDensityDenoiser(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def evaluate_reconstruction(
    model: torch.nn.Module,
    records: list[dict],
    output: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> list[dict]:
    path = output / "reconstruction_metrics.csv"
    rows: list[dict] = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["is_altloc"] = str(row["is_altloc"]).lower() == "true"
            for key in row:
                if key not in {"key", "pdb_id", "is_altloc"}:
                    row[key] = float(row[key])
    completed = {row["key"] for row in rows}
    pending = [record for record in records if record["key"] not in completed]
    loader = DataLoader(
        DensityPairDataset(pending), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
    )
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            masks = batch["local_mask"].to(device, non_blocking=True).bool()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                predictions = model(inputs)
            for index in range(len(inputs)):
                raw, prediction, target, mask = (
                    inputs[index].float(), predictions[index].float(),
                    targets[index].float(), masks[index],
                )
                raw_local_mse = _masked_mse(raw, target, mask)
                denoised_local_mse = _masked_mse(prediction, target, mask)
                rows.append({
                    "key": batch["key"][index],
                    "pdb_id": batch["pdb_id"][index],
                    "is_altloc": bool(batch["is_altloc"][index]),
                    "raw_global_mse": float((raw - target).square().mean()),
                    "denoised_global_mse": float((prediction - target).square().mean()),
                    "raw_global_pearson": _masked_pearson(raw, target, torch.ones_like(mask)),
                    "denoised_global_pearson": _masked_pearson(
                        prediction, target, torch.ones_like(mask)
                    ),
                    "raw_local_mse": raw_local_mse,
                    "denoised_local_mse": denoised_local_mse,
                    "raw_local_pearson": _masked_pearson(raw, target, mask),
                    "denoised_local_pearson": _masked_pearson(prediction, target, mask),
                    "raw_top10_overlap": _top_fraction_overlap(raw, target, mask),
                    "denoised_top10_overlap": _top_fraction_overlap(
                        prediction, target, mask
                    ),
                    "local_mse_fractional_improvement": (
                        raw_local_mse - denoised_local_mse
                    ) / max(raw_local_mse, 1e-12),
                })
            _atomic_csv(path, rows)
            _atomic_json(output / "stage_manifest.json", {
                "status": "reconstruction_running",
                "completed_reconstruction_sites": len(rows),
                "total_reconstruction_sites": len(records),
            })
            print(json.dumps({
                "stage": "reconstruction", "completed": len(rows),
                "total": len(records),
            }), flush=True)
    return rows


def evaluate_landscapes(
    model: torch.nn.Module,
    records: list[dict],
    selections: list[Path],
    output: Path,
    device: torch.device,
    seed: int,
) -> list[dict]:
    path = output / "landscape_metrics.csv"
    rows: list[dict] = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key in row:
                if key.endswith(("top1", "top3", "only")):
                    row[key] = str(row[key]).lower() == "true"
                elif key not in {"site", "residue_name"}:
                    row[key] = float(row[key])
    completed = {row["site"] for row in rows}
    selected = []
    for selection in selections:
        selected.extend(json.loads(selection.read_text())["sites"])
    if len({site["key"] for site in selected}) != len(selected):
        raise ValueError("selection files contain duplicate sites")
    record_by_key = {record["key"]: record for record in records}
    mask = radial_mask(32, 0.5, 4.0, device=device)
    for site in selected:
        if site["key"] in completed:
            continue
        record = record_by_key[site["key"]]
        structure = gemmi.read_structure(site["pdb_path"])
        # The historical U-Net 2 cache used 16 slots, which is sufficient for
        # ARG/MET/ASP but not a two-conformer TRP. Evaluation is not tied to
        # that serialized cache shape, so use enough slots for every panel site.
        label = _build_site(record, structure, seed, max_atom_slots=24)
        if label is None:
            raise RuntimeError(f"could not build candidate landscape for {site['key']}")
        with np.load(record["pair_path"]) as archive:
            raw = torch.from_numpy(archive["input"].astype(np.float32))[None].to(device)
            target = torch.from_numpy(archive["target"].astype(np.float32))[None].to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            denoised = model(raw)
        candidates = render_candidates(
            torch.from_numpy(label["positions"])[None].to(device),
            torch.from_numpy(label["sigma2"])[None].to(device),
            torch.from_numpy(label["weights"])[None].to(device),
            torch.from_numpy(label["atom_mask"])[None].to(device),
        )
        energies = {
            "raw": candidate_energies(raw.float(), candidates, mask)[0].cpu().numpy(),
            "denoised": candidate_energies(
                denoised.float(), candidates, mask
            )[0].cpu().numpy(),
            "target": candidate_energies(
                target.float(), candidates, mask
            )[0].cpu().numpy(),
        }
        row: dict = {
            "site": site["key"],
            "residue_name": site["residue_name"],
            "raw_vs_target_spearman": _rank_correlation(
                energies["raw"], energies["target"]
            ),
            "denoised_vs_target_spearman": _rank_correlation(
                energies["denoised"], energies["target"]
            ),
            "native_render_mse": float(
                (candidates[0, 0] - target[0, 0]).square().mean().cpu()
            ),
        }
        for source, values in energies.items():
            row.update({f"{source}_{key}": value for key, value in _native_metrics(values).items()})
            for label_name, energy in zip(CANDIDATE_LABELS, values):
                row[f"{source}_energy_{label_name}"] = float(energy)
        rows.append(row)
        _atomic_csv(path, rows)
        _atomic_json(output / "stage_manifest.json", {
            "status": "landscape_running",
            "completed_reconstruction_sites": len(records),
            "completed_landscape_sites": len(rows),
            "total_landscape_sites": len(selected),
        })
        print(json.dumps({
            "stage": "landscape", "site": site["key"],
            "completed": len(rows), "total": len(selected),
        }), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extended frozen-U-Net reconstruction and landscape evaluation"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--landscape-seed", type=int, default=20260720)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    records = read_manifest(manifest_path(args.data_root, "test", "crystal"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_model(args.checkpoint, device)
    _atomic_json(args.output / "run_config.json", {
        **vars(args),
        "device": str(device),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_l2": checkpoint.get("best_loss"),
        "test_sites": len(records),
        "methodology": {
            "reconstruction": "untouched 99-protein test split",
            "local_region": "saved atom-local sidechain mask",
            "peak_overlap": (
                "intersection divided by top-set size for the top 10% "
                "target/predicted local voxels"
            ),
            "landscape": "seven frozen A/B, near-native, and random-rotamer candidates",
            "landscape_panel": "20 prospectively selected untouched test proteins",
            "landscape_radius_angstrom": 4.0,
        },
    })
    reconstruction = evaluate_reconstruction(
        model, records, args.output, device, args.batch_size, args.workers
    )
    reconstruction_summary = _summarize_reconstruction(reconstruction)
    _atomic_json(args.output / "reconstruction_summary.json", reconstruction_summary)
    landscapes = evaluate_landscapes(
        model, records, args.selection, args.output, device, args.landscape_seed
    )
    landscape_summary = _summarize_landscape(landscapes)
    _atomic_json(args.output / "landscape_summary.json", landscape_summary)
    final = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_validation_l2": checkpoint.get("best_loss"),
        "reconstruction": reconstruction_summary,
        "landscape": landscape_summary,
    }
    _atomic_json(args.output / "extended_evaluation_summary.json", final)
    _atomic_json(args.output / "stage_manifest.json", {"status": "complete"})
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
