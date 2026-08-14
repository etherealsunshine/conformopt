#!/usr/bin/env python3
"""Fixed-geometry screen for deposited A/B backbone-state support.

This deliberately performs no coordinate optimisation.  It tests whether the
corrected A'' density objective supports one or two deposited conformers before
any recovery benchmark is interpreted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from pathlib import Path

# The pod keeps qFit/CCTBX and CUDA Torch in separate environments.  Load the
# qFit NumPy ABI first, then CUDA Torch, before exposing qFit's extensions.
QFIT_SITE = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/site-packages"
QFIT_DYNLIB = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/lib-dynload"
WORKSPACE = "/home/dev/workspace"
QFIT_SRC = f"{WORKSPACE}/external/qfit-3.0/src"
if os.path.isdir(QFIT_SITE):
    sys.path.insert(0, QFIT_SITE)
    import numpy as np
    sys.path.remove(QFIT_SITE)
    import torch  # noqa: F401  # initialise the CUDA runtime before qFit imports.
    sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]
else:
    import numpy as np

from qfit.xtal.scaler import MapScaler

from occupancy_selection import solve_affine_qp
from run_d1_8d_sequential_poc import SequentialBackbonePOC
from run_d6_tier2_realmap import make_map


SEED = 20260805
FOLDS = 5
TEST_FRACTION = 0.20
EPSILON = 1e-8

DEFAULT_SITES = (
    {"pdb_id": "7T7A", "chain": "A", "resnum": 396},
    {"pdb_id": "5OHJ", "chain": "A", "resnum": 540},
    {"pdb_id": "7UTC", "chain": "A", "resnum": 52},
    {"pdb_id": "6P2N", "chain": "A", "resnum": 161},
)
PANEL_MANIFEST = Path("/home/dev/qfit_unet_data/qfit_audit/d6_tier1_native_bic_v1/panel_manifest.json")


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def site_key(site: dict[str, object]) -> str:
    return f"{site['pdb_id']}_{site['chain']}_{site.get('resname', '')}{site['resnum']}"


def bic(rss: float, n_voxels: int, parameters: int) -> float:
    return float(n_voxels * math.log(max(rss / n_voxels, 1e-300))
                 + parameters * math.log(n_voxels))


def full_structure_scaler_metrics(runner: SequentialBackbonePOC) -> dict[str, float]:
    """Reproduce MapScaler's full-structure support regression without mutation."""
    mtz = Path(f"/home/dev/qfit_unet_data/cache/{runner.split}/mtz/{runner.pdb_id}.mtz")
    xmap, resolution, _, _ = make_map(mtz)
    scaler = MapScaler(xmap)
    transformer = scaler._get_model_transformer(  # pylint: disable=protected-access
        runner.full_structure, transformer="cctbx"
    )
    transformer.mask(0.5 + resolution / 3.0)
    mask = scaler._model_map.array > 0  # pylint: disable=protected-access
    transformer.reset(full=True)
    transformer.density()
    observed = xmap.array[mask].astype(float, copy=False)
    calculated = scaler._model_map.array[mask].astype(float, copy=False)  # pylint: disable=protected-access
    observed_centered = observed - observed.mean()
    calculated_centered = calculated - calculated.mean()
    scale = float(np.dot(calculated_centered, observed_centered)
                  / np.dot(observed_centered, observed_centered))
    offset = float(calculated.mean() - scale * observed.mean())
    return {
        "map_scaler_full_correlation": float(np.corrcoef(observed, calculated)[0, 1]),
        "map_scaler_full_scale": scale,
        "map_scaler_full_offset": offset,
        "map_scaler_full_support_voxels": int(mask.sum()),
    }


def fixed_geometry_fits(runner: SequentialBackbonePOC) -> dict[str, dict[str, object]]:
    window_a = runner.initial_window
    window_b = runner.window_for_deposited_b()
    target = runner.target
    return {
        "A_alone": runner.profile_affine_b_offset(target, [window_a], [0]),
        "B_alone": runner.profile_affine_b_offset(target, [window_b], [1]),
        "AB_fixed": runner.profile_affine_b_offset(
            target, [window_a, window_b], [0, 1], fixed_weights=runner.deposited_occupancies
        ),
        "AB_free": runner.profile_affine_b_offset(target, [window_a, window_b], [0, 1]),
    }


def blocked_splits(runner: SequentialBackbonePOC) -> list[tuple[np.ndarray, np.ndarray]]:
    n_voxels = len(runner.target)
    n_test = round(TEST_FRACTION * n_voxels)
    coordinates = np.argwhere(runner.mask) * np.asarray(runner.qfit.xmap.voxelspacing, float)
    rng = np.random.default_rng(SEED)
    # Preserve the historical stream that generated the prior five slabs.
    for _ in range(FOLDS):
        rng.choice(n_voxels, size=n_test, replace=False)
    all_indices = np.arange(n_voxels)
    answer = []
    for _ in range(FOLDS):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        test = np.sort(np.argsort(coordinates @ direction)[:n_test])
        answer.append((np.setdiff1d(all_indices, test, assume_unique=True), test))
    return answer


def profile_on_training_voxels(runner: SequentialBackbonePOC, windows: list[np.ndarray],
                               slots: list[int], train: np.ndarray) -> dict[str, object]:
    """The runner's B-offset profile restricted to one blocked-CV training set."""
    target = runner.target
    lower = -min(float(runner.slot_b_factors(slot).min()) for slot in slots) + 1e-6
    cache: dict[float, dict[str, object]] = {}

    def evaluate(offset: float) -> dict[str, object]:
        key = round(float(offset), 8)
        if key not in cache:
            models = np.vstack([
                runner.model_density(window, slot=slot, b_offset=key)
                for window, slot in zip(windows, slots)
            ])
            weights, intercept, rss = solve_affine_qp(target[train], models[:, train])
            cache[key] = {
                "b_offset": key, "models": models, "weights": weights,
                "intercept": float(intercept), "rss": float(rss),
            }
        return cache[key]

    probes = {0.0, lower}
    positive = 5.0
    while positive <= 1024.0:
        probes.add(positive)
        positive *= 2.0
    negative = -5.0
    while negative > lower:
        probes.add(negative)
        negative *= 2.0
    grid = sorted(probes)
    fits = [evaluate(offset) for offset in grid]
    best = int(np.argmin([float(fit["rss"]) for fit in fits]))
    if best in (0, len(grid) - 1):
        raise RuntimeError("blocked-CV B-offset profile was not bracketed")
    left, right = grid[best - 1], grid[best + 1]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    first = right - golden * (right - left)
    second = left + golden * (right - left)
    first_fit, second_fit = evaluate(first), evaluate(second)
    for _ in range(18):
        if float(first_fit["rss"]) <= float(second_fit["rss"]):
            right, second, second_fit = second, first, first_fit
            first = right - golden * (right - left)
            first_fit = evaluate(first)
        else:
            left, first, first_fit = first, second, second_fit
            second = left + golden * (right - left)
            second_fit = evaluate(second)
    return evaluate((left + right) / 2.0)


def heldout_comparison(runner: SequentialBackbonePOC, one_label: str) -> list[dict[str, object]]:
    window_a = runner.initial_window
    window_b = runner.window_for_deposited_b()
    one_windows, one_slots = ([window_a], [0]) if one_label == "A_alone" else ([window_b], [1])
    rows = []
    for fold, (train, test) in enumerate(blocked_splits(runner)):
        one = profile_on_training_voxels(runner, one_windows, one_slots, train)
        two = profile_on_training_voxels(runner, [window_a, window_b], [0, 1], train)
        one_residual = runner.target[test] - np.asarray(one["weights"]) @ np.asarray(one["models"])[:, test] - float(one["intercept"])
        two_residual = runner.target[test] - np.asarray(two["weights"]) @ np.asarray(two["models"])[:, test] - float(two["intercept"])
        one_rss = float(np.square(one_residual).sum())
        two_rss = float(np.square(two_residual).sum())
        rows.append({
            "fold": fold, "heldout_voxels": int(len(test)),
            "one_dB_A2": float(one["b_offset"]), "one_weights": np.asarray(one["weights"]).tolist(),
            "one_intercept": float(one["intercept"]), "one_heldout_rss": one_rss,
            "two_dB_A2": float(two["b_offset"]), "two_weights": np.asarray(two["weights"]).tolist(),
            "two_intercept": float(two["intercept"]), "two_heldout_rss": two_rss,
            "two_minus_one_heldout_rss": two_rss - one_rss,
        })
    return rows


def flatten_fit(prefix: str, fit: dict[str, object], n_voxels: int, parameters: int) -> dict[str, object]:
    weights = np.asarray(fit["weights"], dtype=float)
    return {
        f"{prefix}_weights": json.dumps(weights.tolist()), f"{prefix}_total": float(weights.sum()),
        f"{prefix}_dB_A2": float(fit["b_offset"]), f"{prefix}_intercept": float(fit["intercept"]),
        f"{prefix}_rss": float(fit["rss"]), f"{prefix}_bic": bic(float(fit["rss"]), n_voxels, parameters),
    }


def classify(best_one: dict[str, object], two: dict[str, object], cv_rows: list[dict[str, object]], n_voxels: int) -> str:
    one_bic = bic(float(best_one["rss"]), n_voxels, 3)
    two_bic = bic(float(two["rss"]), n_voxels, 4)
    cv_difference = float(np.mean([row["two_minus_one_heldout_rss"] for row in cv_rows]))
    two_wins_bic = two_bic < one_bic - EPSILON
    two_wins_cv = cv_difference < -EPSILON
    one_wins_bic = one_bic <= two_bic + EPSILON
    # A nested two-column model that independently zeroes its extra column has
    # an exact CV tie, which is no evidence for two states and is NOT SUPPORTED.
    one_wins_or_ties_cv = cv_difference >= -EPSILON
    if two_wins_bic and two_wins_cv:
        return "TWO-STATE SUPPORTED"
    if one_wins_bic and one_wins_or_ties_cv:
        return "NOT SUPPORTED"
    return "MARGINAL"


def screen_site(site: dict[str, object], device: str) -> dict[str, object]:
    runner = SequentialBackbonePOC(
        str(site["pdb_id"]), str(site["chain"]), int(site["resnum"]), Path("/private/tmp/deposited_screen"),
        0.25, 2.0, 0.05, residual_scale_mode="none", renderer_backend="torch",
        map_scaler_structure="a_only", mask_scope="window", device=device,
    )
    fits = fixed_geometry_fits(runner)
    n_voxels = len(runner.target)
    one_label = min(("A_alone", "B_alone"), key=lambda label: float(fits[label]["rss"]))
    cv_rows = heldout_comparison(runner, one_label)
    density_a, density_b = np.asarray(fits["AB_free"]["models"], dtype=float)
    mean_density = float(np.mean((density_a + density_b) / 2.0))
    discriminating = np.abs(density_a - density_b) > 0.05 * mean_density
    scaler = full_structure_scaler_metrics(runner)
    row: dict[str, object] = {
        "status": "complete", "site": site_key(site), "pdb_id": runner.pdb_id, "chain": runner.chain,
        "resnum": runner.resnum, "resname": str(runner.central.resn[0]), "resolution_A": runner.resolution,
        "deposited_occupancies": json.dumps(runner.deposited_occupancies.tolist()),
        "mask_voxels": n_voxels, "rho_A_B_correlation": float(np.corrcoef(density_a, density_b)[0, 1]),
        "discriminating_voxel_fraction_5pct": float(discriminating.mean()),
        "best_one_model": one_label,
        **flatten_fit("A_alone", fits["A_alone"], n_voxels, 3),
        **flatten_fit("B_alone", fits["B_alone"], n_voxels, 3),
        **flatten_fit("AB_fixed", fits["AB_fixed"], n_voxels, 4),
        **flatten_fit("AB_free", fits["AB_free"], n_voxels, 4),
        **scaler,
    }
    row["fitted_dB_A2"] = float(fits["AB_free"]["b_offset"])
    row["max_abs_dB_A2_across_four_fits"] = float(max(
        abs(float(fit["b_offset"])) for fit in fits.values()
    ))
    row["dB_over_15_A2"] = bool(row["max_abs_dB_A2_across_four_fits"] > 15.0)
    row["cv_folds"] = cv_rows
    row["cv_one_mean_rss"] = float(np.mean([fold["one_heldout_rss"] for fold in cv_rows]))
    row["cv_two_mean_rss"] = float(np.mean([fold["two_heldout_rss"] for fold in cv_rows]))
    row["cv_two_minus_one"] = [float(fold["two_minus_one_heldout_rss"]) for fold in cv_rows]
    row["cv_two_minus_one_mean"] = float(np.mean(row["cv_two_minus_one"]))
    row["classification"] = classify(fits[one_label], fits["AB_free"], cv_rows, n_voxels)
    return row


def load_sites(panel: str) -> list[dict[str, object]]:
    if panel == "four":
        return [dict(site) for site in DEFAULT_SITES]
    manifest = json.loads(PANEL_MANIFEST.read_text())
    key = "flip_filter" if panel == "flips" else "nonflip_control"
    return [site for site in manifest if site["panel"] == key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("four", "flips", "controls"), default="four")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    checkpoints = args.output / "site_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    sites = load_sites(args.panel)
    atomic_json(args.output / "run_config.json", {
        "operation": "fixed deposited-geometry support screen; no coordinate optimisation",
        "panel": args.panel, "sites_requested": len(sites), "mask": "corrected full-window backbone union",
        "slots": "A uses deposited-A B factors; B uses deposited-B B factors",
        "nuisance_parameters": "profiled global delta_B, affine occupancies, free intercept",
        "BIC": "k=3 one conformer; k=4 two conformers", "cv": "five historical blocked spatial slabs",
    })
    rows = []
    for site in sites:
        checkpoint = checkpoints / f"{site_key(site)}.json"
        if args.resume and checkpoint.exists():
            row = json.loads(checkpoint.read_text())
        else:
            try:
                row = screen_site(site, args.device)
            except Exception as error:  # checkpoint failures rather than hiding panel eligibility.
                row = {"status": "error", "site": site_key(site), "error": repr(error),
                       "traceback": traceback.format_exc()}
            atomic_json(checkpoint, row)
        rows.append(row)
        flat_rows = [{key: value for key, value in item.items() if key not in {"cv_folds", "traceback"}}
                     for item in rows]
        atomic_csv(args.output / "per_site.csv", flat_rows)
        counts = {label: sum(item.get("classification") == label for item in rows)
                  for label in ("TWO-STATE SUPPORTED", "MARGINAL", "NOT SUPPORTED")}
        atomic_json(args.output / "progress.json", {
            "sites_recorded": len(rows), "sites_total": len(sites),
            "complete": sum(item.get("status") == "complete" for item in rows),
            "errors": sum(item.get("status") == "error" for item in rows), "classification_counts": counts,
        })
    print(json.dumps(json.loads((args.output / "progress.json").read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
