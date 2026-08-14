#!/usr/bin/env python3
"""Guarded five-fold CV for the cap-40, lambda-stopped clean-D1 fits."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import qfit  # noqa: F401  # keep qFit/CCTBX imports before CUDA Torch
import numpy as np
import torch

from clean_d1_benchmark import site_key
from fit_provenance import assert_heldout_geometry_provenance
from occupancy_selection import solve_affine_qp
from run_d1_8d_sequential_poc import SequentialBackbonePOC
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential
from run_d1_slot_coordination import build_specs, worker


SITES = ("6ZWK_B_PHE47", "8R7O_C_THR1681")


def make_fold_task(template: dict[str, object], output: Path, fold: int,
                   treatment: str, train: np.ndarray, test: np.ndarray) -> dict[str, object]:
    spec = copy.deepcopy(template)
    spec["output"] = str(output)
    spec["training_indices"] = train.tolist()
    spec["fold"] = fold
    spec["treatment"] = treatment
    spec["test_indices"] = test.tolist()
    if treatment == "dB_zero":
        spec["fixed_b_offset"] = 0.0
    return spec


def fit_and_score(task: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    result = worker(task)
    from run_d1_aprime_sequential import APrimeSequential

    site = tuple(task["site"])
    train = np.asarray(task["training_indices"], dtype=int)
    test = np.asarray(task["test_indices"], dtype=int)
    base = APrimeSequential(
        Path(task["output"]) / "score_base", 80, 6, *site,
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device=task["device"],
        start_pdb=task["start_pdb"], b_factor_mode="single_conformer",
    )
    saved = np.load(Path(task["output"]) / "final_slots.npz")
    assert_heldout_geometry_provenance(result, saved, train, len(base.base.target))
    coordinates = np.stack((saved["slot1_window"], saved["slot2_window"]))
    delta_b = 0.0 if task["treatment"] == "dB_zero" else float(result["final_b_offset_A2"])
    models = base.base.model_density_batch(coordinates, slots=np.array((0, 1)), b_offset=delta_b)
    weights, intercept, train_rss = solve_affine_qp(base.base.target[train], models[:, train])
    heldout = base.base.target[test] - weights @ models[:, test] - intercept
    slot_rmsds = []
    for coordinate in coordinates:
        central = base.base.central_backbone(coordinate)
        slot_rmsds.append({
            "to_A": float(np.sqrt(np.mean(np.sum((central - base.a_backbone) ** 2, axis=1)))),
            "to_B": float(np.sqrt(np.mean(np.sum((central - base.b_backbone) ** 2, axis=1)))),
        })
    assignments = (
        (slot_rmsds[0]["to_A"], slot_rmsds[1]["to_B"]),
        (slot_rmsds[0]["to_B"], slot_rmsds[1]["to_A"]),
    )
    return {
        "site": task["site_key"], "treatment": task["treatment"],
        "fold": int(task["fold"]), "elapsed_s": time.perf_counter() - started,
        "train_voxels": int(len(train)), "heldout_voxels": int(len(test)),
        "delta_B_A2": delta_b, "occupancies_refit_on_training": weights.tolist(),
        "intercept_refit_on_training": float(intercept),
        "training_rss": float(train_rss),
        "heldout_rss": float(np.square(heldout).sum()),
        "slot_rmsds": slot_rmsds,
        "assigned_distances": list(map(float, min(assignments, key=sum))),
        "fit_provenance": result["fit_provenance"],
        "inner_solve_diagnostics": result["inner_solve_diagnostics"],
    }


def oracle_cv(base: APrimeSequential, folds) -> dict[str, object]:
    oracle = SequentialBackbonePOC(
        base.base.pdb_id, base.base.chain, base.base.resnum, base.output / "oracle",
        0.25, 2.0, 0.0, renderer_backend="torch", map_scaler_structure="full",
        mask_scope="window", device=str(base.base.torch_device),
        density_atom_scope="backbone", b_factor_mode="oracle_deposited",
    )
    rows = []
    for fold, (train, test, direction) in enumerate(folds):
        windows = [oracle.initial_window, oracle.window_for_deposited_b()]
        profiled = oracle.profile_affine_b_offset(
            oracle.target[train], windows, [0, 1], voxel_indices=train
        )
        delta_b = float(profiled["b_offset"])
        models = oracle.model_density_batch(windows, slots=np.array((0, 1)), b_offset=delta_b)
        weights, intercept, train_rss = solve_affine_qp(oracle.target[train], models[:, train])
        heldout = oracle.target[test] - weights @ models[:, test] - intercept
        rows.append({
            "fold": fold, "split_direction": direction.tolist(),
            "delta_B_A2": delta_b, "training_rss": float(train_rss),
            "heldout_rss": float(np.square(heldout).sum()),
            "occupancies_refit_on_training": weights.tolist(),
            "intercept_refit_on_training": float(intercept),
        })
    values = np.asarray([row["heldout_rss"] for row in rows])
    return {
        "bound": "deposited geometry and deposited A/B B arrays; unavailable prospectively",
        "folds": rows, "mean_all_five_heldout_rss": float(values.mean()),
        "mean_excluding_fold_0_heldout_rss": float(values[1:].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--flip-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--inner-nfev", type=int, default=40)
    ap.add_argument("--outer-max", type=int, default=50)
    ap.add_argument("--lambda-relative-tolerance", type=float, default=0.01)
    ap.add_argument("--lambda-damping-alpha", type=float, default=1.0)
    ap.add_argument("--lambda-norm-cap", type=float, default=None)
    ap.add_argument("--mirror-eta", type=float, default=0.001)
    ap.add_argument("--amplitude-prior-lambda", type=float, default=0.0)
    ap.add_argument("--dB-zero-only", action="store_true",
                    help="run only the prospective fixed-dB=0 treatment")
    ap.add_argument("--per-slot-trust-radii", action="store_true",
                    help="use independent, ratio-adapted TRF solves for the two slot blocks")
    ap.add_argument("--site", choices=SITES, default=None,
                    help="run only one site")
    ap.add_argument("--fold", type=int, choices=range(5), default=None,
                    help="run only one blocked-CV fold")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {site_key(row): row for row in json.loads(args.manifest.read_text())}
    final = {}
    sites = (args.site,) if args.site is not None else SITES
    for key in sites:
        row = manifest[key]
        site = (str(row["pdb_id"]), str(row["chain"]), int(row["resnum"]))
        start = args.starts / "sites" / key / "neutral_start_aprime_single_slot.pdb"
        site_root = args.output / key
        base = APrimeSequential(
            site_root / "base", 80, 6, *site, renderer_backend="torch",
            residual_scale_mode="none", map_scaler_structure="full",
            mask_scope="window", device=args.device, start_pdb=start,
            b_factor_mode="single_conformer",
        )
        folds = blocked_splits(base.base)
        oracle = oracle_cv(base, folds)
        templates = []
        fold_items = list(enumerate(folds))
        if args.fold is not None:
            fold_items = [(args.fold, folds[args.fold])]
        for fold, (_train, _test, _direction) in fold_items:
            built = build_specs(
                site_root / "templates" / f"fold_{fold}", args.flip_root,
                site=site, mask_scope="window", rama_floor=0.02,
                start_pdb=start, b_factor_mode="single_conformer", device=args.device,
                occupancy_scheme="mirror", mirror_eta=args.mirror_eta,
                inner_nfev=args.inner_nfev, outer_updates=args.outer_max,
                lambda_relative_tolerance=args.lambda_relative_tolerance,
                lambda_damping_alpha=args.lambda_damping_alpha,
                lambda_norm_cap=args.lambda_norm_cap,
                amplitude_prior_lambda=args.amplitude_prior_lambda,
                per_slot_trust_radii=args.per_slot_trust_radii,
            )
            templates.append(next(item for item in built if item["label"] == "D_null_axis2_30deg"))
        treatment_summaries = {}
        treatments = ("dB_zero",) if args.dB_zero_only else ("dB_fitted", "dB_zero")
        for treatment in treatments:
            tasks = []
            for fold, (train, test, _direction) in fold_items:
                tasks.append(fit_and_score)  # retain callable provenance in the manifest
            rows = []
            with ProcessPoolExecutor(max_workers=5, mp_context=mp.get_context("spawn")) as executor:
                futures = []
                for (fold, (train, test, _direction)), template in zip(fold_items, templates):
                    task = make_fold_task(
                        template, site_root / treatment / f"fold_{fold}" / "D_null_axis2_30deg",
                        fold, treatment, train, test,
                    )
                    task["site"] = list(site)
                    task["site_key"] = key
                    task["device"] = args.device
                    task["start_pdb"] = str(start)
                    futures.append(executor.submit(fit_and_score, task))
                for future in as_completed(futures):
                    rows.append(future.result())
            rows.sort(key=lambda row: row["fold"])
            assigned = np.asarray([row["assigned_distances"] for row in rows], dtype=float)
            heldout = np.asarray([row["heldout_rss"] for row in rows], dtype=float)
            threshold = 0.30 * base.ab_distance
            treatment_summaries[treatment] = {
                "threshold_A": float(threshold), "folds": rows,
                "mean_all_five_heldout_rss": float(heldout.mean()),
                "mean_excluding_fold_0_heldout_rss": (
                    float(heldout[1:].mean()) if args.fold is None else None
                ),
                "mean_all_five_assigned_distances_A": assigned.mean(axis=0).tolist(),
                "mean_excluding_fold_0_assigned_distances_A": assigned[1:].mean(axis=0).tolist(),
                "fold_sd_assigned_distances_A": assigned.std(axis=0, ddof=1).tolist(),
                "max_fold_sd_A": float(assigned.std(axis=0, ddof=1).max()),
                "fold_0_known_unknown": True,
            }
        final[key] = {"oracle": oracle, "treatments": treatment_summaries}
    report = {
        "status": "complete", "device": args.device,
        "inner_nfev": args.inner_nfev, "outer_max": args.outer_max,
        "lambda_relative_tolerance": args.lambda_relative_tolerance,
        "lambda_damping_alpha": args.lambda_damping_alpha,
        "lambda_norm_cap": args.lambda_norm_cap,
        "occupancy_scheme": "mirror", "mirror_eta": args.mirror_eta,
        "amplitude_prior_lambda": args.amplitude_prior_lambda,
        "dB_zero_only": args.dB_zero_only,
        "per_slot_trust_radii": args.per_slot_trust_radii,
        "sites": list(sites),
        "rama_floor": 0.02, "results": final,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
