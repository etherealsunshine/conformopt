#!/usr/bin/env python3
"""Five-fold blocked, leakage-corrected A' optimization at 7UTC A:ARG52.

Each child invocation fits both slot geometries using only one fold's training
voxels.  It then renders the pair on the full mask and evaluates the untouched
blocked holdout.  A separate aggregate invocation only reads completed child
artifacts, so the five fits can be sharded safely.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_d1_8d_sequential_poc import atomic_csv, atomic_json, rmsd
from run_d1_aprime_sequential import APrimeSequential
from occupancy_selection import solve_affine_qp


SEED = 20260805
SPLITS = 5
TEST_FRACTION = 0.20


def blocked_splits(base):
    """Reproduce the five spatial-slab splits used by the basin analyses."""
    n_voxels = int(base.mask.sum())
    n_test = round(TEST_FRACTION * n_voxels)
    coordinates = np.argwhere(base.mask) * np.asarray(base.qfit.xmap.voxelspacing, float)
    rng = np.random.default_rng(SEED)
    # Preserve the historical RNG stream: the prior random-voxel experiment
    # consumed these draws before generating the five blocked directions.
    for _ in range(SPLITS):
        rng.choice(n_voxels, size=n_test, replace=False)
    all_indices = np.arange(n_voxels)
    answer = []
    for fold in range(SPLITS):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        test = np.sort(np.argsort(coordinates @ direction)[:n_test])
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        answer.append((train, test, direction))
    return answer


def fair_density(base, coordinates, b_factors):
    """Render one alternate with its deposited atom-wise B factors."""
    density = base.model_density_with_b(coordinates, b_factors)
    return np.maximum(density, base.qfit.options.bulk_solvent_level)


def fit_affine(target, models):
    """Fit the shared affine occupancy objective, including its intercept."""
    weights, intercept, rss = solve_affine_qp(target, models)
    return np.asarray(weights, dtype=float), float(intercept), float(rss)


def rmsds(base, coordinates):
    backbone = base.central_backbone(coordinates)
    return {
        "rmsd_to_deposited_A_A": float(rmsd(backbone, base.a_backbone)),
        "rmsd_to_deposited_B_A": float(rmsd(backbone, base.b_backbone)),
    }


def run_fold(args):
    args.output.mkdir(parents=True, exist_ok=False)
    probe = APrimeSequential(args.output, args.inner_nfev, args.outer_updates,
                             args.pdb_id, args.chain, args.resnum,
                             mask_scope=args.mask_scope, device=args.device)
    probe.rama_floor = args.rama_floor
    train, test, direction = blocked_splits(probe.base)[args.fold]
    # Recreate the runner with the exact same site but a train-only density
    # vector.  No coordinate objective or finite-difference Jacobian can read
    # the test indices through this wrapper.
    runner = APrimeSequential(args.output, args.inner_nfev, args.outer_updates,
                              args.pdb_id, args.chain, args.resnum,
                              training_indices=train, mask_scope=args.mask_scope,
                              device=args.device)
    runner.rama_floor = args.rama_floor
    base = runner.base
    config = {
        "status": "running",
        "operation": "leakage-corrected blocked-CV A' sequential optimization",
        "fold": args.fold,
        "seed": SEED,
        "blocked_split_definition": "lowest 20 percent of voxel positions projected onto a seeded random spatial direction",
        "split_direction": direction.tolist(),
        "train_voxels": int(len(train)),
        "heldout_voxels": int(len(test)),
        "full_mask_voxels": int(base.mask.sum()),
        "mask_scope": args.mask_scope,
        "torch_device": str(base.torch_device) if base.torch_device is not None else None,
        "rama_floor": args.rama_floor,
        "joint_slot2_qp": True,
        "slot2_temporary_occupancy_floor": args.slot2_occupancy_floor,
        "fair_bfactor_rendering": "slot 1 and deposited A use A B factors; slot 2 and deposited B use B B factors",
    }
    atomic_json(args.output / "run_config.json", config)
    result = runner.run(joint_slot2_qp=True,
                        slot2_occupancy_floor=args.slot2_occupancy_floor)

    a_b = np.asarray(base.b_factors_a, dtype=float)
    b_b = np.asarray(base.b_factors_b, dtype=float)
    final = np.load(args.output / "final_slots.npz")
    slot1, slot2 = final["slot1_window"], final["slot2_window"]
    recovered_models = np.vstack((fair_density(base, slot1, a_b),
                                  fair_density(base, slot2, b_b)))
    deposited_models = np.vstack((fair_density(base, base.initial_window, a_b),
                                  fair_density(base, base.window_for_deposited_b(), b_b)))
    y = base.target
    recovered_weights, recovered_intercept, recovered_train_rss = fit_affine(
        y[train], recovered_models[:, train]
    )
    deposited_weights, deposited_intercept, deposited_train_rss = fit_affine(
        y[train], deposited_models[:, train]
    )
    recovered_heldout_rss = float(np.square(
        y[test] - (recovered_weights @ recovered_models[:, test] + recovered_intercept)
    ).sum())
    deposited_heldout_rss = float(np.square(
        y[test] - (deposited_weights @ deposited_models[:, test] + deposited_intercept)
    ).sum())
    heldout = {
        "status": "complete",
            "site": f"{args.pdb_id}_{args.chain}_{base.a_residue.resn[0]}{args.resnum}",
        "fold": args.fold,
        "split_direction": direction.tolist(),
        "train_voxels": int(len(train)),
        "heldout_voxels": int(len(test)),
        "split_trained_pair": {
            "heldout_rss": recovered_heldout_rss,
            "training_rss": recovered_train_rss,
            "occupancies_refit_on_training": recovered_weights.tolist(),
            "intercept_refit_on_training": recovered_intercept,
            "slot1": rmsds(base, slot1),
            "slot2": rmsds(base, slot2),
        },
        "deposited_A_B_fair_B_factors": {
            "heldout_rss": deposited_heldout_rss,
            "training_rss": deposited_train_rss,
            "occupancies_refit_on_training": deposited_weights.tolist(),
            "intercept_refit_on_training": deposited_intercept,
        },
        "deposited_minus_split_trained_heldout_rss": deposited_heldout_rss - recovered_heldout_rss,
        "optimization": {
            "train_only": True,
            "result_verdict": result["verdict"],
            "final_train_objective_qp_rss": result["final_joint_qp_rss"],
        },
    }
    atomic_json(args.output / "heldout_result.json", heldout)
    atomic_json(args.output / "progress.json", {"status": "complete", "fold": args.fold})
    print(json.dumps(heldout, indent=2, sort_keys=True))


def aggregate(output, pdb_id="7UTC", chain="A", resnum=52):
    rows = []
    for fold in range(SPLITS):
        path = output / f"split_{fold}" / "heldout_result.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed fold: {path}")
        record = json.loads(path.read_text())
        trained = record["split_trained_pair"]
        deposited = record["deposited_A_B_fair_B_factors"]
        rows.append({
            "fold": fold,
            "split_trained_heldout_rss": trained["heldout_rss"],
            "deposited_A_B_heldout_rss": deposited["heldout_rss"],
            "deposited_minus_split_trained_heldout_rss": record["deposited_minus_split_trained_heldout_rss"],
            "split_trained_weights": trained["occupancies_refit_on_training"],
            "deposited_A_B_weights": deposited["occupancies_refit_on_training"],
            "slot1_rmsd_to_A_A": trained["slot1"]["rmsd_to_deposited_A_A"],
            "slot1_rmsd_to_B_A": trained["slot1"]["rmsd_to_deposited_B_A"],
            "slot2_rmsd_to_A_A": trained["slot2"]["rmsd_to_deposited_A_A"],
            "slot2_rmsd_to_B_A": trained["slot2"]["rmsd_to_deposited_B_A"],
        })
    deltas = np.asarray([row["deposited_minus_split_trained_heldout_rss"] for row in rows], float)
    summary = {
        "status": "complete",
        "site": f"{pdb_id}_{chain}_{resnum}",
        "comparison": "deposited A+B heldout RSS minus independently split-trained A' pair heldout RSS",
        "fair_bfactor_rendering": "A slot uses A B factors and B slot uses B B factors; all weights refit on each training split",
        "folds": SPLITS,
        "paired_differences_per_fold": deltas.tolist(),
        "paired_difference_mean": float(deltas.mean()),
        "paired_difference_sd": float(deltas.std(ddof=1)),
        "all_five_same_sign": bool(np.all(deltas > 0) or np.all(deltas < 0)),
        "sign": "positive means split-trained A' has lower heldout RSS",
        "per_split": rows,
    }
    atomic_csv(output / "per_split.csv", rows)
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(SPLITS))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--inner-nfev", type=int, default=80)
    parser.add_argument("--outer-updates", type=int, default=6)
    parser.add_argument("--slot2-occupancy-floor", type=float, default=0.02)
    parser.add_argument("--pdb-id", default="7UTC")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--resnum", type=int, default=52)
    parser.add_argument("--mask-scope", choices=("central", "window"), default="central")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rama-floor", type=float, default=0.05)
    args = parser.parse_args()
    if args.aggregate:
        if args.fold is not None:
            parser.error("--aggregate cannot be combined with --fold")
        aggregate(args.output, args.pdb_id, args.chain, args.resnum)
    elif args.fold is None:
        parser.error("provide exactly one --fold, or use --aggregate")
    else:
        run_fold(args)


if __name__ == "__main__":
    main()
