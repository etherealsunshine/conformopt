#!/usr/bin/env python3
"""Fixed-geometry, no-optimisation diagnostics for the A′ 7UTC PoC."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_d1_8d_sequential_poc import (
    EPSILON,
    SequentialBackbonePOC,
    atomic_csv,
    atomic_json,
    rmsd,
)


def pair_qp(base: SequentialBackbonePOC, first: np.ndarray, second: np.ndarray):
    weights, rss = base.joint_qp(first, second)
    return {"occupancies": weights.tolist(), "total_occupancy": float(weights.sum()), "rss": float(rss)}


def single_qp(base: SequentialBackbonePOC, coordinates: np.ndarray):
    occupancy, rss = base.bounded_nnls(base.target, base.model_density(coordinates), 1.0)
    return {"occupancy": occupancy, "rss": rss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sequential-output", type=Path,
        default=Path("/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_sequential_v3"),
    )
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    base = SequentialBackbonePOC("7UTC", "A", 52, args.output, 0.25, 2.0, 0.0, "deposited_ab")
    final = np.load(args.sequential_output / "final_slots.npz")
    slot1, slot2 = final["slot1_window"], final["slot2_window"]
    deposited_a, deposited_b = base.initial_window.copy(), base.window_for_deposited_b()

    converged_pair = pair_qp(base, slot1, slot2)
    deposited_pair = pair_qp(base, deposited_a, deposited_b)
    one_a, one_b = single_qp(base, deposited_a), single_qp(base, deposited_b)

    central_b = base.central_backbone(deposited_b)
    scan = []
    for step in range(args.steps + 1):
        alpha = step / args.steps
        moving_slot2 = (1.0 - alpha) * slot2 + alpha * deposited_b
        result = pair_qp(base, slot1, moving_slot2)
        scan.append({
            "alpha_to_deposited_B": alpha,
            "slot2_central_rmsd_to_B_A": rmsd(base.central_backbone(moving_slot2), central_b),
            **result,
        })

    result = {
        "status": "complete",
        "site": "7UTC_A_ARG52",
        "operation": "fixed-geometry QP objective evaluation only; no coordinate optimisation",
        "map": {
            "residual_scale": base.residual_scale_diagnostic,
            "mask_voxels": int(base.mask.sum()),
            "resolution_A": base.resolution,
            "neighbour_subtraction": True,
        },
        "converged_pair": converged_pair,
        "deposited_A_B_pair": deposited_pair,
        "deposited_pair_minus_converged_pair_rss": deposited_pair["rss"] - converged_pair["rss"],
        "single_deposited_A": one_a,
        "single_deposited_B": one_b,
        "slot2_to_B_scan_with_slot1_frozen": scan,
    }
    atomic_json(args.output / "result.json", result)
    atomic_csv(args.output / "slot2_to_B_scan.csv", scan)
    atomic_json(args.output / "progress.json", {"status": "complete", "scan_points": len(scan)})


if __name__ == "__main__":
    main()
