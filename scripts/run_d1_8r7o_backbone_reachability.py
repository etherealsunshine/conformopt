#!/usr/bin/env python3
"""Targeted deposited-B-initialization reachability test at 8R7O C:1681."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

QFIT_SITE = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/site-packages"
QFIT_DYNLIB = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/lib-dynload"
WORKSPACE = "/home/dev/workspace"
QFIT_SRC = f"{WORKSPACE}/external/qfit-3.0/src"
if os.path.isdir(QFIT_SITE):
    sys.path.insert(0, QFIT_SITE)
    import numpy as np
    sys.path.remove(QFIT_SITE)
    import torch
    sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]
else:
    import numpy as np
    import torch

from run_d1_8d_sequential_poc import atomic_json
from run_d1_aprime_sequential import APrimeSequential, rmsd
from run_d1_slot_coordination import FullJointParameterization, inverse_seed, joint_run


SITE = ("8R7O", "C", 1681)
DEPOSITED_GATE_DB_A2 = -2.84378158


def compact(result: dict[str, object]) -> dict[str, object]:
    return {
        "final_occupancies": result["final_occupancies"],
        "final_selection": result["final_selection"],
        "intercept": result["final_intercept"], "dB_A2": result["final_b_offset_A2"],
        "rss": result["final_rss"], "slot_rmsds": result["slot_rmsds"],
        "minimum_occupancy_seen": result["minimum_occupancy_seen"],
    }


def run_case(root: Path, label: str, fixed_b_offset: float | None) -> dict[str, object]:
    runner = APrimeSequential(root / label / "runner", 8, 6, *SITE,
                              renderer_backend="torch", residual_scale_mode="none",
                              map_scaler_structure="full", mask_scope="window", device="cuda")
    runner.rama_floor = 0.02
    p1 = np.zeros(runner.rotator.ndofs)
    p2 = inverse_seed(runner, runner.base.window_for_deposited_b())
    parameterization = FullJointParameterization(runner.rotator.ndofs)
    initial = runner.torch_forward(parameterization.expand_torch(
        torch.as_tensor(parameterization.pack(p1, p2), dtype=torch.float64)
    )).detach().cpu().numpy()
    initial_pair = float(rmsd(runner.base.central_backbone(initial[0]), runner.base.central_backbone(initial[1])))
    result = joint_run(runner, p1, p2, label, root / label / "endpoint", initial_pair,
                       fixed_b_offset=fixed_b_offset)
    return {
        "initial_slot_to_slot_RMSD_A": initial_pair,
        "fixed_b_offset_A2": fixed_b_offset,
        "result": compact(result),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    atomic_json(args.output / "run_config.json", {
        "site": "8R7O_C_1681", "initialization": "slot 1 deposited A; slot 2 deposited B inverse seed",
        "objective": "backbone-only target/model; all-voxel fit",
        "cases": {"free_dB": "fit global dB", "fixed_dB": DEPOSITED_GATE_DB_A2},
    })
    rows = {}
    for label, fixed in (("deposited_B_init_free_dB", None), ("deposited_B_init_fixed_gate_dB", DEPOSITED_GATE_DB_A2)):
        path = args.output / f"{label}.json"
        if args.resume and path.is_file():
            row = json.loads(path.read_text())
        else:
            row = run_case(args.output, label, fixed)
            atomic_json(path, row)
        rows[label] = row
        atomic_json(args.output / "progress.json", {"status": "running", "completed": list(rows)})
    atomic_json(args.output / "summary.json", {"status": "complete", "cases": rows})
    atomic_json(args.output / "progress.json", {"status": "complete", "completed": list(rows)})


if __name__ == "__main__":
    main()
