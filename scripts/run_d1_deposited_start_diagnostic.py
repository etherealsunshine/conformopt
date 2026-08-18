#!/usr/bin/env python3
"""Measure the calibrated rung-1 objective minimum from deposited A/B."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from run_d1_synthetic_backbone_ladder import (
    SITES,
    _build_site_context,
    _objective_terms,
    _rmsd_report,
)
import numpy as np
import torch

from run_d1_slot_coordination import inverse_seed, joint_run
from d1_population_calibrated_weights import D1_OMEGA_SCALE_DEG, D1_RAMA_FLOOR


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    site = (args.pdb_id, args.chain, args.resnum)
    site_key = f"{args.pdb_id}_{args.chain}_{args.resnum}"
    root = args.output / site_key
    root.mkdir(parents=True, exist_ok=True)

    context_data = _build_site_context(
        site, root, args.device,
        rama_floor=D1_RAMA_FLOOR,
        omega_scale_deg=D1_OMEGA_SCALE_DEG,
    )
    runner = context_data["runner"]
    context = context_data["context"]
    target = np.asarray(context_data["targets"]["1"], dtype=float)
    runner.target = target
    runner.base.target = target.copy()

    deposited_a = runner.base.window_for_deposited_a()
    deposited_b = runner.base.window_for_deposited_b()
    p1 = inverse_seed(runner, deposited_a)
    p2 = inverse_seed(runner, deposited_b)
    deposited_slots = runner.torch_forward(
        torch.as_tensor(np.stack((p1, p2)), dtype=torch.float64)
    ).detach().cpu().numpy()

    aa_models = runner.base.model_density_batch(
        np.stack((runner.initial, runner.initial)), slots=np.asarray((0, 1))
    )
    normalizer = max(float(np.square(
        target - np.asarray((0.5, 0.5)) @ aa_models
        - np.mean(target - np.asarray((0.5, 0.5)) @ aa_models)
    ).sum()), 1e-12)
    deposited_weights = np.asarray(runner.base.deposited_occupancies, dtype=float)
    deposited_terms = _objective_terms(
        runner, target, deposited_slots, deposited_weights, 0.0, normalizer
    )

    run_root = root / "variant_A" / "rung_1_deposited_start"
    runner.output = run_root
    result = joint_run(
        runner, p1, p2, "rung_1_A_deposited_start", run_root,
        float(context["initialization"]["initial_slot_to_slot_backbone_rmsd_A"]),
        fixed_b_offset=0.0,
        occupancy_scheme="mirror", mirror_eta=0.01,
        inner_nfev=args.inner_nfev, outer_updates=200,
        lambda_relative_tolerance=0.01,
        per_slot_trust_radii=True, torch_native_trf=False,
        carry_trust_radii=False,
    )
    saved = np.load(run_root / "final_slots.npz")
    final_slots = np.stack((saved["slot1_window"], saved["slot2_window"]))
    final_weights = np.asarray(result["final_occupancies"], dtype=float)
    final_intercept = float(result["final_intercept"])
    final_terms = _objective_terms(
        runner, target, final_slots, final_weights, final_intercept, normalizer
    )
    recovery = _rmsd_report(runner, final_slots)
    initial_recovery = _rmsd_report(runner, deposited_slots)

    summary = {
        "status": "complete",
        "site": site_key,
        "weights": {
            "rama_floor": D1_RAMA_FLOOR,
            "omega_scale_deg": D1_OMEGA_SCALE_DEG,
        },
        "initialization": {
            "slot1": "deposited A via inverse dihedral seed",
            "slot2": "deposited B via inverse dihedral seed",
            "inverse_seed_slot1_deg": p1.tolist(),
            "inverse_seed_slot2_deg": p2.tolist(),
            "reconstructed_deposited_state": initial_recovery,
        },
        "deposited_objective": {
            "occupancies": deposited_weights.tolist(),
            "intercept_A2": 0.0,
            "terms": deposited_terms,
        },
        "converged_objective": {
            "occupancies": final_weights.tolist(),
            "intercept_A2": final_intercept,
            "terms": final_terms,
        },
        "converged_recovery": recovery,
        "optimizer": {
            "inner_nfev": args.inner_nfev,
            "outer_updates": 200,
            "per_slot_trust_radii": True,
            "carry_trust_radii": False,
            "occupancy_scheme": "mirror",
            "mirror_eta": 0.01,
            "fixed_dB_A2": 0.0,
            "any_inner_evaluation_cap": any(
                bool(item.get("hit_evaluation_cap"))
                for item in result.get("inner_solve_diagnostics", [])
            ),
        },
    }
    atomic_json(root / "summary.json", summary)
    atomic_json(root / "progress.json", {"status": "complete", "site": site_key})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdb-id", choices=[site[0] for site in SITES], required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--resnum", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--inner-nfev", type=int, default=400)
    args = parser.parse_args()
    args.site = (args.pdb_id, args.chain, args.resnum)
    if args.site not in SITES:
        raise ValueError(f"site is not one of the frozen two-site panel: {args.site}")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except Exception as exc:
        atomic_json(args.output / f"{args.pdb_id}_{args.chain}_{args.resnum}" / "failure.json", {
            "status": "error", "error": repr(exc), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
