#!/usr/bin/env python3
"""Short GPU pilot for trust-radius carry-over on the synthetic rung-1 target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_d1_synthetic_backbone_ladder import SITES, _build_site_context
from run_d1_slot_coordination import joint_run
from d1_population_calibrated_weights import D1_OMEGA_SCALE_DEG, D1_RAMA_FLOOR


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    site = (args.pdb_id, args.chain, args.resnum)
    site_key = f"{args.pdb_id}_{args.chain}_{args.resnum}"
    root = args.output / site_key
    context_data = _build_site_context(
        site, root, args.device,
        rama_floor=D1_RAMA_FLOOR,
        omega_scale_deg=D1_OMEGA_SCALE_DEG,
    )
    runner = context_data["runner"]
    context = context_data["context"]
    runner.target = context_data["targets"]["1"]
    runner.base.target = runner.target.copy()
    p1 = [0.0] * 20
    p2 = context["initialization"]["p2_parameters_deg"]
    output = root / "carry_on_pilot"
    result = joint_run(
        runner, p1, p2, "rung_1_carry_on_pilot", output,
        float(context["initialization"]["initial_slot_to_slot_backbone_rmsd_A"]),
        fixed_b_offset=0.0,
        occupancy_scheme="mirror", mirror_eta=0.01,
        inner_nfev=args.inner_nfev, outer_updates=args.outer_updates,
        lambda_relative_tolerance=None,
        per_slot_trust_radii=True, torch_native_trf=False,
        carry_trust_radii=args.carry_trust_radii,
    )
    atomic_json(root / "summary.json", {
        "status": "complete",
        "site": site_key,
        "carry_trust_radii": args.carry_trust_radii,
        "rama_floor": D1_RAMA_FLOOR,
        "omega_scale_deg": D1_OMEGA_SCALE_DEG,
        "outer_updates": args.outer_updates,
        "inner_nfev": args.inner_nfev,
        "inner_diagnostics": result.get("inner_solve_diagnostics", []),
        "final_occupancies": result.get("final_occupancies"),
        "final_rss": result.get("final_rss"),
        "final_energy": result.get("final_energy"),
        "slot_rmsds": result.get("slot_rmsds"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdb-id", choices=[site[0] for site in SITES], required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--resnum", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--inner-nfev", type=int, default=400)
    parser.add_argument("--outer-updates", type=int, default=10)
    parser.add_argument("--carry-trust-radii", action="store_true")
    args = parser.parse_args()
    args.site = (args.pdb_id, args.chain, args.resnum)
    if args.site not in SITES:
        raise ValueError(f"site is not one of the frozen panel: {args.site}")
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
