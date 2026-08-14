#!/usr/bin/env python3
"""Replay one completed clean-D1 A-prime run with convergence diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clean_d1_benchmark import site_key
from run_d1_slot_coordination import build_specs, worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--flip-root", type=Path, required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inner-nfev", type=int, default=8)
    parser.add_argument("--outer-updates", type=int, default=6)
    parser.add_argument("--mirror-eta", type=float, default=0.001)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    site = next(item for item in manifest if site_key(item) == args.site)
    args.output.mkdir(parents=True, exist_ok=False)
    specs = build_specs(
        args.output / "specs", args.flip_root, site=(site["pdb_id"], site["chain"], int(site["resnum"])),
        mask_scope="window", start_pdb=args.start, b_factor_mode="single_conformer",
        device="cpu", occupancy_scheme="mirror", mirror_eta=args.mirror_eta,
    )
    spec = next(item for item in specs if item["label"] == "D_null_axis2_30deg")
    spec["output"] = str(args.output / "D_null_axis2_30deg")
    spec["inner_nfev"] = int(args.inner_nfev)
    spec["outer_updates"] = int(args.outer_updates)
    (args.output / "run_config.json").write_text(json.dumps({
        "site": args.site, "inner_nfev": args.inner_nfev,
        "outer_updates": args.outer_updates, "occupancy_scheme": "mirror",
        "mirror_eta": args.mirror_eta, "start": str(args.start),
        "nullspace": "axis2,30 degrees", "rama_floor": 0.02,
    }, indent=2) + "\n")
    result = worker(spec)
    (args.output / "summary.json").write_text(json.dumps({
        "status": "complete", "site": args.site, "result": result,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
