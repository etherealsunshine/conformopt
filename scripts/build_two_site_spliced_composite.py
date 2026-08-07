from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from density_denoiser.summarize_tmol_margin_sweep import (
    _atomic_csv,
    _atomic_json,
    _matched_rmsd,
    _read_csv,
    build_sweep,
)


REPLACED_SITES = {"7UO8_A_GLN53", "2VFP_A_TYR417"}
BASELINE_ENVIRONMENT_RULE = (
    "2026-07-24-altloc-minstate-water-invariant-v1"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-audit", type=Path, action="append", required=True)
    parser.add_argument("--baseline-run-manifest", type=Path)
    parser.add_argument("--replacement-audit", type=Path, required=True)
    parser.add_argument("--replacement-run-manifest", type=Path, required=True)
    parser.add_argument("--stale-baseline-table", type=Path, required=True)
    parser.add_argument("--comparison-table", type=Path)
    parser.add_argument("--comparison-label", default="comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    conformers: list[dict[str, str]] = []
    ensembles: list[dict[str, str]] = []
    geometry_rules: set[str] = set()
    tmol_rules: set[str] = set()
    provenance_rows: list[dict[str, object]] = []

    replacement_manifest = json.loads(args.replacement_run_manifest.read_text())
    replacement_rule = replacement_manifest["optimizer_physics_environment_rule"]
    baseline_manifest = (
        json.loads(args.baseline_run_manifest.read_text())
        if args.baseline_run_manifest
        else None
    )
    baseline_rule = (
        baseline_manifest["optimizer_physics_environment_rule"]
        if baseline_manifest
        else BASELINE_ENVIRONMENT_RULE
    )
    source_hash_keys = tuple(sorted(replacement_manifest["source_sha256"]))
    if baseline_manifest:
        if baseline_rule != replacement_rule:
            raise ValueError(
                f"mixed optimizer environment rules: "
                f"{baseline_rule} != {replacement_rule}"
            )
        if set(baseline_manifest["source_sha256"]) != set(source_hash_keys):
            raise ValueError("source manifests do not cover the same files")
        for key in source_hash_keys:
            if (
                baseline_manifest["source_sha256"][key]
                != replacement_manifest["source_sha256"][key]
            ):
                raise ValueError(f"source hash mismatch for {key}")
    sources = [
        *[(root, False) for root in args.baseline_audit],
        (args.replacement_audit, True),
    ]
    for root, replacement in sources:
        source_conformers = _read_csv(root / "active_conformer_strict_audit.csv")
        source_ensembles = _read_csv(root / "ensemble_geometry_audit.csv")
        if replacement:
            source_conformers = [
                row for row in source_conformers if row["site"] in REPLACED_SITES
            ]
            source_ensembles = [
                row for row in source_ensembles if row["site"] in REPLACED_SITES
            ]
        else:
            source_conformers = [
                row for row in source_conformers
                if row["site"] not in REPLACED_SITES
            ]
            source_ensembles = [
                row for row in source_ensembles
                if row["site"] not in REPLACED_SITES
            ]
        conformers.extend(source_conformers)
        ensembles.extend(source_ensembles)
        if source_ensembles:
            geometry_rules.add(
                json.loads((root / "strict_summary.json").read_text())[
                    "audit_rule_version"
                ]
            )
            tmol_rules.update(
                row["tmol_environment_rule"]
                for row in _read_csv(root / "tmol_energies.csv")
            )
            for site in sorted({row["site"] for row in source_ensembles}):
                provenance_rows.append(
                    {
                        "site": site,
                        "source_audit_root": str(root),
                        "optimizer_physics_environment_rule": (
                            replacement_rule
                            if replacement
                            else baseline_rule
                        ),
                        "five_site_optimizer_sha256": (
                            replacement_manifest["source_sha256"][
                                "/home/dev/workspace/density_denoiser/"
                                "five_site_optimizer.py"
                            ]
                            if replacement
                            else (
                                baseline_manifest["source_sha256"][
                                    "/home/dev/workspace/density_denoiser/"
                                    "five_site_optimizer.py"
                                ]
                                if baseline_manifest
                                else ""
                            )
                        ),
                        "clash_environment_sha256": (
                            replacement_manifest["source_sha256"][
                                "/home/dev/workspace/density_denoiser/"
                                "clash_environment.py"
                            ]
                            if replacement
                            else (
                                baseline_manifest["source_sha256"][
                                    "/home/dev/workspace/density_denoiser/"
                                    "clash_environment.py"
                                ]
                                if baseline_manifest
                                else ""
                            )
                        ),
                        "all_source_sha256_json": json.dumps(
                            (
                                replacement_manifest
                                if replacement
                                else baseline_manifest
                            )["source_sha256"],
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if baseline_manifest
                        else "",
                        "replacement_site": replacement,
                    }
                )

    if len(geometry_rules) != 1 or len(tmol_rules) != 1:
        raise ValueError(
            f"inconsistent audit rules: geometry={geometry_rules}, tmol={tmol_rules}"
        )
    if len(ensembles) != 1000 or len({row["site"] for row in ensembles}) != 20:
        raise ValueError(
            f"expected 20 sites/1000 starts, got "
            f"{len({row['site'] for row in ensembles})}/{len(ensembles)}"
        )

    stale_both = {
        row["site"]: int(row["both_found_conventional"])
        for row in _read_csv(args.stale_baseline_table)
    }
    per_site, q99_rows = build_sweep(conformers, ensembles, stale_both)
    if args.comparison_table:
        comparison = {
            row["site"]: int(
                row.get("both_found", row.get("both_found_conventional", ""))
            )
            for row in _read_csv(args.comparison_table)
            if row["site"] != "TOTAL"
        }
        if set(comparison) != {row["site"] for row in per_site}:
            raise ValueError("comparison table does not contain the same sites")
        for row in per_site:
            row[f"{args.comparison_label}_both_found"] = comparison[row["site"]]
            row[f"both_found_delta_vs_{args.comparison_label}"] = (
                int(row["both_found"]) - comparison[row["site"]]
            )
    totals: dict[str, object] = {"site": "TOTAL"}
    for key in per_site[0]:
        if key == "site":
            continue
        if key in {"site_positive_margin_q99", "site_q99_effective_tolerance"}:
            totals[key] = "per-site"
        elif key in {"stale_both_found", "both_found_delta_vs_stale"}:
            totals[key] = sum(
                int(row[key]) for row in per_site if row[key] != ""
            )
        else:
            totals[key] = sum(int(row[key]) for row in per_site)

    margin_rows = [
        {
            "candidate_id": row["candidate_id"],
            "site": row["site"],
            "start": int(row["start"]),
            "conformer": int(row["conformer"]),
            "occupancy": float(row["occupancy"]),
            "assignment": row["assignment"],
            "rmsd_to_matched_deposited": (
                _matched_rmsd(row) if row["assignment"] in {"A", "B"} else ""
            ),
            "tmol_energy": row["tmol_energy"],
            "tmol_reference_matched_AB": (
                row["tmol_reference_matched_AB"]
                if row["assignment"] in {"A", "B"}
                else ""
            ),
            "tmol_margin_candidate_minus_matched_deposited": (
                row["tmol_delta_vs_matched_AB"]
                if row["assignment"] in {"A", "B"}
                else ""
            ),
        }
        for row in conformers
    ]

    args.output.mkdir(parents=True)
    _atomic_csv(args.output / "site_rule_provenance.csv", provenance_rows)
    _atomic_csv(args.output / "per_conformer_tmol_margins.csv", margin_rows)
    _atomic_csv(args.output / "per_site_positive_margin_q99.csv", q99_rows)
    _atomic_csv(
        args.output / "per_site_cascade_and_tmol_sweep.csv",
        [*per_site, totals],
    )
    _atomic_json(
        args.output / "summary.json",
        {
            "kind": (
                "derived single-rule 20-site composite; no endpoint files "
                "were overwritten"
                if baseline_manifest
                else "derived two-site splice; no endpoint files were overwritten"
            ),
            "replaced_sites": sorted(REPLACED_SITES),
            "baseline_optimizer_physics_environment_rule": (
                baseline_rule
            ),
            "replacement_optimizer_physics_environment_rule": replacement_rule,
            "geometry_rule": next(iter(geometry_rules)),
            "tmol_rule": next(iter(tmol_rules)),
            "tmol_tolerance_promoted": False,
            "totals": totals,
        },
    )
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
