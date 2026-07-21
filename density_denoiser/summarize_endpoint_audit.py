"""Merge geometry and tmol endpoint audits into strict ensemble success metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--tmol-max-delta", type=float, default=10.0)
    args = parser.parse_args()

    conformers = read_csv(args.audit_root / "active_conformer_geometry_audit.csv")
    ensembles = read_csv(args.audit_root / "ensemble_geometry_audit.csv")
    energies = {
        row["candidate_id"]: row
        for row in read_csv(args.audit_root / "tmol_energies.csv")
    }
    missing = sorted({row["candidate_id"] for row in conformers} - set(energies))
    if missing:
        raise RuntimeError(f"missing tmol scores for {len(missing)} conformers")

    merged = []
    by_ensemble = defaultdict(list)
    for row in conformers:
        energy = energies[row["candidate_id"]]
        value = float(energy["tmol_energy"])
        reference = min(float(energy["tmol_A"]), float(energy["tmol_B"]))
        delta = value - reference
        tmol_valid = math.isfinite(delta) and delta <= args.tmol_max_delta
        strict_valid = (
            as_bool(row["no_sub2A_clash"])
            and as_bool(row["canonical_like_30deg"])
            and tmol_valid
        )
        combined = {
            **row,
            "tmol_energy": value,
            "tmol_reference_best_AB": reference,
            "tmol_delta_vs_best_AB": delta,
            "tmol_valid": tmol_valid,
            "strict_physical_valid": strict_valid,
        }
        merged.append(combined)
        by_ensemble[(row["site"], int(row["start"]))].append(combined)

    ensemble_merged = []
    for row in ensembles:
        key = (row["site"], int(row["start"]))
        active = by_ensemble[key]
        all_tmol = bool(active) and all(item["tmol_valid"] for item in active)
        all_strict = bool(active) and all(item["strict_physical_valid"] for item in active)
        recovered = as_bool(row["geometric_occupancy_success"])
        ensemble_merged.append({
            **row,
            "all_active_tmol_valid": all_tmol,
            "all_active_strict_physical_valid": all_strict,
            "strict_joint_success": recovered and all_strict,
        })

    sites = sorted({row["site"] for row in ensemble_merged})
    per_site = []
    for site in sites:
        site_ensembles = [row for row in ensemble_merged if row["site"] == site]
        site_conformers = [row for row in merged if row["site"] == site]
        per_site.append({
            "site": site,
            "ensembles": len(site_ensembles),
            "active_conformers": len(site_conformers),
            "both_found_conventional": sum(
                as_bool(row["both_found_conventional"]) for row in site_ensembles
            ),
            "recovery_and_occupancy": sum(
                as_bool(row["geometric_occupancy_success"]) for row in site_ensembles
            ),
            "all_active_strict_physical": sum(
                row["all_active_strict_physical_valid"] for row in site_ensembles
            ),
            "strict_joint_success": sum(
                row["strict_joint_success"] for row in site_ensembles
            ),
            "tmol_valid_conformers": sum(row["tmol_valid"] for row in site_conformers),
            "canonical_conformers": sum(
                as_bool(row["canonical_like_30deg"]) for row in site_conformers
            ),
            "clash_free_conformers": sum(
                as_bool(row["no_sub2A_clash"]) for row in site_conformers
            ),
        })

    summary = {
        "criteria": {
            "conventional_rmsd_A": 1.0,
            "both_found": True,
            "occupancy_tolerance": 0.20,
            "direct_and_symmetry_clash_A": 2.0,
            "rotamer_max_deviation_degrees": 30.0,
            "tmol_delta_vs_best_deposited_AB": args.tmol_max_delta,
        },
        "ensembles": len(ensemble_merged),
        "active_conformers": len(merged),
        "both_found_conventional": sum(
            as_bool(row["both_found_conventional"]) for row in ensemble_merged
        ),
        "recovery_and_occupancy": sum(
            as_bool(row["geometric_occupancy_success"]) for row in ensemble_merged
        ),
        "all_active_strict_physical": sum(
            row["all_active_strict_physical_valid"] for row in ensemble_merged
        ),
        "strict_joint_success": sum(
            row["strict_joint_success"] for row in ensemble_merged
        ),
        "tmol_valid_conformers": sum(row["tmol_valid"] for row in merged),
        "canonical_conformers": sum(
            as_bool(row["canonical_like_30deg"]) for row in merged
        ),
        "clash_free_conformers": sum(
            as_bool(row["no_sub2A_clash"]) for row in merged
        ),
        "per_site": per_site,
    }

    for name, rows in (
        ("active_conformer_strict_audit.csv", merged),
        ("ensemble_strict_audit.csv", ensemble_merged),
        ("strict_per_site.csv", per_site),
    ):
        with (args.audit_root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.audit_root / "strict_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
