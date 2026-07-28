"""Merge geometry and tmol endpoint audits into strict ensemble success metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from .audit_five_site_endpoints import CONFORMER_MATCHING_VERSION
from .residue_geometry import AUDIT_RULE_VERSION


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def matched_tmol_evaluation(
    geometry_row: dict[str, str],
    energy_row: dict[str, str],
    maximum_delta: float = 0.0,
) -> tuple[float, float, bool]:
    """Compare an assigned A/B candidate only with its matched control."""
    assignment = geometry_row["assignment"]
    if assignment not in {"A", "B"}:
        return float("nan"), float("nan"), False
    reference = float(energy_row[f"tmol_{assignment}"])
    delta = float(energy_row["tmol_energy"]) - reference
    return reference, delta, math.isfinite(delta) and delta <= maximum_delta


def select_assigned_pair(
    active: list[dict],
    minimum_occupancy: float = 0.10,
) -> dict[str, dict] | None:
    """Select the best-RMSD found conformer for deposited A and B."""
    selected = {}
    for assignment in ("A", "B"):
        candidates = [
            row
            for row in active
            if row["assignment"] == assignment
            and float(row.get("assigned_occupancy", row["occupancy"]))
            > minimum_occupancy
        ]
        if not candidates:
            return None
        selected[assignment] = min(
            candidates,
            key=lambda row: float(row[f"rmsd_to_{assignment}_conventional"]),
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument(
        "--tmol-max-delta",
        type=float,
        default=0.0,
        help="maximum energy above the candidate's matched deposited A/B control",
    )
    parser.add_argument("--found-occupancy", type=float, default=0.10)
    args = parser.parse_args()

    conformers = read_csv(args.audit_root / "active_conformer_geometry_audit.csv")
    ensembles = read_csv(args.audit_root / "ensemble_geometry_audit.csv")
    energies = {
        row["candidate_id"]: row
        for row in read_csv(args.audit_root / "tmol_energies.csv")
    }
    tmol_environment_rules = sorted({
        row.get("tmol_environment_rule", "legacy_shared_base")
        for row in energies.values()
    })
    missing = sorted({row["candidate_id"] for row in conformers} - set(energies))
    if missing:
        raise RuntimeError(f"missing tmol scores for {len(missing)} conformers")

    merged = []
    by_ensemble = defaultdict(list)
    for row in conformers:
        energy = energies[row["candidate_id"]]
        value = float(energy["tmol_energy"])
        reference, delta, tmol_valid = matched_tmol_evaluation(
            row, energy, args.tmol_max_delta
        )
        strict_valid = (
            as_bool(row["no_sub2A_clash"])
            and as_bool(row.get(
                "rotamer_within_allowed_width", row["canonical_like_30deg"]
            ))
            and tmol_valid
        )
        combined = {
            **row,
            "tmol_energy": value,
            "tmol_reference_matched_AB": reference,
            "tmol_delta_vs_matched_AB": delta,
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
        assigned_pair = select_assigned_pair(active, args.found_occupancy)
        pair_complete = assigned_pair is not None
        pair_strict = pair_complete and all(
            item["strict_physical_valid"] for item in assigned_pair.values()
        )
        ensemble_merged.append({
            **row,
            "all_active_tmol_valid": all_tmol,
            "all_active_strict_physical_valid": all_strict,
            "occupancy_conditioned_all_active_strict_physical": (
                recovered and all_strict
            ),
            "assigned_pair_complete": pair_complete,
            "assigned_pair_candidate_A": (
                assigned_pair["A"]["candidate_id"] if pair_complete else ""
            ),
            "assigned_pair_candidate_B": (
                assigned_pair["B"]["candidate_id"] if pair_complete else ""
            ),
            "assigned_pair_strict_physical_valid": pair_strict,
            "assigned_pair_strict_joint_success": recovered and pair_strict,
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
            "occupancy_conditioned_all_active_strict_physical": sum(
                row["occupancy_conditioned_all_active_strict_physical"]
                for row in site_ensembles
            ),
            "assigned_pair_strict_joint_success": sum(
                row["assigned_pair_strict_joint_success"] for row in site_ensembles
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
        "audit_rule_version": AUDIT_RULE_VERSION,
        "conformer_matching_version": CONFORMER_MATCHING_VERSION,
        "criteria": {
            "conventional_rmsd_A": 1.0,
            "both_found": True,
            "occupancy_tolerance": 0.20,
            "direct_and_symmetry_clash_A": 2.0,
            "rotamer_gate": "shared residue/chi centers and per-chi widths",
            "tmol_delta_vs_matched_deposited_AB": args.tmol_max_delta,
            "tmol_environment_rules": tmol_environment_rules,
            "assigned_pair_selection": (
                "optimal one-to-one A/B RMSD assignment among conformers with "
                f"occupancy > {args.found_occupancy}; one conformer per state"
            ),
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
        "occupancy_conditioned_all_active_strict_physical": sum(
            row["occupancy_conditioned_all_active_strict_physical"]
            for row in ensemble_merged
        ),
        "assigned_pair_strict_joint_success": sum(
            row["assigned_pair_strict_joint_success"] for row in ensemble_merged
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
