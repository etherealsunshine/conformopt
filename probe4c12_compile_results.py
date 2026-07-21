#!/usr/bin/env python3
"""Merge Probe 4c.1/4c.2 geometry and corrected tmol endpoint audits."""

import csv
from pathlib import Path


ROOT = Path(__file__).parent
AUDIT = ROOT / "probe4c12_results" / "endpoint_audit"
DESTINATIONS = {
    "probe4c1_kinematic_complex": ROOT / "probe4c1_results",
    "probe4c2_A_soft_synthetic": ROOT / "probe4c2_results/A_synthetic_fobs",
    "probe4c2_B_soft_localized": ROOT / "probe4c2_results/B_localized_sf",
    "probe4c2_C_soft_realspace": ROOT / "probe4c2_results/C_realspace_local",
}


def main() -> None:
    geometry = list(csv.DictReader((AUDIT / "endpoint_metrics.csv").open()))
    energy_rows = list(csv.DictReader((AUDIT / "tmol_energies.csv").open()))
    energies = {(r["experiment"], r["site"], r["start"]): r for r in energy_rows}
    merged = []
    for row in geometry:
        energy = energies[(row["experiment"], row["site"], row["start"])]
        tmol = float(energy["tmol_energy"])
        tmol_a, tmol_b = float(energy["tmol_A"]), float(energy["tmol_B"])
        delta_energy = tmol - min(tmol_a, tmol_b)
        no_clash = (
            float(row["min_direct_distance"]) >= 2.0
            and float(row["min_symmetry_distance"]) >= 2.0
        )
        canonical = row["canonical_like_30deg"] == "True"
        physical_valid = no_clash and delta_energy < 10.0 and canonical
        recovered_b = float(row["rmsd_to_B"]) < 0.50
        merged.append({
            **row,
            "tmol_energy": tmol,
            "tmol_A": tmol_a,
            "tmol_B": tmol_b,
            "tmol_dE_vs_best_deposited": delta_energy,
            "random_tmol_mean": float(energy["random_mean"]),
            "random_tmol_min": float(energy["random_min"]),
            "random_tmol_max": float(energy["random_max"]),
            "no_sub2A_clash": no_clash,
            "physical_valid": physical_valid,
            "recovered_B_lt_0.50A": recovered_b,
            "joint_success": physical_valid and recovered_b,
        })

    fields = list(merged[0])
    with (AUDIT / "physical_audit_all.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    for experiment, destination in DESTINATIONS.items():
        selected = [r for r in merged if r["experiment"] == experiment]
        if len(selected) != 250:
            raise RuntimeError(f"expected 250 rows for {experiment}, got {len(selected)}")
        with (destination / "physical_audit.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
    print(f"wrote {len(merged)} merged endpoint rows")


if __name__ == "__main__":
    main()
