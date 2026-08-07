#!/usr/bin/env python3
"""Merge Probe 4c geometry and corrected tmol audits into deliverable CSVs."""

import csv
from pathlib import Path


ROOT = Path(__file__).parent
AUDIT = ROOT / "probe4c_results" / "endpoint_audit"
DESTINATIONS = {
    "experiment_1_complex_target": ROOT / "probe4c_results/experiment_1_complex_target",
    "experiment_2_A_synthetic_fobs": ROOT / "probe4c_results/experiment_2_regularized/A_synthetic_fobs",
    "experiment_2_B_localized_sf": ROOT / "probe4c_results/experiment_2_regularized/B_localized_sf",
    "experiment_2_C_realspace_local": ROOT / "probe4c_results/experiment_2_regularized/C_realspace_local",
}


def main() -> None:
    geometry = list(csv.DictReader((AUDIT / "endpoint_metrics.csv").open()))
    energy_rows = list(csv.DictReader((AUDIT / "tmol_energies.csv").open()))
    energies = {
        (row["experiment"], row["site"], row["start"]): row for row in energy_rows
    }
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
            "no_sub2A_clash": no_clash,
            "physical_valid": physical_valid,
            "recovered_B_lt_0.50A": recovered_b,
            "joint_success": physical_valid and recovered_b,
        })

    fields = list(merged[0])
    with (AUDIT / "physical_audit_all.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(merged)
    for experiment, destination in DESTINATIONS.items():
        selected = [row for row in merged if row["experiment"] == experiment]
        with (destination / "physical_audit.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(selected)
    print(f"wrote {len(merged)} merged endpoint rows")


if __name__ == "__main__":
    main()
