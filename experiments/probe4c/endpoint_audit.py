#!/usr/bin/env python3
"""Run the validated Probe 4b geometry audit on Probe 4c endpoints."""

from pathlib import Path

from experiments.probe4b import endpoint_audit as audit


ROOT = Path(__file__).resolve().parents[2]
audit.OUT = ROOT / "artifacts" / "probe4c_results" / "endpoint_audit"
audit.ASP_ALLOWS_QUADRATURE = True
audit.EXPERIMENTS = {
    "experiment_1_complex_target": (
        ROOT / "artifacts/probe4c_results/experiment_1_complex_target/altloc_test/trajectories.json"
    ),
    "experiment_2_A_synthetic_fobs": (
        ROOT / "artifacts/probe4c_results/experiment_2_regularized/A_synthetic_fobs/altloc_test/trajectories.json"
    ),
    "experiment_2_B_localized_sf": (
        ROOT / "artifacts/probe4c_results/experiment_2_regularized/B_localized_sf/altloc_test/trajectories.json"
    ),
    "experiment_2_C_realspace_local": (
        ROOT / "artifacts/probe4c_results/experiment_2_regularized/C_realspace_local/altloc_test/trajectories.json"
    ),
}


if __name__ == "__main__":
    audit.main()
