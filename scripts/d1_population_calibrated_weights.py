"""Frozen population-calibrated D1 backbone restraint weights.

These values are derived from the 232,890-row deposited A/B backbone panel:
the omega scale is the central pre-peptide omega p95, and the Ramachandran
floor is the central-conformer p0.5 percentile.  Keep them in one module so
new D1 launchers cannot silently fall back to the old 5 degree / 0.02 values.
"""

from __future__ import annotations


D1_RAMA_FLOOR = 0.0016931321476603015
D1_OMEGA_SCALE_DEG = 13.019150002471157
D1_WEIGHT_CALIBRATION_ID = "backbone_altloc_site_list_v4_central_population_q0p5_p95_v1"
D1_WEIGHT_CALIBRATION_SOURCE_ROWS = 232_890
D1_WEIGHT_CALIBRATION_SOURCE = (
    "backbone_altloc_site_list_v4/candidate_sites.csv; central deposited A/B rows"
)


def d1_weight_provenance() -> dict[str, float | int | str]:
    """Return serializable provenance for run configurations and artifacts."""
    return {
        "calibration_id": D1_WEIGHT_CALIBRATION_ID,
        "rama_floor": D1_RAMA_FLOOR,
        "omega_scale_deg": D1_OMEGA_SCALE_DEG,
        "source_rows": D1_WEIGHT_CALIBRATION_SOURCE_ROWS,
        "source": D1_WEIGHT_CALIBRATION_SOURCE,
        "omega_definition": "central pre-peptide omega trans-planarity p95",
        "rama_definition": "central deposited A/B Ramachandran probability p0.5",
    }
