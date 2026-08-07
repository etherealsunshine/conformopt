from density_denoiser.diagnose_symmetry_endpoint_loss import (
    identify_relevant_failed_starts,
)


def test_relevant_failed_starts_require_recovery_occupancy_and_hard_failure():
    ensemble_rows = [
        {"start": "0", "geometric_occupancy_success": "True"},
        {"start": "1", "geometric_occupancy_success": "False"},
        {"start": "2", "geometric_occupancy_success": "True"},
    ]
    conformer_rows = [
        {"start": "0", "no_symmetry_clash": "False"},
        {"start": "1", "no_symmetry_clash": "False"},
        {"start": "2", "no_symmetry_clash": "True"},
    ]
    assert identify_relevant_failed_starts(ensemble_rows, conformer_rows) == [0]
