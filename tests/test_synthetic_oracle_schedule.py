from density_denoiser.five_site_optimizer import _production_density_schedule


def test_synthetic_oracle_uses_expected_per_residue_schedule() -> None:
    assert _production_density_schedule(3, 500, 1.0, True) == (
        (0.0, 1.0, 500),
    )
    assert _production_density_schedule(4, 500, 1.0, True) == (
        (4.0, 1.0, 100),
        (2.0, 0.1, 100),
        (0.0, 0.01, 100),
    )
