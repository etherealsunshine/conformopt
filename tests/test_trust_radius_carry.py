import os
import sys
from pathlib import Path


# The GPU audit environment needs qFit/CCTBX loaded before NumPy/SciPy/Torch.
# Reuse the production bootstrap so these tests exercise the actual module.
os.environ.setdefault("D1_IMPORT_QFIT_FIRST", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pytest

try:
    from run_d1_synthetic_backbone_ladder import np  # noqa: F401
except (ImportError, OSError):
    pytest.skip("qFit GPU-audit runtime is unavailable", allow_module_level=True)

from scripts.run_d1_slot_coordination import (
    _least_squares_with_trust_trace,
    DEFAULT_GEOMETRY_GRADIENT_OCCUPANCY_FLOOR,
    FullJointParameterization,
    MIN_CARRIED_TRUST_RADIUS_SCALED,
    carryable_trust_radius,
    embed_free_parameters,
    mirror_descent_occupancy_update,
    mirror_initial_occupancies,
    occupancy_decoupled_density_jacobian,
)


def _residual(x: np.ndarray) -> np.ndarray:
    return np.asarray((x[0] - 2.0, 3.0 * x[1] + 1.0), dtype=float)


def _jacobian(_x: np.ndarray) -> np.ndarray:
    return np.asarray(((1.0, 0.0), (0.0, 3.0)), dtype=float)


def test_scipy_trf_accepts_a_carried_initial_radius():
    first_trace = []
    first = _least_squares_with_trust_trace(
        _residual, np.zeros(2), jac=_jacobian, method="trf", x_scale=10.0,
        max_nfev=20, ftol=1e-12, xtol=1e-12, gtol=1e-12,
        trace=first_trace,
    )
    assert first_trace
    assert all("accepted" in row for row in first_trace)

    carried_radius = float(first_trace[-1]["radius_after_scaled"])
    carried_trace = []
    carried = _least_squares_with_trust_trace(
        _residual, first.x + np.asarray((0.1, -0.1)), jac=_jacobian, method="trf", x_scale=10.0,
        max_nfev=20, ftol=1e-12, xtol=1e-12, gtol=1e-12,
        trace=carried_trace, initial_radius=carried_radius,
    )

    assert carried_trace
    assert carried_trace[0]["radius_before_scaled"] == pytest.approx(carried_radius)
    np.testing.assert_allclose(carried.x, np.asarray((2.0, -1.0 / 3.0)), atol=1e-10, rtol=0.0)


def test_free_parameter_embedding_preserves_fixed_coordinates():
    fixed = np.arange(40.0)
    indices = np.arange(20)
    free = np.full(20, -3.5)
    embedded = embed_free_parameters(free, fixed, indices)
    np.testing.assert_allclose(embedded[:20], free)
    np.testing.assert_allclose(embedded[20:], fixed[20:])


def test_mirror_descent_uses_an_explicit_unexplained_occupancy_slack():
    weights = mirror_initial_occupancies(2)
    np.testing.assert_allclose(weights, np.asarray((1.0 / 3.0, 1.0 / 3.0)))
    updated = mirror_descent_occupancy_update(
        target=np.zeros(3), models=np.zeros((2, 3)), weights=weights,
        intercept=0.0, eta=0.01,
    )
    np.testing.assert_allclose(updated, weights)
    assert updated.sum() == pytest.approx(2.0 / 3.0)
    assert 1.0 - updated.sum() == pytest.approx(1.0 / 3.0)


def test_mirror_descent_preserves_a_fixed_slot_occupancy():
    weights = np.asarray((0.3985056, 0.25))
    fixed = np.asarray((0.3985056, np.nan))
    updated = mirror_descent_occupancy_update(
        target=np.asarray((1.0, 0.0)),
        models=np.asarray(((1.0, 0.0), (0.0, 1.0))), weights=weights,
        intercept=0.0, eta=0.01, fixed_weights=fixed,
    )
    assert updated[0] == pytest.approx(0.3985056, abs=1e-12)
    assert 0.0 < updated[1] < 1.0 - updated[0]
    assert updated.sum() < 1.0


def test_underflowed_carried_radius_resets_to_default_only_below_numeric_floor():
    assert carryable_trust_radius(0.1) == pytest.approx(0.1)
    assert carryable_trust_radius(MIN_CARRIED_TRUST_RADIUS_SCALED) == pytest.approx(
        MIN_CARRIED_TRUST_RADIUS_SCALED
    )
    assert carryable_trust_radius(MIN_CARRIED_TRUST_RADIUS_SCALED / 2.0) is None
    assert carryable_trust_radius(1e-100) is None


def test_per_slot_density_jacobian_decoupling_leaves_non_density_rows_unchanged():
    parameterization = FullJointParameterization(20)
    jacobian = np.ones((7, 41), dtype=float)
    transformed = occupancy_decoupled_density_jacobian(
        jacobian, density_rows=3, weights=np.asarray((0.25, 0.50)),
        parameterization=parameterization,
    )
    np.testing.assert_allclose(transformed[:3, :20], 4.0)
    np.testing.assert_allclose(transformed[:3, 20:40], 2.0)
    # The optional global B column and all seam/Rama/omega rows are untouched.
    np.testing.assert_allclose(transformed[:3, 40], 1.0)
    np.testing.assert_allclose(transformed[3:], 1.0)
    np.testing.assert_allclose(jacobian, 1.0)


def test_per_slot_density_jacobian_decoupling_caps_low_occupancy_amplification():
    parameterization = FullJointParameterization(20)
    transformed = occupancy_decoupled_density_jacobian(
        np.ones((2, 40), dtype=float), density_rows=2,
        weights=np.asarray((0.01, 0.50)), parameterization=parameterization,
    )
    np.testing.assert_allclose(
        transformed[:, :20], 1.0 / DEFAULT_GEOMETRY_GRADIENT_OCCUPANCY_FLOOR,
    )
