import itertools

import numpy as np

from scripts.diagnose_frozen_v3_residual_rotamers import (
    canonical_physical_states,
    pearson_correlation,
    scalar_density_fit,
    select_best_fit,
)


def test_scalar_density_fit_recovers_nonnegative_occupancy():
    density = np.asarray([0.0, 1.0, 2.0, 0.5])
    residual = 0.37 * density
    fit = scalar_density_fit(residual, density)
    assert np.isclose(fit["fitted_occupancy_nnls"], 0.37)
    assert np.isclose(fit["fraction_sse_reduced"], 1.0)
    assert np.isclose(fit["pearson_correlation"], 1.0)


def test_scalar_density_fit_clips_negative_insertions():
    density = np.asarray([0.0, 1.0, 2.0])
    fit = scalar_density_fit(-density, density)
    assert fit["unconstrained_fitted_occupancy"] < 0.0
    assert fit["fitted_occupancy_nnls"] == 0.0
    assert fit["sse_reduction"] == 0.0


def test_best_fit_is_selected_by_raw_space_sse_not_pearson():
    residual = np.asarray([1.0, 2.0, 0.0, -1.0])
    densities = [
        np.asarray([0.1, 0.2, 0.0, -0.1]),
        np.asarray([1.0, 1.0, 0.0, 0.0]),
    ]
    fits = [scalar_density_fit(residual, density) for density in densities]
    assert select_best_fit(fits) == 0
    assert np.isclose(pearson_correlation(densities[0], residual), 1.0)


def test_production_canonical_pool_is_cartesian_and_unique():
    states = canonical_physical_states("MET", 3)
    marginal_counts = [
        len(set(state[index] for state in states)) for index in range(3)
    ]
    assert marginal_counts == [3, 3, 3]
    assert len(states) == int(np.prod(marginal_counts))
    assert len(set(itertools.chain(states))) >= 3
