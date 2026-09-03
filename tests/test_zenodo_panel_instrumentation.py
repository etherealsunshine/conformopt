import math
from types import SimpleNamespace

import numpy as np

from scripts.run_qfit_aprime import (
    chi_parameters_to_radians,
    chi_exact_result_needs_lsmr,
    chi_solver_bounds,
    chi_solver_tolerances,
    chi_parameter_scaling,
    is_chi_solver_numerical_failure,
    least_squares_stage_diagnostics,
    normalized_zscore_density_residual,
    resolve_source_pdb,
    rotamer_prior_residual_rows,
    should_subtract_window_sidechains,
    validate_finite_chi_trial,
)
from scripts.run_d1_8d_sequential_poc import SequentialBackbonePOC
from scripts.calibrate_zenodo_rotamer_prior import dihedral_radians, width_scaled_cosine_cost
from scripts.report_zenodo_native_lambda33_six import physical_chi_degrees
from experiments.probe4.core import dihedral
from density_denoiser.residue_geometry import canonical_centers_radians, canonical_width_degrees
from scripts.run_d1_slot_coordination import (
    _zscore_numpy,
    _zscore_torch,
    mirror_descent_ratio_update,
)
from scripts.diagnose_zenodo_chi_sites import (
    quartet_geometry,
    solver_projected_components,
)
import torch


def test_resolve_source_pdb_prefers_frozen_panel_input(tmp_path):
    source_dir = tmp_path / "inputs" / "source"
    source_dir.mkdir(parents=True)
    panel_source = source_dir / "6kve.pdb"
    panel_source.write_text("panel source\n")
    record = {"pdb_id": "6KVE"}
    assert resolve_source_pdb(tmp_path, record) == panel_source


def test_least_squares_stage_diagnostics_preserves_solver_exit():
    result = SimpleNamespace(
        status=2,
        message="`ftol` termination condition is satisfied.",
        nfev=7,
        njev=4,
        optimality=1.25e-7,
        grad=np.asarray([3.0, 4.0]),
    )
    summary = least_squares_stage_diagnostics(result, max_nfev=20)
    assert summary["termination_status"] == 2
    assert summary["termination_message"] == result.message
    assert summary["nfev"] == 7
    assert summary["njev"] == 4
    assert summary["projected_gradient_norm_end"] == result.optimality
    assert summary["gradient_norm_end"] == 5.0
    assert summary["hit_evaluation_cap"] is False
    assert summary["converged_on_gradient"] is True


def test_least_squares_stage_diagnostics_marks_evaluation_cap():
    result = SimpleNamespace(
        status=0,
        message="The maximum number of function evaluations is exceeded.",
        nfev=20,
        njev=19,
        optimality=0.42,
        grad=np.asarray([0.42]),
    )
    summary = least_squares_stage_diagnostics(result, max_nfev=20)
    assert summary["hit_evaluation_cap"] is True
    assert summary["evaluation_cap"] == 20
    assert summary["converged_on_gradient"] is False


def test_all_atom_target_retains_sidechains_and_chi_uses_gradient_only_exit():
    assert should_subtract_window_sidechains("all") is False
    assert should_subtract_window_sidechains("backbone") is True
    assert chi_solver_tolerances() == {"ftol": None, "xtol": None, "gtol": 1.0e-6}
    assert chi_solver_bounds() == (-180.0, 180.0)


def test_chi_trial_guard_and_numerical_failure_classification():
    validate_finite_chi_trial(np.asarray([0.0, 180.0]), "test")
    for value in (np.nan, np.inf, -np.inf):
        with np.testing.assert_raises(FloatingPointError):
            validate_finite_chi_trial(np.asarray([0.0, value]), "test")
    assert is_chi_solver_numerical_failure(FloatingPointError("non-finite trial"))
    assert is_chi_solver_numerical_failure(ValueError("array contains NaN"))
    assert not is_chi_solver_numerical_failure(ValueError("bad topology"))


def test_chi_exact_stagnation_requires_lsmr_retry():
    stalled = SimpleNamespace(status=0, nfev=200, njev=2, optimality=1206.2)
    ordinary_cap = SimpleNamespace(status=0, nfev=200, njev=35, optimality=0.04)
    converged = SimpleNamespace(status=1, nfev=20, njev=20, optimality=8.0e-7)
    assert chi_exact_result_needs_lsmr(stalled, 200)
    assert not chi_exact_result_needs_lsmr(ordinary_cap, 200)
    assert not chi_exact_result_needs_lsmr(converged, 200)


def test_chi_parameter_scaling_uses_downstream_heavy_atom_counts():
    blocks = [
        {
            "slot": 0,
            "resname": "TRP",
            "n_chi": 2,
            "rotations": [
                ("CA", "CB", ("CG", "CD1", "CD2", "NE1")),
                ("CB", "CG", ("CD1", "CD2")),
            ],
        },
        {
            "slot": 1,
            "resname": "VAL",
            "n_chi": 1,
            "rotations": [("CA", "CB", ("CG1", "CG2"))],
        },
    ]
    scales, details = chi_parameter_scaling(blocks)
    assert np.allclose(scales, [0.25, 0.5, 0.5])
    assert [row["downstream_heavy_atom_count"] for row in details] == [4, 2, 2]
    assert [row["slot"] for row in details] == [1, 1, 2]
    assert [row["chi_index"] for row in details] == [1, 2, 1]
    assert all(row["x_scale_units"] == "degrees" for row in details)


def test_chi_solver_degrees_are_converted_once_at_renderer_boundary():
    degrees = torch.tensor([-180.0, -1.0, 0.0, 1.0, 180.0], dtype=torch.float64)
    assert torch.allclose(
        chi_parameters_to_radians(degrees),
        torch.tensor([-math.pi, -math.pi / 180.0, 0.0,
                      math.pi / 180.0, math.pi], dtype=torch.float64),
        atol=1.0e-15,
    )


def test_chi_zscore_residual_has_unit_voxel_count_normalization():
    target = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=torch.float64)
    model = torch.tensor([4.0, 1.0, -2.0, -3.0], dtype=torch.float64,
                         requires_grad=True)
    rows = normalized_zscore_density_residual(model, target)
    model_z = (model - model.mean()) / torch.sqrt(torch.mean((model - model.mean()) ** 2))
    assert torch.allclose(rows, (model_z - target) / math.sqrt(target.numel()))
    assert torch.allclose(torch.dot(rows, rows), torch.mean((model_z - target) ** 2))
    torch.dot(rows, rows).backward()
    assert torch.isfinite(model.grad).all()


def test_renderer_wrap_branch_is_fixed_and_continuous_at_unit_cell_face():
    runner = SequentialBackbonePOC.__new__(SequentialBackbonePOC)
    runner.torch_device = "cpu"
    runner._renderer_cell = torch.eye(3, dtype=torch.float64)
    runner._renderer_fractional_wrap_offsets = None
    reference = torch.tensor([[1.0, 0.25, 0.75]], dtype=torch.float64)
    runner._set_renderer_reference_wrap_offsets(reference)
    assert torch.equal(
        runner._renderer_fractional_wrap_offsets,
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    step = 1.0e-7
    minus = torch.tensor([[1.0 - step, 0.25, 0.75]], dtype=torch.float64)
    plus = torch.tensor([[1.0 + step, 0.25, 0.75]], dtype=torch.float64)
    offsets = runner._renderer_fractional_wrap_offsets
    wrapped_minus = minus - offsets
    wrapped_plus = plus - offsets
    assert np.isclose(float(wrapped_plus[0, 0] - wrapped_minus[0, 0]), 2.0 * step)
    assert np.isclose(float((wrapped_plus[0, 0] - wrapped_minus[0, 0]) / (2.0 * step)), 1.0)


def test_chi_quartet_geometry_flags_only_actual_degeneracy():
    regular = quartet_geometry({
        "A": np.asarray([0.0, 1.0, 0.0]),
        "B": np.asarray([0.0, 0.0, 0.0]),
        "C": np.asarray([1.0, 0.0, 0.0]),
        "D": np.asarray([1.0, 1.0, 1.0]),
    }, ("A", "B", "C", "D"))
    assert regular["complete"] is True
    assert regular["coincident"] is False
    assert regular["collinear_within_0p1_deg"] is False

    degenerate = quartet_geometry({
        "A": np.asarray([-1.0, 0.0, 0.0]),
        "B": np.asarray([0.0, 0.0, 0.0]),
        "C": np.asarray([1.0, 0.0, 0.0]),
        "D": np.asarray([2.0, 0.0, 0.0]),
    }, ("A", "B", "C", "D"))
    assert degenerate["collinear_within_0p1_deg"] is True

    missing = quartet_geometry({"A": np.zeros(3)}, ("A", "B", "C", "D"))
    assert missing["complete"] is False
    assert missing["missing_atoms"] == ["B", "C", "D"]


def test_solver_projected_components_reproduce_bound_factor_only():
    gradient = np.asarray([2.0, -3.0])
    parameters = np.asarray([10.0, -20.0])
    projected = solver_projected_components(
        gradient, parameters, (-180.0, 180.0),
    )
    # Positive g uses distance to lower bound; negative g uses upper bound.
    # x_scale affects the trust-region step, but not reported optimality.
    assert np.allclose(projected, [2.0 * 190.0, -3.0 * 200.0])


def test_rotamer_prior_rows_match_existing_cosine_energy_per_active_slot():
    coordinates = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0], [0.4, 1.2, 0.8]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0], [-0.6, 1.4, 0.5]],
    ], dtype=torch.float64, requires_grad=True)
    blocks = [
        {"slot": slot, "resname": "SER", "n_chi": 1,
         "names": ["CB", "OG"], "indices": np.asarray([2, 3]),
         "fixed": {"N": coordinates[slot, 0], "CA": coordinates[slot, 1]},
         "rotations": [("CA", "CB", ("OG",))]}
        for slot in (0, 1)
    ]
    weight = 2.75
    rows, details = rotamer_prior_residual_rows(
        coordinates, blocks, weight, active_slots={0, 1},
    )
    expected = 0.0
    for slot in (0, 1):
        internal = dihedral(*coordinates[slot]).detach()
        angle = torch.atan2(torch.sin(internal - torch.pi), torch.cos(internal - torch.pi))
        centers = torch.tensor(
            canonical_centers_radians("SER", 0), dtype=torch.float64,
        )
        deltas = torch.atan2(torch.sin(angle - centers), torch.cos(angle - centers))
        expected += weight * (30.0 / canonical_width_degrees("SER", 0)) ** 2 * float(
            (1.0 - torch.cos(deltas)).min()
        )
    assert len(details) == 2
    assert np.isclose(float(torch.dot(rows, rows).detach()), expected)
    torch.dot(rows, rows).backward()
    assert torch.isfinite(coordinates.grad).all()

    one_slot_rows, _ = rotamer_prior_residual_rows(
        coordinates.detach(), blocks, weight, active_slots={1},
    )
    assert one_slot_rows.numel() == 1


def test_rotamer_prior_coordinate_jacobian_matches_finite_difference():
    base = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0], [0.4, 1.2, 0.8]],
    ], dtype=torch.float64)
    blocks = [{
        "slot": 0, "resname": "SER", "n_chi": 1,
        "names": ["CB", "OG"], "indices": np.asarray([2, 3]),
        "fixed": {"N": base[0, 0], "CA": base[0, 1]},
        "rotations": [("CA", "CB", ("OG",))],
    }]

    def rows_from_shift(shift):
        coordinate_delta = torch.zeros_like(base)
        coordinate_delta[0, 3, 2] = shift[0]
        return rotamer_prior_residual_rows(
            base + coordinate_delta, blocks, 1.7, active_slots={0},
        )[0]

    point = torch.tensor([0.03], dtype=torch.float64)
    analytical = torch.autograd.functional.jacobian(rows_from_shift, point).detach().numpy()
    step = 1.0e-6
    plus = rows_from_shift(point + step).detach().numpy()
    minus = rows_from_shift(point - step).detach().numpy()
    finite_difference = (plus - minus) / (2.0 * step)
    assert np.allclose(analytical[:, 0], finite_difference, atol=1.0e-8, rtol=1.0e-7)


def test_rotamer_population_cost_uses_nearest_canonical_center_and_width():
    center = canonical_centers_radians("SER", 0)[0]
    deviation, cost = width_scaled_cosine_cost("SER", 0, center)
    assert np.isclose(deviation, 0.0)
    assert np.isclose(cost, 0.0)
    deviation, cost = width_scaled_cosine_cost("SER", 0, center + np.deg2rad(20.0))
    assert np.isclose(deviation, 20.0)
    assert cost > 0.0


def test_calibration_and_optimizer_use_the_same_physical_chi_convention():
    points = np.asarray([
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.4, 1.2, 0.8],
    ])
    internal = dihedral(*torch.as_tensor(points, dtype=torch.float64))
    optimizer_physical = torch.atan2(
        torch.sin(internal - torch.pi), torch.cos(internal - torch.pi),
    )
    calibration_physical = dihedral_radians([point for point in points])
    difference = math.atan2(
        math.sin(float(optimizer_physical) - calibration_physical),
        math.cos(float(optimizer_physical) - calibration_physical),
    )
    assert np.isclose(difference, 0.0, atol=1.0e-12)


def test_endpoint_rotamer_audit_uses_the_same_physical_chi_convention():
    points = torch.as_tensor([
        [1.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.4, 1.2, 0.8],
    ], dtype=torch.float64)
    audited = physical_chi_degrees(*points)
    expected = math.degrees(dihedral_radians(list(points.numpy())))
    difference = (audited - expected + 180.0) % 360.0 - 180.0
    assert np.isclose(difference, 0.0, atol=1.0e-12)


def test_zscore_density_is_population_normalized_and_differentiable():
    values = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=float)
    normalized, mean, std = _zscore_numpy(values)
    assert mean == 3.75
    assert np.isclose(std, np.sqrt(np.mean((values - mean) ** 2)))
    assert np.isclose(np.mean(normalized), 0.0)
    assert np.isclose(np.mean(normalized ** 2), 1.0)

    x = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(
        lambda v: _zscore_torch(v)[0], x,
    ).detach().numpy()
    step = 1.0e-6
    finite_difference = np.zeros_like(jacobian)
    for index in range(len(values)):
        plus = values.copy(); plus[index] += step
        minus = values.copy(); minus[index] -= step
        finite_difference[:, index] = (
            _zscore_numpy(plus)[0] - _zscore_numpy(minus)[0]
        ) / (2.0 * step)
    assert np.allclose(jacobian, finite_difference, atol=1.0e-8, rtol=1.0e-8)


def test_ratio_mirror_update_pins_total_and_moves_ratio_under_zscore():
    target = np.asarray([1.0, 0.0, 1.0, 0.0])
    models = np.asarray([
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    weights = np.asarray([0.72, 0.28])
    updated, gradient = mirror_descent_ratio_update(
        target, models, weights, 0.0, eta=0.1, density_mode="zscore",
    )
    assert np.all(np.isfinite(gradient))
    assert np.isclose(updated.sum(), 1.0, atol=1.0e-12)
    assert np.all(updated > 0.0)
    assert not np.allclose(updated, weights)
