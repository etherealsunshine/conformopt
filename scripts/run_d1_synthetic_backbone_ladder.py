#!/usr/bin/env python3
"""Synthetic-to-experimental recovery ladder for the two-site A' backbone test.

This controller is intentionally separate from the older D1 exploratory
launchers.  It reuses the production qFit/CCTBX map path, Torch Gaussian
renderer, A' kinematics, and slot-coordination objective, but owns the target
arrays and the rung ordering.  Rung 1 is a forward-model oracle and is a hard
gate: later rungs are not started after a rung-1 failure.

The first implementation is the backbone-only variant from the specification.
The ``--variant`` switch is reserved for the coupled chi extension; it fails
closed until that extension is implemented rather than silently substituting
another objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

QFIT_ENV = os.environ.get("D1_QFIT_ENV", "/home/dev/qfit_unet_data/.venv-qfit-audit-gpu")
TORCH_ENV = os.environ.get("D1_TORCH_ENV", QFIT_ENV)
PYTHON_MINOR = f"python{sys.version_info.major}.{sys.version_info.minor}"
QFIT_SITE = f"{QFIT_ENV}/lib/{PYTHON_MINOR}/site-packages"
QFIT_DYNLIB = f"{QFIT_ENV}/lib/{PYTHON_MINOR}/lib-dynload"
TORCH_SITE = f"{TORCH_ENV}/lib/{PYTHON_MINOR}/site-packages"
WORKSPACE = "/home/dev/workspace"
QFIT_SRC = f"{WORKSPACE}/external/qfit-3.0/src"
IMPORT_QFIT_FIRST = os.environ.get("D1_IMPORT_QFIT_FIRST", "0").lower() in {"1", "true", "yes"}
if os.path.isdir(QFIT_SITE):
    # Keep NumPy/SciPy paired with qFit's compiled extensions.  Torch may
    # intentionally come from a different virtual environment, so import it
    # afterward and leave the module cached while CCTBX resolves against qFit.
    sys.path.insert(0, QFIT_SITE)
    import numpy as np
    sys.path.remove(QFIT_SITE)
    if IMPORT_QFIT_FIRST:
        sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]
        # The CUDA and CCTBX extension stacks in the supplied GPU env have a
        # static-library initialization conflict when Torch is imported first.
        # Initializing mmtbx first is safe and leaves Torch CUDA-enabled.
        import mmtbx.validation.ramalyze  # noqa: F401
    if os.path.isdir(TORCH_SITE):
        sys.path.insert(0, TORCH_SITE)
    # Import CUDA Torch before qFit's compiled extension, matching the
    # established pod launchers.
    import torch  # noqa: F401
    sys.path.remove(TORCH_SITE)
    torch.set_num_threads(int(os.environ.get("D1_TORCH_THREADS", "8")))
    torch.set_num_interop_threads(1)
    if not IMPORT_QFIT_FIRST:
        sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]
else:  # pragma: no cover - local Mac is only a source/edit environment.
    import numpy as np

from scipy.ndimage import gaussian_filter

from density_denoiser.differentiable_renderer import (
    coefficients_for_elements,
    render_cctbx_density,
)
from run_d1_8d_sequential_poc import (
    atomic_csv,
    atomic_json,
    atomic_npz,
    extract_window_neighbors,
)
from run_d1_aprime_sequential import APrimeSequential, internal_geometry, rmsd
from run_d1_reachability import BACKBONE_NAMES, dihedrals
from run_d1_slot_coordination import (
    build_specs,
    joint_evaluate,
    joint_run,
)
from d1_population_calibrated_weights import (
    D1_OMEGA_SCALE_DEG,
    D1_RAMA_FLOOR,
    d1_weight_provenance,
)


SITES = (
    ("7SC4", "B", 2317),
    ("8R7O", "C", 1681),
    ("5OHJ", "A", 540),
)
NEUTRAL_ROOT = Path("/home/dev/qfit_unet_data/qfit_audit/d1_flip_survivor_neutral_starts_v2/sites")
LEGACY_8R7O_NEUTRAL_ROOT = Path("/home/dev/qfit_unet_data/qfit_audit/clean_d1_neutral_starts_v4/sites")
FLIP_ROOT = Path(WORKSPACE) / "data/qfit_2015_s004"
NEUTRAL_NAMES = {
    ("7SC4", "B", 2317): "7SC4_B_PRO2317",
    ("8R7O", "C", 1681): "8R7O_C_THR1681",
    ("5OHJ", "A", 540): "5OHJ_A_SER540",
}
NEUTRAL_PATHS = {
    ("7SC4", "B", 2317): NEUTRAL_ROOT,
    # The frozen production 8R7O control is the v4 neutral artifact; it is
    # the artifact that read back as the specified 5,205-voxel mask.
    ("8R7O", "C", 1681): LEGACY_8R7O_NEUTRAL_ROOT,
    ("5OHJ", "A", 540): NEUTRAL_ROOT,
}
FROZEN_GRID_METADATA = {
    "7SC4": {
        "mask_voxels": 2942,
        "grid_spacing_A_xyz": np.asarray((0.4187333333333333, 0.4584027777777778, 0.44930625)),
    },
    "8R7O": {
        "mask_voxels": 5205,
        "grid_spacing_A_xyz": np.asarray((0.3277, 0.3277, 0.3221)),
    },
    "5OHJ": {
        "mask_voxels": 5153,
        "grid_spacing_A_xyz": np.asarray((0.38381944444444444, 0.3870138888888889, 0.3915972222222222)),
    },
}


def site_name(site: tuple[str, str, int]) -> str:
    return f"{site[0]}_{site[1]}_{site[2]}"


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _estimate_noise_sigma(real_map: np.ndarray, mask: np.ndarray) -> dict[str, float | str | int]:
    """Estimate map noise from robust adjacent-voxel differences off-mask.

    For each Cartesian grid axis, adjacent pairs wholly outside the production
    mask are differenced.  The MAD of the pooled differences divided by
    ``sqrt(2)`` estimates the voxel sigma and is robust to a small amount of
    residual density.  This is measured separately for each site and recorded
    in the rung metadata.
    """
    valid = ~np.asarray(mask, dtype=bool)
    differences = []
    for axis in range(3):
        left = np.take(real_map, range(real_map.shape[axis] - 1), axis=axis)
        right = np.take(real_map, range(1, real_map.shape[axis]), axis=axis)
        left_valid = np.take(valid, range(valid.shape[axis] - 1), axis=axis)
        right_valid = np.take(valid, range(1, valid.shape[axis]), axis=axis)
        values = (right - left)[left_valid & right_valid]
        if len(values):
            differences.append(np.asarray(values, dtype=float))
    pooled = np.concatenate(differences) if differences else np.empty(0, dtype=float)
    if len(pooled) < 10:
        raise RuntimeError("insufficient off-mask adjacent pairs for noise estimate")
    median = float(np.median(pooled))
    mad = float(np.median(np.abs(pooled - median)))
    sigma = 1.4826 * mad / math.sqrt(2.0)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError(f"invalid robust map noise estimate: {sigma}")
    return {
        "method": "1.4826*MAD(adjacent off-mask voxel differences)/sqrt(2)",
        "sigma": sigma,
        "difference_median": median,
        "difference_mad": mad,
        "adjacent_pairs": int(len(pooled)),
    }


def _build_site_context(site: tuple[str, str, int], root: Path, device: str,
                        rama_floor: float = D1_RAMA_FLOOR,
                        omega_scale_deg: float = D1_OMEGA_SCALE_DEG) -> dict[str, object]:
    key = site_name(site)
    neutral = NEUTRAL_PATHS[site] / NEUTRAL_NAMES[site] / "neutral_start_aprime_single_slot.pdb"
    if not neutral.is_file():
        raise FileNotFoundError(f"frozen neutral start not found: {neutral}")
    runner = APrimeSequential(
        root / "context_runner", 1, 1, *site,
        renderer_backend="torch", residual_scale_mode="none",
        # The production neutral-start path owns the frozen all-atom mask.
        map_scaler_structure="full", mask_scope="window", device=device,
        start_pdb=neutral, b_factor_mode="single_conformer",
        density_atom_scope="all",
    )
    runner.rama_floor = float(rama_floor)
    runner.omega_scale_deg = float(omega_scale_deg)
    # The map from the production constructor is already scaled and has had
    # the external seven-residue-scoped neighbour density subtracted.
    real_target = runner.base.target.copy()
    deposited_a = runner.base.window_for_deposited_a()
    deposited_b = runner.base.window_for_deposited_b()
    deposited_models = runner.base.model_density_batch(
        np.stack((deposited_a, deposited_b)), slots=np.asarray((0, 1))
    )
    synthetic_clean = runner.base.deposited_occupancies @ deposited_models
    # ΔB is fixed at zero in the ladder, so fit the intercept at the deposited
    # occupancies and zero-width offset instead of profiling a free ΔB.
    intercept_value = float(np.mean(
        real_target - runner.base.deposited_occupancies @ deposited_models
    ))
    intercept_rss = float(np.square(
        real_target - runner.base.deposited_occupancies @ deposited_models - intercept_value
    ).sum())
    neighbour_structure = extract_window_neighbors(
        runner.base.full_structure, runner.base.window, runner.base.qfit.options.padding
    )
    # Render only the already-selected mask grid.  Calling qFit's full-map
    # transformer here allocates the entire crystallographic cell (several GB
    # for these data) even though rung 4 consumes only mask voxels.
    neighbour_coordinates = torch.as_tensor(
        np.asarray(neighbour_structure.coor, dtype=float),
        dtype=torch.float64, device=runner.base.torch_device,
    )
    neighbour_b = torch.as_tensor(
        np.asarray(neighbour_structure.b, dtype=float),
        dtype=torch.float64, device=runner.base.torch_device,
    )
    neighbour_coefficients = coefficients_for_elements(
        [str(value) for value in neighbour_structure.e],
        dtype=torch.float64, device=runner.base.torch_device,
    )
    with torch.no_grad():
        neighbour_density = render_cctbx_density(
            runner.base._renderer_grid, neighbour_coordinates, neighbour_b,
            neighbour_coefficients, u_base=runner.base._renderer_u_base,
            voxel_chunk=1024,
        ).cpu().numpy()
    noise = _estimate_noise_sigma(runner.base.qfit.xmap.array, runner.base.mask)
    rng_seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    noise_values = rng.normal(0.0, noise["sigma"], size=len(synthetic_clean))
    rung_targets = {
        "1": synthetic_clean,
        "2": synthetic_clean + noise_values,
        "3": synthetic_clean + noise_values + intercept_value,
        # Add an outside-window density contribution, then apply the exact
        # production subtraction to that same atom set.  The cancellation is
        # intentional and is checked/reported rather than hidden.
        "4": synthetic_clean + noise_values + intercept_value
        + neighbour_density - neighbour_density,
        "5": real_target,
    }
    preflight = json.loads((neutral.parent / "preflight.json").read_text())
    initial_p2 = np.asarray(
        preflight["initialisation"]["p2_parameters_deg"], dtype=float
    )
    initial_slots = runner.torch_forward(
        torch.as_tensor(np.vstack((np.zeros(20), initial_p2)), dtype=torch.float64)
    ).detach().cpu().numpy()
    initial_pair_rmsd = float(rmsd(
        runner.base.central_backbone(initial_slots[0]),
        runner.base.central_backbone(initial_slots[1]),
    ))
    cell_matrix = runner.base._renderer_cell.detach().cpu().numpy()
    n_real_xyz = np.asarray(runner.base.qfit.xmap.n_real(), dtype=float)
    # The renderer uses fractional coordinates multiplied by cell_matrix.T;
    # therefore grid axis x/y/z advances along the *columns* of cell_matrix.
    grid_spacing_xyz = np.linalg.norm(cell_matrix, axis=0) / n_real_xyz
    mask_radius = float(0.5 + runner.base.resolution / 3.0)
    expected = FROZEN_GRID_METADATA[site[0]]
    actual_mask_voxels = int(runner.base.mask.sum())
    if actual_mask_voxels != expected["mask_voxels"]:
        raise RuntimeError(
            f"frozen mask mismatch for {key}: {actual_mask_voxels} != "
            f"{expected['mask_voxels']} voxels"
        )
    if not np.allclose(grid_spacing_xyz, expected["grid_spacing_A_xyz"], atol=5e-5, rtol=0.0):
        raise RuntimeError(
            f"frozen grid-spacing mismatch for {key}: {grid_spacing_xyz.tolist()} != "
            f"{expected['grid_spacing_A_xyz'].tolist()}"
        )
    context = {
        "site": key,
        "source_neutral_start": str(neutral),
        "deposited_occupancies_A_B": runner.base.deposited_occupancies.tolist(),
        "resolution_A": float(runner.base.resolution),
        "map_grid_shape_zyx": list(runner.base.qfit.xmap.array.shape),
        "grid_spacing_A_xyz": grid_spacing_xyz.tolist(),
        "grid_spacing_definition": "norms of orthogonalization-matrix columns divided by n_real xyz",
        "u_base_A2": float(runner.base._renderer_u_base),
        "u_base_source": "cctbx.xray.ext.calc_u_base(d_min=resolution, grid_resolution_factor=0.25), matching SequentialBackbonePOC",
        "mask_voxels": actual_mask_voxels,
        "frozen_metadata_check": "passed",
        "mask_radius_A": mask_radius,
        "mask_builder": "SequentialBackbonePOC(mask_scope=window, density_atom_scope=all, start_pdb=neutral) -> full_window_mask -> cctbx.masks.around_atoms",
        "target_renderer": "density_denoiser.differentiable_renderer.render_cctbx_density via APrimeSequential.model_density_batch",
        "target_renderer_function": "density_denoiser.differentiable_renderer.render_cctbx_density",
        "target_atom_scope": "all atoms in both deposited seven-residue windows",
        "target_normalization": "none",
        "rama_floor": float(rama_floor),
        "omega_scale_deg": float(omega_scale_deg),
        "weight_calibration": d1_weight_provenance(),
        "noise": noise,
        "noise_seed": rng_seed,
        "fitted_intercept": intercept_value,
        "deposited_fixed_occupancy_rss_real_map": intercept_rss,
        "neighbour_atom_count": int(neighbour_structure.natoms),
        "neighbour_subtraction_scope": "outside fitted seven-residue window, within qFit padding, bulk_solvent_level=0.0",
        "neighbour_density_sha256": _sha256_array(neighbour_density),
        "rung4_add_then_production_subtract_max_abs_residual": float(
            np.max(np.abs(rung_targets["4"] - rung_targets["3"]))
        ),
        "initialization": {
            "slot1": "neutral start, zero 20-torsion delta",
            "slot2": "neutral start plus 30 degrees along fixed closure-nullspace axis 2",
            "p2_parameters_deg": initial_p2.tolist(),
            "initial_slot_to_slot_backbone_rmsd_A": initial_pair_rmsd,
            "specified_slot_to_slot_backbone_rmsd_A": (
                "measured from the production forward pass; no site-level value is hardcoded"
            ),
        },
    }
    atomic_json(root / "context.json", context)
    atomic_npz(
        root / "targets.npz",
        rung1=rung_targets["1"], rung2=rung_targets["2"], rung3=rung_targets["3"],
        rung4=rung_targets["4"], rung5=rung_targets["5"],
        clean_synthetic=synthetic_clean, noise=noise_values,
        neighbour_density=neighbour_density, deposited_models=deposited_models,
        deposited_A=deposited_a, deposited_B=deposited_b,
        mask_indices=np.argwhere(runner.base.mask),
    )
    return {"runner": runner, "context": context, "targets": rung_targets}


def _rmsd_report(runner: APrimeSequential, slots: np.ndarray) -> dict[str, object]:
    rows = []
    for index, slot in enumerate(slots):
        central = runner.base.central_backbone(slot)
        to_a = float(rmsd(central, runner.a_backbone))
        to_b = float(rmsd(central, runner.b_backbone))
        rows.append({
            "slot": index + 1,
            "to_A_A": to_a,
            "to_B_A": to_b,
            "to_A_fraction_AB": to_a / runner.ab_distance,
            "to_B_fraction_AB": to_b / runner.ab_distance,
            "nearer": "A" if to_a <= to_b else "B",
        })
    return {"slots": rows, "slot_to_slot_backbone_rmsd_A": float(rmsd(
        runner.base.central_backbone(slots[0]), runner.base.central_backbone(slots[1])
    )), "deposited_A_to_B_backbone_rmsd_A": float(runner.ab_distance)}


def _objective_terms(runner: APrimeSequential, target: np.ndarray, slots: np.ndarray,
                     weights: np.ndarray, intercept: float, normalizer: float) -> dict[str, object]:
    models = runner.base.model_density_batch(slots, slots=np.asarray((0, 1)))
    rss = float(np.square(target - weights @ models - intercept).sum())
    seam = []
    for slot in slots:
        value, _, _ = __import__("run_d1_slot_coordination", fromlist=["seam_vector"]).seam_vector(
            runner.initial_backbone, slot[runner.bb_indices]
        )
        seam.append(value)
    seam = np.asarray(seam)
    coordinates_t = torch.as_tensor(slots, dtype=torch.float64, device=runner.base.torch_device)
    _, omega_delta, _, rama_barrier = runner.torch_omega_and_rama(coordinates_t)
    omega_delta = omega_delta.detach().cpu().numpy()
    rama_barrier = rama_barrier.detach().cpu().numpy()
    return {
        "density_rss": rss,
        "density_term_normalized": rss / max(normalizer, 1e-12),
        "seam_term": float(runner.rho / 2.0 * np.square(seam).sum()),
        "seam_vectors_A_equivalent": seam.tolist(),
        "ramachandran_term": float(runner.rama_weight * np.square(rama_barrier).sum()),
        "ramachandran_barriers": rama_barrier.tolist(),
        "omega_term": float(runner.planar_weight * np.square(omega_delta / runner.omega_scale_deg).sum()),
        "omega_delta_deg": omega_delta.tolist(),
        "rotamer_barrier_term": 0.0,
        "objective_without_AL_multiplier": float(
            rss / max(normalizer, 1e-12)
            + runner.rho / 2.0 * np.square(seam).sum()
            + runner.rama_weight * np.square(rama_barrier).sum()
            + runner.planar_weight * np.square(omega_delta / runner.omega_scale_deg).sum()
        ),
    }


def _geometry_report(runner: APrimeSequential, slots: np.ndarray) -> dict[str, object]:
    reports = []
    for slot in slots:
        geometry = internal_geometry(runner.window, runner.initial, slot)
        reports.append({
            "max_abs_bond_length_change_from_neutral_A": geometry["max_abs_bond_length_change_from_A_A"],
            "max_abs_bond_angle_change_from_neutral_deg": geometry["max_abs_bond_angle_change_from_A_deg"],
            "bond_length_changes_A": geometry["bond_length_delta_from_A_A"],
            "bond_angle_changes_deg": geometry["bond_angle_delta_from_A_deg"],
        })
    return {"slots": reports, "reference": "neutral-start NeRF geometry; exact ideal deviations are represented by the neutral-start residual"}


def _run_backbone_rung(runner: APrimeSequential, target: np.ndarray, output: Path,
                       initial_p2: np.ndarray, context: dict[str, object], rung: str,
                       inner_nfev: int, carry_trust_radii: bool = False,
                       rama_floor: float = D1_RAMA_FLOOR,
                       omega_scale_deg: float = D1_OMEGA_SCALE_DEG) -> dict[str, object]:
    runner.target = np.asarray(target, dtype=float)
    runner.base.target = runner.target.copy()
    runner.output = output
    output.mkdir(parents=True, exist_ok=True)
    runner.rama_floor = float(rama_floor)
    runner.omega_scale_deg = float(omega_scale_deg)
    atomic_json(output / "run_config.json", {
        "site": context["site"], "rung": rung, "variant": "A_backbone_only",
        "target_artifact": str(output.parent.parent / "targets.npz"),
        "parameters_per_slot": 20, "parameters_total_torsions": 40,
        "rendering_scope": "all atom",
        "dB": {"mode": "fixed", "value_A2": 0.0},
        "per_slot_trust_radii": True,
        "occupancy_updates": {"method": "multiplicative mirror descent", "eta": 0.01, "gradient_normalization": "unit norm"},
        "rama_floor": float(rama_floor), "omega_scale_deg": float(omega_scale_deg), "omega_weight": 0.05,
        "augmented_lagrangian_rho": 0.755,
        "outer_updates_max": 200, "outer_stop": "lambda norm change <=1%",
        "inner_solver": {"method": "two independent scipy least_squares(method=trf)", "max_nfev": inner_nfev, "xtol": 1e-10, "ftol": 1e-10, "gtol": 1e-10},
        "jacobian": {"mode": "forward-mode", "tangent_chunk_size": int(os.environ.get("D1_JACOBIAN_CHUNK_SIZE", "40"))},
        "carry_trust_radii": bool(carry_trust_radii),
        "initialization": context["initialization"],
    })
    p1 = np.zeros(20, dtype=float)
    result = joint_run(
        runner, p1, initial_p2, f"rung_{rung}_A_backbone_only", output,
        float(context["initialization"]["initial_slot_to_slot_backbone_rmsd_A"]),
        fixed_b_offset=0.0, occupancy_scheme="mirror", mirror_eta=0.01,
        inner_nfev=inner_nfev, outer_updates=200, lambda_relative_tolerance=0.01,
        per_slot_trust_radii=True, torch_native_trf=False,
        carry_trust_radii=carry_trust_radii,
    )
    saved = np.load(output / "final_slots.npz")
    slots = np.stack((saved["slot1_window"], saved["slot2_window"]))
    weights = np.asarray(result["final_occupancies"], dtype=float)
    intercept = float(result["final_intercept"])
    aa_models = runner.base.model_density_batch(
        np.stack((runner.initial, runner.initial)), slots=np.asarray((0, 1))
    )
    normalizer = max(float(np.square(runner.target - np.array([0.5, 0.5]) @ aa_models - np.mean(runner.target - np.array([0.5, 0.5]) @ aa_models)).sum()), 1e-12)
    row = {
        **result,
        "rung": rung,
        "variant": "A_backbone_only",
        "recovery": _rmsd_report(runner, slots),
        "occupancies": {
            "final": weights.tolist(),
            "total_L1_path_from_0p5": float(sum(
                np.abs(current - previous).sum()
                for previous, current in _occupancy_path(result.get("trajectory", []))
            )),
            "trajectory": [values.tolist() for values in _occupancy_values(result.get("trajectory", []))],
        },
        "objective_terms": _objective_terms(runner, runner.target, slots, weights, intercept, normalizer),
        "geometry": _geometry_report(runner, slots),
        "convergence": {
            "outer_updates": result.get("outer_updates_completed"),
            "lambda_rule_fired": bool(result.get("lambda_stop_reached")),
            "inner_diagnostics": result.get("inner_solve_diagnostics", []),
            "any_inner_evaluation_cap": any(bool(item.get("hit_evaluation_cap")) for item in result.get("inner_solve_diagnostics", [])),
        },
        "assignment_consistent": all(
            (entry["nearer"] == expected)
            for entry, expected in zip(_rmsd_report(runner, slots)["slots"], ("A", "B"))
        ),
    }
    atomic_json(output / "ladder_result.json", row)
    return row


def _occupancy_values(trajectory: list[dict[str, object]]) -> list[np.ndarray]:
    values = []
    for item in trajectory:
        if item.get("event") == "mirror_update":
            current = np.asarray(item["occupancies"], dtype=float)
            if not values or not np.allclose(current, values[-1]):
                values.append(current)
    return [np.asarray((0.5, 0.5))] + values


def _occupancy_path(trajectory: list[dict[str, object]]):
    values = _occupancy_values(trajectory)
    return list(zip(values[:-1], values[1:]))


def run(args: argparse.Namespace) -> None:
    root = args.output / site_name(args.site)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "complete.json").is_file() and not args.resume:
        raise FileExistsError(f"completed site exists: {root}; use --resume")
    context_data = _build_site_context(
        args.site, root, args.device,
        rama_floor=args.rama_floor,
        omega_scale_deg=args.omega_scale_deg,
    )
    runner = context_data["runner"]
    context = context_data["context"]
    targets = context_data["targets"]
    if args.preflight_only:
        atomic_json(root / "preflight_only.json", {
            "status": "complete",
            "site": site_name(args.site),
            "mask_voxels": context["mask_voxels"],
            "grid_spacing_A_xyz": context["grid_spacing_A_xyz"],
            "u_base_A2": context["u_base_A2"],
            "target_renderer_function": context["target_renderer_function"],
        })
        atomic_json(root / "progress.json", {
            "status": "preflight_complete", "site": site_name(args.site)
        })
        return
    initial_p2 = np.asarray(context["initialization"]["p2_parameters_deg"], dtype=float)
    atomic_json(root / "progress.json", {"status": "rung_1", "site": site_name(args.site)})
    rows = {}
    for variant in ("A",):
        variant_root = root / f"variant_{variant}"
        row = _run_backbone_rung(
            runner, targets["1"], variant_root / "rung_1", initial_p2, context, "1",
            args.inner_nfev, args.carry_trust_radii,
            args.rama_floor, args.omega_scale_deg,
        )
        rows[f"1_{variant}"] = row
        # Ground truth is intentionally direct: two slots must be assigned to
        # opposite deposited states with a 0.30 A diagnostic threshold.
        slots = row["recovery"]["slots"]
        passed = bool(
            slots[0]["to_A_A"] < 0.30 and slots[1]["to_B_A"] < 0.30
            or slots[0]["to_B_A"] < 0.30 and slots[1]["to_A_A"] < 0.30
        )
        row["rung1_gate"] = {"passed": passed, "threshold_A": 0.30}
        atomic_json(variant_root / "rung_1" / "ladder_result.json", row)
        if not passed:
            atomic_json(root / "summary.json", {"status": "rung_1_failed", "rows": rows})
            atomic_json(root / "progress.json", {"status": "stopped_after_rung_1_failure"})
            return
    if args.rung1_only:
        atomic_json(root / "summary.json", {"status": "rung_1_complete", "rows": rows})
        atomic_json(root / "rung_1_complete.json", {"status": "complete", "site": site_name(args.site)})
        atomic_json(root / "progress.json", {"status": "rung_1_complete", "completed": list(rows)})
        return
    for rung in ("2", "3", "4", "5"):
        atomic_json(root / "progress.json", {"status": f"rung_{rung}", "completed": list(rows)})
        row = _run_backbone_rung(
            runner, targets[rung], root / "variant_A" / f"rung_{rung}",
            initial_p2, context, rung, args.inner_nfev, args.carry_trust_radii,
            args.rama_floor, args.omega_scale_deg,
        )
        rows[f"{rung}_A"] = row
    atomic_json(root / "summary.json", {"status": "complete", "rows": rows})
    atomic_json(root / "complete.json", {"status": "complete", "site": site_name(args.site)})
    atomic_json(root / "progress.json", {"status": "complete", "completed": list(rows)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdb-id", choices=[site[0] for site in SITES], required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--resnum", type=int, required=True)
    parser.add_argument("--variant", choices=("A", "B"), default="A")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Build and checkpoint targets without entering any rung.")
    parser.add_argument("--rung1-only", action="store_true",
                        help="Run Variant A rung 1, then stop before later rungs.")
    parser.add_argument("--inner-nfev", type=int, default=400,
                        help="SciPy TRF evaluation cap per slot and outer update.")
    parser.add_argument("--rama-floor", type=float, default=D1_RAMA_FLOOR,
                        help="Ramachandran population floor for the barrier.")
    parser.add_argument("--omega-scale-deg", type=float, default=D1_OMEGA_SCALE_DEG,
                        help="Omega restraint scale in degrees.")
    parser.add_argument("--carry-trust-radii", action="store_true",
                        help="carry each slot's ending TRF radius into the next outer update.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.site = (args.pdb_id, args.chain, args.resnum)
    if args.site not in SITES:
        raise ValueError(f"site is not one of the frozen two-site panel: {args.site}")
    if args.variant != "A":
        raise RuntimeError("variant B is not yet implemented; no substitute variant was run")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "run_config.json", {
        "operation": "synthetic backbone recovery ladder",
        "site": site_name(args.site), "variant": args.variant,
        "preflight_only": bool(args.preflight_only), "rung1_only": bool(args.rung1_only),
        "carry_trust_radii": bool(args.carry_trust_radii),
        "rama_floor": float(args.rama_floor),
        "omega_scale_deg": float(args.omega_scale_deg),
        "rungs": {"1": "clean production-rendered synthetic", "2": "plus matched noise", "3": "plus fitted intercept", "4": "plus neighbours then exact production subtraction", "5": "real experimental production target"},
        "policy": "stop after rung 1 failure",
    })
    try:
        run(args)
    except Exception as exc:
        atomic_json(args.output / site_name(args.site) / "failure.json", {
            "status": "error", "error": repr(exc), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
