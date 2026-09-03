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
    window_backbone_indices,
)
from run_d1_aprime_sequential import APrimeSequential, internal_geometry, rmsd, seam_vector
from run_d1_reachability import BACKBONE_NAMES, dihedrals
from run_d1_slot_coordination import (
    DEFAULT_STATIONARITY_PROJECTED_GRADIENT_THRESHOLD,
    build_specs,
    inverse_seed,
    joint_evaluate,
    joint_run,
    seam_quadratic_cost,
)
from d1_population_calibrated_weights import (
    D1_OMEGA_SCALE_DEG,
    D1_RAMA_FLOOR,
    d1_weight_provenance,
)


SITES = (
    ("6ZWK", "B", 47),
    ("8R7O", "C", 1681),
    ("7SC4", "B", 2317),
    ("5OHJ", "A", 540),
    ("6I3B", "B", 209),
)
FLIP_ROOT = Path(WORKSPACE) / "data/qfit_2015_s004"
NEUTRAL_ROOT = Path("/home/dev/qfit_unet_data/qfit_audit/d1_neutral_starts_nerf_rebuilt_v4/sites")
NEUTRAL_NAMES = {
    ("6ZWK", "B", 47): "6ZWK_B_47",
    ("8R7O", "C", 1681): "8R7O_C_1681",
    ("7SC4", "B", 2317): "7SC4_B_2317",
    ("5OHJ", "A", 540): "5OHJ_A_540",
    ("6I3B", "B", 209): "6I3B_B_209",
}
NEUTRAL_PATHS = {site: NEUTRAL_ROOT for site in SITES}
FROZEN_GRID_METADATA = {
    "6ZWK": {
        "mask_voxels": 4680,
        "grid_spacing_A_xyz": np.asarray((0.38849777777777783, 0.38849777777777783, 0.36600347222222224)),
        "u_base_A2": 0.03042841785238881,
    },
    "8R7O": {
        "mask_voxels": 5329,
        "grid_spacing_A_xyz": np.asarray((0.3276822916666666, 0.3276822916666666, 0.3221458333333333)),
        "u_base_A2": 0.023468333067598823,
    },
    "7SC4": {
        "mask_voxels": 2880,
        "grid_spacing_A_xyz": np.asarray((0.4187333333333333, 0.4584027777777778, 0.44930625)),
        "u_base_A2": 0.04327628366074445,
    },
    "5OHJ": {
        "mask_voxels": 5213,
        "grid_spacing_A_xyz": np.asarray((0.38381944444444444, 0.3870138888888889, 0.3915972222222222)),
        "u_base_A2": 0.03242230106617484,
    },
    "6I3B": {
        "mask_voxels": 13045,
        "grid_spacing_A_xyz": np.asarray((0.24505729166666668, 0.24459027777777775, 0.24226666666666666)),
        "u_base_A2": 0.012665176468793858,
    },
}


def site_name(site: tuple[str, str, int]) -> str:
    return f"{site[0]}_{site[1]}_{site[2]}"


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _deposited_window_with_fallback(base, state: str) -> tuple[np.ndarray, list[str]]:
    """Return a deposited state window, filling absent altloc atoms by occupancy mean.

    Some deposited windows have a terminal/non-central sidechain atom only in
    altloc B/C, so the extracted A conformer has no coordinate for that atom.
    The all-atom ladder still needs a fixed topology matching the refined
    neutral model.  In that narrow case use the occupancy-weighted coordinate
    over the raw deposited conformers and record the exact fallback keys in
    the run metadata; never silently fabricate a zero or drop a model atom.
    """
    state_structure = base.truth_a_structure if state == "A" else base.b_structure
    state_chain = state_structure[base.chain].conformers[0]
    state_by_key = {}
    for segment in state_chain.segments:
        for residue in segment.residues:
            for atom, coordinate in zip(residue.name.tolist(), residue.coor):
                state_by_key[(residue.id, atom)] = np.asarray(coordinate, dtype=float)

    raw_chain = __import__("qfit.structure", fromlist=["Structure"]).Structure.fromfile(
        str(base.truth_path)
    )[base.chain]
    needed_keys = {
        (residue.id, atom)
        for residue in base.window.residues
        for atom in residue.name.tolist()
    }
    weighted = {}
    for conformer in raw_chain.conformers:
        for segment in conformer.segments:
            for residue in segment.residues:
                if not any((residue.id, atom) in needed_keys for atom in residue.name.tolist()):
                    continue
                for atom, coordinate, occupancy in zip(
                    residue.name.tolist(), residue.coor, residue.q
                ):
                    key = (residue.id, atom)
                    if key not in needed_keys:
                        continue
                    weighted.setdefault(key, []).append(
                        (float(occupancy), np.asarray(coordinate, dtype=float))
                    )

    values = []
    fallback = []
    for residue in base.window.residues:
        for atom in residue.name.tolist():
            key = (residue.id, atom)
            if key in state_by_key:
                values.append(state_by_key[key])
                continue
            observations = weighted.get(key, [])
            if not observations:
                raise RuntimeError(f"deposited {state} is missing atom {key}")
            weights = np.asarray([item[0] for item in observations], dtype=float)
            coordinates = np.asarray([item[1] for item in observations], dtype=float)
            if not np.isfinite(weights).all() or weights.sum() <= 0.0:
                weights = np.ones(len(observations), dtype=float)
            values.append(np.average(coordinates, axis=0, weights=weights))
            fallback.append(f"{key[0]}:{key[1]}")
    return np.asarray(values, dtype=float), fallback


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
    deposited_a, deposited_a_fallback = _deposited_window_with_fallback(runner.base, "A")
    deposited_b, deposited_b_fallback = _deposited_window_with_fallback(runner.base, "B")
    deposited_seam_a = seam_vector(runner.initial_backbone, deposited_a[runner.bb_indices])[0]
    deposited_seam_b = seam_vector(runner.initial_backbone, deposited_b[runner.bb_indices])[0]
    deposited_joint_seam_norm = float(np.linalg.norm(np.concatenate((deposited_seam_a, deposited_seam_b))))
    deposited_models = runner.base.model_density_batch(
        np.stack((deposited_a, deposited_b)), slots=np.asarray((0, 1))
    )
    synthetic_clean = runner.base.deposited_occupancies @ deposited_models
    deposited_synthetic_rss = float(np.square(
        synthetic_clean - runner.base.deposited_occupancies @ deposited_models
    ).sum())
    mean_density = float(np.mean(deposited_models))
    discriminating = np.abs(deposited_models[0] - deposited_models[1]) > 0.05 * mean_density
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
    # The rebuilt/refined neutral-start preflight records geometry and density
    # metadata, but intentionally does not carry the old v1 initialisation
    # payload. Recompute the declared rung-1 slot-2 seed from this exact start
    # so the nullspace axis is tied to the production parameterisation actually
    # used by the run.
    nullspace_specs = build_specs(
        root / "nullspace_specs", FLIP_ROOT,
        site=site, mask_scope="window", rama_floor=rama_floor,
        start_pdb=neutral, b_factor_mode="single_conformer",
        density_atom_scope="all", device=device,
    )
    initial_spec = next(
        spec for spec in nullspace_specs if spec["label"] == "D_null_axis2_30deg"
    )
    initial_p2 = np.asarray(initial_spec["p2"], dtype=float)
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
    if not np.isclose(runner.base._renderer_u_base, expected["u_base_A2"], atol=5e-12, rtol=0.0):
        raise RuntimeError(
            f"production u_base mismatch for {key}: {runner.base._renderer_u_base} != "
            f"{expected['u_base_A2']}"
        )
    mask_indices = np.argwhere(runner.base.mask)
    mask_hash = _sha256_array(mask_indices)
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
        "mask_indices_sha256": mask_hash,
        "target_model_same_mask": bool(len(runner.base.target) == actual_mask_voxels and len(runner.base._renderer_grid) == actual_mask_voxels),
        "target_model_grid_shape_zyx": list(runner.base.qfit.xmap.array.shape),
        "target_model_grid_points": int(len(runner.base._renderer_grid)),
        "deposited_synthetic_density_rss": deposited_synthetic_rss,
        "deposited_window_fallback_atoms": {
            "A": deposited_a_fallback,
            "B": deposited_b_fallback,
            "definition": "occupancy-weighted raw deposited conformers used only where the selected A/B altloc lacks an atom required by the fixed seven-residue model topology",
        },
        "deposited_seam_reference": {
            "definition": "joint norm of the two six-component released-seam vectors at deposited A and B",
            "A_vector_A_equivalent": deposited_seam_a.tolist(),
            "B_vector_A_equivalent": deposited_seam_b.tolist(),
            "A_norm_A": float(np.linalg.norm(deposited_seam_a)),
            "B_norm_A": float(np.linalg.norm(deposited_seam_b)),
            "joint_A_B_norm_A": deposited_joint_seam_norm,
        },
        "discriminating_voxels_5pct_mean_AB": int(discriminating.sum()),
        "discriminating_fraction_5pct_mean_AB": float(discriminating.mean()),
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
    a_window = np.asarray(runner.base.window_for_deposited_a(), dtype=float)
    b_window = np.asarray(runner.base.window_for_deposited_b(), dtype=float)
    backbone_indices = window_backbone_indices(runner.base.window)
    rows = []
    for index, slot in enumerate(slots):
        central = runner.base.central_backbone(slot)
        to_a = float(rmsd(central, runner.a_backbone))
        to_b = float(rmsd(central, runner.b_backbone))
        full_backbone = np.asarray(slot, dtype=float)[backbone_indices]
        full_to_a = float(rmsd(full_backbone, a_window[backbone_indices]))
        full_to_b = float(rmsd(full_backbone, b_window[backbone_indices]))
        all_to_a = float(rmsd(slot, a_window))
        all_to_b = float(rmsd(slot, b_window))
        rows.append({
            "slot": index + 1,
            "to_A_A": to_a,
            "to_B_A": to_b,
            "central": {"to_A_A": to_a, "to_B_A": to_b},
            "full_window_backbone": {"to_A_A": full_to_a, "to_B_A": full_to_b},
            "all_atom": {"to_A_A": all_to_a, "to_B_A": all_to_b},
            "to_A_fraction_AB": to_a / runner.ab_distance,
            "to_B_fraction_AB": to_b / runner.ab_distance,
            "nearer": "A" if to_a <= to_b else "B",
        })
    central_pair = float(rmsd(
        runner.base.central_backbone(slots[0]), runner.base.central_backbone(slots[1])
    ))
    backbone_pair = float(rmsd(
        np.asarray(slots[0])[backbone_indices], np.asarray(slots[1])[backbone_indices]
    ))
    all_atom_pair = float(rmsd(slots[0], slots[1]))
    return {
        "slots": rows,
        "slot_to_slot_backbone_rmsd_A": central_pair,
        "slot_to_slot": {
            "central": central_pair,
            "full_window_backbone": backbone_pair,
            "all_atom": all_atom_pair,
        },
        "deposited_A_to_B_backbone_rmsd_A": float(runner.ab_distance),
    }


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
        "seam_term": seam_quadratic_cost(runner, seam),
        "seam_vectors_A_equivalent": seam.tolist(),
        "ramachandran_term": float(runner.rama_weight * np.square(rama_barrier).sum()),
        "ramachandran_barriers": rama_barrier.tolist(),
        "omega_term": float(runner.planar_weight * np.square(omega_delta / runner.omega_scale_deg).sum()),
        "omega_delta_deg": omega_delta.tolist(),
        "rotamer_barrier_term": 0.0,
        "objective_without_AL_multiplier": float(
            rss / max(normalizer, 1e-12)
            + seam_quadratic_cost(runner, seam)
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
                       omega_scale_deg: float = D1_OMEGA_SCALE_DEG,
                       density_normalizer: float = 1.0,
                       geometry_gradient_mode: str = "standard",
                       geometry_gradient_occupancy_floor: float = 0.05,
                       lambda_damping_alpha: float = 0.3,
                       deposited_seam_tolerance_factor: float = 1.5,
                       stationarity_projected_gradient_threshold: float = DEFAULT_STATIONARITY_PROJECTED_GRADIENT_THRESHOLD,
                       outer_updates: int = 200,
                       seam_rho_vector_values: np.ndarray | None = None,
                       free_parameter_mask: np.ndarray | None = None) -> dict[str, object]:
    runner.target = np.asarray(target, dtype=float)
    runner.base.target = runner.target.copy()
    runner.output = output
    output.mkdir(parents=True, exist_ok=True)
    runner.rama_floor = float(rama_floor)
    runner.omega_scale_deg = float(omega_scale_deg)
    if seam_rho_vector_values is not None:
        seam_rho_vector_values = np.asarray(seam_rho_vector_values, dtype=float)
        if seam_rho_vector_values.shape != (6,) or not np.all(np.isfinite(seam_rho_vector_values)) or np.any(seam_rho_vector_values <= 0.0):
            raise ValueError("seam rho vector must contain six finite positive values")
        runner.rho_vector = seam_rho_vector_values.copy()
    if deposited_seam_tolerance_factor <= 0.0:
        raise ValueError("deposited_seam_tolerance_factor must be positive")
    deposited_joint_seam = float(context["deposited_seam_reference"]["joint_A_B_norm_A"])
    seam_tolerance_A = deposited_seam_tolerance_factor * deposited_joint_seam
    atomic_json(output / "run_config.json", {
        "site": context["site"], "rung": rung, "variant": "A_backbone_only",
        "target_artifact": str(output.parent.parent / "targets.npz"),
        "parameters_per_slot": 20, "parameters_total_torsions": 40,
        "rendering_scope": "all atom",
        "dB": {"mode": "fixed", "value_A2": 0.0},
        "per_slot_trust_radii": True,
        "occupancy_updates": {"method": "multiplicative mirror descent", "eta": 0.01, "gradient_normalization": "unit norm"},
        "rama_floor": float(rama_floor), "omega_scale_deg": float(omega_scale_deg), "omega_weight": 0.05,
        "augmented_lagrangian_rho": float(runner.rho),
        "augmented_lagrangian_rho_vector": (
            None if seam_rho_vector_values is None else seam_rho_vector_values.tolist()
        ),
        "lambda_damping_alpha": float(lambda_damping_alpha),
        "deposited_seam_reference_joint_A": deposited_joint_seam,
        "deposited_seam_tolerance_factor": float(deposited_seam_tolerance_factor),
        "seam_tolerance_A": seam_tolerance_A,
        "stationarity_projected_gradient_threshold": float(stationarity_projected_gradient_threshold),
        "outer_updates_max": int(outer_updates),
        "outer_stop": "seam norm <= deposited-derived tolerance AND projected gradient norm <= configured threshold, otherwise max updates",
        "inner_solver": {"method": "two independent scipy least_squares(method=trf)", "max_nfev": inner_nfev, "xtol": 1e-10, "ftol": 1e-10, "gtol": 1e-10},
        "jacobian": {"mode": "forward-mode", "tangent_chunk_size": int(os.environ.get("D1_JACOBIAN_CHUNK_SIZE", "40"))},
        "carry_trust_radii": bool(carry_trust_radii),
        "free_parameter_mask": (
            None if free_parameter_mask is None else free_parameter_mask.tolist()
        ),
        "free_parameter_count": (
            40 if free_parameter_mask is None else int(np.asarray(free_parameter_mask).sum())
        ),
        "density_normalizer": float(density_normalizer),
        "geometry_gradient_mode": geometry_gradient_mode,
        "geometry_gradient_occupancy_floor": float(geometry_gradient_occupancy_floor),
        "initialization": context["initialization"],
    })
    result_path = output / "result.json"
    slots_path = output / "final_slots.npz"
    if result_path.is_file() and slots_path.is_file():
        # A reporting failure must never repeat an expensive completed solve.
        # This permits a safe resume that only reconstructs ladder artifacts.
        result = json.loads(result_path.read_text())
    else:
        p1 = np.zeros(20, dtype=float)
        result = joint_run(
            runner, p1, initial_p2, f"rung_{rung}_A_backbone_only", output,
            float(context["initialization"]["initial_slot_to_slot_backbone_rmsd_A"]),
            fixed_b_offset=0.0, occupancy_scheme="mirror", mirror_eta=0.01,
            inner_nfev=inner_nfev, outer_updates=outer_updates, seam_tolerance_A=seam_tolerance_A,
            stationarity_projected_gradient_threshold=stationarity_projected_gradient_threshold,
            per_slot_trust_radii=True, torch_native_trf=False,
            carry_trust_radii=carry_trust_radii,
            density_normalizer=density_normalizer,
            geometry_gradient_mode=geometry_gradient_mode,
            geometry_gradient_occupancy_floor=geometry_gradient_occupancy_floor,
            lambda_damping_alpha=lambda_damping_alpha,
            free_parameter_mask=free_parameter_mask,
        )
    saved = np.load(output / "final_slots.npz")
    slots = np.stack((saved["slot1_window"], saved["slot2_window"]))
    weights = np.asarray(result["final_occupancies"], dtype=float)
    intercept = float(result["final_intercept"])
    # Use the exact scalar passed into joint_run.  Recomputing a different
    # post-hoc normalizer here makes the reported objective disagree with the
    # objective that was optimized.
    normalizer = float(result["normalizer_initial_A_A_rss"])
    row = {
        **result,
        "rung": rung,
        "variant": "A_backbone_only",
        "recovery": _rmsd_report(runner, slots),
        "occupancies": {
            "final": weights.tolist(),
            "total": float(weights.sum()),
            "unexplained": float(1.0 - weights.sum()),
            "total_L1_path": float(sum(
                np.abs(current - previous).sum()
                for previous, current in _occupancy_path(result.get("trajectory", []))
            )),
            "trajectory": [values.tolist() for values in _occupancy_values(result.get("trajectory", []))],
        },
        "objective_terms": _objective_terms(runner, runner.target, slots, weights, intercept, normalizer),
        "geometry": _geometry_report(runner, slots),
        "convergence": {
            "outer_updates": result.get("outer_updates_completed"),
            "stopping_rule": result.get("stopping_rule"),
            "seam_tolerance_A": result.get("seam_tolerance_A"),
            "stationarity_projected_gradient_threshold": result.get("stationarity_projected_gradient_threshold"),
            "deposited_seam_reference_joint_A": deposited_joint_seam,
            "deposited_seam_tolerance_factor": float(deposited_seam_tolerance_factor),
            "seam_tolerance_reached": bool(result.get("seam_tolerance_reached")),
            "final_seam_norm_A_equivalent": float(np.linalg.norm(result.get("final_seam_vectors", []))),
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
    return [np.asarray((1.0 / 3.0, 1.0 / 3.0))] + values


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
    free_parameter_mask = None
    if args.slot2_fixed_deposited_b:
        initial_p2 = inverse_seed(runner, runner.base.window_for_deposited_b())
        free_parameter_mask = np.zeros(40, dtype=bool)
        free_parameter_mask[:20] = True
    atomic_json(root / "progress.json", {"status": "rung_1", "site": site_name(args.site)})
    rows = {}
    for variant in ("A",):
        variant_root = root / f"variant_{variant}"
        row = _run_backbone_rung(
            runner, targets["1"], variant_root / "rung_1", initial_p2, context, "1",
            args.inner_nfev, args.carry_trust_radii,
            args.rama_floor, args.omega_scale_deg, args.density_normalizer,
            args.geometry_gradient_mode, args.geometry_gradient_occupancy_floor,
            args.lambda_damping_alpha,
            args.deposited_seam_tolerance_factor,
            args.stationarity_projected_gradient_threshold,
            args.outer_updates,
            args.seam_rho_vector,
            free_parameter_mask=free_parameter_mask,
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
            args.rama_floor, args.omega_scale_deg, args.density_normalizer,
            args.geometry_gradient_mode, args.geometry_gradient_occupancy_floor,
            free_parameter_mask=free_parameter_mask,
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
    parser.add_argument("--density-normalizer", type=float, default=1.0,
                        help="Positive scalar dividing raw density RSS; 1.0 uses raw RSS.")
    parser.add_argument("--geometry-gradient-mode", choices=("standard", "per_slot_occupancy_decoupled"),
                        default="standard",
                        help="Leave the objective unchanged or precondition density geometry columns per slot.")
    parser.add_argument("--geometry-gradient-occupancy-floor", type=float, default=0.05,
                        help="Minimum occupancy used by per-slot density-Jacobian decoupling.")
    parser.add_argument("--lambda-damping-alpha", type=float, default=0.3,
                        help="Multiplier update damping in (0, 1]; 0.3 avoids augmented-Lagrangian cycling.")
    parser.add_argument("--deposited-seam-tolerance-factor", type=float, default=1.5,
                        help="Converge when the joint seam is this multiple of deposited A/B's joint seam norm.")
    parser.add_argument("--stationarity-projected-gradient-threshold", type=float,
                        default=DEFAULT_STATIONARITY_PROJECTED_GRADIENT_THRESHOLD,
                        help="Second stopping condition: projected gradient norm threshold after the inner solve.")
    parser.add_argument("--outer-updates", type=int, default=200,
                        help="Maximum number of augmented-Lagrangian outer updates.")
    parser.add_argument("--seam-rho-vector", type=str, default=None,
                        help="Six comma-separated componentwise seam penalties; default is legacy scalar rho.")
    parser.add_argument("--slot2-fixed-deposited-b", action="store_true",
                        help="Freeze slot 2 at deposited-B torsions and solve only the free-index mask.")
    parser.add_argument("--carry-trust-radii", action="store_true",
                        help="carry each slot's ending TRF radius into the next outer update.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.site = (args.pdb_id, args.chain, args.resnum)
    if args.seam_rho_vector is not None:
        try:
            args.seam_rho_vector = np.asarray(
                [float(value.strip()) for value in args.seam_rho_vector.split(",")],
                dtype=float,
            )
        except ValueError as exc:
            raise ValueError("--seam-rho-vector must be six comma-separated numbers") from exc
        if args.seam_rho_vector.shape != (6,) or not np.all(np.isfinite(args.seam_rho_vector)) or np.any(args.seam_rho_vector <= 0.0):
            raise ValueError("--seam-rho-vector must contain six finite positive values")
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
        "density_normalizer": float(args.density_normalizer),
        "geometry_gradient_mode": args.geometry_gradient_mode,
        "geometry_gradient_occupancy_floor": float(args.geometry_gradient_occupancy_floor),
        "lambda_damping_alpha": float(args.lambda_damping_alpha),
        "deposited_seam_tolerance_factor": float(args.deposited_seam_tolerance_factor),
        "stationarity_projected_gradient_threshold": float(args.stationarity_projected_gradient_threshold),
        "outer_updates": int(args.outer_updates),
        "seam_rho_vector": (
            None if args.seam_rho_vector is None else args.seam_rho_vector.tolist()
        ),
        "slot2_fixed_deposited_b": bool(args.slot2_fixed_deposited_b),
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
