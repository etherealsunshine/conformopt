#!/usr/bin/env python3
"""Checkpointed sequential A-prime runner for qFit model panels.

This runner deliberately keeps the supplied qFit A/B coordinates as stage zero,
uses the existing A-prime backbone objective for backbone blocks, and adds a
Torch-differentiable all-window chi block.  It is an experiment harness: all
stage inputs, objective components, and intermediate coordinates are written
before advancing to the next block.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qfit  # noqa: F401  # CCTBX/qFit must load before NumPy/Torch.
import numpy as np
import torch
from scipy.optimize import least_squares

from experiments.probe4.core import dihedral, torsion_to_coords
from density_denoiser.residue_geometry import (
    CHI_SPECS,
    canonical_centers_radians,
    canonical_width_degrees,
)
from run_d1_conformopt_sequential import APrimeSequential, atomic_json, atomic_npz, seam_vector
from run_d1_slot_coordination import inverse_seed, joint_run, seam_rho_vector
from result_provenance import runner_provenance, sha256_file
from check_runtime import required_runtime_ok, runtime_report


BACKBONE = {"N", "CA", "C", "O"}
ATOM_PEAK_RANDOM_SEED = 20260828
ATOM_PEAK_RANDOM_VOXELS = 100
ATOM_PEAK_MIN_ENRICHMENT = 0.0


def should_subtract_window_sidechains(density_atom_scope: str) -> bool:
    """Only subtract sidechains when the renderer cannot represent them."""
    if density_atom_scope not in {"backbone", "all"}:
        raise ValueError(f"unknown density atom scope: {density_atom_scope}")
    return density_atom_scope == "backbone"


def chi_solver_tolerances() -> dict[str, float | None]:
    """Require χ termination on the gradient criterion or evaluation cap."""
    return {"ftol": None, "xtol": None, "gtol": 1.0e-6}


def chi_solver_bounds() -> tuple[float, float]:
    """Bound torsion deltas to one complete, lossless periodic interval."""
    return (-180.0, 180.0)


def chi_parameters_to_radians(parameters_degrees: torch.Tensor) -> torch.Tensor:
    """Convert the degree-valued SciPy variables for Torch torsion kinematics."""
    return torch.deg2rad(parameters_degrees)


def normalized_zscore_density_residual(
    model_density: torch.Tensor,
    target_z: torch.Tensor,
) -> torch.Tensor:
    """Return the unit-size z-score residual used by the backbone objective."""
    if model_density.shape != target_z.shape:
        raise ValueError("model_density and target_z must have matching shapes")
    model_mean = torch.mean(model_density)
    model_std = torch.sqrt(torch.mean(torch.square(model_density - model_mean)))
    if not bool(torch.isfinite(model_std).detach().cpu()) or float(model_std.detach().cpu()) <= 0.0:
        raise FloatingPointError("non-positive model-density standard deviation")
    return ((model_density - model_mean) / model_std - target_z) / torch.sqrt(
        torch.as_tensor(max(model_density.numel(), 1), dtype=model_density.dtype,
                        device=model_density.device)
    )


def validate_finite_chi_trial(value: np.ndarray, phase: str) -> None:
    """Reject a non-finite SciPy trial before it reaches Torch or JSON."""
    trial = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(trial)):
        raise FloatingPointError(f"non-finite chi {phase} trial")


def is_chi_solver_numerical_failure(exc: BaseException) -> bool:
    """Identify failures for which the rank-robust LSMR retry is valid."""
    if isinstance(exc, (FloatingPointError, np.linalg.LinAlgError)):
        return True
    if not isinstance(exc, ValueError):
        return False
    message = str(exc).lower()
    return any(token in message for token in ("nan", "inf", "non-finite", "nonfinite"))


def chi_exact_result_needs_lsmr(result, max_nfev: int) -> bool:
    """Detect exact-TRF numerical stagnation even when SciPy returns normally.

    The 3FTD chi2 failure consumed the full residual cap while building only
    two Jacobians, emitted trust-region overflow warnings, and returned an
    enormous projected gradient.  That is not a usable capped solve; retry it
    with the rank-robust iterative linear solver.
    """
    optimality = getattr(result, "optimality", None)
    nonstationary = (
        optimality is None
        or not np.isfinite(optimality)
        or float(optimality) > 1.0e-6
    )
    return bool(
        int(getattr(result, "status", -1)) == 0
        and int(getattr(result, "nfev", 0)) >= int(max_nfev)
        and int(getattr(result, "njev", 0) or 0) <= 2
        and nonstationary
    )


def least_squares_stage_diagnostics(result, max_nfev: int) -> dict[str, object]:
    """Serialize solver diagnostics that are otherwise lost at stage end."""
    gradient = getattr(result, "grad", None)
    gradient_norm = None
    if gradient is not None:
        gradient_norm = float(np.linalg.norm(np.asarray(gradient, dtype=float)))
    optimality = getattr(result, "optimality", None)
    projected_gradient = float(optimality) if optimality is not None else None
    return {
        "termination_status": int(result.status),
        "termination_message": str(result.message),
        "nfev": int(result.nfev),
        "njev": int(result.njev or 0),
        # SciPy reports the bound-aware first-order optimality used by TRF.
        "projected_gradient_norm_end": projected_gradient,
        "scipy_optimality": (
            float(optimality) if optimality is not None else None
        ),
        "gradient_norm_end": gradient_norm,
        "evaluation_cap": int(max_nfev),
        "hit_evaluation_cap": bool(
            int(result.status) == 0 and int(result.nfev) >= int(max_nfev)
        ),
        "converged_on_gradient": bool(
            projected_gradient is not None and projected_gradient <= 1.0e-6
        ),
    }


class PairInitialAPrime(APrimeSequential):
    """Use separate A/B full-window templates while retaining A-prime logic."""

    def set_pair_initial(self, pair: np.ndarray) -> None:
        pair = np.asarray(pair, dtype=float)
        if pair.shape != (2, len(self.initial), 3):
            raise ValueError(f"pair initial must have shape (2,{len(self.initial)},3)")
        # The published qFit pair, rather than deposited A/B, is the stage-zero
        # geometry.  Keep all seam/Jacobian references on slot 1 of that pair.
        self.initial = pair[0].copy()
        self.initial_backbone = self.initial[self.bb_indices].copy()
        self._torch_initial = torch.as_tensor(
            self.initial, dtype=torch.float64, device=self.base.torch_device
        )
        self._pair_initial = torch.as_tensor(
            pair, dtype=torch.float64, device=self.base.torch_device
        )

    def torch_forward(self, parameters):  # noqa: D401
        values = torch.as_tensor(
            parameters, dtype=torch.float64, device=self.base.torch_device
        )
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape[-1] != self.rotator.ndofs:
            raise ValueError(f"expected {self.rotator.ndofs} torsions, got {values.shape[-1]}")
        initial = getattr(self, "_pair_initial", None)
        if initial is None or values.shape[0] != 2:
            initial = self._torch_initial.unsqueeze(0).expand(values.shape[0], -1, -1)
        else:
            initial = initial.to(values.device)
        coordinates = initial.clone()
        for parameter, origin, forward, backward, atoms in zip(
            torch.flip(values[:, :self.rotator.phi_psi_ndofs], dims=(-1,)).unbind(-1),
            self._torch_phi_origins, self._torch_phi_forward,
            self._torch_phi_backward, self._torch_phi_atoms,
        ):
            rotation = forward.to(values.device) @ self._torch_rz(
                torch.deg2rad(parameter)
            ) @ backward.to(values.device)
            atoms = atoms.to(values.device)
            moved = coordinates[:, atoms, :] - origin.to(values.device)
            coordinates = coordinates.index_copy(
                1, atoms, moved @ rotation.transpose(-1, -2) + origin.to(values.device)
            )
        for i in reversed(range(self.rotator.omega_ndofs)):
            origin = coordinates[:, self._torch_c_atom_indices[i], :]
            axis = coordinates[:, self._torch_n_atom_indices[i + 1], :] - origin
            rotation = self._torch_rodrigues(
                axis, torch.deg2rad(values[:, self.rotator.phi_psi_ndofs + i])
            )
            atoms = torch.arange(
                self._torch_n_atom_indices[i + 1], coordinates.shape[1],
                dtype=torch.long, device=values.device,
            )
            moved = coordinates[:, atoms, :] - origin.unsqueeze(1)
            coordinates = coordinates.index_copy(
                1, atoms, moved @ rotation.transpose(-1, -2) + origin.unsqueeze(1)
            )
        return coordinates


def install_full_observed_lattice_target(
    runner: PairInitialAPrime,
    source: Path,
    mtz_path: Path,
    mask_radius_A: float,
) -> dict[str, object]:
    """Rebuild the A-prime target and renderer on the full observed grid.

    The historical runner receives qFit's ``make_map`` grid.  Its shape can
    differ from the full-map grid used by the independent scoring audit.  A
    mask radius alone does not fix that: the optimizer would still be fitting
    a different lattice.  This routine constructs the same sigma-scaled
    2mFo-DFc map used by the full-map audit, applies the established MapScaler,
    redoes A-prime's explicit environment subtraction, and installs a
    differentiable renderer whose voxels are exactly that full grid.
    """
    if mask_radius_A <= 0.0:
        raise ValueError("mask_radius_A must be positive")

    from cctbx import maptbx
    from cctbx.array_family import flex
    from iotbx import pdb
    from mmtbx import real_space_correlation, utils
    from cctbx.xray import ext
    from qfit.xtal.volume import SpaceGroup, UnitCell, XMap
    from qfit.xtal.scaler import MapScaler

    ref_input = pdb.input(file_name=str(source))
    ref_xrs = ref_input.xray_structure_simple()
    params = real_space_correlation.master_params().extract()
    params.reflection_file_name = str(mtz_path)
    data = real_space_correlation.extract_data_and_flags(params=params)
    symmetry = data.f_obs.crystal_symmetry()
    ref_xrs = ref_xrs.customized_copy(crystal_symmetry=symmetry)
    false_flags = data.f_obs.array(data=flex.bool(data.f_obs.size(), False))

    def fmodel(xray_structure):
        utils.setup_scattering_dictionaries(
            scattering_table="wk1995", xray_structure=xray_structure, d_min=None,
        )
        return utils.fmodel_simple(
            xray_structures=[xray_structure], scattering_table="wk1995",
            f_obs=data.f_obs, r_free_flags=false_flags,
        )

    observed_fft = fmodel(ref_xrs).electron_density_map().map_coefficients(
        map_type="2mFo-DFc", fill_missing=True, isotropize=True,
    ).fft_map(resolution_factor=0.25)
    observed_fft.apply_sigma_scaling()
    observed_map = observed_fft.real_map_unpadded()
    map_shape_xyz = tuple(int(value) for value in observed_map.focus())
    unit_cell = UnitCell.from_cctbx(symmetry.unit_cell())
    unit_cell.space_group = SpaceGroup.from_cctbx(symmetry.space_group_info())
    full_xmap = XMap.from_cctbx_map(
        observed_map, map_shape_xyz, unit_cell,
        float(data.f_obs.d_min()), origin=np.zeros(3, dtype=float),
    )

    resolution = float(data.f_obs.d_min())
    scale, offset = MapScaler(full_xmap).scale(
        runner.base.full_structure,
        radius=0.5 + resolution / 3.0,
        transformer="cctbx",
    )

    base = runner.base
    qfit = base.qfit
    qfit.xmap = full_xmap
    qfit._xmap_model = full_xmap.zeros_like(full_xmap)  # pylint: disable=protected-access
    qfit._xmap_model2 = full_xmap.zeros_like(full_xmap)  # pylint: disable=protected-access
    qfit._xmap_model.set_space_group("P1")  # pylint: disable=protected-access
    qfit._xmap_model2.set_space_group("P1")  # pylint: disable=protected-access
    qfit._voxel_volume = full_xmap.unit_cell.calc_volume() / full_xmap.array.size  # pylint: disable=protected-access
    qfit._transformer = qfit._get_transformer(  # pylint: disable=protected-access
        qfit.conformer, qfit._xmap_model,  # pylint: disable=protected-access
        smax=qfit._smax, smin=qfit._smin,  # pylint: disable=protected-access
        simple=qfit._simple, em=qfit.options.em,  # pylint: disable=protected-access
    )
    qfit._transformer.initialize()  # pylint: disable=protected-access

    # Reapply the exact A-prime target construction on the new lattice.  The
    # constructor's native-grid subtraction is discarded with its old xmap.
    # Sidechains must remain in an all-atom target: subtracting them while the
    # renderer also models them hollows the map at CB/chi atoms.  That bug was
    # caught by the 6KVE atom-on-peak preflight.
    base._subtract_window_neighbors(base.full_structure)
    if should_subtract_window_sidechains(base.density_atom_scope):
        base._subtract_window_sidechains(base.truth_a_structure)

    active_indices = np.asarray(base.model_atom_indices, dtype=np.int64)
    mask_centers = [
        np.asarray(base.initial_window, dtype=float)[active_indices],
        np.asarray(base.window_for_deposited_b(), dtype=float)[active_indices],
    ]
    sites_cart = flex.vec3_double(np.concatenate(mask_centers, axis=0).tolist())
    mask_indices = maptbx.grid_indices_around_sites(
        unit_cell=unit_cell.to_cctbx(),
        fft_n_real=full_xmap.n_real(), fft_m_real=full_xmap.n_real(),
        sites_cart=sites_cart,
        site_radii=flex.double(len(sites_cart), float(mask_radius_A)),
    )
    # CCTBX selections are in full-map (x, y, z) flattened order.  qFit's
    # NumPy/Torch arrays are (z, y, x), so materialize the boolean mask by
    # converting the exact selected grid coordinates rather than interpolating.
    mask_xyz = np.zeros(tuple(int(value) for value in full_xmap.n_real()), dtype=bool)
    for flattened in mask_indices:
        xyz = np.unravel_index(int(flattened), tuple(int(value) for value in full_xmap.n_real()))
        mask_xyz[xyz] = True
    base.mask = mask_xyz.transpose(2, 1, 0)

    mask_indices_zyx = np.argwhere(base.mask)
    n_real = np.asarray(full_xmap.n_real(), dtype=float)
    fractional = mask_indices_zyx[:, [2, 1, 0]] / n_real[None, :]
    orthogonalization = np.asarray(
        unit_cell.calc_orthogonalization_matrix(), dtype=float,
    ).reshape(3, 3)
    grid_cart = fractional @ orthogonalization.T
    base._renderer_grid = torch.as_tensor(  # pylint: disable=protected-access
        grid_cart, dtype=torch.float64, device=base.torch_device,
    )
    base._renderer_cell = torch.as_tensor(  # pylint: disable=protected-access
        orthogonalization, dtype=torch.float64, device=base.torch_device,
    )
    base._set_renderer_reference_wrap_offsets(  # pylint: disable=protected-access
        np.asarray(base.initial_window, dtype=float)[active_indices]
    )
    base._renderer_u_base = float(  # pylint: disable=protected-access
        ext.calc_u_base(d_min=base.resolution, grid_resolution_factor=0.25),
    )
    base._renderer_b_factors = torch.as_tensor(  # pylint: disable=protected-access
        base.b_factors, dtype=torch.float64, device=base.torch_device,
    )
    from density_denoiser.differentiable_renderer import coefficients_for_elements
    base._renderer_coefficients = coefficients_for_elements(  # pylint: disable=protected-access
        [str(base.window.e[index]).strip() for index in active_indices],
        dtype=torch.float64, device=base.torch_device,
    )
    base._renderer_atom_indices = active_indices.copy()  # pylint: disable=protected-access
    base.target = full_xmap.array[base.mask].astype(float, copy=True)
    base.target_before_residual_scaling = base.target.copy()
    base.residual_target_multiplier = 1.0
    base.mask_cache_report = {
        "cache_path": None,
        "used_cached_mask": False,
        "verified_against_recomputed": True,
        "grid": "full_observed_map",
        "radius_A": float(mask_radius_A),
        "voxel_count": int(base.mask.sum()),
    }
    runner.target = base.target.copy()
    return {
        "map_source": "deposited-source 2mFo-DFc from panel MTZ; full-map CCTBX FFT grid",
        "map_grid_shape_xyz": list(map_shape_xyz),
        "map_resolution_A": resolution,
        "map_scale": float(scale),
        "map_offset": float(offset),
        "window_sidechain_subtraction": (
            "deposited_A_sidechains_subtracted"
            if should_subtract_window_sidechains(base.density_atom_scope)
            else "retained_for_all_atom_renderer"
        ),
        "mask_radius_A": float(mask_radius_A),
        "mask_voxels": int(base.mask.sum()),
        "renderer_grid": "exact full observed-map lattice; no qFit extracted-grid interpolation",
        "unit_cell_parameters_A_deg": list(unit_cell.to_cctbx().parameters()),
    }


def site_label(record: dict[str, str]) -> str:
    return f"{record['pdb_id']}_{record['chain']}_{record['resname']}{record['residue_number']}"


def resolve_source_pdb(panel: Path, record: dict[str, str]) -> Path:
    """Resolve the deposited source structure from the supplied panel."""
    pdb_id = str(record["pdb_id"])
    candidates = [
        panel / "inputs/source" / f"{pdb_id.lower()}.pdb",
        panel / "inputs/source" / f"{pdb_id.upper()}.pdb",
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"No source PDB for {pdb_id} in panel inputs/source")
    return source


def native_deposited_map_configuration(runner: PairInitialAPrime) -> dict[str, object]:
    """Describe the deposited-coefficient qFit map without rebuilding it."""
    base = runner.base
    return {
        "map_protocol": "native_deposited_coefficients",
        "map_source": str(base.map_source),
        "map_grid_shape_xyz": [int(value) for value in base.qfit.xmap.n_real()],
        "map_resolution_A": float(base.resolution),
        "map_scale": float(base.map_scale),
        "map_offset": float(base.map_offset),
        "mask_radius_A": float(base.qfit._rmask),  # pylint: disable=protected-access
        "mask_voxels": int(np.count_nonzero(base.mask)),
        "renderer_grid": "native qFit deposited-coefficient lattice",
        "window_sidechain_subtraction": (
            "deposited_A_sidechains_subtracted"
            if should_subtract_window_sidechains(base.density_atom_scope)
            else "retained_for_all_atom_renderer"
        ),
    }


def atom_on_peak_preflight(
    runner: PairInitialAPrime,
    source_pdb: Path,
    map_path: Path,
    map_configuration: dict[str, object],
    output: Path,
    random_seed: int = ATOM_PEAK_RANDOM_SEED,
    random_voxels: int = ATOM_PEAK_RANDOM_VOXELS,
) -> dict[str, object]:
    """Gate optimization on deposited atoms being denser than mask baseline."""
    base = runner.base
    active_indices = np.asarray(base.model_atom_indices, dtype=np.int64)
    atom_metadata = []
    for residue in runner.window.residues:
        for name, element, global_index in zip(residue.name, residue.e, residue.selection):
            local_index = int(np.searchsorted(runner.window.selection, int(global_index)))
            atom_metadata.append((
                local_index,
                str(residue.resn[0]).strip(),
                int(residue.id[0]),
                str(residue.id[1]).strip(),
                str(name).strip(),
                str(element).strip(),
            ))
    metadata_by_index = {row[0]: row[1:] for row in atom_metadata}

    deposited_pairs = np.stack((
        np.asarray(base.initial_window, dtype=float),
        np.asarray(base.window_for_deposited_b(), dtype=float),
    ))
    selected_atoms = []
    seen_coordinates = set()
    for slot, altloc in enumerate(("A", "B")):
        for local_index in active_indices:
            metadata = metadata_by_index[int(local_index)]
            if metadata[-1].upper() in {"H", "D"}:
                continue
            coordinate = deposited_pairs[slot, local_index]
            coordinate_key = tuple(np.round(coordinate, 3).tolist())
            if coordinate_key in seen_coordinates:
                continue
            seen_coordinates.add(coordinate_key)
            selected_atoms.append({
                "slot": altloc,
                "resname": metadata[0],
                "residue_number": metadata[1],
                "insertion_code": metadata[2],
                "atom_name": metadata[3],
                "element": metadata[4],
                "cartesian_A": np.asarray(coordinate, dtype=float),
            })
    if not selected_atoms:
        raise RuntimeError("atom-on-peak preflight has no deposited heavy atoms")

    xmap = base.qfit.xmap
    shape_xyz = np.asarray(xmap.n_real(), dtype=np.int64)
    xrs_box, dxyz = base.qfit._transformer._get_xray_structure_in_box()  # pylint: disable=protected-access
    shifts = np.asarray(dxyz, dtype=float)
    shift = np.asarray(shifts[0], dtype=float) if shifts.ndim == 2 else np.asarray(shifts)
    if shift.shape == ():
        shift = np.zeros(3, dtype=float)
    unit_cell = xrs_box.unit_cell()

    def nearest_index(cartesian: np.ndarray) -> np.ndarray:
        shifted = np.asarray(cartesian, dtype=float) + shift
        fractional = np.asarray(unit_cell.fractionalize(tuple(shifted)), dtype=float)
        return np.rint(fractional * shape_xyz).astype(np.int64) % shape_xyz

    def density_values(indices_xyz: np.ndarray) -> np.ndarray:
        indices_xyz = np.asarray(indices_xyz, dtype=np.int64)
        return np.asarray(xmap.array, dtype=float)[tuple(indices_xyz[:, [2, 1, 0]].T)]

    atom_indices = np.asarray([nearest_index(atom["cartesian_A"]) for atom in selected_atoms])
    atom_values = density_values(atom_indices)
    mask_indices_xyz = np.argwhere(np.asarray(base.mask, dtype=bool))[:, [2, 1, 0]]
    mask_values = density_values(mask_indices_xyz)
    mask_mean = float(mask_values.mean())
    mask_std = float(mask_values.std(ddof=0))
    if not np.isfinite(mask_std) or mask_std <= 0.0:
        raise RuntimeError(f"atom-on-peak mask has invalid density spread {mask_std}")

    atom_flat = set(np.ravel_multi_index(atom_indices.T, tuple(shape_xyz)))
    mask_flat = np.ravel_multi_index(mask_indices_xyz.T, tuple(shape_xyz))
    candidates = mask_indices_xyz[[int(value) not in atom_flat for value in mask_flat]]
    if len(candidates) < random_voxels:
        raise RuntimeError(
            f"atom-on-peak preflight has {len(candidates)} baseline candidates, "
            f"fewer than requested {random_voxels}"
        )
    selected = np.random.default_rng(int(random_seed)).choice(
        len(candidates), size=int(random_voxels), replace=False,
    )
    baseline_indices = candidates[np.sort(selected)]
    baseline_values = density_values(baseline_indices)
    enrichment = float(atom_values.mean() - baseline_values.mean())
    enrichment_z = float(enrichment / mask_std)

    assignments = []
    for atom, index, value in zip(selected_atoms, atom_indices, atom_values):
        assignments.append({
            **{key: atom[key] for key in (
                "slot", "resname", "residue_number", "insertion_code", "atom_name", "element"
            )},
            "cartesian_A": atom["cartesian_A"].tolist(),
            "voxel_index_xyz": index.tolist(),
            "observed_density": float(value),
            "mask_standardized_density": float((value - mask_mean) / mask_std),
        })
    passed = bool(np.isfinite(enrichment) and enrichment > ATOM_PEAK_MIN_ENRICHMENT)
    report = {
        "status": "passed" if passed else "failed",
        "mandatory_gate": True,
        "gate": "mean deposited-heavy-atom density must exceed deterministic non-atom mask baseline",
        "minimum_raw_enrichment_exclusive": ATOM_PEAK_MIN_ENRICHMENT,
        "map_protocol": map_configuration.get("map_protocol"),
        "map_source": map_configuration.get("map_source"),
        "map_grid_shape_xyz": shape_xyz.tolist(),
        "mask_voxels": int(len(mask_indices_xyz)),
        "mask_density_mean": mask_mean,
        "mask_density_std": mask_std,
        "atom_centres": {
            "count": int(len(atom_values)),
            "mean_observed_density": float(atom_values.mean()),
            "mean_mask_standardized_density": float(((atom_values - mask_mean) / mask_std).mean()),
        },
        "random_non_atom_mask_voxels": {
            "count": int(len(baseline_values)),
            "seed": int(random_seed),
            "mean_observed_density": float(baseline_values.mean()),
            "mean_mask_standardized_density": float(((baseline_values - mask_mean) / mask_std).mean()),
            "definition": "mask voxels excluding nearest-grid assignments of deposited heavy atoms",
        },
        "atom_minus_random_enrichment": {
            "raw_density": enrichment,
            "mask_standardized_density": enrichment_z,
        },
        "atom_assignments": assignments,
        "provenance": {
            "source_pdb": {"path": str(source_pdb), "sha256": sha256_file(source_pdb)},
            "map_mtz": {"path": str(map_path), "sha256": sha256_file(map_path)},
            "full_mask_voxel_count": int(len(mask_indices_xyz)),
            "target_mask_voxel_count": int(len(base.target)),
            "endpoint_npz": {},
            "folds": {"not_applicable": True, "reason": "pre-optimization map gate"},
        },
    }
    atomic_json(output, report)
    return report


def read_published_qfit_pair(path: Path, runner: PairInitialAPrime) -> tuple[np.ndarray, np.ndarray]:
    """Read the published qFit A/B pair in the runner's window atom order."""
    by_key: dict[tuple[str, int, str, str, str], tuple[np.ndarray, float]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 66:
            continue
        try:
            key = (line[21].strip(), int(line[22:26]), line[26].strip(),
                   line[12:16].strip(), line[16].strip())
            xyz = np.asarray([float(line[30:38]), float(line[38:46]),
                              float(line[46:54])], dtype=float)
            b_factor = float(line[60:66])
        except (IndexError, ValueError):
            continue
        by_key.setdefault(key, (xyz, b_factor))

    names = []
    residue_keys = []
    for residue in runner.window.residues:
        for name in residue.name:
            names.append(str(name).strip())
            residue_keys.append((runner.base.chain, int(residue.id[0]),
                                 str(residue.id[1]).strip()))
    slots, b_factors = [], []
    missing = []
    for altloc in ("A", "B"):
        coords, b_values = [], []
        for name, residue_key in zip(names, residue_keys):
            value = by_key.get((*residue_key, name, altloc))
            if value is None:
                value = by_key.get((*residue_key, name, ""))
            if value is None:
                missing.append((*residue_key, name, altloc))
                continue
            coords.append(value[0])
            b_values.append(value[1])
        if len(coords) != len(names):
            continue
        slots.append(np.asarray(coords, dtype=float))
        b_factors.append(np.asarray(b_values, dtype=float))
    if missing or len(slots) != 2:
        raise RuntimeError(
            f"published qFit pair is incomplete for {path}: "
            f"missing={len(missing)} slots={len(slots)}"
        )
    return np.stack(slots), np.stack(b_factors)


def chi_blocks(runner: PairInitialAPrime, pair: np.ndarray):
    """Construct all sidechain chi blocks for both slots in current coordinates."""
    blocks = []
    for slot in range(2):
        current = np.asarray(pair[slot], dtype=float)
        for residue in runner.window.residues:
            resname = str(residue.resn[0])
            spec = CHI_SPECS.get(resname)
            if spec is None:
                continue
            names = [str(name) for name in residue.name
                     if str(name) not in BACKBONE and not str(name).startswith(("H", "D"))]
            if not names or any(name not in names for rot in spec["rotations"] for name in rot[2]):
                raise RuntimeError(f"incomplete chi atom set for {resname} {residue.id}")
            indices = np.asarray([
                int(np.searchsorted(runner.window.selection, int(index)))
                for index in residue.selection
                if str(residue.name[list(residue.selection).index(index)]) in names
            ], dtype=int)
            # The selection-order expression above is intentionally replaced
            # by a name-indexed map for qFit structures with omitted atoms.
            name_to_index = {str(name): int(np.searchsorted(
                runner.window.selection, int(global_index)
            )) for name, global_index in zip(residue.name, residue.selection)}
            indices = np.asarray([name_to_index[name] for name in names], dtype=int)
            template = torch.as_tensor(current[indices], dtype=torch.float64,
                                       device=runner.base.torch_device)
            fixed = {
                name: torch.as_tensor(current[index], dtype=torch.float64,
                                      device=runner.base.torch_device)
                for name, index in name_to_index.items() if name not in names
            }
            blocks.append({"slot": slot, "resname": resname, "names": names,
                           "indices": indices, "template": template, "fixed": fixed,
                           "rotations": list(spec["rotations"]),
                           "n_chi": len(spec["rotations"])})
    return blocks


def chi_parameter_scaling(
    blocks: list[dict[str, object]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Return per-angle scales based on downstream heavy-atom counts."""
    scales: list[float] = []
    details: list[dict[str, object]] = []
    parameter = 0
    for block in blocks:
        rotations = list(block["rotations"])
        if len(rotations) != int(block["n_chi"]):
            raise ValueError(
                f"chi topology mismatch for {block['resname']}: "
                f"{len(rotations)} rotations != {block['n_chi']} chis"
            )
        for chi_index, rotation in enumerate(rotations, start=1):
            downstream = tuple(str(name) for name in rotation[2])
            count = len(downstream)
            if count < 1:
                raise ValueError(
                    f"chi{chi_index} for {block['resname']} has no downstream heavy atoms"
                )
            scale = 1.0 / float(count)
            scales.append(scale)
            details.append({
                "parameter_index": parameter,
                "slot": int(block["slot"]) + 1,
                "resname": str(block["resname"]),
                "chi_index": chi_index,
                "axis_atoms": [str(rotation[0]), str(rotation[1])],
                "downstream_heavy_atoms": list(downstream),
                "downstream_heavy_atom_count": count,
                "x_scale": scale,
                "x_scale_units": "degrees",
            })
            parameter += 1
    return np.asarray(scales, dtype=float), details


def rotamer_prior_residual_rows(
    coordinates: torch.Tensor,
    blocks: list[dict[str, object]],
    weight: float,
    active_slots: set[int] | None = None,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    """Return exact least-squares rows for the production cosine rotamer prior.

    Squaring a row gives

      weight * (30 / allowed_width_deg)^2 * (1 - cos(delta_to_center))

    which is the sidechain optimizer's existing prior.  Each chi in every
    active slot contributes once; occupancies do not enter the residual.
    """
    if weight < 0.0:
        raise ValueError("rotamer prior weight must be nonnegative")
    rows: list[torch.Tensor] = []
    details: list[dict[str, object]] = []
    if weight == 0.0:
        return torch.empty(0, dtype=coordinates.dtype, device=coordinates.device), details
    prefactor = torch.sqrt(torch.as_tensor(
        2.0 * weight, dtype=coordinates.dtype, device=coordinates.device,
    ))
    for block in blocks:
        slot = int(block["slot"])
        if active_slots is not None and slot not in active_slots:
            continue
        indices = torch.as_tensor(
            block["indices"], dtype=torch.long, device=coordinates.device,
        )
        residue_xyz = coordinates[slot].index_select(0, indices)
        atom_xyz = {
            str(name): residue_xyz[index]
            for index, name in enumerate(block["names"])
        }
        atom_xyz.update({
            str(name): torch.as_tensor(
                value, dtype=coordinates.dtype, device=coordinates.device,
            )
            for name, value in dict(block.get("fixed", {})).items()
        })
        resname = str(block["resname"])
        for chi_index, quartet in enumerate(CHI_SPECS[resname]["dihedrals"]):
            missing = [name for name in quartet if name not in atom_xyz]
            if missing:
                raise KeyError(
                    f"missing rotamer-prior atoms for {resname} chi{chi_index + 1}: {missing}"
                )
            # probe4.core.dihedral uses the internal Rodrigues convention.  The
            # production rotamer authority converts it to the physical PDB chi
            # convention by subtracting pi and wrapping.
            internal = dihedral(*(atom_xyz[name] for name in quartet))
            value = torch.atan2(
                torch.sin(internal - torch.pi),
                torch.cos(internal - torch.pi),
            )
            centers = torch.as_tensor(
                canonical_centers_radians(resname, chi_index),
                dtype=value.dtype, device=value.device,
            )
            deltas = torch.atan2(torch.sin(value - centers), torch.cos(value - centers))
            costs = 1.0 - torch.cos(deltas)
            nearest = torch.argmin(costs)
            delta = deltas[nearest]
            width = float(canonical_width_degrees(resname, chi_index))
            rows.append(prefactor * (30.0 / width) * torch.sin(delta / 2.0))
            details.append({
                "slot": slot + 1,
                "resname": resname,
                "chi_index": chi_index + 1,
                "allowed_width_degrees": width,
                "nearest_center_radians": float(centers[nearest].detach().cpu()),
            })
    if not rows:
        return torch.empty(0, dtype=coordinates.dtype, device=coordinates.device), details
    return torch.stack(rows), details


def chi_stage(runner: PairInitialAPrime, pair: np.ndarray, occupancy: np.ndarray,
              output: Path, max_nfev: int = 20) -> tuple[np.ndarray, dict[str, object]]:
    stage_started = time.perf_counter()
    blocks = chi_blocks(runner, pair)
    offsets = []
    cursor = 0
    for block in blocks:
        offsets.append((cursor, cursor + block["n_chi"]))
        cursor += block["n_chi"]
    if not blocks:
        raise RuntimeError("no χ-bearing residues in seven-residue window")
    block_summary = [{"slot": block["slot"] + 1, "resname": block["resname"],
                      "n_chi": block["n_chi"], "names": block["names"]}
                     for block in blocks]
    progress = {
        "status": "running", "phase": "initializing", "evaluations": 0,
        "jacobian_evaluations": 0, "n_parameters": int(cursor),
        "max_nfev": int(max_nfev), "blocks": block_summary,
        "stage_elapsed_s": 0.0,
    }
    atomic_json(output / "chi_progress.json", progress)
    target = torch.as_tensor(runner.target, dtype=torch.float64, device=runner.base.torch_device)
    density_mode = str(getattr(runner, "density_objective_mode", "raw"))
    if density_mode not in {"raw", "zscore"}:
        raise ValueError(f"unknown density objective mode: {density_mode}")
    target_z = None
    if density_mode == "zscore":
        target_mean = torch.mean(target)
        target_std = torch.sqrt(torch.mean(torch.square(target - target_mean)))
        target_z = (target - target_mean) / target_std
    occ = torch.as_tensor(occupancy, dtype=torch.float64, device=target.device)
    active_slots = {
        int(index) for index, value in enumerate(np.asarray(occupancy, dtype=float))
        if value > 0.02
    }
    b_factors = torch.as_tensor(
        np.stack((runner.base.b_factors_a_model, runner.base.b_factors_b_model)),
        dtype=torch.float64, device=target.device,
    )
    current = np.asarray(pair, dtype=float).copy()
    evaluations = 0
    jacobian_evaluations = 0

    def build(value: torch.Tensor) -> torch.Tensor:
        coords = torch.as_tensor(current, dtype=torch.float64, device=target.device).clone()
        for block, (start, stop) in zip(blocks, offsets):
            xyz = torsion_to_coords(
                block["template"], block["names"],
                chi_parameters_to_radians(value[start:stop]),
                block["rotations"], block["fixed"],
            )
            indices = torch.as_tensor(block["indices"], dtype=torch.long, device=target.device)
            coords[block["slot"]] = coords[block["slot"]].index_copy(0, indices, xyz)
        return coords

    def density_residual(value: torch.Tensor) -> torch.Tensor:
        coords = build(value)
        models = runner.base.model_density_torch(
            coords[:, runner.base.model_atom_indices], b_factors=b_factors,
        )
        model_density = (occ[:, None] * models).sum(0)
        if density_mode == "raw":
            intercept = torch.mean(target - model_density)
            rows = [target - model_density - intercept]
        else:
            rows = [normalized_zscore_density_residual(model_density, target_z)]
        if getattr(runner, "clash_context", None) is not None and runner.clash_weight > 0.0:
            rows.append(runner.clash_context.residual(
                coords, runner.clash_weight,
                normalize_by_pair_count=getattr(runner, "clash_normalize_by_pair_count", False),
            ))
        rotamer_rows, _ = rotamer_prior_residual_rows(
            coords, blocks, float(getattr(runner, "rotamer_weight", 0.0)), active_slots,
        )
        if rotamer_rows.numel():
            rows.append(rotamer_rows)
        return torch.cat(rows)

    def residual_numpy(value: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return density_residual(torch.as_tensor(value, dtype=torch.float64,
                                                    device=target.device)).cpu().numpy()

    def jacobian_numpy(value: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(value, dtype=torch.float64, device=target.device)
        jac = torch.autograd.functional.jacobian(density_residual, tensor,
                                                 vectorize=True, strategy="forward-mode")
        return jac.detach().cpu().numpy()

    x0 = np.zeros(cursor, dtype=float)
    x_scale, x_scale_details = chi_parameter_scaling(blocks)
    def residual(value):
        nonlocal evaluations
        validate_finite_chi_trial(value, "residual")
        evaluations += 1
        started = time.perf_counter()
        atomic_json(output / "chi_progress.json", {
            **progress, "phase": "residual_running", "evaluations": evaluations,
            "jacobian_evaluations": jacobian_evaluations,
            "last_update": evaluations, "parameters": value.tolist(),
            "stage_elapsed_s": time.perf_counter() - stage_started,
        })
        result = residual_numpy(value)
        elapsed = time.perf_counter() - started
        checkpoint = {
            **progress, "status": "running", "phase": "residual_complete",
            "evaluations": evaluations, "jacobian_evaluations": jacobian_evaluations,
            "last_update": evaluations, "parameters": value.tolist(),
            "last_residual_wall_s": elapsed,
            "stage_elapsed_s": time.perf_counter() - stage_started,
        }
        atomic_json(output / "chi_progress.json", checkpoint)
        atomic_json(output / f"checkpoint_eval_{evaluations:03d}.json", checkpoint)
        return result

    def jacobian(value):
        nonlocal jacobian_evaluations
        validate_finite_chi_trial(value, "jacobian")
        jacobian_evaluations += 1
        started = time.perf_counter()
        atomic_json(output / "chi_progress.json", {
            **progress, "phase": "jacobian_running", "evaluations": evaluations,
            "jacobian_evaluations": jacobian_evaluations,
            "last_update": evaluations, "parameters": value.tolist(),
            "stage_elapsed_s": time.perf_counter() - stage_started,
        })
        result = jacobian_numpy(value)
        elapsed = time.perf_counter() - started
        atomic_json(output / "chi_progress.json", {
            **progress, "status": "running", "phase": "jacobian_complete",
            "evaluations": evaluations, "jacobian_evaluations": jacobian_evaluations,
            "last_update": evaluations, "parameters": value.tolist(),
            "last_jacobian_wall_s": elapsed,
            "stage_elapsed_s": time.perf_counter() - stage_started,
        })
        return result

    tolerances = chi_solver_tolerances()
    bounds = chi_solver_bounds()
    solver_attempts: list[dict[str, object]] = []
    try:
        result = least_squares(
            residual, x0, jac=jacobian, method="trf", tr_solver="exact",
            x_scale=x_scale, bounds=bounds, max_nfev=max_nfev, **tolerances,
        )
        solver_attempts.append({"tr_solver": "exact", "status": "complete"})
    except Exception as exc:
        if not is_chi_solver_numerical_failure(exc):
            raise
        solver_attempts.append({
            "tr_solver": "exact", "status": "numerical_failure",
            "error": repr(exc),
        })
        atomic_json(output / "chi_solver_retry.json", {
            "status": "retrying", "failed_tr_solver": "exact",
            "retry_tr_solver": "lsmr", "reason": repr(exc),
            "bounds_degrees": list(bounds),
            "evaluations_before_retry": evaluations,
            "jacobian_evaluations_before_retry": jacobian_evaluations,
        })
        result = least_squares(
            residual, x0, jac=jacobian, method="trf", tr_solver="lsmr",
            x_scale=x_scale, bounds=bounds, max_nfev=max_nfev, **tolerances,
        )
        solver_attempts.append({"tr_solver": "lsmr", "status": "complete"})
    if chi_exact_result_needs_lsmr(result, max_nfev):
        solver_attempts[-1]["status"] = "numerical_stagnation"
        solver_attempts[-1]["nfev"] = int(result.nfev)
        solver_attempts[-1]["njev"] = int(result.njev or 0)
        solver_attempts[-1]["projected_gradient_norm_end"] = float(result.optimality)
        atomic_json(output / "chi_solver_retry.json", {
            "status": "retrying", "failed_tr_solver": "exact",
            "retry_tr_solver": "lsmr",
            "reason": "exact TRF exhausted the residual cap with <=2 Jacobians and remained nonstationary",
            "bounds_degrees": list(bounds),
            "evaluations_before_retry": evaluations,
            "jacobian_evaluations_before_retry": jacobian_evaluations,
            "exact_result": solver_attempts[-1],
        })
        result = least_squares(
            residual, x0, jac=jacobian, method="trf", tr_solver="lsmr",
            x_scale=x_scale, bounds=bounds, max_nfev=max_nfev, **tolerances,
        )
        solver_attempts.append({"tr_solver": "lsmr", "status": "complete"})
    if chi_exact_result_needs_lsmr(result, max_nfev):
        solver_attempts[-1]["status"] = "numerical_stagnation"
        solver_attempts[-1]["nfev"] = int(result.nfev)
        solver_attempts[-1]["njev"] = int(result.njev or 0)
        solver_attempts[-1]["projected_gradient_norm_end"] = float(result.optimality)
        atomic_json(output / "chi_solver_dogbox_retry.json", {
            "status": "retrying", "failed_tr_solver": "lsmr",
            "retry_method": "dogbox",
            "reason": "LSMR TRF also exhausted the residual cap with <=2 Jacobians and remained nonstationary",
            "bounds_degrees": list(bounds),
            "evaluations_before_retry": evaluations,
            "jacobian_evaluations_before_retry": jacobian_evaluations,
            "lsmr_result": solver_attempts[-1],
        })
        result = least_squares(
            residual, x0, jac=jacobian, method="dogbox",
            x_scale=x_scale, bounds=bounds, max_nfev=max_nfev, **tolerances,
        )
        solver_attempts.append({"method": "dogbox", "status": "complete"})
    validate_finite_chi_trial(result.x, "endpoint")
    solver_diagnostics = least_squares_stage_diagnostics(result, max_nfev)
    final = build(torch.as_tensor(result.x, dtype=torch.float64, device=target.device))
    final_np = final.detach().cpu().numpy()
    final_rotamer_rows, final_rotamer_details = rotamer_prior_residual_rows(
        final, blocks, float(getattr(runner, "rotamer_weight", 0.0)), active_slots,
    )
    final_rotamer_energy = float(torch.dot(
        final_rotamer_rows, final_rotamer_rows,
    ).detach().cpu()) if final_rotamer_rows.numel() else 0.0
    atomic_npz(output / "final_slots.npz", slot1_window=final_np[0], slot2_window=final_np[1],
               chi_parameters=result.x, initial_slots=current)
    atomic_json(output / "chi_result.json", {
        "status": "complete", "n_parameters": int(cursor), "n_blocks": len(blocks),
        "nfev": int(result.nfev), "njev": int(result.njev or 0),
        "x_scale": x_scale.tolist(), "x_scale_details": x_scale_details,
        "parameter_units": "degrees", "renderer_torsion_units": "radians_after_deg2rad",
        "x_scale_units": "degrees", "zscore_density_residual_normalization": "1/sqrt(n_voxels)",
        "bounds_degrees": list(bounds), "solver_attempts": solver_attempts,
        "solver_tolerances": tolerances,
        "termination_contract": "gradient criterion or evaluation cap; ftol/xtol disabled",
        "parameters": result.x.tolist(),
        "rotamer_prior_weight": float(getattr(runner, "rotamer_weight", 0.0)),
        "rotamer_prior_residual_count": int(final_rotamer_rows.numel()),
        "rotamer_prior_energy_end": final_rotamer_energy,
        "rotamer_prior_details": final_rotamer_details,
        **solver_diagnostics,
        "blocks": [{"slot": b["slot"] + 1, "resname": b["resname"],
                    "n_chi": b["n_chi"], "names": b["names"]} for b in blocks],
    })
    atomic_json(output / "chi_progress.json", {
        **progress, "status": "complete", "phase": "complete",
        "evaluations": evaluations, "jacobian_evaluations": jacobian_evaluations,
        "last_update": evaluations, "nfev": int(result.nfev),
        "njev": int(result.njev or 0),
        **solver_diagnostics,
        "bounds_degrees": list(bounds), "solver_attempts": solver_attempts,
        "stage_elapsed_s": time.perf_counter() - stage_started,
    })
    return final_np, {"n_parameters": int(cursor), **solver_diagnostics,
                      "x_scale": x_scale.tolist(),
                      "x_scale_details": x_scale_details,
                      "parameter_units": "degrees",
                      "renderer_torsion_units": "radians_after_deg2rad",
                      "x_scale_units": "degrees",
                      "zscore_density_residual_normalization": "1/sqrt(n_voxels)",
                      "bounds_degrees": list(bounds),
                      "solver_attempts": solver_attempts,
                      "solver_tolerances": tolerances,
                      "rotamer_prior_weight": float(getattr(runner, "rotamer_weight", 0.0)),
                      "rotamer_prior_residual_count": int(final_rotamer_rows.numel()),
                      "rotamer_prior_energy_end": final_rotamer_energy,
                      "termination_contract": "gradient criterion or evaluation cap; ftol/xtol disabled"}


def objective(runner: PairInitialAPrime, pair: np.ndarray, occupancy: np.ndarray) -> dict[str, object]:
    models = runner.base.model_density_batch(pair, slots=np.array((0, 1)))
    density_mode = str(getattr(runner, "density_objective_mode", "raw"))
    model_density = np.asarray(occupancy @ models, dtype=float)
    raw_density = np.asarray(runner.target - model_density, dtype=float)
    if density_mode == "raw":
        intercept = float(np.mean(raw_density))
        density = raw_density - intercept
    elif density_mode == "zscore":
        target_z = (runner.target - np.mean(runner.target)) / np.sqrt(
            np.mean(np.square(runner.target - np.mean(runner.target)))
        )
        model_z = (model_density - np.mean(model_density)) / np.sqrt(
            np.mean(np.square(model_density - np.mean(model_density)))
        )
        intercept = 0.0
        density = (model_z - target_z) / np.sqrt(max(len(target_z), 1))
    else:
        raise ValueError(f"unknown density objective mode: {density_mode}")
    seam = []
    for coords in pair:
        vector, _, _ = seam_vector(runner.initial_backbone, coords[runner.bb_indices])
        seam.append(vector)
    seam = np.asarray(seam)
    rho = seam_rho_vector(runner)
    seam_energy = float(np.sum((np.sqrt(rho / 2.0) * seam) ** 2))
    rama_energy = 0.0
    planar_energy = 0.0
    rama_scores = []
    for coords in pair:
        _, omega_delta, scores, barrier = runner.omega_and_rama(coords)
        rama_energy += float(runner.rama_weight * np.square(barrier).sum())
        planar_energy += float(runner.planar_weight * np.square(omega_delta / runner.omega_scale_deg).sum())
        rama_scores.append(scores)
    components = {"density_energy": float(np.dot(density, density)),
                  "seam_penalty_energy": seam_energy, "rama_energy": rama_energy,
                  "planarity_energy": planar_energy}
    if getattr(runner, "clash_context", None) is not None and runner.clash_weight > 0.0:
        clash = runner.clash_context.residual(
            torch.as_tensor(pair, dtype=torch.float64), runner.clash_weight,
            normalize_by_pair_count=getattr(runner, "clash_normalize_by_pair_count", False),
        ).detach().cpu().numpy()
        components["clash_penalty_energy"] = float(np.dot(clash, clash))
    else:
        components["clash_penalty_energy"] = 0.0
    rotamer_blocks = chi_blocks(runner, pair)
    active_slots = {
        int(index) for index, value in enumerate(np.asarray(occupancy, dtype=float))
        if value > 0.02
    }
    rotamer_rows, rotamer_details = rotamer_prior_residual_rows(
        torch.as_tensor(pair, dtype=torch.float64, device=runner.base.torch_device),
        rotamer_blocks,
        float(getattr(runner, "rotamer_weight", 0.0)),
        active_slots,
    )
    components["rotamer_prior_energy"] = (
        float(torch.dot(rotamer_rows, rotamer_rows).detach().cpu())
        if rotamer_rows.numel() else 0.0
    )
    components["total_energy"] = float(sum(components.values()))
    return {"occupancies": occupancy.tolist(), "intercept": intercept,
            "density_objective_mode": density_mode,
            "intercept_mode": "profiled" if density_mode == "raw" else "retained_zero",
            "raw_density_energy": float(np.dot(raw_density - (np.mean(raw_density) if density_mode == "raw" else 0.0),
                                                raw_density - (np.mean(raw_density) if density_mode == "raw" else 0.0))),
            "components": components, "n_density_voxels": int(len(density)),
            "seam_norm_A_equivalent": float(np.linalg.norm(seam)),
            "rama_scores": rama_scores,
            "rotamer_prior_residual_count": int(rotamer_rows.numel()),
            "rotamer_prior_details": rotamer_details}


def write_stage_pdb(runner: PairInitialAPrime, pair: np.ndarray, occupancy: np.ndarray,
                    source: Path, output: Path) -> None:
    """Write a minimal full-model PDB with refined A/B window coordinates."""
    key_to_template = {}
    for line in source.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        key = (line[21].strip(), int(line[22:26]), line[26].strip(), line[12:16].strip())
        key_to_template.setdefault(key, line)
    window_keys = []
    for residue in runner.window.residues:
        for name, index in zip(residue.name, residue.selection):
                window_keys.append((runner.base.chain, int(residue.id[0]), str(residue.id[1]).strip(), str(name).strip()))
    lines = []
    window_key_set = set(window_keys)
    for line in source.read_text().splitlines():
        # The window atoms are rewritten below.  Keeping source ANISOU lines
        # would leave them detached from their corresponding ATOM records,
        # which makes strict PDB readers (including Phenix) reject the model.
        if line.startswith("ANISOU"):
            continue
        if not line.startswith(("ATOM", "HETATM")):
            lines.append(line)
            continue
        key = (line[21].strip(), int(line[22:26]), line[26].strip(), line[12:16].strip())
        if key not in window_key_set:
            lines.append(line)
    serial = 1
    for slot, altloc in enumerate(("A", "B")):
        for residue in runner.window.residues:
            for name, index, coord in zip(residue.name, residue.selection, pair[slot][
                    [int(np.searchsorted(runner.window.selection, int(i))) for i in residue.selection]]):
                name = str(name)
                key = (runner.base.chain, int(residue.id[0]), str(residue.id[1]).strip(), name.strip())
                template = key_to_template.get(key)
                if template is None:
                    continue
                # Preserve the five-column atom-name field (columns 12-16),
                # then write the new altloc in column 17.
                line = template[:6] + f"{serial:5d}" + template[11:16] + altloc + template[17:30]
                line += f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                line += f"{float(occupancy[slot]):6.2f}" + template[60:]
                lines.append(line); serial += 1
    output.write_text("\n".join(lines) + "\nEND\n")


def run_site(record: dict[str, str], panel: Path, out_root: Path,
             inner_nfev: int, outer_updates: int, chi_nfev: int,
             clash_weight: float = 0.0, density_mode: str = "raw",
             free_occupancy_ratio: bool = False,
             fitting_mask_radius_A: float = 1.5,
             normalize_clash_by_pair_count: bool = False,
             rotamer_weight: float = 0.0,
             rotamer_calibration: Path | None = None,
             map_protocol: str = "native_deposited",
             device: str = "auto",
             preflight_only: bool = False) -> dict[str, object]:
    if map_protocol not in {"native_deposited", "rebuilt_fmodel"}:
        raise ValueError(f"unknown map protocol: {map_protocol}")
    label = site_label(record); out = out_root / label; out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "status.json", {"status": "running", "site": label})
    atomic_json(out / "run_config.json", {
        "site": label, "inner_nfev": inner_nfev, "outer_updates": outer_updates,
        "chi_nfev": chi_nfev, "clash_weight": float(clash_weight),
        "rotamer_weight": float(rotamer_weight),
        "rotamer_prior": (
            "per-active-slot, per-chi, occupancy-unweighted; "
            "lambda*(30/allowed_width_deg)^2*(1-cos(delta_to_nearest_center))"
        ),
        "rotamer_calibration": (
            {"path": str(rotamer_calibration), "sha256": sha256_file(rotamer_calibration)}
            if rotamer_calibration is not None else None
        ),
        "clash_lambda_eff": float(clash_weight),
        "clash_normalize_by_pair_count": bool(normalize_clash_by_pair_count),
        "clash_residual_normalization": (
            "sqrt(lambda_eff / (2 * monitored_pair_count_one_slot))"
            if normalize_clash_by_pair_count else "none"
        ),
        "clash_pair_count_one_slot": None,
        "clash_residual_count_two_slots": None,
        "clash_pair_cutoff_A": 4.5, "clash_threshold_scale": 0.75,
        "clash_residual": "sqrt(weight)*max(0.75*(vdW_i+vdW_j)-distance, 0)",
        "clash_graph_filter": "exclude graph-distance 1-2 and 1-3; heavy atoms only",
        "density_objective_mode": density_mode,
        "occupancy_ratio_mode": (
            "mirror_ratio_total_pinned" if free_occupancy_ratio else "fixed_inherited_qfit"
        ),
        "density_intercept": "retained_zero" if density_mode == "zscore" else "profiled",
        "density_zscore_std": "population_std_over_frozen_mask" if density_mode == "zscore" else None,
        "density_zscore_residual_normalization": (
            "1/sqrt(n_voxels)" if density_mode == "zscore" else None
        ),
        "chi_parameter_units": "degrees",
        "chi_renderer_units": "radians_after_deg2rad",
        "chi_x_scale": "1 / downstream heavy-atom count for each chi",
        "chi_x_scale_units": "degrees",
        "renderer_periodic_wrap": "fixed branch from stage-zero reference coordinates",
        "chi_solver_tolerances": chi_solver_tolerances(),
        "chi_termination_contract": "gradient criterion or evaluation cap; ftol/xtol disabled",
        "seam_rho": [7.42, 15.87, 46.91, 79.14, 148.52, 207.27],
        "fitting_mask_radius_A": float(fitting_mask_radius_A),
        "map_protocol": map_protocol,
        "device": device,
        "fitting_mask_grid": (
            "native_deposited_coefficient_lattice"
            if map_protocol == "native_deposited" else "rebuilt_fmodel_full_map"
        ),
        "optimization_target_grid": (
            "native_deposited_coefficient_lattice"
            if map_protocol == "native_deposited" else "rebuilt_fmodel_full_map"
        ),
        "preflight_only": bool(preflight_only),
        "atom_on_peak_gate": "mandatory; positive mean atom-minus-random enrichment",
    })
    source = resolve_source_pdb(panel, record)
    qfit_candidates = (
        panel / "inputs/qfit" / f"{record['pdb_id']}_qFit.pdb",
        panel / "inputs/qfit" / f"{record['pdb_id'].lower()}_qFit.pdb",
    )
    qfit_source = next((path for path in qfit_candidates if path.is_file()), None)
    if qfit_source is None:
        raise FileNotFoundError(f"qFit model missing: {qfit_candidates[0]}")
    map_candidates = (
        panel / "inputs/map_mtz" / f"{record['pdb_id'].lower()}.mtz",
        panel / "inputs/map_mtz" / f"{record['pdb_id'].upper()}.mtz",
    )
    map_path = next((path for path in map_candidates if path.is_file()), None)
    if map_path is None:
        raise FileNotFoundError(f"MTZ missing: {map_candidates[0]}")
    runner = PairInitialAPrime(
        out / "backbone_to_conformopt", inner_nfev, outer_updates,
        record["pdb_id"], record["chain"], int(record["residue_number"]),
        renderer_backend="torch", map_scaler_structure="full", mask_scope="window",
        density_atom_scope="all", b_factor_mode="deposited_A_B", device=device,
        clash_weight=clash_weight,
        source_pdb=source, mtz_path=map_path,
    )
    runner.clash_normalize_by_pair_count = bool(normalize_clash_by_pair_count)
    runner.rotamer_weight = float(rotamer_weight)
    if runner.clash_context is not None:
        run_config = json.loads((out / "run_config.json").read_text())
        run_config["clash_pair_count_one_slot"] = int(runner.clash_context.pair_count)
        run_config["clash_residual_count_two_slots"] = int(
            runner.clash_context.residual_pair_count
        )
        atomic_json(out / "run_config.json", run_config)
    if density_mode not in {"raw", "zscore"}:
        raise ValueError(f"unknown density objective mode: {density_mode}")
    runner.density_objective_mode = density_mode
    # The z-score experiment is explicitly the corrected per-component seam
    # contract, not the historical scalar-rho run.
    runner.rho_vector = np.asarray(
        [7.42, 15.87, 46.91, 79.14, 148.52, 207.27], dtype=float
    )
    qfit_pair, qfit_b_factors = read_published_qfit_pair(qfit_source, runner)
    runner.set_pair_initial(qfit_pair)
    # The qFit model is the actual stage-zero model, including its own B
    # factors.  Deposited B remains available as the independent reference in
    # the base object, but must not silently replace qFit's B for the start.
    runner.base.b_factors_a_model = qfit_b_factors[0, runner.base.model_atom_indices].copy()
    runner.base.b_factors_b_model = qfit_b_factors[1, runner.base.model_atom_indices].copy()
    runner.base.b_factors = runner.base.b_factors_a_model.copy()
    runner.base.b_factor_mode = "published_qfit_A_for_slot1_published_qfit_B_for_slot2"
    runner.base._renderer_b_factors = torch.as_tensor(
        runner.base.b_factors, dtype=torch.float64, device=runner.base.torch_device
    )
    if map_protocol == "native_deposited":
        map_configuration = native_deposited_map_configuration(runner)
    else:
        map_configuration = install_full_observed_lattice_target(
            runner, source, map_path, fitting_mask_radius_A
        )
        map_configuration["map_protocol"] = "rebuilt_fmodel"
    # The published qFit pair, not the deposited control, is stage zero for
    # this experiment.  Freeze the periodic renderer branch to slot 1 of that
    # pair; alternate slots are local perturbations of the same atom images.
    runner.base._set_renderer_reference_wrap_offsets(  # pylint: disable=protected-access
        qfit_pair[0, runner.base.model_atom_indices]
    )
    run_config = json.loads((out / "run_config.json").read_text())
    run_config["map_configuration"] = map_configuration
    run_config["actual_fitting_mask_radius_A"] = float(map_configuration["mask_radius_A"])
    run_config["actual_mask_voxels"] = int(map_configuration["mask_voxels"])
    atomic_json(out / "run_config.json", run_config)
    atom_peak = atom_on_peak_preflight(
        runner, source, map_path, map_configuration,
        out / "atom_on_peak_preflight.json",
    )
    if atom_peak["status"] != "passed":
        raise RuntimeError(
            "mandatory atom-on-peak preflight failed: "
            f"enrichment={atom_peak['atom_minus_random_enrichment']['raw_density']}"
        )
    if preflight_only:
        result = {
            "status": "complete",
            "site": label,
            "mode": "preflight_only",
            "map_protocol": map_protocol,
            "map_configuration": map_configuration,
            "atom_on_peak_preflight": atom_peak,
            "provenance": atom_peak["provenance"],
        }
        atomic_json(out / "result.json", result)
        atomic_json(out / "status.json", result)
        return result
    raw_occupancy = np.asarray([
        float(json.loads(record["qfit_occupancies"])["A"]),
        float(json.loads(record["qfit_occupancies"])["B"]),
    ])
    # qFit occupancies in the inventory are read from float32 PDB fields;
    # values such as 0.72+0.28 can therefore exceed one by ~3e-8.  Preserve
    # the inherited pair while removing only this serialization roundoff so
    # the fixed mirror pair satisfies its exact-unit-sum contract.
    if np.isclose(raw_occupancy.sum(), 1.0, atol=1e-6, rtol=0.0):
        occupancy = raw_occupancy / raw_occupancy.sum()
    else:
        occupancy = raw_occupancy.copy()
    qfit_occupancy = occupancy.copy()
    atomic_npz(out / "qfit_input.npz", slot1_window=qfit_pair[0], slot2_window=qfit_pair[1],
               occupancies=occupancy, target=runner.target,
               fit_mask_indices=np.flatnonzero(runner.base.mask).astype(np.int64),
               fit_grid_shape_xyz=np.asarray(runner.base.qfit.xmap.n_real(), dtype=np.int64))
    atomic_json(out / "qfit_input_objective.json", objective(runner, qfit_pair, occupancy))
    p1 = np.zeros(runner.rotator.ndofs); p2 = np.zeros(runner.rotator.ndofs)
    occupancy_scheme = "mirror_ratio" if free_occupancy_ratio else "mirror"
    fixed_occupancy_weights = None if free_occupancy_ratio else occupancy
    backbone_out = out / "conformopt_backbone_1"; backbone_out.mkdir()
    backbone_result = joint_run(
        runner, p1, p2, "conformopt_backbone_1", backbone_out,
        float(runner.ab_distance), fixed_b_offset=0.0, occupancy_scheme=occupancy_scheme,
        mirror_eta=0.1, initial_occupancy_weights=occupancy,
        fixed_occupancy_weights=fixed_occupancy_weights, per_slot_trust_radii=True,
        carry_trust_radii=True, geometry_gradient_mode="per_slot_occupancy_decoupled",
        lambda_damping_alpha=0.3, density_normalizer=1.0,
        inner_nfev=inner_nfev, outer_updates=outer_updates, tr_solver="exact",
    )
    if free_occupancy_ratio:
        occupancy = np.asarray(backbone_result["final_occupancies"], dtype=float)
    pair = np.stack((np.load(backbone_out / "final_slots.npz")["slot1_window"],
                     np.load(backbone_out / "final_slots.npz")["slot2_window"]))
    atomic_json(out / "conformopt_backbone_1_objective.json", objective(runner, pair, occupancy))
    # Strict backbone-only control: apply the second backbone block directly
    # to B1, with no intervening chi update.  The production sequence below
    # remains B1 -> chi1 -> B2 -> chi2; this side branch isolates how much of
    # the endpoint signal the independently converged backbone path supplies.
    backbone_1_pair = pair.copy()
    runner.set_pair_initial(backbone_1_pair)
    backbone_only_second = out / "conformopt_backbone_only_2"; backbone_only_second.mkdir()
    backbone_only_result_2 = joint_run(
        runner, np.zeros(runner.rotator.ndofs), np.zeros(runner.rotator.ndofs),
        "conformopt_backbone_only_2", backbone_only_second,
        float(runner.ab_distance), fixed_b_offset=0.0,
        occupancy_scheme=occupancy_scheme, mirror_eta=0.1,
        initial_occupancy_weights=occupancy,
        fixed_occupancy_weights=fixed_occupancy_weights,
        per_slot_trust_radii=True, carry_trust_radii=True,
        geometry_gradient_mode="per_slot_occupancy_decoupled",
        lambda_damping_alpha=0.3, density_normalizer=1.0,
        inner_nfev=inner_nfev, outer_updates=outer_updates, tr_solver="exact",
    )
    backbone_only_occupancy = (
        np.asarray(backbone_only_result_2["final_occupancies"], dtype=float)
        if free_occupancy_ratio else occupancy.copy()
    )
    backbone_only_pair = np.stack((
        np.load(backbone_only_second / "final_slots.npz")["slot1_window"],
        np.load(backbone_only_second / "final_slots.npz")["slot2_window"],
    ))
    atomic_json(
        out / "conformopt_backbone_only_2_objective.json",
        objective(runner, backbone_only_pair, backbone_only_occupancy),
    )
    # Restore the production branch to the B1 checkpoint before chi1.
    runner.set_pair_initial(backbone_1_pair)
    pair = backbone_1_pair
    chi_out = out / "conformopt_sidechain_chi"; chi_out.mkdir()
    pair, chi_result = chi_stage(runner, pair, occupancy, chi_out, chi_nfev)
    atomic_json(out / "conformopt_sidechain_chi_objective.json", objective(runner, pair, occupancy))
    runner.set_pair_initial(pair)
    second = out / "conformopt_backbone_2"; second.mkdir()
    backbone_result_2 = joint_run(runner, np.zeros(runner.rotator.ndofs), np.zeros(runner.rotator.ndofs),
              "conformopt_backbone_2", second, float(runner.ab_distance),
              fixed_b_offset=0.0, occupancy_scheme=occupancy_scheme, mirror_eta=0.1,
              initial_occupancy_weights=occupancy, fixed_occupancy_weights=fixed_occupancy_weights,
              per_slot_trust_radii=True, carry_trust_radii=True,
              geometry_gradient_mode="per_slot_occupancy_decoupled",
              lambda_damping_alpha=0.3, density_normalizer=1.0,
              inner_nfev=inner_nfev, outer_updates=outer_updates, tr_solver="exact")
    if free_occupancy_ratio:
        occupancy = np.asarray(backbone_result_2["final_occupancies"], dtype=float)
    pair = np.stack((np.load(second / "final_slots.npz")["slot1_window"],
                     np.load(second / "final_slots.npz")["slot2_window"]))
    atomic_json(out / "conformopt_backbone_2_objective.json", objective(runner, pair, occupancy))
    final_chi = out / "conformopt_sidechain_chi_2"; final_chi.mkdir()
    pair, _ = chi_stage(runner, pair, occupancy, final_chi, chi_nfev)
    atomic_json(out / "conformopt_sidechain_chi_2_objective.json", objective(runner, pair, occupancy))
    phenix_pdb = out / "phenix_input.pdb"; write_stage_pdb(runner, pair, occupancy, source, phenix_pdb)
    phenix_dir = out / "phenix"; phenix_dir.mkdir()
    command = ["phenix.refine", str(phenix_pdb),
               str(map_path),
               f"output.prefix={phenix_dir / 'refined'}",
               "strategy=individual_sites+individual_adp"]
    try:
        completed = subprocess.run(command, cwd=phenix_dir, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=3600, check=False)
        (phenix_dir / "phenix.log").write_text(completed.stdout)
        phenix_status = {"status": "complete" if completed.returncode == 0 else "failed",
                         "returncode": completed.returncode, "command": command}
    except FileNotFoundError as exc:
        phenix_status = {"status": "unavailable", "reason": str(exc), "command": command}
    except subprocess.TimeoutExpired:
        phenix_status = {"status": "timeout", "command": command}
    atomic_json(phenix_dir / "status.json", phenix_status)
    provenance = runner_provenance(
        runner,
        source,
        map_path,
        {
            "qfit_input": out / "qfit_input.npz",
            "conformopt_backbone_1": backbone_out / "final_slots.npz",
            "conformopt_backbone_only_2": backbone_only_second / "final_slots.npz",
            "conformopt_sidechain_chi_1": chi_out / "final_slots.npz",
            "conformopt_backbone_2": second / "final_slots.npz",
            "conformopt_sidechain_chi_2": final_chi / "final_slots.npz",
        },
    )
    result = {"status": "complete", "site": label, "qfit_input": objective(runner, qfit_pair, qfit_occupancy),
              "conformopt_backbone_1": json.loads((out / "conformopt_backbone_1_objective.json").read_text()),
              "conformopt_backbone_only_2": json.loads(
                  (out / "conformopt_backbone_only_2_objective.json").read_text()
              ),
              "conformopt_sidechain_chi_1": json.loads((out / "conformopt_sidechain_chi_objective.json").read_text()),
              "conformopt_backbone_2": json.loads((out / "conformopt_backbone_2_objective.json").read_text()),
              "conformopt_sidechain_chi_2": json.loads((out / "conformopt_sidechain_chi_2_objective.json").read_text()),
              "chi_integration": chi_result, "phenix": phenix_status,
              "clash_weight": float(clash_weight),
              "rotamer_weight": float(rotamer_weight),
              "rotamer_calibration": (
                  {"path": str(rotamer_calibration), "sha256": sha256_file(rotamer_calibration)}
                  if rotamer_calibration is not None else None
              ),
              "clash_lambda_eff": float(clash_weight),
              "clash_normalize_by_pair_count": bool(normalize_clash_by_pair_count),
              "clash_pair_count_one_slot": (
                  int(runner.clash_context.pair_count)
                  if runner.clash_context is not None else None
              ),
              "clash_residual_count_two_slots": (
                  int(runner.clash_context.residual_pair_count)
                  if runner.clash_context is not None else None
              ),
              "density_objective_mode": density_mode,
              "occupancy_ratio_mode": (
                  "mirror_ratio_total_pinned" if free_occupancy_ratio else "fixed_inherited_qfit"
              ),
              "final_occupancies": occupancy.tolist(),
              "fitting_mask_radius_A": float(map_configuration["mask_radius_A"]),
              "map_protocol": map_protocol,
              "map_configuration": map_configuration,
              "full_lattice_configuration": (
                  map_configuration if map_protocol == "rebuilt_fmodel" else None
              ),
              "atom_on_peak_preflight": atom_peak,
              "clash_pair_cutoff_A": 4.5, "clash_threshold_scale": 0.75,
              "provenance": provenance}
    atomic_json(out / "result.json", result); atomic_json(out / "status.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site", action="append")
    parser.add_argument("--inner-nfev", type=int, default=8)
    parser.add_argument("--outer-updates", type=int, default=6)
    parser.add_argument("--chi-nfev", type=int, default=20)
    parser.add_argument("--clash-weight", type=float, default=0.0)
    parser.add_argument("--rotamer-weight", type=float, default=0.0)
    parser.add_argument("--rotamer-calibration", type=Path)
    parser.add_argument("--density-mode", choices=("raw", "zscore"), default="raw")
    parser.add_argument("--free-occupancy-ratio", action="store_true",
                        help="free the two-slot ratio with total occupancy pinned at one")
    parser.add_argument("--fitting-mask-radius", type=float, default=1.5,
                        help="fitting-mask radius in Angstroms on the full observed-map lattice")
    parser.add_argument("--normalize-clash-by-pair-count", action="store_true",
                        help="divide clash residual weight by the two-slot monitored-pair count")
    parser.add_argument(
        "--map-protocol", choices=("native_deposited", "rebuilt_fmodel"),
        default="native_deposited",
        help="use deposited map coefficients by default; rebuilt fmodel is diagnostic-only",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto",
        help="Torch device for differentiable density calculations (default: auto)",
    )
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="run the mandatory atom-on-peak gate and stop before optimization",
    )
    args = parser.parse_args()
    if args.fitting_mask_radius <= 0.0:
        raise ValueError("--fitting-mask-radius must be positive")
    if args.rotamer_weight < 0.0:
        raise ValueError("--rotamer-weight must be nonnegative")
    if args.rotamer_weight > 0.0 and args.rotamer_calibration is None:
        raise ValueError("a positive --rotamer-weight requires --rotamer-calibration provenance")
    if args.rotamer_calibration is not None and not args.rotamer_calibration.is_file():
        raise FileNotFoundError(args.rotamer_calibration)
    args.output.mkdir(parents=True, exist_ok=False)
    environment = runtime_report()
    atomic_json(args.output / "environment.json", environment)
    if args.clash_weight > 0.0 and not required_runtime_ok(
        environment, clash_weight=args.clash_weight
    ):
        raise RuntimeError(
            "clash-weighted optimization requires qFit/CCTBX, NumPy, SciPy, "
            "PyTorch, Gemmi, and PHENIX_ROOT; run scripts/check_runtime.py for details"
        )
    status_path = args.output / "status.txt"
    lock_path = args.output / "controller.lock"
    status_path.write_text("optimizing\n")
    (args.output / "controller.pid").write_text(f"{os.getpid()}\n")
    lock_path.write_text("active\n")
    try:
        rows = list(csv.DictReader((args.panel / "selected_sites.csv").open()))
        if args.site:
            rows = [row for row in rows if site_label(row) in set(args.site)]
        results = []
        for row in rows:
            try:
                results.append(run_site(row, args.panel, args.output, args.inner_nfev,
                                        args.outer_updates, args.chi_nfev, args.clash_weight,
                                        args.density_mode, args.free_occupancy_ratio,
                                        args.fitting_mask_radius,
                                        args.normalize_clash_by_pair_count,
                                        args.rotamer_weight,
                                        args.rotamer_calibration,
                                        args.map_protocol, args.device, args.preflight_only))
            except Exception as exc:  # checkpoint failure and continue to next site
                label = site_label(row); site_dir = args.output / label; site_dir.mkdir(exist_ok=True)
                atomic_json(site_dir / "status.json", {"status": "failed", "site": label,
                                                         "error": repr(exc)})
                results.append({"status": "failed", "site": label, "error": repr(exc)})
            atomic_json(args.output / "progress.json", {"status": "running",
                                                         "completed_sites": len(results),
                                                         "total_sites": len(rows), "results": results})
        final_status = "complete" if all(
            result.get("status") == "complete" for result in results
        ) else "failed"
        atomic_json(args.output / "progress.json", {"status": final_status,
                                                     "completed_sites": len(results),
                                                     "total_sites": len(rows),
                                                     "results": results})
        status_path.write_text(f"{final_status}\n")
        return 0 if final_status == "complete" else 1
    except BaseException:
        status_path.write_text("failed\n")
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
