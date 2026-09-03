#!/usr/bin/env python3
"""One-site sequential two-slot backbone-nullspace proof of concept.

The two slots are 7-residue qFit backbone windows (14 phi/psi parameters per
slot).  At every finite-difference update the torsion gradient is projected
onto ``null(compute_jacobian(...))``.  Slot 1 is fitted to the qFit-scaled,
neighbour-subtracted real map, frozen, and slot 2 is fitted to that residual.
Occupancies and a free map intercept are re-solved by a bounded affine QP at
each objective evaluation.

This is a measurement/optimization wrapper around qFit primitives, not a
replacement for ``BackboneRotator`` or its closure Jacobian.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

from mmtbx.validation import ramalyze
from qfit.backbone import compute_jacobian
from qfit.qfit import QFitOptions, QFitRotamericResidue
from qfit.samplers import BackboneRotator
from qfit.structure import Structure
from qfit.xtal.scaler import MapScaler

from cctbx import masks
from cctbx.array_family import flex as flex_array
from cctbx.xray import ext
from density_denoiser.differentiable_renderer import (
    coefficients_for_elements,
    render_cctbx_density,
)

from run_d1_reachability import BACKBONE_NAMES, dihedrals, local_index, rmsd
from run_d1_tier_a_flips import atom_local_index, source_path
from occupancy_selection import solve_affine_qp
from run_d6_tier2_realmap import make_map
from result_provenance import runner_provenance


EPSILON = 1e-8


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".tmp", delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    temporary.replace(path)


def reordered_coordinates(reference, other) -> np.ndarray:
    by_name = {name: coor for name, coor in zip(other.name.tolist(), other.coor)}
    missing = set(reference.name.tolist()) - set(by_name)
    if missing:
        raise ValueError(f"B altloc lacks atoms: {sorted(missing)}")
    return np.asarray([by_name[name] for name in reference.name.tolist()], dtype=float)


def backbone_coordinates(residue) -> np.ndarray:
    return np.asarray([residue.coor[atom_local_index(residue, name)] for name in BACKBONE_NAMES])


def atom_window_indices(window, residue, names: tuple[str, ...]) -> list[int]:
    return [local_index(window, residue, name) for name in names]


def window_backbone_indices(window) -> np.ndarray:
    """Return N/CA/C/O indices for every residue in a qFit window."""
    return np.asarray(
        [index for residue in window.residues
         for index in atom_window_indices(window, residue, BACKBONE_NAMES)],
        dtype=np.int64,
    )


def full_window_mask(transformer, xmap, coordinate_sets, radius):
    """Build a qFit-compatible union-of-spheres mask around each atom."""
    xrs, dxyz = transformer._get_xray_structure_in_box()  # pylint: disable=protected-access
    # The box extraction applies one common Cartesian translation to the
    # structure.  qFit's get_conformers_mask applies the corresponding
    # per-atom vector; the source structures here are already in that frame.
    shift = np.asarray(dxyz[0], dtype=float)
    total = None
    for coordinates in coordinate_sets:
        sites_cart = flex_array.vec3_double((np.asarray(coordinates) + shift).tolist())
        sites_frac = xrs.unit_cell().fractionalize(sites_cart)
        rmax = flex_array.double(sites_frac.size(), float(radius))
        selected = masks.around_atoms(
            xrs.unit_cell(), 1, sites_frac, rmax, xmap.n_real(), 0, 0
        ).data
        selected = (selected == 0).as_numpy_array()
        total = selected if total is None else (total | selected)
    # CCTBX mask grids use (x, y, z); qFit's NumPy map arrays use (z, y, x).
    return np.asarray(total, dtype=bool).swapaxes(0, 2)


def extract_window_neighbors(structure, window, distance):
    """Return atoms outside ``window`` within ``distance`` of any window atom.

    qFit's :meth:`Structure.extract_neighbors` intentionally excludes only its
    fitted *central residue*, because native qFit renders that residue alone.
    A'' is different for the expanded-mask modes: it parameterises and renders
    a peptide window.  Reusing qFit's central-residue subtraction would remove
    density for atoms that remain in A'' rho_calc, hollowing the target.
    Keep this scope change in A'' rather than altering qFit's correct native
    central-residue behaviour.
    """
    window_ids = {
        (str(residue.chain[0]), int(residue.resi[0]))
        for residue in window.residues
    }
    outside_window = np.asarray([
        (str(chain), int(resi)) not in window_ids
        for chain, resi in zip(structure.chain, structure.resi)
    ], dtype=bool)
    candidates = structure.copy().get_selected_structure(
        flex_array.bool(outside_window).iselection()
    )
    nearby = np.zeros(candidates.natoms, dtype=bool)
    for coordinate in window.coor:
        distances = np.linalg.norm(candidates.coor - np.asarray(coordinate), axis=1)
        nearby |= distances < distance
    return candidates.copy().get_selected_structure(
        flex_array.bool(nearby).iselection()
    ).with_symmetry(structure.crystal_symmetry)


def extract_window_sidechains(structure, window):
    """Return deposited-A window atoms other than N/CA/C/O.

    The expanded A'' window objective has torsions only for the backbone.  Its
    fixed sidechains must therefore be removed from the experimental residual
    just like external neighbours: retaining their density would charge an
    unchangeable A-sidechain mismatch to the backbone fit.  This is purposely
    separate from qFit's central-residue neighbour extraction, whose scope is
    correct for native qFit.
    """
    window_ids = {
        (str(residue.chain[0]), int(residue.resi[0]))
        for residue in window.residues
    }
    selected = np.asarray([
        ((str(chain), int(resi)) in window_ids) and str(atom) not in BACKBONE_NAMES
        for chain, resi, atom in zip(structure.chain, structure.resi, structure.name)
    ], dtype=bool)
    return structure.copy().get_selected_structure(
        flex_array.bool(selected).iselection()
    ).with_symmetry(structure.crystal_symmetry)


def rama_category(window, index: int, omega: np.ndarray) -> str:
    residue = window.residues[index]
    name = residue.resn[0]
    if name == "GLY":
        return "glycine"
    if name == "PRO":
        # The peptide before an internal PRO is omega[index - 1].
        return "cis-proline" if abs(float(omega[index - 1])) < 30.0 else "trans-proline"
    if index + 1 < len(window.residues) and window.residues[index + 1].resn[0] == "PRO":
        return "pre-proline"
    if name in {"ILE", "VAL"}:
        return "isoleucine or valine"
    return "general"


class SequentialBackbonePOC:
    def __init__(self, pdb_id: str, chain: str, resnum: int, output: Path, fd_step_deg: float,
                 max_step_deg: float, rama_weight: float, residual_scale_mode: str = "none",
                 renderer_backend: str = "torch", map_scaler_structure: str = "a_only",
                 mask_scope: str = "central", device: str = "auto",
                 density_atom_scope: str = "backbone", start_pdb: str | Path | None = None,
                 b_factor_mode: str | None = None,
                 mask_indices_cache: str | Path | None = None,
                 verify_mask_cache: bool = True,
                 source_pdb: str | Path | None = None,
                 mtz_path: str | Path | None = None):
        if source_pdb is None:
            path, split = source_path(pdb_id)
        else:
            path = str(Path(source_pdb))
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            split = "explicit"
        self.output = output
        self.pdb_id, self.chain, self.resnum, self.split = pdb_id, chain, resnum, split
        self.truth_path = Path(path)
        self.start_path = None if start_pdb is None else Path(start_pdb)
        if self.start_path is not None and not self.start_path.is_file():
            raise FileNotFoundError(self.start_path)
        self.requested_b_factor_mode = b_factor_mode or (
            "deposited_A_B" if self.start_path is None else "single_conformer"
        )
        if self.requested_b_factor_mode not in {"deposited_A_B", "single_conformer", "oracle_deposited"}:
            raise ValueError(f"unknown b-factor mode: {self.requested_b_factor_mode}")
        self.fd_step_deg, self.max_step_deg, self.rama_weight = fd_step_deg, max_step_deg, rama_weight
        self.residual_scale_mode = residual_scale_mode
        if map_scaler_structure not in {"a_only", "full"}:
            raise ValueError(f"unknown map_scaler_structure: {map_scaler_structure}")
        self.map_scaler_structure = map_scaler_structure
        if renderer_backend not in {"torch", "cctbx"}:
            raise ValueError(f"unknown renderer backend: {renderer_backend}")
        self.renderer_backend = renderer_backend
        if mask_scope not in {"central", "window", "three"}:
            raise ValueError(f"unknown mask scope: {mask_scope}")
        if mask_scope in {"window", "three"} and renderer_backend != "torch":
            raise ValueError("window masks currently require the Torch renderer")
        self.mask_scope = mask_scope
        if density_atom_scope not in {"backbone", "all"}:
            raise ValueError(f"unknown density atom scope: {density_atom_scope}")
        if density_atom_scope == "all" and mask_scope != "window":
            raise ValueError("all-atom density scope is only defined for the seven-residue window")
        self.density_atom_scope = density_atom_scope
        if renderer_backend == "torch":
            import torch

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch_device = torch.device(device)
        else:
            self.torch_device = None
        residue_id = (resnum, "")
        truth_structure = Structure.fromfile(path)
        self.truth_a_structure = truth_structure.extract("altloc", ("", "A"))
        self.b_structure = truth_structure.extract("altloc", ("", "B"))
        if self.start_path is None:
            self.a_structure = self.truth_a_structure
            self.full_structure = truth_structure
        else:
            # The benchmark's geometry start is a separately refined,
            # single-conformer model.  Deposited A/B remain truth-only
            # references for scoring and occupancy metadata.
            self.a_structure = Structure.fromfile(str(self.start_path)).extract("altloc", ("", "A"))
            self.full_structure = Structure.fromfile(str(self.start_path))
        self.a_residue = self.a_structure[chain].conformers[0][residue_id]
        self.b_residue = self.b_structure[chain].conformers[0][residue_id]
        self.deposited_occupancies = np.array([
            float(np.median(self.truth_a_structure[chain].conformers[0][residue_id].q)),
            float(np.median(self.b_residue.q)),
        ])
        configured_mtz_root = os.environ.get("D1_MTZ_ROOT")
        mtz = Path(mtz_path) if mtz_path is not None else (
            Path(configured_mtz_root) / f"{pdb_id.lower()}.mtz"
            if configured_mtz_root
            else Path(f"/home/dev/qfit_unet_data/cache/{split}/mtz/{pdb_id}.mtz")
        )
        if not mtz.exists():
            raise FileNotFoundError(mtz)
        self.mtz_path = mtz
        xmap, self.resolution, self.n_reflections, self.map_source = make_map(mtz)
        radius = 0.5 + self.resolution / 3.0
        scale, offset = MapScaler(xmap).scale(
            self.full_structure if map_scaler_structure == "full" else self.a_structure,
            radius=radius, transformer="cctbx"
        )
        self.map_scale, self.map_offset = float(scale), float(offset)
        options = QFitOptions()
        options.qp_solver = options.miqp_solver = "CVXPYSolver"
        # Native qFit is central-residue fitting, so its neighbour subtraction
        # correctly excludes only that residue.  Expanded A'' objectives render
        # the entire seven-residue window and must instead retain every window
        # atom in their target; see _subtract_window_neighbors below.
        options.subtract = self.mask_scope == "central"
        # Adopt the intercept-fitted occupancy convention: disable qFit's
        # fixed bulk-solvent floor for target and calculated model.
        options.bulk_solvent_level = 0.0
        qfit_structure = self.full_structure if map_scaler_structure == "full" else self.a_structure
        self.qfit = QFitRotamericResidue(self.a_residue, qfit_structure, xmap, options)
        index = self.qfit.segment.find(self.qfit.residue.id)
        required = self.qfit.options.neighbor_residues_required
        if index < required or index + required >= len(self.qfit.segment):
            raise RuntimeError("7-residue strict window unavailable for requested PoC site")
        self.window = self.qfit.segment[index - required:index + required + 1]
        if len(self.window.residues) != 7:
            raise RuntimeError("expected a seven-residue qFit window")
        self.subtraction_scope = "central_residue" if options.subtract else "window_plus_A_sidechains"
        self.subtracted_atom_count = None
        self.subtracted_window_sidechain_atom_count = 0
        if not options.subtract:
            # Target construction is independent of the prospective start.
            # With a published qFit start, ``qfit_structure`` contains qFit's
            # alternate coordinates; subtract the deposited environment so
            # the experiment does not change its density target merely by
            # changing the initialization model.
            self._subtract_window_neighbors(truth_structure)
            if density_atom_scope == "backbone":
                # The B sidechain can occupy different voxels, so this A-position
                # subtraction is necessarily an approximation.  It is still the
                # only consistent residual for a backbone-only model: neither
                # slot has chi parameters with which to explain sidechain density.
                self._subtract_window_sidechains(self.truth_a_structure)
        self.initial_window = self.window.coor.copy()
        self.central = self.window.residues[3]
        self.b_factors_a = np.asarray(self.window.b, dtype=float).copy()
        self.b_factors_b = self._deposited_b_model_b_factors()
        self.central_indices = atom_window_indices(self.window, self.central, tuple(self.central.name.tolist()))
        self.central_backbone_indices = atom_window_indices(self.window, self.central, BACKBONE_NAMES)
        self.a_central = self.a_residue.coor.copy()
        self.b_central = reordered_coordinates(self.a_residue, self.b_residue)
        self.b_backbone = backbone_coordinates(self.b_residue)
        self.a_backbone = backbone_coordinates(
            self.truth_a_structure[chain].conformers[0][residue_id]
        )
        self.b_factors = self.a_residue.b.copy()  # qFit backbone sampler retains the input B array.
        if self.mask_scope == "central":
            mask_coordinates = [self.a_central] if self.start_path is not None else [self.a_central, self.b_central]
            self.model_atom_indices = np.asarray(self.central_indices, dtype=np.int64)
            self.b_factors = np.asarray(self.a_residue.b, dtype=float).copy()
            elements = list(self.qfit._transformer.structure.e)  # pylint: disable=protected-access
        elif self.mask_scope == "three":
            active_residues = self.window.residues[2:5]
            window_selection = np.asarray(self.window.selection, dtype=np.int64)
            active_indices = np.asarray(
                [int(np.searchsorted(window_selection, index))
                 for residue in active_residues for index in residue.selection],
                dtype=np.int64,
            )
            active_backbone_indices = np.asarray(
                [index for residue in active_residues
                 for index in atom_window_indices(self.window, residue, BACKBONE_NAMES)],
                dtype=np.int64,
            )
            mask_coordinates = ([self.initial_window[active_backbone_indices]] if self.start_path is not None else [
                self.initial_window[active_backbone_indices], self.window_for_deposited_b()[active_backbone_indices]
            ])
            self.model_atom_indices = active_indices
            self.b_factors = self.b_factors_a[active_indices].copy()
            elements = [self.window.e[index] for index in active_indices]
        else:
            if density_atom_scope == "backbone":
                # Score and render only N/CA/C/O in the complete seven-residue
                # window.  Frozen sidechains have been subtracted at deposited-A
                # positions from the target above; retaining them in rho_calc
                # would charge their unparameterised mismatch to backbone torsions.
                active_indices = window_backbone_indices(self.window)
            else:
                # Diagnostic reconstruction of the previous all-atom scope:
                # both residual and rho_calc include the frozen sidechains.
                # This is not an A'' production objective because chi angles
                # remain unparameterised.
                active_indices = np.arange(len(self.window.coor), dtype=np.int64)
            mask_coordinates = ([self.initial_window[active_indices]] if self.start_path is not None else [
                self.initial_window[active_indices], self.window_for_deposited_b()[active_indices]
            ])
            self.model_atom_indices = active_indices
            self.b_factors = self.b_factors_a[active_indices].copy()
            elements = [self.window.e[index] for index in active_indices]
        # A'' renders two independent slots.  The deposited B factors are
        # therefore slot-specific: applying A's vector to both slots sharpens
        # or broadens B incorrectly whenever the deposited altlocs differ.
        # Keep the vectors in model-atom order, including the central/three
        # residue sub-selections used by the smaller diagnostic masks.
        self.b_factors_a_model = self.b_factors_a[self.model_atom_indices].copy()
        self.oracle_b_factors_b_model = self.b_factors_b[self.model_atom_indices].copy()
        if self.requested_b_factor_mode in {"deposited_A_B", "oracle_deposited"}:
            self.b_factors_b_model = self.oracle_b_factors_b_model.copy()
            self.b_factor_mode = "deposited_A_for_slot1_deposited_B_for_slot2"
        else:
            # Prospective fitting has one B-factor vector: the neutral
            # single-conformer start.  The deposited B vector remains retained
            # only as an explicitly labelled oracle bound.
            self.b_factors_b_model = self.b_factors_a_model.copy()
            self.b_factor_mode = "single_conformer_start_for_both_slots"
        self.b_factors = self.b_factors_a_model.copy()
        self.mask_cache_report = {
            "cache_path": None,
            "used_cached_mask": False,
            "verified_against_recomputed": False,
        }
        if mask_indices_cache is not None:
            cache_path = Path(mask_indices_cache)
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            with np.load(cache_path, allow_pickle=False) as cached_npz:
                if "mask_indices" not in cached_npz:
                    raise KeyError(f"{cache_path} does not contain mask_indices")
                cached_indices = np.asarray(cached_npz["mask_indices"], dtype=np.int64)
            if cached_indices.ndim != 2 or cached_indices.shape[1] != 3:
                raise ValueError(f"mask_indices must have shape [N,3], got {cached_indices.shape}")
            shape = np.asarray(self.qfit.xmap.array.shape, dtype=np.int64)
            if np.any(cached_indices < 0) or np.any(cached_indices >= shape[None, :]):
                raise ValueError(f"cached mask indices exceed map shape {tuple(shape)}")
            cached_mask = np.zeros(tuple(shape), dtype=bool)
            cached_mask[tuple(cached_indices.T)] = True
            same = None
            recomputed_voxels = None
            if verify_mask_cache:
                recomputed_mask = (
                    self.qfit._transformer.get_conformers_mask(  # pylint: disable=protected-access
                        mask_coordinates, self.qfit._rmask  # pylint: disable=protected-access
                    )
                    if self.mask_scope == "central" else
                    full_window_mask(self.qfit._transformer, self.qfit.xmap,
                                     mask_coordinates, self.qfit._rmask)  # pylint: disable=protected-access
                )
                same = bool(np.array_equal(cached_mask, recomputed_mask))
                recomputed_voxels = int(recomputed_mask.sum())
                if not same:
                    raise ValueError(
                        f"cached mask mismatch for {cache_path}: "
                        f"cached={int(cached_mask.sum())}, recomputed={recomputed_voxels}"
                    )
            self.mask = cached_mask
            self.mask_cache_report = {
                "cache_path": str(cache_path),
                "used_cached_mask": True,
                "verified_against_recomputed": bool(same) if same is not None else False,
                "cached_voxels": int(cached_mask.sum()),
                "recomputed_voxels": recomputed_voxels,
                "indices_exact": bool(same) if same is not None else None,
                "verification_skipped": not verify_mask_cache,
            }
        else:
            recomputed_mask = (
                self.qfit._transformer.get_conformers_mask(  # pylint: disable=protected-access
                    mask_coordinates, self.qfit._rmask  # pylint: disable=protected-access
                )
                if self.mask_scope == "central" else
                full_window_mask(self.qfit._transformer, self.qfit.xmap,
                                 mask_coordinates, self.qfit._rmask)  # pylint: disable=protected-access
            )
            self.mask = recomputed_mask
        self._renderer_grid = None
        self._renderer_u_base = None
        self._renderer_b_factors = None
        self._renderer_coefficients = None
        self._renderer_cell = None
        self._renderer_fractional_wrap_offsets = None
        self._renderer_atom_indices = self.model_atom_indices.copy()
        if self.renderer_backend == "torch":
            # xmap.array is indexed [z, y, x], while fractional coordinates
            # and the CCTBX unit-cell axes are ordered [x, y, z].  Construct
            # the exact masked grid once and retain it for every candidate.
            import torch

            mask_indices = np.argwhere(self.mask)
            n_real = np.asarray(self.qfit.xmap.n_real(), dtype=float)
            fractional = mask_indices[:, [2, 1, 0]] / n_real[None, :]
            unit_cell = self.qfit._transformer.structure.to_xray_structure().unit_cell()  # pylint: disable=protected-access
            orthogonalization = np.asarray(
                unit_cell.orthogonalization_matrix(), dtype=float
            ).reshape(3, 3)
            grid_cart = fractional @ orthogonalization.T
            self._renderer_grid = torch.as_tensor(
                grid_cart, dtype=torch.float64, device=self.torch_device
            )
            self._renderer_cell = torch.as_tensor(
                orthogonalization, dtype=torch.float64, device=self.torch_device
            )
            self._set_renderer_reference_wrap_offsets(
                self.initial_window[self._renderer_atom_indices]
            )
            self._renderer_u_base = float(
                ext.calc_u_base(d_min=self.resolution, grid_resolution_factor=0.25)
            )
            self._renderer_b_factors = torch.as_tensor(
                self.b_factors, dtype=torch.float64, device=self.torch_device
            )
            self._renderer_coefficients = coefficients_for_elements(
                elements, dtype=torch.float64, device=self.torch_device
            )
        self.target = self.qfit.xmap.array[self.mask].astype(float, copy=True)
        self.target_before_residual_scaling = self.target.copy()
        self.residual_target_multiplier = 1.0
        self.residual_scale_diagnostic = {"mode": residual_scale_mode}
        if residual_scale_mode == "deposited_ab":
            # MapScaler calibrates the full map before qFit removes neighbours.
            # This known A/B control records the needed residual-map amplitude;
            # it is intentionally opt-in and is not prospective recovery.
            deposited_a = self.model_density(self.initial_window, slot=0)
            deposited_b = self.model_density(self.window_for_deposited_b(), slot=1)
            deposited_ensemble = np.dot(
                self.deposited_occupancies, np.vstack([deposited_a, deposited_b])
            )
            residual_to_model = float(np.dot(self.target, deposited_ensemble) /
                                      np.dot(deposited_ensemble, deposited_ensemble))
            if not np.isfinite(residual_to_model) or residual_to_model <= EPSILON:
                raise RuntimeError(f"invalid residual-map scale: {residual_to_model}")
            self.residual_target_multiplier = 1.0 / residual_to_model
            self.target *= self.residual_target_multiplier
            self.residual_scale_diagnostic.update({
                "fit": "through-origin least squares: residual_map = factor * deposited_A_B_model",
                "residual_map_to_model_factor": residual_to_model,
                "model_to_residual_map_factor": self.residual_target_multiplier,
                "deposited_occupancies_A_B": self.deposited_occupancies.tolist(),
            })
        elif residual_scale_mode != "none":
            raise ValueError(f"unknown residual scale mode: {residual_scale_mode}")
        self.rama_eval = ramalyze.ramachandran_eval.RamachandranEval()
        self.window.coor = self.initial_window.copy()

    def _subtract_window_neighbors(self, structure) -> None:
        """Apply A''-specific neighbour subtraction after the window is known."""
        subtract_structure = extract_window_neighbors(
            structure, self.window, self.qfit.options.padding
        )
        self.subtracted_atom_count = int(subtract_structure.natoms)
        self._subtract_density_structure(subtract_structure)

    def _subtract_window_sidechains(self, structure) -> None:
        """Subtract immutable deposited-A sidechains from the A'' residual."""
        subtract_structure = extract_window_sidechains(structure, self.window)
        self.subtracted_window_sidechain_atom_count = int(subtract_structure.natoms)
        before = self.qfit.xmap.array.copy()
        self._subtract_density_structure(subtract_structure)
        self.start_sidechain_density_full_map = before - self.qfit.xmap.array

    def _subtract_density_structure(self, subtract_structure) -> None:
        """Subtract one explicit atom set using qFit's CCTBX transformer."""
        if subtract_structure.natoms == 0:
            return
        subtransformer = self.qfit._get_transformer(  # pylint: disable=protected-access
            subtract_structure,
            self.qfit._xmap_model2,  # pylint: disable=protected-access
            smax=self.qfit._smax,  # pylint: disable=protected-access
            smin=self.qfit._smin,  # pylint: disable=protected-access
            simple=self.qfit._simple,  # pylint: disable=protected-access
            em=self.qfit.options.em,
        )
        subtransformer.initialize()
        subtransformer.reset(full=True)
        subtransformer.density()
        # The A'' intercept convention disables qFit's fixed solvent floor.
        # Retain the guard so this helper remains internally consistent if a
        # diagnostic explicitly enables a nonzero floor.
        if not self.qfit.options.em and self.qfit.options.bulk_solvent_level > 0.0:
            np.maximum(
                subtransformer.xmap.array,
                self.qfit.options.bulk_solvent_level,
                out=subtransformer.xmap.array,
            )
        self.qfit._subtransformer = subtransformer  # pylint: disable=protected-access
        self.qfit.xmap.array -= subtransformer.xmap.array

    def _render_structure_full_map(self, structure) -> np.ndarray:
        """Render a structure on the qFit map grid without changing the target."""
        transformer = self.qfit._get_transformer(  # pylint: disable=protected-access
            structure, self.qfit._xmap_model2,  # pylint: disable=protected-access
            smax=self.qfit._smax, smin=self.qfit._smin,  # pylint: disable=protected-access
            simple=self.qfit._simple,  # pylint: disable=protected-access
            em=self.qfit.options.em,
        )
        transformer.initialize()
        transformer.reset(full=True)
        transformer.density()
        return transformer.xmap.array.copy()

    def start_sidechain_subtraction_mismatch(self) -> dict[str, float | int | str]:
        """Quantify start-sidechain versus deposited-A subtraction at A geometry."""
        if self.start_path is None:
            return {"status": "not_applicable", "reason": "oracle/deposited-A input"}
        deposited = extract_window_sidechains(self.truth_a_structure, self.window)
        deposited_density = self._render_structure_full_map(deposited)
        start_density = np.asarray(self.start_sidechain_density_full_map)
        delta = (start_density - deposited_density)[self.mask]
        deposited_target = self.qfit.xmap.array[self.mask] + start_density[self.mask] - deposited_density[self.mask]
        return {
            "status": "complete",
            "definition": "RMS of residual-map change caused by using start sidechains instead of deposited-A sidechains, evaluated on the benchmark mask",
            "rms_density_units": float(np.sqrt(np.mean(delta * delta))),
            "mean_abs_density_units": float(np.mean(np.abs(delta))),
            "relative_to_deposited_A_sidechain_rms": float(
                np.sqrt(np.mean(delta * delta)) /
                max(float(np.sqrt(np.mean(deposited_density[self.mask] ** 2))), 1e-12)
            ),
            "mask_voxels": int(self.mask.sum()),
            "start_sidechain_atoms": int(self.subtracted_window_sidechain_atom_count),
            "deposited_A_sidechain_atoms": int(deposited.natoms),
            "deposited_A_fixed_geometry_target_mean": float(np.mean(deposited_target)),
        }

    def central_coordinates(self, window_coordinates: np.ndarray) -> np.ndarray:
        return np.asarray(window_coordinates[self.central_indices], dtype=float)

    def slot_b_factors(self, slot: int, b_offset: float = 0.0) -> np.ndarray:
        """Return the deposited slot-specific B vector plus a global offset."""
        if slot not in (0, 1):
            raise ValueError(f"slot must be 0 (A) or 1 (B), got {slot}")
        base = self.b_factors_a_model if slot == 0 else self.b_factors_b_model
        values = base + float(b_offset)
        if np.any(values <= 0.0):
            raise ValueError("global B offset makes one or more B factors non-positive")
        return values

    def model_density(self, window_coordinates: np.ndarray, *, slot: int = 0,
                      b_offset: float = 0.0) -> np.ndarray:
        return self.model_density_batch(
            np.asarray(window_coordinates), slots=np.asarray([slot]), b_offset=b_offset
        )[0]

    def model_density_batch(self, window_coordinates: np.ndarray, *,
                            slots: np.ndarray | None = None,
                            b_offset: float = 0.0) -> np.ndarray:
        """Render one or more slot-labelled windows on the fixed mask."""
        windows = np.asarray(window_coordinates, dtype=float)
        if windows.ndim == 2:
            windows = windows[None, ...]
        if windows.ndim != 3:
            raise ValueError("window_coordinates must have shape [atoms, 3] or [batch, atoms, 3]")
        if slots is None:
            slots = np.zeros(len(windows), dtype=int)
        slots = np.asarray(slots, dtype=int)
        if slots.shape != (len(windows),) or np.any((slots < 0) | (slots > 1)):
            raise ValueError("slots must contain one 0/1 label per window")
        b_factors = np.vstack([self.slot_b_factors(int(slot), b_offset) for slot in slots])
        if self.renderer_backend == "torch":
            import torch

            with torch.no_grad():
                density = self.model_density_torch(torch.as_tensor(
                    windows[:, self.model_atom_indices], dtype=torch.float64,
                    device=self.torch_device,
                ), b_factors=b_factors)
            values = density.cpu().numpy()
            return (np.maximum(values, self.qfit.options.bulk_solvent_level)
                    if self.qfit.options.bulk_solvent_level > 0.0 else values)
        densities = [density[self.mask].astype(float, copy=False) for density in
                     self.qfit._transformer.get_conformers_densities(  # pylint: disable=protected-access
            list(windows[:, self.model_atom_indices]),
            list(b_factors),
        )]
        values = np.asarray(densities)
        return (np.maximum(values, self.qfit.options.bulk_solvent_level)
                if self.qfit.options.bulk_solvent_level > 0.0 else values)

    def model_density_with_b(self, window_coordinates: np.ndarray, b_factors: np.ndarray) -> np.ndarray:
        """Render a window with explicitly selected atom-wise B factors."""
        import torch

        if self.renderer_backend != "torch":
            raise RuntimeError("model_density_with_b currently requires renderer_backend='torch'")
        coordinates = torch.as_tensor(
            np.asarray(window_coordinates)[self.model_atom_indices],
            dtype=torch.float64, device=self.torch_device,
        )
        with torch.no_grad():
            return self.model_density_torch(coordinates, b_factors=b_factors).cpu().numpy()

    def profile_affine_b_offset(self, target: np.ndarray, windows: list[np.ndarray],
                                slots: list[int], max_total: float = 1.0,
                                fixed_weights: np.ndarray | None = None,
                                voxel_indices: np.ndarray | None = None) -> dict[str, object]:
        """Profile a site-wide B offset with occupancies and intercept.

        ``delta_b`` is one physically interpretable map/model-width parameter,
        shared by every slot.  At each trial offset the affine occupancy QP
        profiles occupancies and the unconstrained intercept.  This keeps it
        out of the torsion gradient while making it part of the density model.
        """
        if len(windows) != len(slots) or not windows:
            raise ValueError("windows and slots must be non-empty and have the same length")
        target = np.asarray(target, dtype=float)
        indices = (None if voxel_indices is None
                   else np.asarray(voxel_indices, dtype=int))
        if indices is not None and target.shape != (len(indices),):
            raise ValueError("target must be restricted to voxel_indices before B-offset profiling")
        fixed = None if fixed_weights is None else np.asarray(fixed_weights, dtype=float)
        if fixed is not None:
            if fixed.shape != (len(windows),) or np.any(fixed < 0.0) or fixed.sum() > max_total + EPSILON:
                raise ValueError("fixed occupancy weights violate the affine-QP constraints")
        minimum_b = min(float(self.slot_b_factors(slot).min()) for slot in slots)
        lower = -minimum_b + 1e-6
        cache: dict[float, dict[str, object]] = {}

        def evaluate_offset(offset: float) -> dict[str, object]:
            key = round(float(offset), 8)
            if key not in cache:
                models = np.vstack([
                    self.model_density(window, slot=slot, b_offset=key)
                    for window, slot in zip(windows, slots)
                ])
                fitting_models = models if indices is None else models[:, indices]
                if fixed is None:
                    weights, intercept, rss = solve_affine_qp(
                        target, fitting_models, max_total=max_total
                    )
                else:
                    weights = fixed.copy()
                    intercept = float(np.mean(target - weights @ fitting_models))
                    rss = float(np.square(target - weights @ fitting_models - intercept).sum())
                cache[key] = {
                    "b_offset": key, "models": models, "weights": weights,
                    "intercept": intercept, "rss": rss,
                }
            return cache[key]

        # Establish a data-driven bracket on each side of zero by doubling a
        # 5 A^2 probe.  The limit is only a numerical safety stop; a minimum
        # still descending there is reported as unbracketed rather than used.
        probes = {0.0, lower}
        positive = 5.0
        while positive <= 1024.0:
            probes.add(positive)
            positive *= 2.0
        negative = -5.0
        while negative > lower:
            probes.add(negative)
            negative *= 2.0
        grid = sorted(probes)
        grid_fits = [evaluate_offset(offset) for offset in grid]
        best_index = int(np.argmin([float(fit["rss"]) for fit in grid_fits]))
        if best_index in (0, len(grid) - 1):
            # A boundary minimum is a measured, converged profile result, not
            # a harness crash.  Preserve it with an explicit non-bracketed
            # flag so the screen can apply the |dB| criterion only when the
            # fit is genuinely available.
            answer = grid_fits[best_index].copy()
            answer["profile_interval_A2"] = [float(grid[best_index]), float(grid[best_index])]
            answer["profile_density_renders"] = int(len(cache) * len(windows))
            answer["profile_converged"] = True
            answer["profile_bracketed"] = False
            return answer
        left, right = grid[best_index - 1], grid[best_index + 1]
        golden = (math.sqrt(5.0) - 1.0) / 2.0
        first = right - golden * (right - left)
        second = left + golden * (right - left)
        first_fit, second_fit = evaluate_offset(first), evaluate_offset(second)
        for _ in range(18):
            if float(first_fit["rss"]) <= float(second_fit["rss"]):
                right, second, second_fit = second, first, first_fit
                first = right - golden * (right - left)
                first_fit = evaluate_offset(first)
            else:
                left, first, first_fit = first, second, second_fit
                second = left + golden * (right - left)
                second_fit = evaluate_offset(second)
        answer = evaluate_offset((left + right) / 2.0).copy()
        answer["profile_interval_A2"] = [float(left), float(right)]
        answer["profile_density_renders"] = int(len(cache) * len(windows))
        answer["profile_converged"] = True
        answer["profile_bracketed"] = True
        return answer

    def _set_renderer_reference_wrap_offsets(self, reference_coordinates) -> None:
        """Freeze each atom's periodic image at a reference geometry.

        A live ``fractional - floor(fractional)`` is discontinuous when an
        atom lies on a unit-cell face: an infinitesimal Cartesian perturbation
        can move it by a complete unit cell in the differentiable graph.  The
        periodic image is a discrete rendering choice, not an optimisable
        coordinate.  Choose it once from the stage-zero geometry and retain
        that branch for all subsequent derivatives.
        """
        import torch

        if self._renderer_cell is None:
            raise RuntimeError("renderer cell must be installed before wrap offsets")
        reference = torch.as_tensor(
            reference_coordinates, dtype=torch.float64, device=self.torch_device,
        )
        if reference.ndim != 2 or reference.shape[-1] != 3:
            raise ValueError("reference_coordinates must have shape [atoms, 3]")
        with torch.no_grad():
            fractional = torch.linalg.solve(
                self._renderer_cell, reference.transpose(-1, -2)
            ).transpose(-1, -2)
            self._renderer_fractional_wrap_offsets = torch.floor(fractional)

    def model_density_torch(self, coordinates, b_factors=None):
        """Differentiable density for the configured atom set on the fixed mask."""
        if self.renderer_backend != "torch":
            raise RuntimeError("model_density_torch requires renderer_backend='torch'")
        import torch

        coordinates = coordinates.to(dtype=torch.float64, device=self.torch_device)
        if coordinates.ndim == 2:
            coordinates = coordinates.unsqueeze(0)
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape [atoms, 3] or [batch, atoms, 3]")
        # Keep atoms in the same unit cell as the map and include neighboring
        # periodic images.  The latter matters only when a candidate crosses
        # a cell boundary, but costs nothing to express in the differentiable
        # graph and matches the periodic map convention.
        cell = self._renderer_cell
        fractional = torch.linalg.solve(cell, coordinates.transpose(-1, -2)).transpose(-1, -2)
        if self._renderer_fractional_wrap_offsets is None:
            raise RuntimeError("renderer reference wrap offsets were not initialized")
        wrap_offsets = self._renderer_fractional_wrap_offsets.to(device=coordinates.device)
        if wrap_offsets.shape != coordinates.shape[-2:]:
            raise ValueError(
                "renderer wrap-offset atom count does not match coordinates: "
                f"{tuple(wrap_offsets.shape)} != {tuple(coordinates.shape[-2:])}"
            )
        fractional = fractional - wrap_offsets.unsqueeze(0)
        base_cart = torch.matmul(fractional, cell.T)
        if self.mask_scope in {"window", "three"}:
            # The diagnostic full-window mask is made in the extracted qFit
            # box, whose scored voxels and all seven-residue coordinates are
            # inside one periodic image.  Avoid expanding 52 atoms to 1,404
            # images in every autodiff graph; the central-image equivalence is
            # checked at the deposited geometry before the rerun is accepted.
            image_tuples = [(0, 0, 0)]
        else:
            image_tuples = [(i, j, k) for i in (-1, 0, 1)
                            for j in (-1, 0, 1) for k in (-1, 0, 1)]
        shifts = torch.tensor(image_tuples, dtype=torch.float64, device=coordinates.device)
        image_shifts = torch.matmul(shifts, cell.T)
        n_images = len(image_tuples)
        atom_xyz = (base_cart.unsqueeze(-2) + image_shifts.view(1, 1, n_images, 3)).reshape(
            coordinates.shape[0], -1, 3
        )
        if b_factors is None:
            b_factors = self._renderer_b_factors
        else:
            b_factors = torch.as_tensor(b_factors, dtype=torch.float64, device=coordinates.device)
        if b_factors.ndim == 1:
            b_factors = b_factors.unsqueeze(0)
        if (b_factors.ndim != 2 or b_factors.shape[1] != coordinates.shape[1]
                or b_factors.shape[0] not in (1, coordinates.shape[0])):
            raise ValueError("b_factors must have shape [atoms] or [batch, atoms]")
        coefficients = self._renderer_coefficients.to(device=coordinates.device)
        b_factors = b_factors.repeat_interleave(n_images, dim=1)
        coefficients = coefficients.repeat_interleave(n_images, dim=0).unsqueeze(0).expand(
            coordinates.shape[0], -1, -1, -1
        )
        return render_cctbx_density(
            self._renderer_grid.to(device=coordinates.device), atom_xyz, b_factors,
            coefficients, u_base=self._renderer_u_base,
            exp_table_one_over_step_size=0.0,
            voxel_chunk=(1024 if self.mask_scope in {"window", "three"} else 4096),
        )

    def apply_increment(self, starting_coordinates: np.ndarray, torsions_deg: np.ndarray) -> np.ndarray:
        original = self.window.coor.copy()
        try:
            self.window.coor = starting_coordinates.copy()
            BackboneRotator(self.window)(torsions_deg)
            return self.window.coor.copy()
        finally:
            self.window.coor = original

    def rama_penalty(self, window_coordinates: np.ndarray) -> tuple[float, list[float]]:
        original = self.window.coor.copy()
        try:
            self.window.coor = window_coordinates
            phi_psi, omega = dihedrals(self.window)
            scores = []
            # End residues lack one defining atom; qFit nevertheless exposes
            # their torsions, while the physical Ramachandran prior applies to
            # the five internal residues only.
            for index in range(1, len(self.window.residues) - 1):
                phi, psi = float(phi_psi[2 * index]), float(phi_psi[2 * index + 1])
                score = float(self.rama_eval.evaluate(rama_category(self.window, index, omega), [phi, psi]))
                scores.append(score)
            return float(np.mean([-math.log(max(score, 1e-6)) for score in scores])), scores
        finally:
            self.window.coor = original

    @staticmethod
    def bounded_nnls(target: np.ndarray, model: np.ndarray, capacity: float) -> tuple[float, float, float]:
        upper_bound = max(0.0, capacity)
        # SciPy rejects equal lower/upper bounds.  In a sequential fit a first
        # slot may legitimately consume all available occupancy; the unique
        # feasible second-slot solution is then zero occupancy.
        if upper_bound <= EPSILON:
            intercept = float(np.mean(target))
            return 0.0, intercept, float(np.square(target - intercept).sum())
        weights, intercept, rss = solve_affine_qp(
            target, np.asarray(model)[None, :], upper_bounds=np.array([upper_bound])
        )
        return float(weights[0]), float(intercept), float(rss)

    def closure_project(self, window_coordinates: np.ndarray, gradient: np.ndarray) -> tuple[np.ndarray, int, float]:
        original = self.window.coor.copy()
        try:
            self.window.coor = window_coordinates
            selection = np.sort(self.window.select("name", ("N", "CA", "C")))
            jacobian = compute_jacobian(self.window.get_xyz(selection))
        finally:
            self.window.coor = original
        basis = null_space(jacobian)
        projected = basis @ (basis.T @ gradient)
        return projected, int(basis.shape[1]), float(np.linalg.norm(jacobian @ projected))

    def evaluate(self, moving_window: np.ndarray, target: np.ndarray, capacity: float,
                 normalizer: float, slot: int) -> dict[str, object]:
        affine = self.profile_affine_b_offset(target, [moving_window], [slot], capacity)
        model = np.asarray(affine["models"])[0]
        occupancy = float(np.asarray(affine["weights"])[0])
        intercept = float(affine["intercept"])
        density_rss = float(affine["rss"])
        rama, scores = self.rama_penalty(moving_window)
        return {
            "objective": density_rss / normalizer + self.rama_weight * rama,
            "density_rss": density_rss, "occupancy": occupancy, "intercept": intercept,
            "b_offset_A2": float(affine["b_offset"]),
            "rama_penalty": rama, "rama_scores": scores, "model": model,
        }

    def optimize_slot(self, stage: str, initial: np.ndarray, target: np.ndarray, capacity: float,
                      slot: int,
                      steps: int, trajectory: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, object]]:
        current = initial.copy()
        baseline = self.evaluate(current, target, capacity, normalizer=1.0, slot=slot)
        normalizer = max(float(baseline["density_rss"]), EPSILON)
        state = self.evaluate(current, target, capacity, normalizer, slot)
        consecutive_stalls = 0
        for step in range(1, steps + 1):
            gradient = np.zeros(14, dtype=float)
            for column in range(14):
                delta = np.zeros(14, dtype=float)
                delta[column] = self.fd_step_deg
                plus = self.evaluate(self.apply_increment(current, delta), target, capacity, normalizer, slot)
                minus = self.evaluate(self.apply_increment(current, -delta), target, capacity, normalizer, slot)
                gradient[column] = (float(plus["objective"]) - float(minus["objective"])) / (2.0 * self.fd_step_deg)
            projected, null_dimension, projection_residual = self.closure_project(current, gradient)
            max_component = float(np.max(np.abs(projected)))
            if max_component <= 1e-12:
                break
            # Normalize in torsion space before line search.  The previous
            # min(1, max_step/max_component) form left a small finite-
            # difference gradient virtually unscaled (0.016 degrees in the
            # smoke run), rather than taking the configured projected step.
            direction = -projected * (self.max_step_deg / max_component)
            chosen = state
            chosen_coordinates = current
            chosen_scale = 0.0
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = self.apply_increment(current, scale * direction)
                trial = self.evaluate(candidate, target, capacity, normalizer, slot)
                if float(trial["objective"]) < float(chosen["objective"]) - 1e-10:
                    chosen, chosen_coordinates, chosen_scale = trial, candidate, scale
                    break
            if chosen_scale == 0.0:
                consecutive_stalls += 1
            else:
                current, state, consecutive_stalls = chosen_coordinates, chosen, 0
            central_backbone = self.central_coordinates(current)[[self.central.name.tolist().index(name) for name in BACKBONE_NAMES]]
            trajectory.append({
                "stage": stage, "step": step, "objective": float(state["objective"]),
                "density_rss": float(state["density_rss"]), "occupancy": float(state["occupancy"]),
                "intercept": float(state["intercept"]), "b_offset_A2": float(state["b_offset_A2"]),
                "rama_penalty": float(state["rama_penalty"]), "rama_min_score": float(min(state["rama_scores"])),
                "gradient_norm": float(np.linalg.norm(gradient)), "projected_gradient_norm": float(np.linalg.norm(projected)),
                "null_dimension": null_dimension, "J_times_projected_gradient_norm": projection_residual,
                "accepted_scale": chosen_scale, "applied_max_torsion_deg": float(np.max(np.abs(chosen_scale * direction))),
                "slot_backbone_rmsd_to_A_A": rmsd(central_backbone, self.a_backbone),
                "slot_backbone_rmsd_to_B_A": rmsd(central_backbone, self.b_backbone),
            })
            self.checkpoint(stage, step, current, trajectory)
            if consecutive_stalls >= 5:
                break
        return current, state

    def checkpoint(self, stage: str, step: int, current: np.ndarray, trajectory: list[dict[str, object]]) -> None:
        atomic_npz(self.output / "checkpoint.npz", current_window=current, initial_window=self.initial_window)
        atomic_csv(self.output / "trajectory.csv", trajectory)
        atomic_json(self.output / "progress.json", {
            "status": "running", "stage": stage, "step": step, "trajectory_rows": len(trajectory),
        })

    def joint_qp(self, slot1: np.ndarray, slot2: np.ndarray) -> dict[str, object]:
        return self.profile_affine_b_offset(self.target, [slot1, slot2], [0, 1])

    def run(self, steps_per_stage: int) -> dict[str, object]:
        deposited_windows = [self.initial_window, self.window_for_deposited_b()]
        deposited_calibration = self.profile_affine_b_offset(
            self.target, deposited_windows, [0, 1], fixed_weights=self.deposited_occupancies
        )
        deposited_models = np.asarray(deposited_calibration["models"])
        deposited_weights, deposited_intercept, deposited_rss = solve_affine_qp(self.target, deposited_models)
        trajectory: list[dict[str, object]] = []
        slot1, stage1 = self.optimize_slot("slot1_fit", self.initial_window, self.target, 1.0, 0, steps_per_stage, trajectory)
        slot1_model = np.asarray(stage1["model"])
        frozen_occupancy, frozen_intercept, frozen_rss = (
            float(stage1["occupancy"]), float(stage1["intercept"]), float(stage1["density_rss"])
        )
        residual_target = self.target - frozen_occupancy * slot1_model - frozen_intercept
        slot2, stage2 = self.optimize_slot(
            "slot2_residual_fit", self.initial_window, residual_target,
            max(0.0, 1.0 - frozen_occupancy), 1, steps_per_stage, trajectory,
        )
        joint = self.joint_qp(slot1, slot2)
        joint_weights = np.asarray(joint["weights"])
        joint_intercept, joint_rss = float(joint["intercept"]), float(joint["rss"])
        slot1_backbone = self.central_backbone(slot1)
        slot2_backbone = self.central_backbone(slot2)
        result = {
            "status": "complete", "site": f"{self.pdb_id}_{self.chain}_ARG{self.resnum}",
            "map": {"source": self.map_source, "resolution_A": self.resolution, "n_reflections": self.n_reflections,
                    "qfit_map_scaler_factor": self.map_scale, "qfit_map_scaler_offset": self.map_offset,
            "qfit_neighbour_subtraction": True,
            "neighbour_subtraction_scope": self.subtraction_scope,
            "neighbour_subtracted_atom_count": self.subtracted_atom_count,
            "window_A_sidechain_subtracted_atom_count": self.subtracted_window_sidechain_atom_count,
            "mask_scope": self.mask_scope,
            "model_atom_count": int(len(self.model_atom_indices)),
            "density_atom_scope": (
                "backbone_N_CA_C_O" if self.mask_scope == "window" else "configured_mask_atom_set"
            ),
            "mask_voxels": int(self.mask.sum()),
                    "residual_scale": self.residual_scale_diagnostic},
            "parameterization": {"slots": 2, "phi_psi_parameters_per_slot": 14, "closure_null_dimension": 8,
                                 "density_renderer": self.renderer_backend,
                                 "occupancy_solver": "intercept-fitted bounded QP every objective evaluation; intercept-fitted QP final joint solve",
                                 "global_B_offset": "one site-wide B offset profiled with occupancies and intercept",
                                 "bulk_solvent_floor": 0.0,
                                 "sequential": "slot 1 fit then frozen geometry/occupancy; slot 2 fits residual",
                                 "ramachandran": "mmtbx RamachandranEval negative-log probability barrier from first step",
                                 "ramachandran_weight": self.rama_weight},
            "deposited_occupancies_A_B": self.deposited_occupancies.tolist(),
            "deposited_A_B_qp_occupancies": deposited_weights.tolist(),
            "deposited_A_B_qp_intercept": deposited_intercept,
            "deposited_A_B_qp_rss": deposited_rss,
            "deposited_A_B_fixed_occupancy_B_offset_A2": float(deposited_calibration["b_offset"]),
            "deposited_A_B_fixed_occupancy_B_offset_interval_A2": deposited_calibration["profile_interval_A2"],
            "slot1": {"steps": sum(row["stage"] == "slot1_fit" for row in trajectory), "frozen_occupancy": frozen_occupancy,
                      "intercept": frozen_intercept,
                      "rss": frozen_rss, "backbone_rmsd_to_A_A": rmsd(slot1_backbone, self.a_backbone),
                      "backbone_rmsd_to_B_A": rmsd(slot1_backbone, self.b_backbone)},
            "slot2": {"steps": sum(row["stage"] == "slot2_residual_fit" for row in trajectory), "residual_occupancy": float(stage2["occupancy"]),
                      "intercept": float(stage2["intercept"]),
                      "backbone_rmsd_to_A_A": rmsd(slot2_backbone, self.a_backbone),
                      "backbone_rmsd_to_B_A": rmsd(slot2_backbone, self.b_backbone),
                      "reaches_deposited_B_under_1A": rmsd(slot2_backbone, self.b_backbone) < 1.0},
            "final_joint_qp_occupancies_slot1_slot2": joint_weights.tolist(),
            "final_joint_qp_intercept": joint_intercept, "final_joint_qp_rss": joint_rss,
            "final_joint_qp_B_offset_A2": float(joint["b_offset"]),
            "final_joint_qp_B_offset_interval_A2": joint["profile_interval_A2"],
            "trajectory_rows": len(trajectory),
        }
        atomic_npz(self.output / "final_slots.npz", slot1_window=slot1, slot2_window=slot2, deposited_A_window=self.initial_window)
        atomic_csv(self.output / "trajectory.csv", trajectory)
        result["provenance"] = runner_provenance(
            self,
            self.truth_path,
            self.mtz_path,
            {"final_slots": self.output / "final_slots.npz"},
        )
        atomic_json(self.output / "result.json", result)
        atomic_json(self.output / "progress.json", {"status": "complete", "trajectory_rows": len(trajectory)})
        return result

    def central_backbone(self, window_coordinates: np.ndarray) -> np.ndarray:
        central = self.central_coordinates(window_coordinates)
        names = self.central.name.tolist()
        return np.asarray([central[names.index(name)] for name in BACKBONE_NAMES])

    def _deposited_b_model_b_factors(self) -> np.ndarray:
        """Return deposited-B B factors in full window-coordinate order."""
        b_chain = self.b_structure[self.chain].conformers[0]
        b_segment = next(segment for segment in b_chain.segments
                         if any(residue.id == self.b_residue.id for residue in segment.residues))
        b_by_key = {
            (residue.id, name): float(b_value)
            for residue in b_segment.residues
            for name, b_value in zip(residue.name.tolist(), residue.b)
        }
        a_chain = self.truth_a_structure[self.chain].conformers[0]
        a_by_key = {
            (residue.id, name): float(b_value)
            for segment in a_chain.segments
            for residue in segment.residues
            for name, b_value in zip(residue.name.tolist(), residue.b)
        }
        values = []
        fallback_keys = []
        for residue in self.window.residues:
            for name in residue.name.tolist():
                key = (residue.id, name)
                if key in b_by_key:
                    values.append(b_by_key[key])
                elif key in a_by_key:
                    # Flanking residues are often single-conformer in the
                    # deposited model.  Their A/B coordinates are still
                    # valid, but a deposited-B B factor does not exist.
                    # Keep the deposited-A value for that frozen atom rather
                    # than rejecting an otherwise eligible seven-residue
                    # window.
                    values.append(a_by_key[key])
                    fallback_keys.append(key)
                else:
                    raise RuntimeError(f"deposited B is missing B factor for {key}")
        self.deposited_b_fallback_keys = fallback_keys
        return np.asarray(values, dtype=float)

    def window_for_deposited_b(self) -> np.ndarray:
        """Return full deposited-B window for fixed-mask/QP calibration only."""
        b_chain = self.b_structure[self.chain].conformers[0]
        a_chain = self.truth_a_structure[self.chain].conformers[0]
        b_by_key = {
            (residue.id, atom): np.asarray(coor, dtype=float)
            for segment in b_chain.segments
            for residue in segment.residues
            for atom, coor in zip(residue.name.tolist(), residue.coor)
        }
        a_by_key = {
            (residue.id, atom): np.asarray(coor, dtype=float)
            for segment in a_chain.segments
            for residue in segment.residues
            for atom, coor in zip(residue.name.tolist(), residue.coor)
        }
        values = []
        fallback_keys = []
        for residue in self.window.residues:
            for atom in residue.name.tolist():
                key = (residue.id, atom)
                if key in b_by_key:
                    values.append(b_by_key[key])
                elif key in a_by_key:
                    # Single-conformer flanks have no independent deposited-B
                    # coordinates; use their deposited-A coordinates for the
                    # fixed seven-residue mask reference.
                    values.append(a_by_key[key])
                    fallback_keys.append(key)
                else:
                    raise RuntimeError(f"deposited B is missing coordinates for {key}")
        self.deposited_b_fallback_coordinate_keys = fallback_keys
        return np.asarray(values, dtype=float)

    def window_for_deposited_a(self) -> np.ndarray:
        """Return the deposited-A window, independent of the optimization start."""
        a_chain = self.truth_a_structure[self.chain].conformers[0]
        a_residue = a_chain[(self.resnum, "")]
        a_segment = next(segment for segment in a_chain.segments
                         if any(residue.id == a_residue.id for residue in segment.residues))
        index = a_segment.find(a_residue.id)
        window = a_segment[index - 3:index + 4]
        if len(window.residues) != 7 or [res.id for res in window.residues] != [res.id for res in self.window.residues]:
            raise RuntimeError("deposited-A strict window does not match the optimization window")
        a_by_key = {
            (residue.id, atom): np.asarray(coor, dtype=float)
            for residue in window.residues
            for atom, coor in zip(residue.name.tolist(), residue.coor)
        }
        return np.asarray([
            a_by_key[(residue.id, atom)]
            for residue in self.window.residues for atom in residue.name.tolist()
        ], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdb-id", default="7UTC")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--resnum", type=int, default=52)
    parser.add_argument("--steps-per-stage", type=int, default=60)
    parser.add_argument("--fd-step-deg", type=float, default=0.25)
    parser.add_argument("--max-step-deg", type=float, default=2.0)
    parser.add_argument("--rama-weight", type=float, default=0.05)
    parser.add_argument("--residual-scale", choices=("none", "deposited_ab"), default="none")
    parser.add_argument("--renderer-backend", choices=("torch", "cctbx"), default="torch",
                        help="Calculated-density backend (default: differentiable Torch renderer).")
    parser.add_argument("--mask-scope", choices=("central", "window"), default="central",
                        help="Mask/model atom scope; window is a diagnostic full-window objective.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "run_config.json", {
        **{key: value for key, value in vars(args).items() if key != "output"},
        "output": str(args.output),
    })
    experiment = SequentialBackbonePOC(
        args.pdb_id, args.chain, args.resnum, args.output, args.fd_step_deg,
        args.max_step_deg, args.rama_weight, args.residual_scale,
        args.renderer_backend, mask_scope=args.mask_scope,
    )
    result = experiment.run(args.steps_per_stage)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
