#!/usr/bin/env python3
"""A′ sequential two-slot real-map PoC with free omega and AL seam control."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from qfit.samplers import BackboneRotator

from torch_rama import TorchRamaEvaluator

from run_d1_8d_sequential_poc import (
    EPSILON, SequentialBackbonePOC, atomic_csv, atomic_json, atomic_npz,
    backbone_coordinates, rama_category, rmsd,
)
from run_d1_aprime_representability_gate import PhiPsiOmegaRotator, frame, rotation_vector
from run_d1_reachability import (
    BACKBONE_NAMES,
    dihedrals,
    residue_local_index,
    wrapped_delta,
)
from qfit.structure.math import dihedral_angle
from run_d1_tier_a_flips import atom_local_index
from d1_population_calibrated_weights import D1_OMEGA_SCALE_DEG, D1_RAMA_FLOOR
from aprime_clash import build_context_for_runner
from result_provenance import runner_provenance
from occupancy_selection import (
    DEFAULT_CARDINALITY_CAP,
    DEFAULT_MIN_OCCUPANCY,
    LEGACY_CULL_THRESHOLD,
    diagnose_affine_cardinality_caps,
    evaluate_qfit_coupled_thresholds,
    legacy_cull,
    select_decoupled_affine_miqp,
    solve_affine_qp,
)


def backbone_indices(window):
    indices = []
    for residue in window.residues:
        for name in BACKBONE_NAMES:
            global_index = int(residue.select("name", name)[0])
            local = int(np.searchsorted(window.selection, global_index))
            indices.append(local)
    return indices


def seam_vector(initial_backbone, current_backbone, lever_arm=1.5):
    a_terminal, terminal = initial_backbone[-4:], current_backbone[-4:]
    translation = terminal[0] - a_terminal[0]
    rotation = rotation_vector(frame(*a_terminal[:3]).T @ frame(*terminal[:3]))
    return np.concatenate((translation, lever_arm * rotation)), translation, rotation


def _angle_deg(a, b, c):
    u, v = a - b, c - b
    cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def internal_geometry(window, initial, current):
    """Measure bond-length and bond-angle changes in an explicit state."""
    original = window.coor.copy()
    try:
        records = {}
        for label, coordinates in (("initial", initial), ("final", current)):
            window.coor = coordinates.copy()
            lengths, angles = {}, {}
            for i, residue in enumerate(window.residues):
                for name1, name2 in (("N", "CA"), ("CA", "C"), ("C", "O")):
                    a, b = residue.coor[atom_local_index(residue, name1)], residue.coor[atom_local_index(residue, name2)]
                    lengths[f"{i}:{name1}-{name2}"] = float(np.linalg.norm(a - b))
                for name1, name2, name3 in (("N", "CA", "C"), ("CA", "C", "O")):
                    a, b, c = (residue.coor[atom_local_index(residue, name)] for name in (name1, name2, name3))
                    angles[f"{i}:{name1}-{name2}-{name3}"] = _angle_deg(a, b, c)
                if i < len(window.residues) - 1:
                    next_residue = window.residues[i + 1]
                    c_atom = residue.coor[atom_local_index(residue, "C")]
                    n_atom = next_residue.coor[atom_local_index(next_residue, "N")]
                    ca_atom = next_residue.coor[atom_local_index(next_residue, "CA")]
                    lengths[f"{i}:C-N"] = float(np.linalg.norm(c_atom - n_atom))
                    angles[f"{i}:C-N-CA"] = _angle_deg(c_atom, n_atom, ca_atom)
            records[f"bond_lengths_{label}_A"] = lengths
            records[f"bond_angles_{label}_deg"] = angles
        length_deltas = {key: records["bond_lengths_final_A"][key] - value
                         for key, value in records["bond_lengths_initial_A"].items()}
        angle_deltas = {key: records["bond_angles_final_deg"][key] - value
                        for key, value in records["bond_angles_initial_deg"].items()}
        max_length = max(length_deltas, key=lambda key: abs(length_deltas[key]))
        max_angle = max(angle_deltas, key=lambda key: abs(angle_deltas[key]))
        return {**records, "bond_length_delta_from_A_A": length_deltas,
                "bond_angle_delta_from_A_deg": angle_deltas,
                "max_abs_bond_length_change_from_A_A": float(abs(length_deltas[max_length])),
                "max_bond_length_label": max_length,
                "max_abs_bond_angle_change_from_A_deg": float(abs(angle_deltas[max_angle])),
                "max_bond_angle_label": max_angle}
    finally:
        window.coor = original


class APrimeSequential:
    def __init__(self, output: Path, inner_nfev: int, outer_updates: int,
                 pdb_id: str = "7UTC", chain: str = "A", resnum: int = 52,
                 training_indices=None, renderer_backend: str = "torch",
                 residual_scale_mode: str = "none", map_scaler_structure: str = "a_only",
                 mask_scope: str = "central", device: str = "auto",
                 start_pdb: str | Path | None = None,
                 b_factor_mode: str | None = None,
                 density_atom_scope: str = "backbone",
                 mask_indices_cache: str | Path | None = None,
                 verify_mask_cache: bool = True,
                 clash_weight: float = 0.0,
                 clash_pair_cutoff_A: float = 4.5,
                 clash_threshold_scale: float = 0.75,
                 source_pdb: str | Path | None = None,
                 mtz_path: str | Path | None = None):
        self.output = output
        self.inner_nfev, self.outer_updates = inner_nfev, outer_updates
        self.base = SequentialBackbonePOC(
            pdb_id, chain, resnum, output, 0.25, 2.0, 0.0,
            residual_scale_mode, renderer_backend,
            map_scaler_structure=map_scaler_structure,
            mask_scope=mask_scope,
            device=device,
            start_pdb=start_pdb,
            b_factor_mode=b_factor_mode,
            density_atom_scope=density_atom_scope,
            mask_indices_cache=mask_indices_cache,
            verify_mask_cache=verify_mask_cache,
            source_pdb=source_pdb,
            mtz_path=mtz_path,
        )
        self.window, self.initial = self.base.window, self.base.initial_window.copy()
        self.rotator = PhiPsiOmegaRotator(self.window)
        self._init_torch_kinematics()
        self.bb_indices = backbone_indices(self.window)
        self.initial_backbone = self.initial[self.bb_indices]
        self.a_backbone, self.b_backbone = self.base.a_backbone, self.base.b_backbone
        self.ab_distance = rmsd(self.a_backbone, self.b_backbone)
        self.a_phi_psi, self.a_omega = dihedrals(self.window)
        # The strict seven-residue window has fixed flanking atoms in the
        # qFit segment.  A' releases the seam, so the edge residue phi/psi
        # values must use those frozen flanks rather than dihedrals()'s zero
        # placeholders for an incomplete window.
        segment_residues = self.base.qfit.segment.residues
        window_start = next(
            (index for index, residue in enumerate(segment_residues)
             if residue.id == self.window.residues[0].id),
            None,
        )
        if window_start is None:
            raise RuntimeError("could not locate A' Rama window in qFit segment")
        # A strict centre +/-3 window can coincide with a real polymer
        # segment end.  In that case phi(0) and/or psi(6) are physically
        # undefined: no external atom exists from which to derive them.  Keep
        # their barrier rows explicitly zero and report them as undefined;
        # inventing a flanking atom or silently replacing them by qFit's zero
        # placeholder would be less defensible.  All defined phi/psi pairs,
        # including both components of the five internal residues, are still
        # scored by the same residue-aware table.
        self._rama_previous_c = None
        self._rama_following_n = None
        if window_start > 0:
            previous = segment_residues[window_start - 1]
            self._rama_previous_c = previous.coor[residue_local_index(previous, "C")].copy()
        if window_start + len(self.window.residues) < len(segment_residues):
            following = segment_residues[window_start + len(self.window.residues)]
            self._rama_following_n = following.coor[residue_local_index(following, "N")].copy()
        self.rama_edge_components_defined = {
            "phi_offset_minus3": self._rama_previous_c is not None,
            "psi_offset_plus3": self._rama_following_n is not None,
        }
        self.rho_reference_seam_A = 1.6275900803874028
        # E_density is normalized to one at a slot's start.  At the measured
        # B-like seam, rho/2*||g||² is therefore also one.
        self.rho = 2.0 / self.rho_reference_seam_A ** 2
        self.rama_floor, self.rama_weight = D1_RAMA_FLOOR, 0.10
        self.torch_rama = TorchRamaEvaluator(self.base.torch_device)
        # A' retains an explicit soft omega restraint, but the 20 degree
        # historical scale is too weak for a released-seam seven-residue
        # window.  Five degrees still permits a peptide-flip transition while
        # pricing large departures.  The panel runner records this setting in
        # every checkpointed candidate.
        self.omega_scale_deg, self.planar_weight = D1_OMEGA_SCALE_DEG, 0.05
        self.training_indices = (None if training_indices is None
                                 else np.asarray(training_indices, dtype=int))
        # Coordinate optimization must only see these voxels.  Keep the base
        # object untouched so callers can render full-mask models for a
        # genuinely held-out evaluation after fitting.
        self.target = (self.base.target if self.training_indices is None
                       else self.base.target[self.training_indices].copy())
        if not np.isfinite(clash_weight) or clash_weight < 0.0:
            raise ValueError("clash_weight must be finite and non-negative")
        self.clash_weight = float(clash_weight)
        self.clash_context = (
            None if self.clash_weight == 0.0 else
            build_context_for_runner(
                self, pair_cutoff_A=clash_pair_cutoff_A,
                threshold_scale=clash_threshold_scale,
            )
        )
        self.trajectory = []

    def _init_torch_kinematics(self):
        """Cache qFit's fixed rotation operators for a Torch FK graph.

        qFit's BackboneRotator is a sequence of rigid rotations with fixed
        atom selections.  The six omega rotations in ``PhiPsiOmegaRotator``
        are downstream rotations whose axes are read from the current graph.
        Storing only the qFit operators and index sets keeps the Torch graph
        differentiable while preserving the existing coordinate convention.
        """
        import torch

        rotator = BackboneRotator(self.window)
        device = self.base.torch_device
        self._torch_initial = torch.as_tensor(self.initial, dtype=torch.float64, device=device)
        self._torch_phi_origins = [torch.as_tensor(origin, dtype=torch.float64, device=device)
                                   for origin in rotator._origins]  # pylint: disable=protected-access
        self._torch_phi_forward = [torch.as_tensor(aligner.forward_rotation, dtype=torch.float64, device=device)
                                   for aligner in rotator._aligners]  # pylint: disable=protected-access
        self._torch_phi_backward = [torch.as_tensor(aligner.backward_rotation, dtype=torch.float64, device=device)
                                    for aligner in rotator._aligners]  # pylint: disable=protected-access
        self._torch_phi_atoms = [torch.as_tensor(selection, dtype=torch.long)
                                 for selection in rotator._atoms_to_rotate]  # pylint: disable=protected-access
        # qFit's selections are global structure indices, while ``window.coor``
        # is stored in window-selection order.  Use the same searchsorted map
        # as the existing NumPy wrapper; subtracting the first global index is
        # wrong when atoms are omitted from the source structure.
        window_selection = np.asarray(self.window.selection, dtype=np.int64)
        self._torch_phi_atoms = [torch.as_tensor(
            np.searchsorted(window_selection, indices.detach().cpu().numpy()
                            if hasattr(indices, "detach") else np.asarray(indices)),
            dtype=torch.long,
        ) for indices in self._torch_phi_atoms]
        self._torch_c_atom_indices = [int(np.searchsorted(
            window_selection, int(residue.select("name", "C")[0])
        )) for residue in self.window.residues]
        self._torch_n_atom_indices = [int(np.searchsorted(
            window_selection, int(residue.select("name", "N")[0])
        )) for residue in self.window.residues]
        self._torch_ca_atom_indices = [int(np.searchsorted(
            window_selection, int(residue.select("name", "CA")[0])
        )) for residue in self.window.residues]

    @staticmethod
    def _torch_rz(theta):
        import torch

        zero = torch.zeros_like(theta)
        one = torch.ones_like(theta)
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.stack((
            torch.stack((c, -s, zero), dim=-1),
            torch.stack((s, c, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ), dim=-2)

    @staticmethod
    def _torch_rodrigues(axis, theta):
        import torch

        axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
        x, y, z = axis.unbind(-1)
        c, s = torch.cos(theta), torch.sin(theta)
        one_c = 1.0 - c
        return torch.stack((
            torch.stack((c + x*x*one_c, x*y*one_c - z*s, x*z*one_c + y*s), dim=-1),
            torch.stack((y*x*one_c + z*s, c + y*y*one_c, y*z*one_c - x*s), dim=-1),
            torch.stack((z*x*one_c - y*s, z*y*one_c + x*s, c + z*z*one_c), dim=-1),
        ), dim=-2)

    def torch_forward(self, parameters):
        """Differentiable equivalent of ``PhiPsiOmegaRotator.__call__``."""
        import torch

        values = torch.as_tensor(
            parameters, dtype=torch.float64, device=self.base.torch_device
        )
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.shape[-1] != self.rotator.ndofs:
            raise ValueError(f"expected {self.rotator.ndofs} torsions, got {values.shape[-1]}")
        coordinates = self._torch_initial.to(device=values.device).unsqueeze(0).expand(values.shape[0], -1, -1).clone()

        # This is the same reversed qFit order as BackboneRotator.__call__.
        for parameter, origin, forward, backward, atoms in zip(
                torch.flip(values[:, :self.rotator.phi_psi_ndofs], dims=(-1,)).unbind(-1),
                self._torch_phi_origins, self._torch_phi_forward,
                self._torch_phi_backward, self._torch_phi_atoms):
            rotation = forward.to(values.device) @ self._torch_rz(
                torch.deg2rad(parameter)
            ) @ backward.to(values.device)
            moved = coordinates[:, atoms.to(values.device), :] - origin.to(values.device)
            coordinates = coordinates.index_copy(
                1, atoms.to(values.device), moved @ rotation.transpose(-1, -2) + origin.to(values.device)
            )

        # Match the downstream C-terminal-to-N-terminal omega loop.  Unlike
        # phi/psi, each axis is a function of the current graph.
        for i in reversed(range(self.rotator.omega_ndofs)):
            origin = coordinates[:, self._torch_c_atom_indices[i], :]
            axis = coordinates[:, self._torch_n_atom_indices[i + 1], :] - origin
            rotation = self._torch_rodrigues(axis, torch.deg2rad(values[:, self.rotator.phi_psi_ndofs + i]))
            atoms = torch.arange(
                self._torch_n_atom_indices[i + 1], coordinates.shape[1],
                dtype=torch.long, device=values.device,
            )
            moved = coordinates[:, atoms, :] - origin.unsqueeze(1)
            coordinates = coordinates.index_copy(
                1, atoms, moved @ rotation.transpose(-1, -2) + origin.unsqueeze(1)
            )
        return coordinates

    @staticmethod
    def _torch_dihedral(a, b, c, d):
        import torch

        b0 = a - b
        b1 = c - b
        b2 = d - c
        b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1e-12)
        v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
        w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
        return torch.rad2deg(torch.atan2(
            (torch.cross(b1, v, dim=-1) * w).sum(-1), (v * w).sum(-1)
        ))

    def _torch_omega(self, coordinates):
        import torch

        values = []
        for i in range(self.rotator.omega_ndofs):
            values.append(self._torch_dihedral(
                coordinates[:, self._torch_ca_atom_indices[i], :],
                coordinates[:, self._torch_c_atom_indices[i], :],
                coordinates[:, self._torch_n_atom_indices[i + 1], :],
                coordinates[:, self._torch_ca_atom_indices[i + 1], :],
            ))
        return torch.stack(values, dim=-1)

    def _torch_phi_psi(self, coordinates):
        """Return differentiable (phi, psi) values for each window residue."""
        import torch

        values = []
        previous_c = (None if self._rama_previous_c is None else
                      torch.as_tensor(self._rama_previous_c, dtype=torch.float64,
                                      device=coordinates.device))
        following_n = (None if self._rama_following_n is None else
                       torch.as_tensor(self._rama_following_n, dtype=torch.float64,
                                       device=coordinates.device))
        for index in range(len(self.window.residues)):
            if index == 0:
                c0 = previous_c
            else:
                c0 = coordinates[:, self._torch_c_atom_indices[index - 1], :]
            n = coordinates[:, self._torch_n_atom_indices[index], :]
            ca = coordinates[:, self._torch_ca_atom_indices[index], :]
            c = coordinates[:, self._torch_c_atom_indices[index], :]
            if c0 is None:
                phi = torch.zeros(coordinates.shape[0], dtype=coordinates.dtype,
                                  device=coordinates.device)
            else:
                if c0.ndim == 1:
                    c0 = c0.unsqueeze(0).expand(coordinates.shape[0], -1)
                phi = self._torch_dihedral(c0, n, ca, c)

            if index == len(self.window.residues) - 1:
                n1 = following_n
            else:
                n1 = coordinates[:, self._torch_n_atom_indices[index + 1], :]
            if n1 is None:
                psi = torch.zeros(coordinates.shape[0], dtype=coordinates.dtype,
                                  device=coordinates.device)
            else:
                if n1.ndim == 1:
                    n1 = n1.unsqueeze(0).expand(coordinates.shape[0], -1)
                psi = self._torch_dihedral(n, ca, c, n1)
            values.append(torch.stack((phi, psi), dim=-1))
        return torch.stack(values, dim=1)

    def torch_rama_barrier(self, coordinates, omega=None):
        """Evaluate all Rama barriers in Torch, retaining gradients."""
        import torch

        if omega is None:
            omega = self._torch_omega(coordinates)
        phi_psi = self._torch_phi_psi(coordinates)
        barriers = torch.zeros(
            (coordinates.shape[0], len(self.window.residues)),
            dtype=coordinates.dtype, device=coordinates.device,
        )
        scores = [[None] * len(self.window.residues) for _ in range(coordinates.shape[0])]
        omega_numpy = omega.detach().cpu().numpy()
        for index in range(len(self.window.residues)):
            undefined = ((index == 0 and self._rama_previous_c is None)
                         or (index == len(self.window.residues) - 1
                             and self._rama_following_n is None))
            if undefined:
                continue
            # The residue category is discrete (not part of the derivative).
            # Select it independently for each slot so a cis/trans boundary
            # cannot make slot 2 inherit slot 1's table.
            for batch_index in range(coordinates.shape[0]):
                category = rama_category(self.window, index, omega_numpy[batch_index])
                score, barrier = self.torch_rama.barrier(
                    category, phi_psi[batch_index:batch_index + 1, index, 0],
                    phi_psi[batch_index:batch_index + 1, index, 1], self.rama_floor
                )
                barriers[batch_index, index] = barrier[0]
                scores[batch_index][index] = float(score[0].detach().cpu())
        return barriers, scores

    def torch_omega_and_rama(self, coordinates):
        """Return Torch omega residuals and Rama barriers plus score diagnostics."""
        import torch

        omega = self._torch_omega(coordinates)
        omega_delta = torch.remainder(
            omega - torch.as_tensor(self.a_omega, dtype=coordinates.dtype,
                                    device=coordinates.device) + 180.0, 360.0
        ) - 180.0
        barriers, scores = self.torch_rama_barrier(coordinates, omega)
        return omega, omega_delta, scores, barriers

    @staticmethod
    def _torch_seam(initial_backbone, current_backbone):
        import torch

        def frame(n, ca, c):
            x = ca - n
            x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
            y = c - n
            y = y - x * (x * y).sum(-1, keepdim=True)
            y = y / torch.linalg.vector_norm(y, dim=-1, keepdim=True).clamp_min(1e-12)
            return torch.stack((x, y, torch.cross(x, y, dim=-1)), dim=-1)

        terminal = current_backbone[:, -4:, :]
        a_terminal = initial_backbone[-4:, :].unsqueeze(0)
        translation = terminal[:, 0, :] - a_terminal[:, 0, :]
        rotation = frame(a_terminal[:, 0, :], a_terminal[:, 1, :], a_terminal[:, 2, :]).transpose(-1, -2) @ frame(
            terminal[:, 0, :], terminal[:, 1, :], terminal[:, 2, :]
        )
        skew = torch.stack((
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ), dim=-1)
        sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
        cosine = 0.5 * (torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1) - 1.0)
        theta = torch.atan2(sine, cosine)
        factor = torch.where(sine > 1e-8, theta / (2.0 * sine), torch.full_like(theta, 0.5))
        return torch.cat((translation, 1.5 * factor.unsqueeze(-1) * skew), dim=-1)

    def _autodiff_residual_jacobian(self, parameters, state, target, normalizer, lambdas,
                                    b_offset: float = 0.0):
        """Build J with one Torch reverse-mode graph, not 40 FD renders.

        Occupancy is deliberately profiled outside the geometry gradient: the
        QP is solved by the existing NumPy/CVXPY path and its current weight is
        held fixed while differentiating the residual.  This is the requested
        occupancy-out-of-gradient treatment.  The active Torch path evaluates
        the tabulated Rama term on-device with bilinear interpolation and
        autodiff.  The CCTBX branch remains only as a compatibility fallback
        for the CPU/CCTBX renderer.
        """
        import torch

        if self.base.renderer_backend != "torch":
            # CCTBX has no autodiff path.  The residual/objective is unchanged;
            # only the geometry Jacobian construction differs.
            occupancy = float(state["occupancy"])
            intercept = float(state["intercept"])

            def cctbx_residual(value):
                coordinates = self.forward(value)
                model = self.base.model_density(coordinates, slot=0, b_offset=b_offset)
                density = (target - occupancy * model - intercept) / math.sqrt(normalizer)
                backbone = coordinates[self.bb_indices]
                g, _, _ = seam_vector(self.initial_backbone, backbone)
                seam = math.sqrt(self.rho / 2.0) * (g + lambdas / self.rho)
                omega, omega_delta, _, rama_barrier = self.omega_and_rama(coordinates)
                del omega
                rama = math.sqrt(self.rama_weight) * rama_barrier
                planar = math.sqrt(self.planar_weight) * omega_delta / self.omega_scale_deg
                return np.concatenate((density, seam, rama, planar))

            step = 0.25
            columns = []
            for column in range(len(parameters)):
                direction = np.zeros(len(parameters))
                direction[column] = step
                columns.append((cctbx_residual(np.asarray(parameters) + direction) -
                                cctbx_residual(np.asarray(parameters) - direction)) / (2.0 * step))
            return np.asarray(columns).T

        device = self.base.torch_device
        value = torch.tensor(
            np.asarray(parameters, dtype=float), dtype=torch.float64,
            device=device, requires_grad=True,
        )
        target_t = torch.as_tensor(target, dtype=torch.float64, device=device)
        lambda_t = torch.as_tensor(lambdas, dtype=torch.float64, device=device)
        occupancy = float(state["occupancy"])
        intercept = float(state["intercept"])
        initial_bb = self._torch_initial[self.bb_indices]

        def torch_residual(torsions):
            coordinates = self.torch_forward(torsions)[0]
            model = self.base.model_density_torch(
                coordinates[self.base.model_atom_indices],
                b_factors=self.base.slot_b_factors(0, b_offset),
            )[0]
            if self.training_indices is not None:
                model = model[torch.as_tensor(
                    self.training_indices, dtype=torch.long, device=device
                )]
            if self.base.qfit.options.bulk_solvent_level > 0.0:
                model = torch.maximum(model, torch.as_tensor(
                    self.base.qfit.options.bulk_solvent_level,
                    dtype=torch.float64, device=device,
                ))
            density = (target_t - occupancy * model - intercept) / math.sqrt(normalizer)
            backbone = coordinates[self.bb_indices].unsqueeze(0)
            seam = math.sqrt(self.rho / 2.0) * (self._torch_seam(initial_bb, backbone)[0] + lambda_t / self.rho)
            omega = self._torch_omega(coordinates.unsqueeze(0))[0]
            omega_delta = omega - torch.as_tensor(
                self.a_omega, dtype=torch.float64, device=device
            )
            planar = math.sqrt(self.planar_weight) * omega_delta / self.omega_scale_deg
            rama = math.sqrt(self.rama_weight) * self.torch_rama_barrier(
                coordinates.unsqueeze(0), omega.unsqueeze(0)
            )[0][0]
            return torch.cat((density, seam, rama, planar))

        # There are only 20 torsion inputs but a full-window mask can contain
        # several thousand density residuals.  Carrying all 20 forward-mode
        # tangents through one graph can exceed the pod memory limit, so the
        # diagnostic full-window path evaluates small tangent blocks.  This is
        # still autodiff (not finite differences) and keeps the occupancy
        # profile outside the geometry graph.
        if self.base.mask_scope == "window":
            jacobian = np.zeros((len(target) + 6 + len(self.window.residues) + self.rotator.omega_ndofs,
                                 len(parameters)), dtype=float)
            # With the full-window renderer restricted to the verified central
            # periodic image, the complete 20-direction JVP fits comfortably;
            # retain the smaller block as the safe fallback for larger masks.
            chunk_size = 20 if len(target) <= 4000 else 4
            base = value.detach()
            for start in range(0, len(parameters), chunk_size):
                stop = min(start + chunk_size, len(parameters))
                chunk = value[start:stop].detach().clone().requires_grad_(True)

                def chunk_residual(block):
                    full = torch.cat((base[:start], block, base[stop:]))
                    return torch_residual(full)

                block_jacobian = torch.autograd.functional.jacobian(
                    chunk_residual, chunk, create_graph=False, vectorize=True,
                    strategy="forward-mode",
                )
                jacobian[:, start:stop] = block_jacobian.detach().cpu().numpy()
        else:
            jacobian = torch.autograd.functional.jacobian(
                torch_residual, value, create_graph=False, vectorize=True,
                strategy="forward-mode",
            ).detach().cpu().numpy()
        return jacobian

    def forward(self, parameters):
        return self.rotator(np.asarray(parameters, dtype=float))

    def model_density(self, coordinates, b_offset: float = 0.0):
        density = self.base.model_density(coordinates, slot=0, b_offset=b_offset)
        return density if self.training_indices is None else density[self.training_indices]

    def omega_and_rama(self, coordinates):
        original = self.window.coor.copy()
        try:
            self.window.coor = coordinates.copy()
            phi_psi, omega = dihedrals(self.window)
            # dihedrals() returns zero placeholders for phi(0) and psi(6),
            # because the strict window does not own the flanking atoms.
            # Those atoms are fixed outside the optimization and are supplied
            # explicitly here, so all seven optimized residue phi/psi pairs
            # are scored by the residue-type-aware table.
            if self._rama_previous_c is not None:
                first = self.window.residues[0]
                phi_psi[0] = dihedral_angle(np.asarray([
                    self._rama_previous_c,
                    first.coor[residue_local_index(first, "N")],
                    first.coor[residue_local_index(first, "CA")],
                    first.coor[residue_local_index(first, "C")],
                ]))
            if self._rama_following_n is not None:
                last = self.window.residues[-1]
                phi_psi[-1] = dihedral_angle(np.asarray([
                    last.coor[residue_local_index(last, "N")],
                    last.coor[residue_local_index(last, "CA")],
                    last.coor[residue_local_index(last, "C")],
                    self._rama_following_n,
                ]))
            scores = []
            for index in range(len(self.window.residues)):
                if ((index == 0 and self._rama_previous_c is None)
                        or (index == len(self.window.residues) - 1 and self._rama_following_n is None)):
                    scores.append(None)
                else:
                    score = float(self.base.rama_eval.evaluate(
                        rama_category(self.window, index, omega),
                        [float(phi_psi[2 * index]), float(phi_psi[2 * index + 1])],
                    ))
                    scores.append(score)
        finally:
            self.window.coor = original
        omega_delta = wrapped_delta(self.a_omega, omega)
        rama_barrier = np.zeros(len(scores), dtype=float)
        defined = np.asarray([score is not None for score in scores], dtype=bool)
        score_values = np.asarray([1.0 if score is None else score for score in scores], dtype=float)
        rama_barrier[defined] = np.maximum(
            0.0, np.log(self.rama_floor / np.maximum(score_values[defined], 1e-12))
        )
        return omega, omega_delta, scores, rama_barrier

    def evaluate(self, parameters, target, capacity, normalizer, lambdas):
        coordinates = self.forward(parameters)
        model = self.model_density(coordinates)
        occupancy, intercept, rss = self.base.bounded_nnls(target, model, capacity)
        backbone = coordinates[self.bb_indices]
        g, translation, rotation = seam_vector(self.initial_backbone, backbone)
        omega, omega_delta, rama_scores, rama_barrier = self.omega_and_rama(coordinates)
        density_residual = (target - occupancy * model - intercept) / math.sqrt(normalizer)
        seam_residual = math.sqrt(self.rho / 2.0) * (g + lambdas / self.rho)
        rama_residual = math.sqrt(self.rama_weight) * rama_barrier
        planar_residual = math.sqrt(self.planar_weight) * omega_delta / self.omega_scale_deg
        residual = np.concatenate((density_residual, seam_residual, rama_residual, planar_residual))
        central = self.base.central_coordinates(coordinates)
        names = self.base.central.name.tolist()
        central_bb = np.asarray([central[names.index(name)] for name in BACKBONE_NAMES])
        return {"coordinates": coordinates, "model": model, "occupancy": occupancy,
                "intercept": intercept, "rss": rss,
                "g": g, "translation": translation, "rotation": rotation, "omega": omega,
                "omega_delta": omega_delta, "rama_scores": rama_scores, "rama_barrier": rama_barrier,
                "residual": residual, "energy": float(np.dot(residual, residual)),
                "density_energy": float(rss / normalizer), "central_bb": central_bb,
                "rmsd_A": rmsd(central_bb, self.a_backbone), "rmsd_B": rmsd(central_bb, self.b_backbone)}

    def evaluate_single_slot(self, parameters, target, normalizer, lambdas):
        """Evaluate the neutral-start objective: one torsion slot plus global dB."""
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (self.rotator.ndofs + 1,):
            raise ValueError(f"expected {self.rotator.ndofs + 1} single-slot parameters")
        torsions = parameters[:self.rotator.ndofs]
        b_offset = float(parameters[-1])
        coordinates = self.forward(torsions)
        model = self.model_density(coordinates, b_offset=b_offset)
        occupancy, intercept, rss = self.base.bounded_nnls(target, model, 1.0)
        backbone = coordinates[self.bb_indices]
        g, translation, rotation = seam_vector(self.initial_backbone, backbone)
        omega, omega_delta, rama_scores, rama_barrier = self.omega_and_rama(coordinates)
        density_residual = (target - occupancy * model - intercept) / math.sqrt(normalizer)
        seam_residual = math.sqrt(self.rho / 2.0) * (g + lambdas / self.rho)
        rama_residual = math.sqrt(self.rama_weight) * rama_barrier
        planar_residual = math.sqrt(self.planar_weight) * omega_delta / self.omega_scale_deg
        residual = np.concatenate((density_residual, seam_residual, rama_residual, planar_residual))
        central = self.base.central_coordinates(coordinates)
        names = self.base.central.name.tolist()
        central_bb = np.asarray([central[names.index(name)] for name in BACKBONE_NAMES])
        return {"coordinates": coordinates, "model": model, "occupancy": occupancy,
                "intercept": intercept, "rss": rss, "b_offset_A2": b_offset,
                "g": g, "translation": translation, "rotation": rotation, "omega": omega,
                "omega_delta": omega_delta, "rama_scores": rama_scores,
                "rama_barrier": rama_barrier, "residual": residual,
                "energy": float(np.dot(residual, residual)),
                "density_energy": float(rss / normalizer), "central_bb": central_bb,
                "rmsd_A": rmsd(central_bb, self.a_backbone),
                "rmsd_B": rmsd(central_bb, self.b_backbone)}

    def _single_slot_fixed_affine_residual(self, parameters, target, normalizer, lambdas,
                                           occupancy, intercept):
        """Residual with affine terms fixed, used for the global-dB Jacobian column."""
        parameters = np.asarray(parameters, dtype=float)
        coordinates = self.forward(parameters[:self.rotator.ndofs])
        model = self.model_density(coordinates, b_offset=float(parameters[-1]))
        g, _, _ = seam_vector(self.initial_backbone, coordinates[self.bb_indices])
        omega, omega_delta, _, rama_barrier = self.omega_and_rama(coordinates)
        density = (target - occupancy * model - intercept) / math.sqrt(normalizer)
        return np.concatenate((
            density,
            math.sqrt(self.rho / 2.0) * (g + lambdas / self.rho),
            math.sqrt(self.rama_weight) * rama_barrier,
            math.sqrt(self.planar_weight) * omega_delta / self.omega_scale_deg,
        ))

    def single_slot_residual_jacobian(self, parameters, target, normalizer, lambdas):
        """Jacobian for one-slot fitting: Torch autodiff for torsions, FD for dB."""
        parameters = np.asarray(parameters, dtype=float)
        state = self.evaluate_single_slot(parameters, target, normalizer, lambdas)
        torsion_jacobian = self._autodiff_residual_jacobian(
            parameters[:self.rotator.ndofs], state, target, normalizer, lambdas,
            b_offset=float(parameters[-1]),
        )
        step = 0.25
        plus = parameters.copy(); plus[-1] += step
        minus = parameters.copy(); minus[-1] -= step
        d_b = (
            self._single_slot_fixed_affine_residual(
                plus, target, normalizer, lambdas, state["occupancy"], state["intercept"]
            ) - self._single_slot_fixed_affine_residual(
                minus, target, normalizer, lambdas, state["occupancy"], state["intercept"]
            )
        ) / (2.0 * step)
        return np.column_stack((torsion_jacobian, d_b))

    def fit_single_slot(self, target=None):
        """Refine the collapsed single-conformer model to convergence on all mask voxels."""
        if target is None:
            target = self.base.target
        target = np.asarray(target, dtype=float)
        if self.training_indices is not None:
            raise ValueError("neutral start must use all voxels, not a held-out/training split")
        parameters = np.zeros(self.rotator.ndofs + 1, dtype=float)
        lower_b = -float(np.min(self.base.b_factors_a_model)) + 1e-6
        upper_b = 200.0
        lambdas = np.zeros(6, dtype=float)
        start = self.evaluate_single_slot(parameters, target, 1.0, lambdas)
        normalizer = max(float(start["rss"]), EPSILON)
        trajectory = []
        final_result = None
        for outer in range(1, self.outer_updates + 1):
            evaluations = 0

            def residual_function(value):
                nonlocal evaluations
                state = self.evaluate_single_slot(value, target, normalizer, lambdas)
                evaluations += 1
                trajectory.append({
                    "outer_update": outer, "evaluation": evaluations,
                    "energy": state["energy"], "density_energy": state["density_energy"],
                    "rss": state["rss"], "occupancy": state["occupancy"],
                    "intercept": state["intercept"], "dB_A2": state["b_offset_A2"],
                    "rmsd_to_A_A": state["rmsd_A"], "rmsd_to_B_A": state["rmsd_B"],
                    "seam_A_equivalent": state["g"].tolist(),
                    "omega_deg": state["omega"].tolist(),
                    "rama_probabilities": state["rama_scores"],
                })
                return state["residual"]

            final_result = least_squares(
                residual_function, parameters, method="trf",
                jac=lambda value: self.single_slot_residual_jacobian(
                    value, target, normalizer, lambdas
                ),
                bounds=(np.r_[np.full(self.rotator.ndofs, -np.inf), lower_b],
                        np.r_[np.full(self.rotator.ndofs, np.inf), upper_b]),
                x_scale=np.r_[np.full(self.rotator.ndofs, 10.0), 5.0],
                max_nfev=self.inner_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10,
            )
            parameters = final_result.x.copy()
            state = self.evaluate_single_slot(parameters, target, normalizer, lambdas)
            lambdas = lambdas + self.rho * state["g"]
            trajectory.append({
                "outer_update": outer, "event": "AL_update",
                "lm_status": int(final_result.status), "lm_message": final_result.message,
                "lm_nfev": int(final_result.nfev), "dB_A2": state["b_offset_A2"],
                "occupancy": state["occupancy"], "intercept": state["intercept"],
                "rmsd_to_A_A": state["rmsd_A"], "rmsd_to_B_A": state["rmsd_B"],
                "seam_A_equivalent": state["g"].tolist(),
                "lambda_after_update": lambdas.tolist(),
                "omega_deg": state["omega"].tolist(),
                "rama_probabilities": state["rama_scores"],
            })
            atomic_npz(self.output / "single_slot_checkpoint.npz",
                       parameters=parameters, initial_window=self.initial)
            atomic_csv(self.output / "single_slot_trajectory.csv", trajectory)
            atomic_json(self.output / "single_slot_progress.json", {
                "status": "running", "outer_update": outer,
                "trajectory_rows": len(trajectory),
            })
        final = self.evaluate_single_slot(parameters, target, normalizer, lambdas)
        result = {
            "status": "complete", "parameters": parameters.tolist(),
            "torsions_deg": parameters[:-1].tolist(), "b_offset_A2": float(parameters[-1]),
            "occupancy": float(final["occupancy"]), "intercept": float(final["intercept"]),
            "rss": float(final["rss"]), "energy": float(final["energy"]),
            "rmsd_to_A_A": float(final["rmsd_A"]), "rmsd_to_B_A": float(final["rmsd_B"]),
            "final_lambdas": lambdas.tolist(), "normalizer_initial_rss": normalizer,
            "inner_max_nfev": self.inner_nfev, "outer_updates": self.outer_updates,
            "last_lm_status": int(final_result.status),
            "last_lm_message": final_result.message,
            "last_lm_nfev": int(final_result.nfev), "trajectory_rows": len(trajectory),
        }
        atomic_npz(self.output / "single_slot_final.npz",
                   parameters=parameters, initial_window=self.initial,
                   final_window=final["coordinates"])
        atomic_csv(self.output / "single_slot_trajectory.csv", trajectory)
        result["provenance"] = runner_provenance(
            self,
            self.base.truth_path,
            self.base.mtz_path,
            {"single_slot_final": self.output / "single_slot_final.npz"},
        )
        atomic_json(self.output / "single_slot_result.json", result)
        atomic_json(self.output / "single_slot_progress.json", {
            "status": "complete", "trajectory_rows": len(trajectory),
        })
        return parameters, final, result

    @staticmethod
    def joint_qp_weights(target, models, lower_bounds=(0.0, 0.0)):
        """qFit-compatible two-slot QP with an optional temporary slot-2 floor."""
        lower = np.asarray(lower_bounds, dtype=float)
        return solve_affine_qp(target, models, lower_bounds=lower)

    def evaluate_joint_slot2(self, parameters, target, frozen_slot1_model, normalizer, lambdas,
                             slot2_occupancy_floor):
        """Score slot 2 while its slot-1 geometry remains frozen but both weights move."""
        coordinates = self.forward(parameters)
        moving_model = self.model_density(coordinates)
        models = np.vstack((frozen_slot1_model, moving_model))
        occupancies, intercept, rss = self.joint_qp_weights(
            target, models, lower_bounds=(0.0, slot2_occupancy_floor),
        )
        backbone = coordinates[self.bb_indices]
        g, translation, rotation = seam_vector(self.initial_backbone, backbone)
        omega, omega_delta, rama_scores, rama_barrier = self.omega_and_rama(coordinates)
        density_residual = (target - occupancies @ models - intercept) / math.sqrt(normalizer)
        seam_residual = math.sqrt(self.rho / 2.0) * (g + lambdas / self.rho)
        rama_residual = math.sqrt(self.rama_weight) * rama_barrier
        planar_residual = math.sqrt(self.planar_weight) * omega_delta / self.omega_scale_deg
        residual = np.concatenate((density_residual, seam_residual, rama_residual, planar_residual))
        central = self.base.central_coordinates(coordinates)
        names = self.base.central.name.tolist()
        central_bb = np.asarray([central[names.index(name)] for name in BACKBONE_NAMES])
        return {"coordinates": coordinates, "model": moving_model, "models": models,
                "occupancy": occupancies, "intercept": intercept, "rss": rss, "g": g, "translation": translation,
                "rotation": rotation, "omega": omega, "omega_delta": omega_delta,
                "rama_scores": rama_scores, "rama_barrier": rama_barrier, "residual": residual,
                "energy": float(np.dot(residual, residual)), "density_energy": float(rss / normalizer),
                "central_bb": central_bb, "rmsd_A": rmsd(central_bb, self.a_backbone),
                "rmsd_B": rmsd(central_bb, self.b_backbone)}

    def scalar_gradient(self, parameters, target, capacity, normalizer, lambdas):
        state = self.evaluate(parameters, target, capacity, normalizer, lambdas)
        jacobian = self._autodiff_residual_jacobian(
            parameters, state, target, normalizer, lambdas,
        )
        return 2.0 * jacobian.T @ state["residual"]

    def residual_jacobian(self, parameters, target, capacity, normalizer, lambdas):
        """Autodiff residual Jacobian with occupancy profiled outside geometry."""
        state = self.evaluate(parameters, target, capacity, normalizer, lambdas)
        return self._autodiff_residual_jacobian(
            parameters, state, target, normalizer, lambdas,
        )

    def checkpoint(self, stage, outer, parameters):
        atomic_npz(self.output / "checkpoint.npz", parameters=parameters, initial_window=self.initial)
        atomic_csv(self.output / "trajectory.csv", self.trajectory)
        atomic_json(self.output / "progress.json", {"status": "running", "stage": stage, "outer_update": outer, "trajectory_rows": len(self.trajectory)})

    def fit_slot(self, stage, target, capacity):
        parameters, lambdas = np.zeros(self.rotator.ndofs), np.zeros(6)
        start = self.evaluate(parameters, target, capacity, 1.0, lambdas)
        normalizer = max(start["rss"], EPSILON)
        # The full-window rerun already spends most of its cost on the actual
        # LM Jacobian.  These start/end norms are diagnostics only, so avoid
        # two additional full Torch autodiff graphs in that mode.
        start_gradient_norm = (
            None if self.base.mask_scope == "window" else
            float(np.linalg.norm(self.scalar_gradient(parameters, target, capacity, normalizer, lambdas)))
        )
        final_result = None
        for outer in range(1, self.outer_updates + 1):
            evaluations = 0
            def residual_function(value):
                nonlocal evaluations
                state = self.evaluate(value, target, capacity, normalizer, lambdas)
                evaluations += 1
                self.trajectory.append({"stage": stage, "outer_update": outer, "evaluation": evaluations,
                    "energy": state["energy"], "density_energy": state["density_energy"], "rss": state["rss"],
                    "occupancy": state["occupancy"], "intercept": state["intercept"],
                    "rmsd_to_A_A": state["rmsd_A"], "rmsd_to_B_A": state["rmsd_B"],
                    "seam_A_equivalent": state["g"].tolist(), "omega_deg": state["omega"].tolist(),
                    "rama_probabilities": state["rama_scores"]})
                return state["residual"]
            final_result = least_squares(
                residual_function, parameters, method="trf",
                jac=lambda value: self.residual_jacobian(value, target, capacity, normalizer, lambdas), x_scale=10.0,
                                         max_nfev=self.inner_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            parameters = final_result.x.copy()
            state = self.evaluate(parameters, target, capacity, normalizer, lambdas)
            lambdas = lambdas + self.rho * state["g"]
            self.trajectory.append({"stage": stage, "outer_update": outer, "event": "AL_update",
                "lm_status": int(final_result.status), "lm_message": final_result.message, "lm_nfev": int(final_result.nfev),
                "lm_gradient_norm": float(np.linalg.norm(final_result.grad)), "occupancy": state["occupancy"],
                "intercept": state["intercept"],
                "rmsd_to_A_A": state["rmsd_A"], "rmsd_to_B_A": state["rmsd_B"], "seam_A_equivalent": state["g"].tolist(),
                "lambda_after_update": lambdas.tolist(), "omega_deg": state["omega"].tolist(), "rama_probabilities": state["rama_scores"]})
            self.checkpoint(stage, outer, parameters)
        final = self.evaluate(parameters, target, capacity, normalizer, lambdas)
        gradient_norm_end = (
            None if self.base.mask_scope == "window" else
            float(np.linalg.norm(self.scalar_gradient(parameters, target, capacity, normalizer, lambdas)))
        )
        return parameters, final, {"gradient_norm_start": start_gradient_norm,
                                    "gradient_norm_end": gradient_norm_end,
                                    "outer_updates": self.outer_updates, "inner_max_nfev": self.inner_nfev,
                                    "last_lm_status": int(final_result.status), "last_lm_message": final_result.message,
                                    "last_lm_nfev": int(final_result.nfev), "final_lambdas": lambdas.tolist(),
                                    "normalizer_initial_rss": normalizer}

    def fit_slot2_with_joint_qp(self, frozen_slot1_model, slot2_occupancy_floor):
        """Fit slot 2 with slot-1 coordinates frozen and both QP weights free."""
        stage = "slot2_fit_frozen_geometry_joint_qp"
        parameters, lambdas = np.zeros(self.rotator.ndofs), np.zeros(6)
        evaluate = lambda value, norm, multipliers: self.evaluate_joint_slot2(
            value, self.target, frozen_slot1_model, norm, multipliers, slot2_occupancy_floor,
        )
        start = evaluate(parameters, 1.0, lambdas)
        normalizer = max(start["rss"], EPSILON)

        def scalar_gradient(value, multipliers):
            gradient = np.zeros_like(value)
            for index in range(len(value)):
                plus, minus = value.copy(), value.copy()
                plus[index] += 0.25
                minus[index] -= 0.25
                gradient[index] = (evaluate(plus, normalizer, multipliers)["energy"] -
                                   evaluate(minus, normalizer, multipliers)["energy"]) / 0.5
            return gradient

        start_gradient_norm = (
            None if self.base.mask_scope == "window" else
            float(np.linalg.norm(scalar_gradient(parameters, lambdas)))
        )
        final_result = None
        for outer in range(1, self.outer_updates + 1):
            evaluations = 0

            def residual_function(value):
                nonlocal evaluations
                state = evaluate(value, normalizer, lambdas)
                evaluations += 1
                self.trajectory.append({"stage": stage, "outer_update": outer, "evaluation": evaluations,
                    "energy": state["energy"], "density_energy": state["density_energy"], "rss": state["rss"],
                    "occupancy": state["occupancy"].tolist(), "intercept": state["intercept"],
                    "rmsd_to_A_A": state["rmsd_A"],
                    "rmsd_to_B_A": state["rmsd_B"], "seam_A_equivalent": state["g"].tolist(),
                    "omega_deg": state["omega"].tolist(), "rama_probabilities": state["rama_scores"]})
                return state["residual"]

            def jacobian_function(value):
                columns = []
                for index in range(len(value)):
                    plus, minus = value.copy(), value.copy()
                    plus[index] += 0.25
                    minus[index] -= 0.25
                    columns.append((evaluate(plus, normalizer, lambdas)["residual"] -
                                    evaluate(minus, normalizer, lambdas)["residual"]) / 0.5)
                return np.asarray(columns).T

            final_result = least_squares(
                residual_function, parameters, method="trf", jac=jacobian_function, x_scale=10.0,
                max_nfev=self.inner_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10,
            )
            parameters = final_result.x.copy()
            state = evaluate(parameters, normalizer, lambdas)
            lambdas = lambdas + self.rho * state["g"]
            self.trajectory.append({"stage": stage, "outer_update": outer, "event": "AL_update",
                "lm_status": int(final_result.status), "lm_message": final_result.message,
                "lm_nfev": int(final_result.nfev), "lm_gradient_norm": float(np.linalg.norm(final_result.grad)),
                "occupancy": state["occupancy"].tolist(), "intercept": state["intercept"],
                "rmsd_to_A_A": state["rmsd_A"],
                "rmsd_to_B_A": state["rmsd_B"], "seam_A_equivalent": state["g"].tolist(),
                "lambda_after_update": lambdas.tolist(), "omega_deg": state["omega"].tolist(),
                "rama_probabilities": state["rama_scores"]})
            self.checkpoint(stage, outer, parameters)
        final = evaluate(parameters, normalizer, lambdas)
        gradient_norm_end = (
            None if self.base.mask_scope == "window" else
            float(np.linalg.norm(scalar_gradient(parameters, lambdas)))
        )
        return parameters, final, {"gradient_norm_start": start_gradient_norm,
                                    "gradient_norm_end": gradient_norm_end,
                                    "outer_updates": self.outer_updates, "inner_max_nfev": self.inner_nfev,
                                    "last_lm_status": int(final_result.status), "last_lm_message": final_result.message,
                                    "last_lm_nfev": int(final_result.nfev), "final_lambdas": lambdas.tolist(),
                                    "normalizer_initial_rss": normalizer,
                                    "slot2_temporary_occupancy_floor": slot2_occupancy_floor}

    def slot_report(self, state, convergence):
        g, translation, rotation = state["g"], state["translation"], state["rotation"]
        geometry = internal_geometry(self.window, self.initial, state["coordinates"])
        occupancy = np.asarray(state["occupancy"])
        return {"rmsd_to_A_A": state["rmsd_A"], "rmsd_to_B_A": state["rmsd_B"],
                "fraction_of_A_to_B_distance_covered": 1.0 - state["rmsd_B"] / self.ab_distance,
                "occupancy_single_slot": float(occupancy) if occupancy.ndim == 0 else None,
                "joint_occupancies_during_stage": occupancy.tolist() if occupancy.ndim else None,
                "intercept": float(state["intercept"]),
                "density_rss": state["rss"],
                "seam_translation_A": translation.tolist(), "seam_rotation_deg": np.degrees(rotation).tolist(),
                "seam_A_equivalent": g.tolist(), "seam_sigma_translation_0p02A": (translation / .02).tolist(),
                "seam_sigma_rotation_1p5deg": (np.degrees(rotation) / 1.5).tolist(),
                "omega_deg": state["omega"].tolist(), "omega_deviation_from_A_deg": state["omega_delta"].tolist(),
                "rama_probabilities_window": state["rama_scores"], "rama_below_0p05": [value < .05 for value in state["rama_scores"]],
                "convergence": convergence, "internal_geometry": geometry}

    def final_occupancy_selection(self, models, continuous_weights, cardinality_cap, t_min):
        """Select fixed final geometries; never called from a geometry gradient."""
        n_atoms = len(self.base.a_residue.coor)
        decoupled = select_decoupled_affine_miqp(
            self.target,
            models,
            cardinality_cap=cardinality_cap,
            t_min=t_min,
            n_atoms=n_atoms,
        )
        legacy_weights = legacy_cull(continuous_weights, LEGACY_CULL_THRESHOLD)
        decoupled["continuous_qp_weights"] = continuous_weights.tolist()
        _, continuous_intercept, continuous_rss = solve_affine_qp(
            self.target, models, lower_bounds=np.zeros(len(models)),
            upper_bounds=np.ones(len(models)),
        )
        decoupled["continuous_qp_intercept"] = continuous_intercept
        decoupled["continuous_qp_rss"] = continuous_rss
        decoupled["legacy_0.09_cull"] = {
            "threshold": LEGACY_CULL_THRESHOLD,
            "weights": legacy_weights.tolist(),
            "surviving_slots": np.flatnonzero(legacy_weights > 0.0).tolist(),
        }
        decoupled["qfit_native_coupled_thresholds"] = evaluate_qfit_coupled_thresholds(
            self.target,
            models,
            n_atoms=n_atoms,
        )
        decoupled["bic_by_cardinality_cap"] = diagnose_affine_cardinality_caps(
            self.target,
            models,
            cardinality_caps=(1, 2, 3, 4),
            t_min=t_min,
            n_atoms=n_atoms,
        )
        return decoupled

    def run(
        self,
        joint_slot2_qp=False,
        slot2_occupancy_floor=0.0,
        selection_k=DEFAULT_CARDINALITY_CAP,
        selection_t_min=DEFAULT_MIN_OCCUPANCY,
    ):
        if abs(float(self.rama_floor) - D1_RAMA_FLOOR) > 1e-12:
            raise RuntimeError(
                f"benchmark Rama floor is fixed at calibrated value {D1_RAMA_FLOOR}"
            )
        slot1, state1, convergence1 = self.fit_slot("slot1_fit", self.target, 1.0)
        if joint_slot2_qp:
            slot2, state2, convergence2 = self.fit_slot2_with_joint_qp(
                state1["model"], slot2_occupancy_floor,
            )
            slot2_protocol = ("slot-1 geometry frozen; both QP occupancies refit at every slot-2 "
                              "objective/Jacobian evaluation; slot-2 floor released for final joint QP")
        else:
            residual = self.target - state1["occupancy"] * state1["model"] - state1["intercept"]
            slot2, state2, convergence2 = self.fit_slot(
                "slot2_residual_fit", residual, max(0.0, 1.0 - state1["occupancy"]),
            )
            slot2_protocol = "slot 1 fit alone, frozen; slot 2 fit to residual"
        models = self.base.model_density_batch(
            np.stack((state1["coordinates"], state2["coordinates"]))
        )
        if self.training_indices is not None:
            models = models[:, self.training_indices]
        weights, intercept, final_rss = solve_affine_qp(self.target, models)
        if self.base.mask_scope == "window":
            # Full-window endpoint runs must not be held hostage by the
            # optional MIQP/coupled-threshold comparison.  Preserve the
            # continuous affine-QP endpoint here; run final selection as a
            # separate fixed-model diagnostic after the endpoint is written.
            occupancy_selection = {
                "status": "deferred_fixed_model_diagnostic",
                "weights": weights.tolist(),
                "selected_slots": np.flatnonzero(weights > 0.0).tolist(),
                "cardinality_cap": selection_k,
                "t_min": selection_t_min,
            }
        else:
            occupancy_selection = self.final_occupancy_selection(
                models, weights, selection_k, selection_t_min
            )
        slot_reports = {"slot1": self.slot_report(state1, convergence1), "slot2": self.slot_report(state2, convergence2)}
        assignments = [(slot_reports["slot1"]["rmsd_to_B_A"], slot_reports["slot2"]["rmsd_to_A_A"]),
                       (slot_reports["slot1"]["rmsd_to_A_A"], slot_reports["slot2"]["rmsd_to_B_A"])]
        passed = any(first < .3 and second < .3 for first, second in assignments)
        one_state = any(value < .3 for pair in assignments for value in pair)
        verdict = "PASS" if passed else ("PARTIAL" if one_state else "FAIL")
        result = {"status": "complete", "site": f"{self.base.pdb_id}_{self.base.chain}_{self.base.a_residue.resn[0]}{self.base.resnum}", "verdict": verdict,
            "parameterization": {"phi_psi": 14, "internal_omega": 6, "total_per_slot": 20,
                "sequential": slot2_protocol, "joint_slot2_qp": joint_slot2_qp,
                "slot2_temporary_occupancy_floor": slot2_occupancy_floor if joint_slot2_qp else None,
            "optimizer": "SciPy trust-region Gauss-Newton least squares",
                "occupancy_solver": "nonnegative affine QP with free intercept; occupancies profiled outside geometry gradient",
                "bulk_solvent_floor": 0.0,
                "map_scaler_structure": self.base.map_scaler_structure,
                "AL_outer_update_every_inner_function_evaluations": self.inner_nfev,
                "rho": self.rho, "rho_reference": "rho/2 * 1.62759A² = normalized initial density energy 1",
                "rotation_lever_arm_A": 1.5, "rama_floor": self.rama_floor,
                "rama_weight": self.rama_weight, "omega_restraint": "soft omega deviation from deposited A, scale 20 degrees",
                "planar_weight": self.planar_weight,
            "density_renderer": self.base.renderer_backend},
            "start_model": {
                "path": str(self.base.start_path) if self.base.start_path is not None else None,
                "b_factor_mode": self.base.b_factor_mode,
                "per_slot_b_factor_refinement": False,
                "oracle_b_factors_retained_only_for_bound": self.base.start_path is not None,
            },
            "map": {"residual_scale": self.base.residual_scale_diagnostic, "mask_voxels": int(len(self.target)),
                    "full_mask_voxels": int(self.base.mask.sum()),
                    "resolution_A": self.base.resolution, "neighbour_subtraction": True,
                    "torch_device": (str(self.base.torch_device)
                                      if self.base.torch_device is not None else None)},
            "deposited": {"occupancies_A_B": self.base.deposited_occupancies.tolist(), "central_A_to_B_rmsd_A": self.ab_distance},
            "slots": slot_reports, "final_joint_occupancies_slot1_slot2": weights.tolist(),
            "final_joint_total_occupancy": float(weights.sum()), "final_joint_qp_intercept": intercept,
            "final_joint_qp_rss": final_rss,
            "final_selected_occupancies_slot1_slot2": occupancy_selection["weights"],
            "final_selected_slots": occupancy_selection["selected_slots"],
            "occupancy_selection": occupancy_selection,
            "success_assignment_distances": {"slot1_to_B_slot2_to_A": assignments[0], "slot1_to_A_slot2_to_B": assignments[1]},
            "trajectory_rows": len(self.trajectory)}
        atomic_npz(
            self.output / "final_slots.npz",
            slot1_window=state1["coordinates"],
            slot2_window=state2["coordinates"],
            slot1_parameters=slot1,
            slot2_parameters=slot2,
            deposited_A_window=self.initial,
        )
        atomic_csv(self.output / "trajectory.csv", self.trajectory)
        result["provenance"] = runner_provenance(
            self,
            self.base.truth_path,
            self.base.mtz_path,
            {"final_slots": self.output / "final_slots.npz"},
        )
        atomic_json(self.output / "result.json", result)
        atomic_json(self.output / "progress.json", {"status": "complete", "verdict": verdict, "trajectory_rows": len(self.trajectory)})
        return result


def run_fixed_geometry_objective(
    output: Path,
    sequential_output: Path,
    steps: int,
    selection_k: int = DEFAULT_CARDINALITY_CAP,
    selection_t_min: float = DEFAULT_MIN_OCCUPANCY,
    pdb_id: str = "7UTC",
    chain: str = "A",
    resnum: int = 52,
    renderer_backend: str = "torch",
):
    """Evaluate the calibrated objective at fixed A/B and recovered geometries."""
    if steps < 1:
        raise ValueError("--scan-steps must be positive")
    runner = APrimeSequential(
        output,
        inner_nfev=80,
        outer_updates=6,
        pdb_id=pdb_id,
        chain=chain,
        resnum=resnum,
        renderer_backend=renderer_backend,
    )
    base = runner.base
    final = np.load(sequential_output / "final_slots.npz")
    slot1, slot2 = final["slot1_window"], final["slot2_window"]
    deposited_a, deposited_b = base.initial_window.copy(), base.window_for_deposited_b()

    def pair(first, second):
        weights, intercept, rss = base.joint_qp(first, second)
        return {"occupancies": weights.tolist(), "total_occupancy": float(weights.sum()),
                "intercept": float(intercept), "rss": float(rss)}

    def single(coordinates):
        occupancy, intercept, rss = base.bounded_nnls(
            base.target, base.model_density(coordinates), 1.0
        )
        return {"occupancy": occupancy, "intercept": intercept, "rss": rss}

    converged = pair(slot1, slot2)
    deposited = pair(deposited_a, deposited_b)
    converged_models = np.vstack((base.model_density(slot1), base.model_density(slot2)))
    converged_weights = np.asarray(converged["occupancies"], dtype=float)
    converged_selection = runner.final_occupancy_selection(
        converged_models, converged_weights, selection_k, selection_t_min
    )
    central_b = base.central_backbone(deposited_b)
    scan = []
    for step in range(steps + 1):
        alpha = step / steps
        moving_slot2 = (1.0 - alpha) * slot2 + alpha * deposited_b
        scan.append({
            "alpha_to_deposited_B": alpha,
            "slot2_central_rmsd_to_B_A": rmsd(base.central_backbone(moving_slot2), central_b),
            **pair(slot1, moving_slot2),
        })

    result = {
        "status": "complete",
        "site": f"{pdb_id}_{chain}_{base.a_residue.resn[0]}{resnum}",
        "operation": "fixed-geometry QP objective evaluation only; no coordinate optimisation",
        "map": {"residual_scale": base.residual_scale_diagnostic, "mask_voxels": int(base.mask.sum()),
                "resolution_A": base.resolution, "neighbour_subtraction": True},
        "density_renderer": base.renderer_backend,
        "converged_pair": converged,
        "converged_occupancy_selection": converged_selection,
        "deposited_A_B_pair": deposited,
        "deposited_pair_minus_converged_pair_rss": deposited["rss"] - converged["rss"],
        "single_deposited_A": single(deposited_a),
        "single_deposited_B": single(deposited_b),
        "slot2_to_B_scan_with_slot1_frozen": scan,
    }
    atomic_json(output / "result.json", result)
    atomic_csv(output / "slot2_to_B_scan.csv", scan)
    atomic_json(output / "progress.json", {"status": "complete", "scan_points": len(scan)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inner-nfev", type=int, default=80)
    parser.add_argument("--outer-updates", type=int, default=6)
    parser.add_argument("--objective-eval-only", action="store_true")
    parser.add_argument("--scan-steps", type=int, default=10)
    parser.add_argument("--pdb-id", default="7UTC")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--resnum", type=int, default=52)
    parser.add_argument("--joint-slot2-qp", action="store_true",
                        help="Freeze slot-1 coordinates only; refit both QP weights throughout slot-2 fitting.")
    parser.add_argument("--slot2-occupancy-floor", type=float, default=0.0,
                        help="Temporary lower bound for slot-2 QP weight during --joint-slot2-qp fitting.")
    parser.add_argument("--selection-k", type=int, default=DEFAULT_CARDINALITY_CAP,
                        help="Independent final MIQP cardinality cap K (default: 4).")
    parser.add_argument("--selection-t-min", type=float, default=DEFAULT_MIN_OCCUPANCY,
                        help="Independent final MIQP minimum nonzero occupancy (default: 0.02).")
    parser.add_argument("--renderer-backend", choices=("torch", "cctbx"), default="torch",
                        help="Calculated-density backend (default: differentiable Torch renderer).")
    parser.add_argument("--mask-scope", choices=("central", "window"), default="central",
                        help="Mask/model atom scope; window uses the configured full-window mask.")
    parser.add_argument("--rama-floor", type=float, default=0.05,
                        help="Minimum Ramachandran probability used by the barrier.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Torch device for density/FK autodiff (default: auto).")
    parser.add_argument("--map-scaler-structure", choices=("a_only", "full"), default="a_only",
                        help="Structure used for MapScaler and neighbour subtraction.")
    parser.add_argument("--residual-scale-mode", choices=("none", "deposited_ab"), default="none",
                        help="Residual target scaling; use none for shared-target comparisons.")
    parser.add_argument("--sequential-output", type=Path,
                        default=Path("/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_sequential_v3"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    atomic_json(args.output / "run_config.json", run_config)
    if args.objective_eval_only:
        result = run_fixed_geometry_objective(
            args.output,
            args.sequential_output,
            args.scan_steps,
            args.selection_k,
            args.selection_t_min,
            args.pdb_id,
            args.chain,
            args.resnum,
            args.renderer_backend,
        )
    else:
        runner = APrimeSequential(
            args.output, args.inner_nfev, args.outer_updates,
            args.pdb_id, args.chain, args.resnum,
            renderer_backend=args.renderer_backend,
            residual_scale_mode=args.residual_scale_mode,
            map_scaler_structure=args.map_scaler_structure,
            mask_scope=args.mask_scope,
            device=args.device,
        )
        runner.rama_floor = args.rama_floor
        result = runner.run(
            args.joint_slot2_qp,
            args.slot2_occupancy_floor,
            args.selection_k,
            args.selection_t_min,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
