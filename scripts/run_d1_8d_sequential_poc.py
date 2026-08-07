#!/usr/bin/env python3
"""One-site sequential two-slot backbone-nullspace proof of concept.

The two slots are 7-residue qFit backbone windows (14 phi/psi parameters per
slot).  At every finite-difference update the torsion gradient is projected
onto ``null(compute_jacobian(...))``.  Slot 1 is fitted to the qFit-scaled,
neighbour-subtracted real map, frozen, and slot 2 is fitted to that residual.
Occupancies are re-solved by bounded NNLS at each objective evaluation.

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
from scipy.optimize import lsq_linear

from mmtbx.validation import ramalyze
from qfit.backbone import compute_jacobian
from qfit.qfit import QFitOptions, QFitRotamericResidue
from qfit.samplers import BackboneRotator
from qfit.solvers import get_qp_solver_class
from qfit.structure import Structure
from qfit.xtal.scaler import MapScaler

from run_d1_reachability import BACKBONE_NAMES, dihedrals, local_index, rmsd
from run_d1_tier_a_flips import atom_local_index, source_path
from run_d6_tier2_realmap import make_map


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
                 max_step_deg: float, rama_weight: float, residual_scale_mode: str = "none"):
        path, split = source_path(pdb_id)
        self.output = output
        self.pdb_id, self.chain, self.resnum, self.split = pdb_id, chain, resnum, split
        self.fd_step_deg, self.max_step_deg, self.rama_weight = fd_step_deg, max_step_deg, rama_weight
        self.residual_scale_mode = residual_scale_mode
        residue_id = (resnum, "")
        self.a_structure = Structure.fromfile(path).extract("altloc", ("", "A"))
        self.b_structure = Structure.fromfile(path).extract("altloc", ("", "B"))
        self.a_residue = self.a_structure[chain].conformers[0][residue_id]
        self.b_residue = self.b_structure[chain].conformers[0][residue_id]
        self.deposited_occupancies = np.array([
            float(np.median(self.a_residue.q)), float(np.median(self.b_residue.q))
        ])
        mtz = Path(f"/home/dev/qfit_unet_data/cache/{split}/mtz/{pdb_id}.mtz")
        if not mtz.exists():
            raise FileNotFoundError(mtz)
        xmap, self.resolution, self.n_reflections, self.map_source = make_map(mtz)
        radius = 0.5 + self.resolution / 3.0
        scale, offset = MapScaler(xmap).scale(
            self.a_structure, radius=radius, transformer="cctbx"
        )
        self.map_scale, self.map_offset = float(scale), float(offset)
        options = QFitOptions()
        options.qp_solver = options.miqp_solver = "CVXPYSolver"
        options.subtract = True
        self.qfit = QFitRotamericResidue(self.a_residue, self.a_structure, xmap, options)
        index = self.qfit.segment.find(self.qfit.residue.id)
        required = self.qfit.options.neighbor_residues_required
        if index < required or index + required >= len(self.qfit.segment):
            raise RuntimeError("7-residue strict window unavailable for requested PoC site")
        self.window = self.qfit.segment[index - required:index + required + 1]
        if len(self.window.residues) != 7:
            raise RuntimeError("expected a seven-residue qFit window")
        self.initial_window = self.window.coor.copy()
        self.central = self.window.residues[3]
        self.central_indices = atom_window_indices(self.window, self.central, tuple(self.central.name.tolist()))
        self.central_backbone_indices = atom_window_indices(self.window, self.central, BACKBONE_NAMES)
        self.a_central = self.a_residue.coor.copy()
        self.b_central = reordered_coordinates(self.a_residue, self.b_residue)
        self.b_backbone = backbone_coordinates(self.b_residue)
        self.a_backbone = backbone_coordinates(self.a_residue)
        self.b_factors = self.a_residue.b.copy()  # qFit backbone sampler retains the input B array.
        self.mask = self.qfit._transformer.get_conformers_mask(  # pylint: disable=protected-access
            [self.a_central, self.b_central], self.qfit._rmask  # pylint: disable=protected-access
        )
        self.target = self.qfit.xmap.array[self.mask].astype(float, copy=True)
        self.target_before_residual_scaling = self.target.copy()
        self.residual_target_multiplier = 1.0
        self.residual_scale_diagnostic = {"mode": residual_scale_mode}
        if residual_scale_mode == "deposited_ab":
            # MapScaler calibrates the full map before qFit removes neighbours.
            # This known A/B control records the needed residual-map amplitude;
            # it is intentionally opt-in and is not prospective recovery.
            deposited_a = self.model_density(self.initial_window)
            deposited_b = self.model_density(self.window_for_deposited_b())
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

    def central_coordinates(self, window_coordinates: np.ndarray) -> np.ndarray:
        return np.asarray(window_coordinates[self.central_indices], dtype=float)

    def model_density(self, window_coordinates: np.ndarray) -> np.ndarray:
        central = self.central_coordinates(window_coordinates)
        density = next(self.qfit._transformer.get_conformers_densities(  # pylint: disable=protected-access
            [central], [self.b_factors]
        ))[self.mask].astype(float, copy=False)
        # Mirrors qFit _convert's bulk-solvent floor.
        return np.maximum(density, self.qfit.options.bulk_solvent_level)

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
    def bounded_nnls(target: np.ndarray, model: np.ndarray, capacity: float) -> tuple[float, float]:
        upper_bound = max(0.0, capacity)
        # SciPy rejects equal lower/upper bounds.  In a sequential fit a first
        # slot may legitimately consume all available occupancy; the unique
        # feasible second-slot solution is then zero occupancy.
        if upper_bound <= EPSILON:
            return 0.0, float(np.square(target).sum())
        fit = lsq_linear(model[:, None], target, bounds=(0.0, upper_bound), method="bvls")
        weight = float(fit.x[0])
        return weight, float(np.square(target - weight * model).sum())

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
                 normalizer: float) -> dict[str, object]:
        model = self.model_density(moving_window)
        occupancy, density_rss = self.bounded_nnls(target, model, capacity)
        rama, scores = self.rama_penalty(moving_window)
        return {
            "objective": density_rss / normalizer + self.rama_weight * rama,
            "density_rss": density_rss, "occupancy": occupancy,
            "rama_penalty": rama, "rama_scores": scores, "model": model,
        }

    def optimize_slot(self, stage: str, initial: np.ndarray, target: np.ndarray, capacity: float,
                      steps: int, trajectory: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, object]]:
        current = initial.copy()
        baseline = self.evaluate(current, target, capacity, normalizer=1.0)
        normalizer = max(float(baseline["density_rss"]), EPSILON)
        state = self.evaluate(current, target, capacity, normalizer)
        consecutive_stalls = 0
        for step in range(1, steps + 1):
            gradient = np.zeros(14, dtype=float)
            for column in range(14):
                delta = np.zeros(14, dtype=float)
                delta[column] = self.fd_step_deg
                plus = self.evaluate(self.apply_increment(current, delta), target, capacity, normalizer)
                minus = self.evaluate(self.apply_increment(current, -delta), target, capacity, normalizer)
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
                trial = self.evaluate(candidate, target, capacity, normalizer)
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

    def joint_qp(self, slot1: np.ndarray, slot2: np.ndarray) -> tuple[np.ndarray, float]:
        models = np.vstack([self.model_density(slot1), self.model_density(slot2)])
        solver = get_qp_solver_class("CVXPYSolver")(self.target, models)
        solver.solve_qp()
        return np.asarray(solver.weights, dtype=float), float(solver.objective_value)

    def run(self, steps_per_stage: int) -> dict[str, object]:
        deposited_models = np.vstack([self.model_density(self.initial_window), self.model_density(self.window_for_deposited_b())])
        deposited_solver = get_qp_solver_class("CVXPYSolver")(self.target, deposited_models)
        deposited_solver.solve_qp()
        trajectory: list[dict[str, object]] = []
        slot1, stage1 = self.optimize_slot("slot1_fit", self.initial_window, self.target, 1.0, steps_per_stage, trajectory)
        slot1_model = self.model_density(slot1)
        frozen_occupancy, frozen_rss = self.bounded_nnls(self.target, slot1_model, 1.0)
        residual_target = self.target - frozen_occupancy * slot1_model
        slot2, stage2 = self.optimize_slot(
            "slot2_residual_fit", self.initial_window, residual_target,
            max(0.0, 1.0 - frozen_occupancy), steps_per_stage, trajectory,
        )
        joint_weights, joint_rss = self.joint_qp(slot1, slot2)
        slot1_backbone = self.central_backbone(slot1)
        slot2_backbone = self.central_backbone(slot2)
        result = {
            "status": "complete", "site": f"{self.pdb_id}_{self.chain}_ARG{self.resnum}",
            "map": {"source": self.map_source, "resolution_A": self.resolution, "n_reflections": self.n_reflections,
                    "qfit_map_scaler_factor": self.map_scale, "qfit_map_scaler_offset": self.map_offset,
                    "qfit_neighbour_subtraction": True, "mask_voxels": int(self.mask.sum()),
                    "residual_scale": self.residual_scale_diagnostic},
            "parameterization": {"slots": 2, "phi_psi_parameters_per_slot": 14, "closure_null_dimension": 8,
                                 "occupancy_solver": "bounded NNLS every objective evaluation; qFit CVXPY QP final joint solve",
                                 "sequential": "slot 1 fit then frozen geometry/occupancy; slot 2 fits residual",
                                 "ramachandran": "mmtbx RamachandranEval negative-log probability barrier from first step",
                                 "ramachandran_weight": self.rama_weight},
            "deposited_occupancies_A_B": self.deposited_occupancies.tolist(),
            "deposited_A_B_qp_occupancies": np.asarray(deposited_solver.weights, dtype=float).tolist(),
            "deposited_A_B_qp_rss": float(deposited_solver.objective_value),
            "slot1": {"steps": sum(row["stage"] == "slot1_fit" for row in trajectory), "frozen_occupancy": frozen_occupancy,
                      "rss": frozen_rss, "backbone_rmsd_to_A_A": rmsd(slot1_backbone, self.a_backbone),
                      "backbone_rmsd_to_B_A": rmsd(slot1_backbone, self.b_backbone)},
            "slot2": {"steps": sum(row["stage"] == "slot2_residual_fit" for row in trajectory), "residual_occupancy": float(stage2["occupancy"]),
                      "backbone_rmsd_to_A_A": rmsd(slot2_backbone, self.a_backbone),
                      "backbone_rmsd_to_B_A": rmsd(slot2_backbone, self.b_backbone),
                      "reaches_deposited_B_under_1A": rmsd(slot2_backbone, self.b_backbone) < 1.0},
            "final_joint_qp_occupancies_slot1_slot2": joint_weights.tolist(), "final_joint_qp_rss": joint_rss,
            "trajectory_rows": len(trajectory),
        }
        atomic_npz(self.output / "final_slots.npz", slot1_window=slot1, slot2_window=slot2, deposited_A_window=self.initial_window)
        atomic_csv(self.output / "trajectory.csv", trajectory)
        atomic_json(self.output / "result.json", result)
        atomic_json(self.output / "progress.json", {"status": "complete", "trajectory_rows": len(trajectory)})
        return result

    def central_backbone(self, window_coordinates: np.ndarray) -> np.ndarray:
        central = self.central_coordinates(window_coordinates)
        names = self.central.name.tolist()
        return np.asarray([central[names.index(name)] for name in BACKBONE_NAMES])

    def window_for_deposited_b(self) -> np.ndarray:
        """Use deposited B only for fixed-mask/QP calibration, never as a search start."""
        b_structure = self.b_structure
        b_residue = b_structure[self.chain].conformers[0][(self.resnum, "")]
        b_segment = next(segment for segment in b_structure.segments if b_residue in segment)
        index = b_segment.find(b_residue.id)
        window = b_segment[index - 3:index + 4]
        if len(window.residues) != 7 or [res.id for res in window.residues] != [res.id for res in self.window.residues]:
            raise RuntimeError("deposited-B strict window does not match deposited A")
        # The density objective only needs central-residue coordinates; retain
        # A coordinates at all other positions to keep array ordering explicit.
        combined = self.initial_window.copy()
        b_by_name = {name: coor for name, coor in zip(b_residue.name.tolist(), b_residue.coor)}
        for local, name in zip(self.central_indices, self.central.name.tolist()):
            combined[local] = b_by_name[name]
        return combined


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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "run_config.json", {
        **{key: value for key, value in vars(args).items() if key != "output"},
        "output": str(args.output),
    })
    experiment = SequentialBackbonePOC(
        args.pdb_id, args.chain, args.resnum, args.output, args.fd_step_deg,
        args.max_step_deg, args.rama_weight, args.residual_scale,
    )
    result = experiment.run(args.steps_per_stage)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
