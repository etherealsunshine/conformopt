#!/usr/bin/env python3
"""A′ sequential two-slot real-map PoC with free omega and AL seam control."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import cvxpy as cp
from scipy.optimize import least_squares

from qfit.solvers import get_qp_solver_class

from run_d1_8d_sequential_poc import (
    EPSILON, SequentialBackbonePOC, atomic_csv, atomic_json, atomic_npz,
    backbone_coordinates, rama_category, rmsd,
)
from run_d1_aprime_representability_gate import PhiPsiOmegaRotator, frame, rotation_vector
from run_d1_reachability import BACKBONE_NAMES, dihedrals, wrapped_delta
from run_d1_tier_a_flips import atom_local_index
from occupancy_selection import (
    DEFAULT_CARDINALITY_CAP,
    DEFAULT_MIN_OCCUPANCY,
    LEGACY_CULL_THRESHOLD,
    evaluate_qfit_coupled_thresholds,
    diagnose_cardinality_caps,
    legacy_cull,
    select_decoupled_miqp,
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
                 training_indices=None):
        self.output = output
        self.inner_nfev, self.outer_updates = inner_nfev, outer_updates
        self.base = SequentialBackbonePOC(pdb_id, chain, resnum, output, 0.25, 2.0, 0.0, "deposited_ab")
        self.window, self.initial = self.base.window, self.base.initial_window.copy()
        self.rotator = PhiPsiOmegaRotator(self.window)
        self.bb_indices = backbone_indices(self.window)
        self.initial_backbone = self.initial[self.bb_indices]
        self.a_backbone, self.b_backbone = self.base.a_backbone, self.base.b_backbone
        self.ab_distance = rmsd(self.a_backbone, self.b_backbone)
        self.a_phi_psi, self.a_omega = dihedrals(self.window)
        self.rho_reference_seam_A = 1.6275900803874028
        # E_density is normalized to one at a slot's start.  At the measured
        # B-like seam, rho/2*||g||² is therefore also one.
        self.rho = 2.0 / self.rho_reference_seam_A ** 2
        self.rama_floor, self.rama_weight = 0.05, 0.10
        self.omega_scale_deg, self.planar_weight = 20.0, 0.05
        self.training_indices = (None if training_indices is None
                                 else np.asarray(training_indices, dtype=int))
        # Coordinate optimization must only see these voxels.  Keep the base
        # object untouched so callers can render full-mask models for a
        # genuinely held-out evaluation after fitting.
        self.target = (self.base.target if self.training_indices is None
                       else self.base.target[self.training_indices].copy())
        self.trajectory = []

    def forward(self, parameters):
        return self.rotator(np.asarray(parameters, dtype=float))

    def model_density(self, coordinates):
        density = self.base.model_density(coordinates)
        return density if self.training_indices is None else density[self.training_indices]

    def omega_and_rama(self, coordinates):
        original = self.window.coor.copy()
        try:
            self.window.coor = coordinates.copy()
            phi_psi, omega = dihedrals(self.window)
            scores = []
            for index in range(1, 6):
                score = float(self.base.rama_eval.evaluate(rama_category(self.window, index, omega), [float(phi_psi[2 * index]), float(phi_psi[2 * index + 1])]))
                scores.append(score)
        finally:
            self.window.coor = original
        omega_delta = wrapped_delta(self.a_omega, omega)
        rama_barrier = np.maximum(0.0, np.log(self.rama_floor / np.maximum(np.asarray(scores), 1e-12)))
        return omega, omega_delta, scores, rama_barrier

    def evaluate(self, parameters, target, capacity, normalizer, lambdas):
        coordinates = self.forward(parameters)
        model = self.model_density(coordinates)
        occupancy, rss = self.base.bounded_nnls(target, model, capacity)
        backbone = coordinates[self.bb_indices]
        g, translation, rotation = seam_vector(self.initial_backbone, backbone)
        omega, omega_delta, rama_scores, rama_barrier = self.omega_and_rama(coordinates)
        density_residual = (target - occupancy * model) / math.sqrt(normalizer)
        seam_residual = math.sqrt(self.rho / 2.0) * (g + lambdas / self.rho)
        rama_residual = math.sqrt(self.rama_weight) * rama_barrier
        planar_residual = math.sqrt(self.planar_weight) * omega_delta / self.omega_scale_deg
        residual = np.concatenate((density_residual, seam_residual, rama_residual, planar_residual))
        central = self.base.central_coordinates(coordinates)
        names = self.base.central.name.tolist()
        central_bb = np.asarray([central[names.index(name)] for name in BACKBONE_NAMES])
        return {"coordinates": coordinates, "model": model, "occupancy": occupancy, "rss": rss,
                "g": g, "translation": translation, "rotation": rotation, "omega": omega,
                "omega_delta": omega_delta, "rama_scores": rama_scores, "rama_barrier": rama_barrier,
                "residual": residual, "energy": float(np.dot(residual, residual)),
                "density_energy": float(rss / normalizer), "central_bb": central_bb,
                "rmsd_A": rmsd(central_bb, self.a_backbone), "rmsd_B": rmsd(central_bb, self.b_backbone)}

    @staticmethod
    def joint_qp_weights(target, models, lower_bounds=(0.0, 0.0)):
        """qFit-compatible two-slot QP with an optional temporary slot-2 floor."""
        matrix = np.asarray(models, dtype=float).T
        lower = np.asarray(lower_bounds, dtype=float)
        if lower.sum() > 1.0:
            raise ValueError("joint occupancy lower bounds exceed the unit occupancy budget")
        weights = cp.Variable(matrix.shape[1])
        problem = cp.Problem(cp.Minimize(cp.sum_squares(np.asarray(target) - matrix @ weights)),
                             [weights >= lower, cp.sum(weights) <= 1.0])
        problem.solve(solver=cp.OSQP, warm_start=True, polish=True)
        if weights.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"joint occupancy QP failed: {problem.status}")
        answer = np.maximum(np.asarray(weights.value, dtype=float), lower)
        return answer, float(np.square(np.asarray(target) - matrix @ answer).sum())

    def evaluate_joint_slot2(self, parameters, target, frozen_slot1_model, normalizer, lambdas,
                             slot2_occupancy_floor):
        """Score slot 2 while its slot-1 geometry remains frozen but both weights move."""
        coordinates = self.forward(parameters)
        moving_model = self.model_density(coordinates)
        models = np.vstack((frozen_slot1_model, moving_model))
        occupancies, rss = self.joint_qp_weights(
            target, models, lower_bounds=(0.0, slot2_occupancy_floor),
        )
        backbone = coordinates[self.bb_indices]
        g, translation, rotation = seam_vector(self.initial_backbone, backbone)
        omega, omega_delta, rama_scores, rama_barrier = self.omega_and_rama(coordinates)
        density_residual = (target - occupancies @ models) / math.sqrt(normalizer)
        seam_residual = math.sqrt(self.rho / 2.0) * (g + lambdas / self.rho)
        rama_residual = math.sqrt(self.rama_weight) * rama_barrier
        planar_residual = math.sqrt(self.planar_weight) * omega_delta / self.omega_scale_deg
        residual = np.concatenate((density_residual, seam_residual, rama_residual, planar_residual))
        central = self.base.central_coordinates(coordinates)
        names = self.base.central.name.tolist()
        central_bb = np.asarray([central[names.index(name)] for name in BACKBONE_NAMES])
        return {"coordinates": coordinates, "model": moving_model, "models": models,
                "occupancy": occupancies, "rss": rss, "g": g, "translation": translation,
                "rotation": rotation, "omega": omega, "omega_delta": omega_delta,
                "rama_scores": rama_scores, "rama_barrier": rama_barrier, "residual": residual,
                "energy": float(np.dot(residual, residual)), "density_energy": float(rss / normalizer),
                "central_bb": central_bb, "rmsd_A": rmsd(central_bb, self.a_backbone),
                "rmsd_B": rmsd(central_bb, self.b_backbone)}

    def scalar_gradient(self, parameters, target, capacity, normalizer, lambdas):
        gradient = np.zeros_like(parameters)
        step = 0.25
        for index in range(len(parameters)):
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += step; minus[index] -= step
            gradient[index] = (self.evaluate(plus, target, capacity, normalizer, lambdas)["energy"] - self.evaluate(minus, target, capacity, normalizer, lambdas)["energy"]) / (2.0 * step)
        return gradient

    def residual_jacobian(self, parameters, target, capacity, normalizer, lambdas):
        """Centred, absolute 0.25-degree Jacobian for qFit's coordinate path.

        SciPy's default relative step is effectively zero at this all-zero
        start, and qFit's coordinate updates then appear constant.  Using an
        explicit absolute step makes this a conventional finite-difference
        Gauss--Newton/LM iteration and matches the gradient diagnostic.
        """
        step = 0.25
        jacobian = []
        for index in range(len(parameters)):
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += step
            minus[index] -= step
            derivative = (
                self.evaluate(plus, target, capacity, normalizer, lambdas)["residual"]
                - self.evaluate(minus, target, capacity, normalizer, lambdas)["residual"]
            ) / (2.0 * step)
            jacobian.append(derivative)
        return np.asarray(jacobian).T

    def checkpoint(self, stage, outer, parameters):
        atomic_npz(self.output / "checkpoint.npz", parameters=parameters, initial_window=self.initial)
        atomic_csv(self.output / "trajectory.csv", self.trajectory)
        atomic_json(self.output / "progress.json", {"status": "running", "stage": stage, "outer_update": outer, "trajectory_rows": len(self.trajectory)})

    def fit_slot(self, stage, target, capacity):
        parameters, lambdas = np.zeros(self.rotator.ndofs), np.zeros(6)
        start = self.evaluate(parameters, target, capacity, 1.0, lambdas)
        normalizer = max(start["rss"], EPSILON)
        start_gradient_norm = float(np.linalg.norm(self.scalar_gradient(parameters, target, capacity, normalizer, lambdas)))
        final_result = None
        for outer in range(1, self.outer_updates + 1):
            evaluations = 0
            def residual_function(value):
                nonlocal evaluations
                state = self.evaluate(value, target, capacity, normalizer, lambdas)
                evaluations += 1
                self.trajectory.append({"stage": stage, "outer_update": outer, "evaluation": evaluations,
                    "energy": state["energy"], "density_energy": state["density_energy"], "rss": state["rss"],
                    "occupancy": state["occupancy"], "rmsd_to_A_A": state["rmsd_A"], "rmsd_to_B_A": state["rmsd_B"],
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
                "rmsd_to_A_A": state["rmsd_A"], "rmsd_to_B_A": state["rmsd_B"], "seam_A_equivalent": state["g"].tolist(),
                "lambda_after_update": lambdas.tolist(), "omega_deg": state["omega"].tolist(), "rama_probabilities": state["rama_scores"]})
            self.checkpoint(stage, outer, parameters)
        final = self.evaluate(parameters, target, capacity, normalizer, lambdas)
        return parameters, final, {"gradient_norm_start": start_gradient_norm,
                                    "gradient_norm_end": float(np.linalg.norm(self.scalar_gradient(parameters, target, capacity, normalizer, lambdas))),
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

        start_gradient_norm = float(np.linalg.norm(scalar_gradient(parameters, lambdas)))
        final_result = None
        for outer in range(1, self.outer_updates + 1):
            evaluations = 0

            def residual_function(value):
                nonlocal evaluations
                state = evaluate(value, normalizer, lambdas)
                evaluations += 1
                self.trajectory.append({"stage": stage, "outer_update": outer, "evaluation": evaluations,
                    "energy": state["energy"], "density_energy": state["density_energy"], "rss": state["rss"],
                    "occupancy": state["occupancy"].tolist(), "rmsd_to_A_A": state["rmsd_A"],
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
                "occupancy": state["occupancy"].tolist(), "rmsd_to_A_A": state["rmsd_A"],
                "rmsd_to_B_A": state["rmsd_B"], "seam_A_equivalent": state["g"].tolist(),
                "lambda_after_update": lambdas.tolist(), "omega_deg": state["omega"].tolist(),
                "rama_probabilities": state["rama_scores"]})
            self.checkpoint(stage, outer, parameters)
        final = evaluate(parameters, normalizer, lambdas)
        return parameters, final, {"gradient_norm_start": start_gradient_norm,
                                    "gradient_norm_end": float(np.linalg.norm(scalar_gradient(parameters, lambdas))),
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
                "density_rss": state["rss"],
                "seam_translation_A": translation.tolist(), "seam_rotation_deg": np.degrees(rotation).tolist(),
                "seam_A_equivalent": g.tolist(), "seam_sigma_translation_0p02A": (translation / .02).tolist(),
                "seam_sigma_rotation_1p5deg": (np.degrees(rotation) / 1.5).tolist(),
                "omega_deg": state["omega"].tolist(), "omega_deviation_from_A_deg": state["omega_delta"].tolist(),
                "rama_probabilities_internal": state["rama_scores"], "rama_below_0p05": [value < .05 for value in state["rama_scores"]],
                "convergence": convergence, "internal_geometry": geometry}

    def final_occupancy_selection(self, models, continuous_weights, cardinality_cap, t_min):
        """Select fixed final geometries; never called from a geometry gradient."""
        n_atoms = len(self.base.a_residue.coor)
        decoupled = select_decoupled_miqp(
            self.target,
            models,
            cardinality_cap=cardinality_cap,
            t_min=t_min,
            n_atoms=n_atoms,
        )
        legacy_weights = legacy_cull(continuous_weights, LEGACY_CULL_THRESHOLD)
        decoupled["continuous_qp_weights"] = continuous_weights.tolist()
        decoupled["continuous_qp_rss"] = float(
            np.square(self.target - continuous_weights @ models).sum()
        )
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
        decoupled["bic_by_cardinality_cap"] = diagnose_cardinality_caps(
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
        slot1, state1, convergence1 = self.fit_slot("slot1_fit", self.target, 1.0)
        if joint_slot2_qp:
            slot2, state2, convergence2 = self.fit_slot2_with_joint_qp(
                state1["model"], slot2_occupancy_floor,
            )
            slot2_protocol = ("slot-1 geometry frozen; both QP occupancies refit at every slot-2 "
                              "objective/Jacobian evaluation; slot-2 floor released for final joint QP")
        else:
            residual = self.target - state1["occupancy"] * state1["model"]
            slot2, state2, convergence2 = self.fit_slot(
                "slot2_residual_fit", residual, max(0.0, 1.0 - state1["occupancy"]),
            )
            slot2_protocol = "slot 1 fit alone, frozen; slot 2 fit to residual"
        models = np.vstack((self.model_density(state1["coordinates"]), self.model_density(state2["coordinates"])))
        solver = get_qp_solver_class("CVXPYSolver")(self.target, models); solver.solve_qp()
        weights = np.asarray(solver.weights, dtype=float)
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
                "AL_outer_update_every_inner_function_evaluations": self.inner_nfev,
                "rho": self.rho, "rho_reference": "rho/2 * 1.62759A² = normalized initial density energy 1",
                "rotation_lever_arm_A": 1.5, "rama_floor": self.rama_floor,
                "rama_weight": self.rama_weight, "omega_restraint": "soft omega deviation from deposited A, scale 20 degrees",
                "planar_weight": self.planar_weight},
            "map": {"residual_scale": self.base.residual_scale_diagnostic, "mask_voxels": int(len(self.target)),
                    "full_mask_voxels": int(self.base.mask.sum()),
                    "resolution_A": self.base.resolution, "neighbour_subtraction": True},
            "deposited": {"occupancies_A_B": self.base.deposited_occupancies.tolist(), "central_A_to_B_rmsd_A": self.ab_distance},
            "slots": slot_reports, "final_joint_occupancies_slot1_slot2": weights.tolist(),
            "final_joint_total_occupancy": float(weights.sum()), "final_joint_qp_rss": float(solver.objective_value),
            "final_selected_occupancies_slot1_slot2": occupancy_selection["weights"],
            "final_selected_slots": occupancy_selection["selected_slots"],
            "occupancy_selection": occupancy_selection,
            "success_assignment_distances": {"slot1_to_B_slot2_to_A": assignments[0], "slot1_to_A_slot2_to_B": assignments[1]},
            "trajectory_rows": len(self.trajectory)}
        atomic_npz(self.output / "final_slots.npz", slot1_window=state1["coordinates"], slot2_window=state2["coordinates"], deposited_A_window=self.initial)
        atomic_csv(self.output / "trajectory.csv", self.trajectory)
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
    )
    base = runner.base
    final = np.load(sequential_output / "final_slots.npz")
    slot1, slot2 = final["slot1_window"], final["slot2_window"]
    deposited_a, deposited_b = base.initial_window.copy(), base.window_for_deposited_b()

    def pair(first, second):
        weights, rss = base.joint_qp(first, second)
        return {"occupancies": weights.tolist(), "total_occupancy": float(weights.sum()), "rss": float(rss)}

    def single(coordinates):
        occupancy, rss = base.bounded_nnls(base.target, base.model_density(coordinates), 1.0)
        return {"occupancy": occupancy, "rss": rss}

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
        )
    else:
        result = APrimeSequential(
            args.output, args.inner_nfev, args.outer_updates,
            args.pdb_id, args.chain, args.resnum,
        ).run(
            args.joint_slot2_qp,
            args.slot2_occupancy_floor,
            args.selection_k,
            args.selection_t_min,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
