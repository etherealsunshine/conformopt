#!/usr/bin/env python3
"""A′ no-density reachability gate: qFit phi/psi plus six free omega torsions."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from qfit.samplers import BackboneRotator
from qfit.structure import Structure

from run_d1_reachability import BACKBONE_NAMES, dihedrals, wrapped_delta
from run_d1_tier_a_flips import atom_local_index, source_path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def local_indices(segment, residue, names=BACKBONE_NAMES) -> list[int]:
    answer = []
    for name in names:
        global_index = int(residue.select("name", name)[0])
        position = int(np.searchsorted(segment.selection, global_index))
        if position == len(segment.selection) or segment.selection[position] != global_index:
            raise ValueError(f"{residue.id} {name} not in window")
        answer.append(position)
    return answer


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=1))))


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)],
    ])


def frame(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    x = ca - n; x /= np.linalg.norm(x)
    y = c - n; y -= x * np.dot(x, y); y /= np.linalg.norm(y)
    return np.column_stack((x, y, np.cross(x, y)))


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cosine))
    if theta < 1e-12:
        return np.zeros(3)
    vector = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    return theta * vector / (2.0 * np.sin(theta))


class PhiPsiOmegaRotator:
    """qFit's 14 φ/ψ rotations followed by six C(i)-N(i+1) omega rotations.

    qFit's ``BackboneRotator`` is retained unchanged for the 14 φ/ψ degrees.
    The wrapper then applies downstream C-N rotations from C terminus to N
    terminus, so every omega axis is read from its current upstream geometry.
    """

    def __init__(self, segment):
        self.segment = segment
        self.initial = segment.coor.copy()
        self.phi_psi_ndofs = BackboneRotator(segment).ndofs
        self.omega_ndofs = len(segment.residues) - 1
        self.ndofs = self.phi_psi_ndofs + self.omega_ndofs

    def __call__(self, parameters_deg: np.ndarray) -> np.ndarray:
        if len(parameters_deg) != self.ndofs:
            raise ValueError(f"expected {self.ndofs} torsions, got {len(parameters_deg)}")
        self.segment.coor = self.initial.copy()
        BackboneRotator(self.segment)(np.asarray(parameters_deg[:self.phi_psi_ndofs], dtype=float))
        # Apply C-terminal first. Rotating a downstream residue does not move
        # any upstream C-N axis subsequently used by the loop.
        for i in reversed(range(self.omega_ndofs)):
            theta = np.deg2rad(float(parameters_deg[self.phi_psi_ndofs + i]))
            if theta == 0.0:
                continue
            left, right = self.segment.residues[i], self.segment.residues[i + 1]
            origin = left.coor[atom_local_index(left, "C")]
            axis = right.coor[atom_local_index(right, "N")] - origin
            axis /= np.linalg.norm(axis)
            selection = np.concatenate([np.asarray(residue.selection, dtype=int) for residue in self.segment.residues[i + 1:]])
            coordinates = self.segment.get_xyz(selection) - origin
            self.segment.set_xyz(coordinates @ rotation_matrix(axis, theta).T + origin, selection)
        return self.segment.coor.copy()


class Gate:
    def __init__(self, pdb_id: str, chain: str, resnum: int):
        path, split = source_path(pdb_id)
        self.pdb_id, self.chain, self.resnum, self.split = pdb_id, chain, resnum, split
        residue_id = (resnum, "")
        self.a_structure = Structure.fromfile(path).extract("altloc", ("", "A"))
        self.b_structure = Structure.fromfile(path).extract("altloc", ("", "B"))
        a_residue = self.a_structure[chain].conformers[0][residue_id]
        b_residue = self.b_structure[chain].conformers[0][residue_id]
        a_chain, b_chain = self.a_structure[chain].conformers[0], self.b_structure[chain].conformers[0]
        self.a_segment = next(segment for segment in a_chain.segments if any(residue.id == residue_id for residue in segment.residues))
        self.b_segment = next(segment for segment in b_chain.segments if any(residue.id == residue_id for residue in segment.residues))
        a_index, b_index = self.a_segment.find(residue_id), self.b_segment.find(residue_id)
        self.window = self.a_segment[a_index - 3:a_index + 4]
        self.b_window = self.b_segment[b_index - 3:b_index + 4]
        if len(self.window.residues) != 7 or len(self.b_window.residues) != 7 or [r.id for r in self.window.residues] != [r.id for r in self.b_window.residues]:
            raise RuntimeError("A/B strict seven-residue windows do not match")
        self.initial = self.window.coor.copy()
        self.indices = [index for residue in self.window.residues for index in local_indices(self.window, residue)]
        self.central_indices = local_indices(self.window, self.window.residues[3])
        self.target = np.asarray([coordinate for residue in self.b_window.residues for coordinate in [
            residue.coor[atom_local_index(residue, name)] for name in BACKBONE_NAMES
        ]])
        self.rotator = PhiPsiOmegaRotator(self.window)
        self.a_phi_psi, self.a_omega = dihedrals(self.window)
        self.b_phi_psi, self.b_omega = dihedrals(self.b_window)

    def forward(self, parameters: np.ndarray) -> np.ndarray:
        return self.rotator(parameters)[self.indices].copy()

    def seam(self, coordinates: np.ndarray) -> dict[str, object]:
        terminal = coordinates[-4:]
        a_terminal = self.initial[self.indices][-4:]
        translation = terminal[0] - a_terminal[0]
        orientation = rotation_vector(frame(*a_terminal[:3]).T @ frame(*terminal[:3]))
        lever_arm = 1.5
        return {
            "translation_A": translation.tolist(), "rotation_rad": orientation.tolist(),
            "rotation_deg": np.degrees(orientation).tolist(),
            "rotation_A_equivalent_at_1p5A": (lever_arm * orientation).tolist(),
            "six_component_A_equivalent": np.concatenate((translation, lever_arm * orientation)).tolist(),
            "norm_A_equivalent": float(np.linalg.norm(np.concatenate((translation, lever_arm * orientation)))),
        }

    def solve(self, initial: np.ndarray) -> dict[str, object]:
        result = least_squares(lambda parameters: (self.forward(parameters) - self.target).ravel(), initial,
                               method="lm", x_scale=10.0, max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        coordinates = self.forward(result.x)
        return {"parameters_deg": result.x.tolist(), "cost": float(result.cost), "nfev": int(result.nfev),
                "optimality": float(result.optimality), "status": int(result.status), "message": result.message,
                "full_window_backbone_rmsd_A": rmsd(coordinates, self.target),
                "central_backbone_rmsd_A": rmsd(coordinates[12:16], self.target[12:16]),
                "seam": self.seam(coordinates)}

    def validation(self) -> dict[str, object]:
        zero_error = float(np.max(np.abs(self.forward(np.zeros(self.rotator.ndofs)) - self.initial[self.indices])))
        command = np.zeros(self.rotator.ndofs); command[14:] = 1.0
        measured_phi_psi, measured_omega = dihedrals(self.coordinates_to_window(command))
        omega_error = wrapped_delta(self.a_omega + 1.0, measured_omega)
        return {"zero_parameter_max_abs_coordinate_error_A": zero_error,
                "one_degree_each_internal_omega_max_abs_error_deg": float(np.max(np.abs(omega_error)))}

    def coordinates_to_window(self, parameters: np.ndarray):
        self.rotator(parameters)
        return self.window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pdb-id", default="7UTC")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--resnum", default=52, type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gate = Gate(args.pdb_id, args.chain, args.resnum)
    torsion_seed = np.concatenate((wrapped_delta(gate.a_phi_psi, gate.b_phi_psi), wrapped_delta(gate.a_omega, gate.b_omega)))
    trials = {"zero": gate.solve(np.zeros(gate.rotator.ndofs)), "deposited_dihedral_delta": gate.solve(torsion_seed)}
    best_name, best = min(trials.items(), key=lambda item: item[1]["full_window_backbone_rmsd_A"])
    result = {
        "status": "complete", "site": f"{args.pdb_id}_{args.chain}_{args.resnum}",
        "parameter_count": {"qfit_BackboneRotator_phi_psi": gate.rotator.phi_psi_ndofs,
                            "new_internal_omega": gate.rotator.omega_ndofs, "A_prime_total": gate.rotator.ndofs},
        "parameterization": "qFit BackboneRotator phi/psi then six downstream C-N omega rotations; no density, no seam, no Rama",
        "validation": gate.validation(), "deposited_torsion_delta_seed_deg": torsion_seed.tolist(),
        "trials": trials, "best_trial": best_name, "best": best,
    }
    atomic_json(args.output / "result.json", result)
    atomic_json(args.output / "progress.json", {"status": "complete", "best_trial": best_name})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
