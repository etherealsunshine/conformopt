#!/usr/bin/env python3
"""Reconstruct and audit every Probe 4b learned endpoint conformation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import gemmi
import numpy as np
import torch
from scipy.spatial import cKDTree
from SFC_Torch import SFcalculator

from probe4_core import dihedral, torsion_to_coords, wrap_angles


ROOT = Path(__file__).parent
OUT = ROOT / "probe4b_results" / "endpoint_audit"
EXPERIMENTS = {
    "A_synthetic_fobs": ROOT / "probe4b_results/experiment_A_synthetic_fobs/altloc_test/trajectories.json",
    "B_localized_sf": ROOT / "probe4b_results/experiment_B_localized_sf/altloc_test/trajectories.json",
    "C_realspace_local": ROOT / "probe4b_results/experiment_C_realspace_local/altloc_test/trajectories.json",
}
ASP_ALLOWS_QUADRATURE = False
SPECS = {
    "ARG": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
                      ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
        "rotations": (("CA", "CB", ("CG", "CD", "NE", "CZ", "NH1", "NH2")),
                      ("CB", "CG", ("CD", "NE", "CZ", "NH1", "NH2")),
                      ("CG", "CD", ("NE", "CZ", "NH1", "NH2")),
                      ("CD", "NE", ("CZ", "NH1", "NH2"))),
    },
    "MET": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"),
                      ("CB", "CG", "SD", "CE")),
        "rotations": (("CA", "CB", ("CG", "SD", "CE")),
                      ("CB", "CG", ("SD", "CE")), ("CG", "SD", ("CE",))),
    },
    "ASP": {
        "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
        "rotations": (("CA", "CB", ("CG", "OD1", "OD2")),
                      ("CB", "CG", ("OD1", "OD2"))),
    },
}


def atom_map(residue: gemmi.Residue, alt: str) -> dict[str, torch.Tensor]:
    result = {}
    for atom in residue:
        atom_alt = atom.altloc if atom.altloc not in ("\x00", " ") else ""
        if atom_alt in ("", alt):
            result[atom.name.strip()] = torch.tensor(atom.pos.tolist(), dtype=torch.float32)
    return result


def canonical_centers(resname: str, chi_index: int) -> list[tuple[str, float]]:
    # Terminal carboxylate/guanidinium torsions are planar and symmetry-related.
    if resname == "ASP" and chi_index == 1 and ASP_ALLOWS_QUADRATURE:
        return [("p0", 0.0), ("p90", 90.0), ("p-90", -90.0),
                ("p180", 180.0), ("p-180", -180.0)]
    if (resname == "ASP" and chi_index == 1) or (resname == "ARG" and chi_index == 3):
        return [("p0", 0.0), ("p180", 180.0), ("p-180", -180.0)]
    if resname == "MET" and chi_index == 2:
        return [("m-", -90.0), ("m+", 90.0), ("t", 180.0), ("t", -180.0)]
    return [("g-", -60.0), ("g+", 60.0), ("t", 180.0), ("t", -180.0)]


def angular_delta_deg(value: float, target: float) -> float:
    return abs(((value - target + 180.0) % 360.0) - 180.0)


def classify_rotamer(resname: str, angles: list[float]) -> tuple[str, list[float], bool]:
    labels, deviations = [], []
    for index, angle in enumerate(angles):
        candidates = canonical_centers(resname, index)
        label, center = min(candidates, key=lambda item: angular_delta_deg(angle, item[1]))
        labels.append(label)
        deviations.append(angular_delta_deg(angle, center))
    return "/".join(labels), deviations, max(deviations) <= 30.0


def write_segment_pdb(
    path: Path,
    structure: gemmi.Structure,
    chain_name: str,
    site_number: int | None = None,
    candidate_names: list[str] | None = None,
    candidate_coords: np.ndarray | None = None,
) -> None:
    candidate_lookup = {
        name: candidate_coords[index]
        for index, name in enumerate(candidate_names or [])
    }
    lines, serial = [], 1
    chain = next(chain for chain in structure[0] if chain.name == chain_name)
    for residue in chain:
        if not 108 <= residue.seqid.num <= 136:
            continue
        chosen = {}
        for atom in residue:
            alt = atom.altloc if atom.altloc not in ("\x00", " ") else ""
            name = atom.name.strip()
            if alt == "" or alt == "A":
                chosen[name] = atom
        # 2O1K omits the terminal phenolic oxygen on Tyr131.  tmol can build
        # missing leaf atoms, but OH is treated as the parent of a hydrogen and
        # therefore must be present in the input.  Place it in the idealized
        # continuation of the aromatic ring, 1.36 A beyond CZ.
        synthetic_atoms = {}
        if residue.name == "TYR" and "OH" not in chosen and {"CE1", "CE2", "CZ"} <= chosen.keys():
            ce_midpoint = 0.5 * (
                np.asarray(chosen["CE1"].pos.tolist()) + np.asarray(chosen["CE2"].pos.tolist())
            )
            cz = np.asarray(chosen["CZ"].pos.tolist())
            direction = cz - ce_midpoint
            synthetic_atoms["OH"] = cz + 1.36 * direction / np.linalg.norm(direction)
        ordered_atoms = [(name, atom, None) for name, atom in chosen.items()]
        ordered_atoms.extend((name, chosen["CZ"], xyz) for name, xyz in synthetic_atoms.items())
        for name, atom, synthetic_xyz in ordered_atoms:
            default_xyz = synthetic_xyz if synthetic_xyz is not None else np.asarray(atom.pos.tolist())
            xyz = candidate_lookup.get(name, default_xyz) if residue.seqid.num == site_number else default_xyz
            element = ("O" if name == "OH" and synthetic_xyz is not None else atom.element.name.upper()).rjust(2)
            lines.append(
                f"ATOM  {serial:5d} {name:^4s} {residue.name:>3s} {chain_name:1s}"
                f"{residue.seqid.num:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                f"{1.00:6.2f}{atom.b_iso:6.2f}          {element:>2s}"
            )
            serial += 1
    lines.extend(("TER", "END"))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    torch.manual_seed(91)
    OUT.mkdir(parents=True, exist_ok=True)
    visualization = OUT / "visualization"
    visualization.mkdir(exist_ok=True)
    structure = gemmi.read_structure(str(ROOT / "data/2O1K.pdb"))
    cell = structure.cell
    spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
    all_atoms = [(chain, residue, atom) for chain in structure[0] for residue in chain for atom in residue]

    trajectories = {name: json.loads(path.read_text()) for name, path in EXPERIMENTS.items()}

    # Fixed observed-map coefficients for a quantitative density check.
    sfcalc = SFcalculator(str(ROOT / "data/2O1K.pdb"), str(ROOT / "data/2O1K.mtz"), device=torch.device("cpu"))
    fobs = sfcalc.Fo.detach()
    fcalc = sfcalc.calc_fprotein(Return=True).detach()
    hkl = torch.tensor(sfcalc.HKL_array, dtype=torch.float32)
    valid = torch.isfinite(fobs) & (fobs > 0)
    calc_amp = fcalc.abs()
    calc_to_obs = (calc_amp[valid] @ fobs[valid]) / calc_amp[valid].square().sum()
    phase = fcalc / calc_amp.clamp_min(1e-8)
    map_coefficients = (2 * fobs - calc_to_obs * calc_amp) * phase
    orth_to_frac = sfcalc.orth2frac_tensor.detach()

    def raw_density(points: torch.Tensor) -> torch.Tensor:
        frac = points @ orth_to_frac.T
        phase_values = 2 * torch.pi * (hkl[valid] @ frac.T)
        return 2 * torch.real(map_coefficients[valid] @ torch.exp(-1j * phase_values)) / int(valid.sum())

    # Generate tmol-compatible A-conformer base segments for both chains.
    for chain_name in ("A", "B"):
        write_segment_pdb(visualization / f"base_chain_{chain_name}.pdb", structure, chain_name)

    endpoint_rows, site_summaries, tmol_inputs = [], [], {"sites": []}
    for chain in structure[0]:
        for residue in chain:
            key = f"{chain.name}_{residue.name}{residue.seqid.num}"
            if key not in {row["site"] for records in trajectories.values() for row in records}:
                continue
            spec = SPECS[residue.name]
            map_a, map_b = atom_map(residue, "A"), atom_map(residue, "B")
            indices = [
                index for index, (candidate_chain, candidate_residue, atom) in enumerate(all_atoms)
                if candidate_chain.name == chain.name
                and candidate_residue.seqid.num == residue.seqid.num
                and atom.altloc == "B"
                and atom.name.strip() not in {"N", "CA", "C", "O"}
            ]
            names = [all_atoms[index][2].name.strip() for index in indices]
            template = torch.stack([map_a[name] for name in names])
            deposited_b = torch.stack([map_b[name] for name in names])
            fixed = {name: value for name, value in map_a.items() if name not in names}

            def coords(delta: torch.Tensor) -> torch.Tensor:
                return torsion_to_coords(template, names, delta, list(spec["rotations"]), fixed)

            def actual_chi(candidate: torch.Tensor) -> list[float]:
                lookup = dict(fixed)
                lookup.update({name: candidate[index] for index, name in enumerate(names)})
                return [
                    # probe4_core's signed internal torsion convention differs
                    # from the crystallographic chi convention by 180 degrees.
                    math.degrees(float(wrap_angles(dihedral(*(lookup[name] for name in quartet)) - torch.pi)))
                    for quartet in spec["dihedrals"]
                ]

            chi_a = actual_chi(template)
            chi_b = actual_chi(deposited_b)
            _, dev_a, canonical_a = classify_rotamer(residue.name, chi_a)
            _, dev_b, canonical_b = classify_rotamer(residue.name, chi_b)

            # Environment for direct and crystallographic-symmetry clash checks.
            environment_xyz, environment_labels, environment_elements = [], [], []
            for candidate_chain, candidate_residue, atom in all_atoms:
                if candidate_chain.name == chain.name and candidate_residue.seqid.num == residue.seqid.num:
                    continue
                environment_xyz.append(atom.pos.tolist())
                environment_labels.append(f"{candidate_chain.name}:{candidate_residue.name}{candidate_residue.seqid.num}:{atom.name.strip()}")
                environment_elements.append(atom.element.name.upper())
            environment_xyz_np = np.asarray(environment_xyz)
            direct_tree = cKDTree(environment_xyz_np)

            symmetry_xyz, symmetry_labels = [], []
            for operation_index, operation in enumerate(spacegroup.operations()):
                for tx in (-1, 0, 1):
                    for ty in (-1, 0, 1):
                        for tz in (-1, 0, 1):
                            identity = operation_index == 0 and tx == ty == tz == 0
                            if identity:
                                continue
                            for candidate_chain, candidate_residue, atom in all_atoms:
                                fractional = cell.fractionalize(atom.pos)
                                transformed = operation.apply_to_xyz(fractional.tolist())
                                transformed = gemmi.Fractional(transformed[0] + tx, transformed[1] + ty, transformed[2] + tz)
                                orthogonal = cell.orthogonalize(transformed)
                                symmetry_xyz.append(orthogonal.tolist())
                                symmetry_labels.append(
                                    f"sym{operation_index}[{tx},{ty},{tz}]/{candidate_chain.name}:"
                                    f"{candidate_residue.name}{candidate_residue.seqid.num}:{atom.name.strip()}"
                                )
            symmetry_xyz_np = np.asarray(symmetry_xyz)
            symmetry_tree = cKDTree(symmetry_xyz_np)

            # Reference density distribution and A/B atom-density support.
            grid_axis = torch.linspace(-4, 4, 9)
            grid = torch.cartesian_prod(grid_axis, grid_axis, grid_axis) + map_a["CA"]
            grid_density = raw_density(grid)
            grid_mean, grid_std = grid_density.mean(), grid_density.std().clamp_min(1e-6)

            def density_z(candidate: torch.Tensor) -> float:
                return float(((raw_density(candidate) - grid_mean) / grid_std).mean())

            def spatial_metrics(candidate: torch.Tensor) -> dict:
                xyz = candidate.detach().numpy()
                direct_distances, direct_indices = direct_tree.query(xyz, k=1)
                symmetry_distances, symmetry_indices = symmetry_tree.query(xyz, k=1)
                direct_atom = int(np.argmin(direct_distances))
                symmetry_atom = int(np.argmin(symmetry_distances))
                polar_contacts = []
                for atom_index, name in enumerate(names):
                    element = all_atoms[indices[atom_index]][2].element.name.upper()
                    if element not in {"N", "O", "S"}:
                        continue
                    nearby = direct_tree.query_ball_point(xyz[atom_index], 3.5)
                    for neighbor_index in nearby:
                        distance = float(np.linalg.norm(xyz[atom_index] - environment_xyz_np[neighbor_index]))
                        if environment_elements[neighbor_index] in {"N", "O", "S"} and 2.2 <= distance <= 3.5:
                            polar_contacts.append(f"{name}--{environment_labels[neighbor_index]}:{distance:.2f}A")
                return {
                    "min_direct_distance": float(direct_distances[direct_atom]),
                    "closest_direct_atom": f"{names[direct_atom]}--{environment_labels[int(direct_indices[direct_atom])]}",
                    "min_symmetry_distance": float(symmetry_distances[symmetry_atom]),
                    "closest_symmetry_atom": f"{names[symmetry_atom]}--{symmetry_labels[int(symmetry_indices[symmetry_atom])]}",
                    "polar_contacts": polar_contacts,
                    "density_z_mean": density_z(candidate),
                }

            controls = {"A": template, "B": deposited_b}
            random_rotamers = []
            for _ in range(10):
                desired = []
                for chi_index in range(len(spec["dihedrals"])):
                    centers = canonical_centers(residue.name, chi_index)
                    desired.append(math.radians(centers[torch.randint(len(centers), (1,)).item()][1]))
                # Determine rotation direction for this implementation from a finite difference.
                signs = []
                base_angles = torch.tensor([math.radians(value) for value in chi_a])
                for chi_index in range(len(desired)):
                    probe = torch.zeros(len(desired)); probe[chi_index] = 0.01
                    moved = torch.tensor([math.radians(value) for value in actual_chi(coords(probe))])
                    signs.append(1.0 if float(wrap_angles(moved[chi_index] - base_angles[chi_index])) > 0 else -1.0)
                delta = torch.tensor(signs) * wrap_angles(torch.tensor(desired) - base_angles)
                random_rotamers.append(coords(delta).detach())

            site_tmol = {
                "site": key,
                "chain": chain.name,
                "residue_number": residue.seqid.num,
                "residue_type": residue.name,
                "atom_names": names,
                "A": template.tolist(),
                "B": deposited_b.tolist(),
                "random_rotamers": [value.tolist() for value in random_rotamers],
                "experiments": {},
            }

            comparison_coords = {"A": template, "B": deposited_b}
            for experiment, records in trajectories.items():
                site_records = [row for row in records if row["site"] == key]
                endpoint_coords = [coords(torch.tensor(row["final_chi"])).detach() for row in site_records]
                site_tmol["experiments"][experiment] = [value.tolist() for value in endpoint_coords]
                comparison_coords[experiment] = endpoint_coords[0]
                for row, candidate in zip(site_records, endpoint_coords):
                    angles = actual_chi(candidate)
                    state, deviations, canonical_like = classify_rotamer(residue.name, angles)
                    spatial = spatial_metrics(candidate)
                    endpoint_rows.append({
                        "experiment": experiment,
                        "site": key,
                        "start": row["start"],
                        "rmsd_to_A": row["rmsd_to_A"],
                        "rmsd_to_B": row["rmsd_to_B"],
                        "chi_degrees": ";".join(f"{value:.2f}" for value in angles),
                        "nearest_state": state,
                        "canonical_deviation_degrees": ";".join(f"{value:.2f}" for value in deviations),
                        "canonical_like_30deg": canonical_like,
                        **{key2: value for key2, value in spatial.items() if key2 != "polar_contacts"},
                        "polar_contacts": ";".join(spatial["polar_contacts"]),
                    })
            tmol_inputs["sites"].append(site_tmol)

            # Reference metrics plus representative learned structures.
            reference = {}
            for label, candidate in {**controls, **comparison_coords}.items():
                state, deviations, canonical_like = classify_rotamer(residue.name, actual_chi(candidate))
                reference[label] = {
                    "chi_degrees": actual_chi(candidate),
                    "nearest_state": state,
                    "canonical_deviations": deviations,
                    "canonical_like_30deg": canonical_like,
                    **spatial_metrics(candidate),
                }
            site_summaries.append({"site": key, "references": reference})

            # PDBs load as aligned objects in either PyMOL or ChimeraX.
            safe_key = key.replace("_", "-")
            for label, candidate in comparison_coords.items():
                write_segment_pdb(
                    visualization / f"{safe_key}_{label}.pdb",
                    structure, chain.name, residue.seqid.num, names, candidate.numpy(),
                )

    fieldnames = list(endpoint_rows[0])
    with (OUT / "endpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(endpoint_rows)
    (OUT / "endpoint_summary.json").write_text(json.dumps(site_summaries, indent=2) + "\n")
    (OUT / "tmol_inputs.json").write_text(json.dumps(tmol_inputs) + "\n")

    # Ready-to-run visualization scripts. Density-map loading is left as an
    # explicit optional line because no PyMOL/ChimeraX executable is installed.
    pymol_lines = ["reinitialize", "bg_color white", "set stick_radius, 0.18"]
    colors = {"A": "green", "B": "cyan", "A_synthetic_fobs": "magenta", "B_localized_sf": "orange", "C_realspace_local": "yellow"}
    for site_summary in site_summaries:
        key = site_summary["site"]; safe_key = key.replace("_", "-")
        residue_number = int("".join(char for char in key if char.isdigit()))
        for label, color in colors.items():
            obj = f"{safe_key}_{label}".replace("-", "_")
            pymol_lines.append(f"load {safe_key}_{label}.pdb, {obj}")
            pymol_lines.append(f"hide everything, {obj}")
            pymol_lines.append(f"show sticks, {obj} and resi {residue_number}")
            pymol_lines.append(f"color {color}, {obj} and resi {residue_number}")
        pymol_lines.append(f"show cartoon, {safe_key}_A".replace("-", "_"))
    pymol_lines.extend(("set ray_opaque_background, off", "zoom", "orient"))
    (visualization / "probe4b_endpoints.pml").write_text("\n".join(pymol_lines) + "\n")

    chimerax_lines = ["set bgColor white"]
    model = 1
    for site_summary in site_summaries:
        safe_key = site_summary["site"].replace("_", "-")
        for label, color in colors.items():
            chimerax_lines.append(f"open {safe_key}_{label}.pdb")
            chimerax_lines.append(f"color #{model} {color}")
            model += 1
    chimerax_lines.extend(("style stick", "view"))
    (visualization / "probe4b_endpoints.cxc").write_text("\n".join(chimerax_lines) + "\n")
    print(f"wrote {len(endpoint_rows)} endpoint rows to {OUT}")


if __name__ == "__main__":
    main()
