"""Audit paired SampleWorks ensembles against deposited alternate conformers.

SampleWorks emits independent full-structure models rather than occupancy-bearing
slots.  This audit therefore reports structural A/B recovery across the sampled
ensemble, conventional chemically equivalent side-chain RMSD, rotamer geometry,
and direct/crystallographic-symmetry clashes.  It deliberately does not invent
crystallographic occupancies for the samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import gemmi
import numpy as np
import torch

from density_denoiser.audit_five_site_endpoints import classify_rotamer
from density_denoiser.five_site_optimizer import dihedral, wrap_angles
from density_denoiser.residue_geometry import CHI_SPECS, symmetry_aware_rmsd


BACKBONE = {"N", "CA", "C", "O", "OXT"}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _find_residue(model, chain_name: str, residue_number: int):
    for chain in model:
        if chain.name != chain_name:
            continue
        for residue in chain:
            if residue.seqid.num == residue_number:
                return residue
    raise KeyError(f"residue not found: {chain_name}:{residue_number}")


def _atom_map(residue, altloc: str | None = None) -> dict[str, torch.Tensor]:
    result = {}
    for atom in residue:
        if atom.element.name == "H":
            continue
        atom_altloc = str(atom.altloc).strip("\x00 ")
        if altloc is not None and atom_altloc not in {"", altloc}:
            continue
        result[atom.name.strip()] = torch.tensor(atom.pos.tolist(), dtype=torch.float64)
    return result


def _nearest(candidate: np.ndarray, environment: np.ndarray) -> tuple[float, int, int]:
    distances = np.linalg.norm(candidate[:, None, :] - environment[None, :, :], axis=-1)
    flat = int(np.argmin(distances))
    atom_index, environment_index = np.unravel_index(flat, distances.shape)
    return float(distances[atom_index, environment_index]), int(atom_index), int(environment_index)


def _superpose_from_backbone(
    candidate_map: dict[str, torch.Tensor],
    reference_map: dict[str, torch.Tensor],
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Express generated side-chain coordinates in the deposited local backbone frame."""
    names = [name for name in ("N", "CA", "C", "O") if name in candidate_map]
    mobile = torch.stack([candidate_map[name] for name in names])
    reference = torch.stack([reference_map[name] for name in names])
    mobile_center = mobile.mean(dim=0)
    reference_center = reference.mean(dim=0)
    left, _, right_transpose = torch.linalg.svd(
        (mobile - mobile_center).T @ (reference - reference_center)
    )
    rotation = left @ right_transpose
    if torch.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    return (coordinates - mobile_center) @ rotation + reference_center


def _symmetry_environment(
    model,
    cell: gemmi.UnitCell,
    spacegroup,
    center: np.ndarray,
) -> tuple[np.ndarray | None, list[str]]:
    heavy_atoms = [
        (chain, residue, atom)
        for chain in model
        for residue in chain
        for atom in residue
        if atom.element.name != "H"
    ]
    coordinates, labels = [], []
    for operation_index, operation in enumerate(spacegroup.operations()):
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                for tz in (-1, 0, 1):
                    if operation_index == 0 and tx == ty == tz == 0:
                        continue
                    for chain, residue, atom in heavy_atoms:
                        transformed = operation.apply_to_xyz(cell.fractionalize(atom.pos).tolist())
                        position = cell.orthogonalize(gemmi.Fractional(
                            transformed[0] + tx,
                            transformed[1] + ty,
                            transformed[2] + tz,
                        ))
                        xyz = np.asarray(position.tolist())
                        if np.linalg.norm(xyz - center) <= 8.0:
                            coordinates.append(xyz)
                            labels.append(
                                f"sym{operation_index}[{tx},{ty},{tz}]/"
                                f"{chain.name}:{residue.name}{residue.seqid.num}:{atom.name.strip()}"
                            )
    return (np.asarray(coordinates) if coordinates else None), labels


def audit(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    truth = gemmi.read_structure(str(args.truth))
    truth_residue = _find_residue(truth[0], args.truth_chain, args.truth_residue)
    map_a = _atom_map(truth_residue, "A")
    map_b = _atom_map(truth_residue, "B")
    names = [
        atom.name.strip()
        for atom in truth_residue
        if str(atom.altloc).strip("\x00 ") == "B"
        and atom.element.name != "H"
        and atom.name.strip() not in BACKBONE
    ]
    if not names:
        raise ValueError("deposited B has no explicit side-chain heavy atoms")
    reference_a = torch.stack([map_a[name] for name in names])
    reference_b = torch.stack([map_b[name] for name in names])
    spec = CHI_SPECS[truth_residue.name]
    spacegroup = gemmi.find_spacegroup_by_name(truth.spacegroup_hm)
    if spacegroup is None:
        raise ValueError(f"unknown truth space group: {truth.spacegroup_hm}")

    all_rows: list[dict] = []
    condition_summaries = {}
    for condition, ensemble_path in (("raw", args.raw), ("denoised", args.denoised)):
        ensemble = gemmi.read_structure(str(ensemble_path))
        condition_rows = []
        for model_index, model in enumerate(ensemble):
            residue = _find_residue(model, args.output_chain, args.output_residue)
            candidate_map = _atom_map(residue)
            missing = [name for name in names if name not in candidate_map]
            if missing:
                raise ValueError(f"model {model_index} missing target atoms: {missing}")
            candidate = torch.stack([candidate_map[name] for name in names])
            rmsd_a = float(symmetry_aware_rmsd(candidate, reference_a, names, truth_residue.name))
            rmsd_b = float(symmetry_aware_rmsd(candidate, reference_b, names, truth_residue.name))

            local_candidate = _superpose_from_backbone(candidate_map, map_a, candidate)
            local_rmsd_a = float(symmetry_aware_rmsd(
                local_candidate, reference_a, names, truth_residue.name
            ))
            local_rmsd_b = float(symmetry_aware_rmsd(
                local_candidate, reference_b, names, truth_residue.name
            ))

            backbone_names = [name for name in ("N", "CA", "C", "O") if name in candidate_map]
            backbone_rmsd = float(torch.sqrt(torch.mean(torch.sum(torch.stack([
                candidate_map[name] - map_a[name] for name in backbone_names
            ]).square(), dim=-1))))
            chi_degrees = [
                math.degrees(float(wrap_angles(
                    dihedral(*(candidate_map[name] for name in quartet)) - torch.pi
                )))
                for quartet in spec["dihedrals"]
            ]
            rotamer, deviations, canonical = classify_rotamer(truth_residue.name, chi_degrees)

            direct_atoms = [
                (chain, other_residue, atom)
                for chain in model
                for other_residue in chain
                for atom in other_residue
                if atom.element.name != "H"
                and not (
                    chain.name == args.output_chain
                    and other_residue.seqid.num == args.output_residue
                )
            ]
            direct_xyz = np.asarray([atom.pos.tolist() for _, _, atom in direct_atoms])
            direct_labels = [
                f"{chain.name}:{other.name}{other.seqid.num}:{atom.name.strip()}"
                for chain, other, atom in direct_atoms
            ]
            xyz = candidate.numpy()
            direct_distance, direct_atom, direct_environment = _nearest(xyz, direct_xyz)

            symmetry_xyz, symmetry_labels = _symmetry_environment(
                model, truth.cell, spacegroup, xyz.mean(axis=0)
            )
            if symmetry_xyz is None:
                symmetry_distance = float("nan")
                closest_symmetry = ""
            else:
                symmetry_distance, symmetry_atom, symmetry_environment = _nearest(xyz, symmetry_xyz)
                closest_symmetry = f"{names[symmetry_atom]}--{symmetry_labels[symmetry_environment]}"
            no_clash = direct_distance >= args.clash_cutoff and (
                math.isnan(symmetry_distance) or symmetry_distance >= args.clash_cutoff
            )
            assignment = (
                "A" if rmsd_a < args.rmsd_cutoff and rmsd_a <= rmsd_b
                else "B" if rmsd_b < args.rmsd_cutoff
                else "other"
            )
            row = {
                "condition": condition,
                "sample": model_index,
                "assignment": assignment,
                "rmsd_to_A_conventional": rmsd_a,
                "rmsd_to_B_conventional": rmsd_b,
                "local_backbone_aligned_rmsd_to_A": local_rmsd_a,
                "local_backbone_aligned_rmsd_to_B": local_rmsd_b,
                "target_backbone_rmsd": backbone_rmsd,
                "chi_degrees": ";".join(f"{value:.4f}" for value in chi_degrees),
                "nearest_rotamer": rotamer,
                "rotamer_deviation_degrees": ";".join(f"{value:.4f}" for value in deviations),
                "canonical_like_30deg": canonical,
                "min_direct_distance": direct_distance,
                "closest_direct_atom": f"{names[direct_atom]}--{direct_labels[direct_environment]}",
                "min_symmetry_distance": symmetry_distance,
                "closest_symmetry_atom": closest_symmetry,
                "no_sub2A_clash": no_clash,
                "geometry_physical_valid": bool(canonical and no_clash),
            }
            condition_rows.append(row)
            all_rows.append(row)
            _atomic_csv(args.output / "sample_geometry_audit.csv", all_rows)
            _atomic_json(args.output / "progress.json", {
                "status": "auditing",
                "condition": condition,
                "completed_models": model_index + 1,
                "total_models_per_condition": len(ensemble),
            })

        found_a = any(row["assignment"] == "A" for row in condition_rows)
        found_b = any(row["assignment"] == "B" for row in condition_rows)
        local_a_samples = sum(
            row["local_backbone_aligned_rmsd_to_A"] < args.rmsd_cutoff
            and row["local_backbone_aligned_rmsd_to_A"]
            <= row["local_backbone_aligned_rmsd_to_B"]
            for row in condition_rows
        )
        local_b_samples = sum(
            row["local_backbone_aligned_rmsd_to_B"] < args.rmsd_cutoff
            for row in condition_rows
        )
        condition_summaries[condition] = {
            "samples": len(condition_rows),
            "A_samples": sum(row["assignment"] == "A" for row in condition_rows),
            "B_samples": sum(row["assignment"] == "B" for row in condition_rows),
            "other_samples": sum(row["assignment"] == "other" for row in condition_rows),
            "found_A": found_a,
            "found_B": found_b,
            "both_states_found_across_ensemble": found_a and found_b,
            "geometry_physical_valid_samples": sum(
                row["geometry_physical_valid"] for row in condition_rows
            ),
            "best_rmsd_to_A": min(row["rmsd_to_A_conventional"] for row in condition_rows),
            "best_rmsd_to_B": min(row["rmsd_to_B_conventional"] for row in condition_rows),
            "local_backbone_aligned_A_samples": local_a_samples,
            "local_backbone_aligned_B_samples": local_b_samples,
            "local_backbone_aligned_best_rmsd_to_A": min(
                row["local_backbone_aligned_rmsd_to_A"] for row in condition_rows
            ),
            "local_backbone_aligned_best_rmsd_to_B": min(
                row["local_backbone_aligned_rmsd_to_B"] for row in condition_rows
            ),
            "median_target_backbone_rmsd": float(np.median([
                row["target_backbone_rmsd"] for row in condition_rows
            ])),
        }

    payload = {
        "status": "complete",
        "site": f"{args.truth_chain}_{truth_residue.name}{args.truth_residue}",
        "rmsd_definition": "sqrt(mean_atoms(sum_xyz(delta^2))), equivalent labels minimized",
        "rmsd_cutoff_angstrom": args.rmsd_cutoff,
        "clash_cutoff_angstrom": args.clash_cutoff,
        "occupancy_audit": "not applicable: SampleWorks samples are not crystallographic occupancy slots",
        "conditions": condition_summaries,
    }
    _atomic_json(args.output / "summary.json", payload)
    _atomic_json(args.output / "progress.json", {"status": "complete", "models": len(all_rows)})
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--truth", type=Path, required=True)
    result.add_argument("--raw", type=Path, required=True)
    result.add_argument("--denoised", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--truth-chain", default="B")
    result.add_argument("--truth-residue", type=int, default=447)
    result.add_argument("--output-chain", default="Bxp")
    result.add_argument("--output-residue", type=int, default=47)
    result.add_argument("--rmsd-cutoff", type=float, default=1.0)
    result.add_argument("--clash-cutoff", type=float, default=2.0)
    return result


if __name__ == "__main__":
    audit(parser().parse_args())
