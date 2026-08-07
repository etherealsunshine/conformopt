"""Diagnose production-mask density loss and canonical reachability coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np
import torch

from density_denoiser.data_pipeline import _grid_coordinates
from density_denoiser.five_site_optimizer import (
    _alt_atom_map,
    _canonical_state_pool,
    _unique_canonical_centers_radians,
)
from density_denoiser.residue_geometry import (
    CHI_SPECS,
    canonical_width_degrees,
)
from experiments.probe4.core import dihedral, torsion_to_coords, wrap_angles


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    return float(np.dot(left, right) / (
        np.linalg.norm(left) * np.linalg.norm(right)
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument(
        "--ensemble-table", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-radius", type=float, default=4.0)
    parser.add_argument("--padding", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    records = {}
    for path in args.selection:
        for record in json.loads(path.read_text())["sites"]:
            records[record["key"]] = record

    ensemble_by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in args.ensemble_table:
        for row in read_csv(path):
            ensemble_by_site[row["site"]].append(row)
    if set(ensemble_by_site) != set(records):
        raise RuntimeError(
            f"ensemble/selection mismatch: {sorted(set(records) ^ set(ensemble_by_site))}"
        )

    device = torch.device("cpu")
    site_rows: list[dict[str, object]] = []
    outside_rows: list[dict[str, object]] = []
    canonical_exception_rows: list[dict[str, object]] = []
    deposited_state_rows: list[dict[str, object]] = []

    for site, record in sorted(records.items()):
        structure = gemmi.read_structure(record["pdb_path"])
        residue = next(
            residue
            for chain in structure[0] if chain.name == record["chain"]
            for residue in chain
            if residue.seqid.num == int(record["residue_number"])
            and residue.seqid.icode == record["insertion_code"]
        )
        map_a = _alt_atom_map(residue, "A", device)
        map_b = _alt_atom_map(residue, "B", device)
        b_atoms = [
            atom for atom in residue
            if atom.altloc == "B"
            and atom.element.name != "H"
            and atom.name.strip() not in {"N", "CA", "C", "O"}
        ]
        names = [atom.name.strip() for atom in b_atoms]
        a_atoms = [
            atom for atom in residue
            if atom.altloc == "A" and atom.name.strip() in names
        ]
        occ_a = float(np.median([atom.occ for atom in a_atoms]))
        occ_b = float(np.median([atom.occ for atom in b_atoms]))
        equal_occupancy = math.isclose(occ_a, occ_b, abs_tol=1e-6)
        minor = "equal" if equal_occupancy else ("A" if occ_a < occ_b else "B")
        major = "equal" if equal_occupancy else ("B" if minor == "A" else "A")

        pair = np.load(record["pair_path"], allow_pickle=False)
        metadata = json.loads(str(pair["metadata"].item()))
        center = np.asarray(metadata["center"], dtype=np.float32)
        patch_center = np.asarray(
            metadata.get("patch_center_crystal", metadata["center"]),
            dtype=np.float32,
        )
        coordinates = _grid_coordinates(patch_center, 32, 0.5, None)
        production_mask = (
            np.linalg.norm(coordinates - center, axis=-1)
            <= args.production_radius
        )
        full_grid = torch.tensor(
            coordinates.reshape(-1, 3), dtype=torch.float32
        )

        xyz_a = torch.stack([map_a[name] for name in names])
        xyz_b = torch.stack([map_b[name] for name in names])
        sigma2 = torch.tensor([
            max(float(atom.b_iso) / (8.0 * math.pi**2), 0.04)
            for atom in b_atoms
        ])
        weights = torch.tensor([
            atom.element.atomic_number * atom.occ / max(occ_b, 1e-6)
            for atom in b_atoms
        ])

        def atom_density(xyz: torch.Tensor) -> np.ndarray:
            distance2 = (
                full_grid[:, None, :] - xyz[None, :, :]
            ).square().sum(dim=-1)
            normalization = (2.0 * math.pi * sigma2).pow(-1.5)
            values = (
                weights[None, :]
                * normalization[None, :]
                * torch.exp(-distance2 / (2.0 * sigma2[None, :]))
            ).sum(dim=1)
            return values.reshape(coordinates.shape[:-1]).numpy()

        density_a = atom_density(xyz_a)
        density_b = atom_density(xyz_b)
        outside_fraction_a = float(
            density_a[~production_mask].sum() / density_a.sum()
        )
        outside_fraction_b = float(
            density_b[~production_mask].sum() / density_b.sum()
        )

        for state, xyz in (("A", xyz_a), ("B", xyz_b)):
            for atom_name, atom_xyz in zip(names, xyz):
                radius = float(torch.linalg.vector_norm(
                    atom_xyz - torch.tensor(center)
                ))
                if radius > args.production_radius:
                    outside_rows.append({
                        "site": site,
                        "residue_type": residue.name,
                        "state": state,
                        "occupancy_rank": (
                            "equal"
                            if equal_occupancy
                            else ("minor" if state == minor else "major")
                        ),
                        "atom": atom_name,
                        "distance_from_mask_center_A": radius,
                        "excess_A": radius - args.production_radius,
                    })

        spec = CHI_SPECS[residue.name]
        template = xyz_a
        fixed_lookup = {
            name: value for name, value in map_a.items() if name not in names
        }

        def coordinates_from_chi(delta: torch.Tensor) -> torch.Tensor:
            return torsion_to_coords(
                template,
                names,
                wrap_angles(delta),
                list(spec["rotations"]),
                fixed_lookup,
            )

        def physical_chi(candidate: torch.Tensor) -> torch.Tensor:
            lookup = dict(fixed_lookup)
            lookup.update({
                name: candidate[index] for index, name in enumerate(names)
            })
            return torch.stack([
                wrap_angles(
                    dihedral(*(lookup[name] for name in quartet)) - torch.pi
                )
                for quartet in spec["dihedrals"]
            ])

        base_physical = physical_chi(xyz_a)
        directions = []
        for index in range(len(spec["rotations"])):
            probe = torch.zeros(len(spec["rotations"]))
            probe[index] = 0.01
            moved = physical_chi(coordinates_from_chi(probe))
            direction = torch.sign(wrap_angles(
                moved[index] - base_physical[index]
            ))
            directions.append(float(direction) or 1.0)
        directions = torch.tensor(directions)
        _, canonical_physical = _canonical_state_pool(
            residue.name, len(spec["rotations"]), device=device
        )
        canonical_delta = directions * wrap_angles(
            canonical_physical - base_physical
        )
        canonical_xyz = torch.stack([
            coordinates_from_chi(delta) for delta in canonical_delta
        ])
        canonical_union = canonical_xyz.reshape(-1, 3)

        state_outside_canonical = {}
        for state, xyz in (("A", xyz_a), ("B", xyz_b)):
            minimum_any = torch.cdist(xyz, canonical_union).min(dim=1).values
            same_name = torch.stack([
                torch.linalg.vector_norm(
                    canonical_xyz[:, atom_index, :] - xyz[atom_index],
                    dim=1,
                ).min()
                for atom_index in range(len(names))
            ])
            state_outside_canonical[state] = int(
                (minimum_any > args.padding).sum()
            )
            state_physical = physical_chi(xyz)
            chi_deviations = []
            for chi_index, value in enumerate(state_physical):
                centers = torch.tensor(
                    _unique_canonical_centers_radians(
                        residue.name, chi_index
                    )
                )
                chi_deviations.append(float(torch.rad2deg(
                    torch.abs(wrap_angles(value - centers)).min()
                )))
            state_rmsds = torch.sqrt(torch.mean(torch.sum(
                (canonical_xyz - xyz[None, :, :]).square(), dim=-1
            ), dim=-1))
            deposited_state_rows.append({
                "site": site,
                "residue_type": residue.name,
                "state": state,
                "occupancy_rank": (
                    "equal"
                    if equal_occupancy
                    else ("minor" if state == minor else "major")
                ),
                "chi_deviation_from_nearest_center_degrees": ";".join(
                    f"{value:.6f}" for value in chi_deviations
                ),
                "maximum_chi_deviation_degrees": max(chi_deviations),
                "production_marginal_rotamer_pass": all(
                    deviation <= canonical_width_degrees(
                        residue.name, chi_index
                    )
                    for chi_index, deviation in enumerate(chi_deviations)
                ),
                "nearest_enumerated_fixed_label_rmsd_A": float(
                    state_rmsds.min()
                ),
                "atoms_outside_1A_canonical_union": int(
                    (minimum_any > args.padding).sum()
                ),
            })
            for atom_index, atom_name in enumerate(names):
                if minimum_any[atom_index] > args.padding:
                    canonical_exception_rows.append({
                        "site": site,
                        "residue_type": residue.name,
                        "state": state,
                        "occupancy_rank": (
                            "equal"
                            if equal_occupancy
                            else ("minor" if state == minor else "major")
                        ),
                        "atom": atom_name,
                        "nearest_any_canonical_atom_A": float(
                            minimum_any[atom_index]
                        ),
                        "nearest_same_atom_canonical_A": float(
                            same_name[atom_index]
                        ),
                        "padding_A": args.padding,
                        "same_atom_sets": True,
                    })

        ensembles = ensemble_by_site[site]
        if equal_occupancy:
            found_minor = found_major = major_only = None
        else:
            found_minor = sum(
                row[f"found_{minor}_conventional"].lower() == "true"
                for row in ensembles
            )
            found_major = sum(
                row[f"found_{major}_conventional"].lower() == "true"
                for row in ensembles
            )
            major_only = sum(
                row[f"found_{major}_conventional"].lower() == "true"
                and row[f"found_{minor}_conventional"].lower() != "true"
                for row in ensembles
            )

        relative_a = occ_a / (occ_a + occ_b)
        relative_b = occ_b / (occ_a + occ_b)
        target = relative_a * density_a + relative_b * density_b
        collapsed = (
            density_a if major in {"A", "equal"} else density_b
        )
        production_signal = 1.0 - correlation(
            target[production_mask], collapsed[production_mask]
        )

        site_rows.append({
            "site": site,
            "residue_type": residue.name,
            "minor_state": minor,
            "major_state": major,
            "minor_occupancy": min(relative_a, relative_b),
            "major_occupancy": max(relative_a, relative_b),
            "A_density_outside_fraction": outside_fraction_a,
            "B_density_outside_fraction": outside_fraction_b,
            "minor_density_outside_fraction": (
                ""
                if equal_occupancy
                else (
                    outside_fraction_a if minor == "A" else outside_fraction_b
                )
            ),
            "major_density_outside_fraction": (
                ""
                if equal_occupancy
                else (
                    outside_fraction_b if minor == "A" else outside_fraction_a
                )
            ),
            "minor_found": found_minor,
            "major_found": found_major,
            "minor_failure_rate": (
                "" if equal_occupancy else 1.0 - found_minor / len(ensembles)
            ),
            "major_only_minor_misses": major_only,
            "major_only_minor_miss_rate": (
                "" if equal_occupancy else major_only / len(ensembles)
            ),
            "production_major_collapse_correlation_loss": production_signal,
            "deposited_A_atoms_outside_canonical_union": (
                state_outside_canonical["A"]
            ),
            "deposited_B_atoms_outside_canonical_union": (
                state_outside_canonical["B"]
            ),
            "canonical_states_enumerated": int(canonical_xyz.shape[0]),
        })

    minor_out = np.asarray([
        float(row["minor_density_outside_fraction"]) for row in site_rows
        if row["minor_state"] != "equal"
    ])
    minor_failure = np.asarray([
        float(row["minor_failure_rate"]) for row in site_rows
        if row["minor_state"] != "equal"
    ])
    major_only_failure = np.asarray([
        float(row["major_only_minor_miss_rate"]) for row in site_rows
        if row["minor_state"] != "equal"
    ])
    args.output.mkdir(parents=True)
    atomic_csv(args.output / "production_mask_by_site.csv", site_rows)
    atomic_csv(args.output / "production_mask_outside_atoms.csv", outside_rows)
    atomic_csv(
        args.output / "canonical_reachability_exceptions.csv",
        canonical_exception_rows,
    )
    atomic_csv(
        args.output / "deposited_state_canonical_coverage.csv",
        deposited_state_rows,
    )
    atomic_json(args.output / "summary.json", {
        "diagnostic_only": True,
        "production_changed": False,
        "metric_changed": False,
        "production_radius_A": args.production_radius,
        "canonical_union_padding_A": args.padding,
        "outside_deposited_atoms": len(outside_rows),
        "canonical_union_deposited_exceptions": len(
            canonical_exception_rows
        ),
        "minor_density_outside_vs_minor_failure_pearson": float(
            np.corrcoef(minor_out, minor_failure)[0, 1]
        ),
        "minor_density_outside_vs_major_only_miss_pearson": float(
            np.corrcoef(minor_out, major_only_failure)[0, 1]
        ),
    })


if __name__ == "__main__":
    main()
