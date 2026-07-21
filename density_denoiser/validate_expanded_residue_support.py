from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import gemmi
import torch

from probe4_core import dihedral, torsion_to_coords, wrap_angles

from .five_site_optimizer import _alt_atom_map
from .residue_geometry import CHI_SPECS, symmetry_aware_rmsd


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate production kinematics for a frozen residue panel"
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-kinematic-rmsd", type=float, default=0.1)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    rows: list[dict] = []
    for record in selection["sites"]:
        structure = gemmi.read_structure(record["pdb_path"])
        residue = next(
            residue for chain in structure[0] if chain.name == record["chain"]
            for residue in chain
            if residue.seqid.num == int(record["residue_number"])
            and residue.seqid.icode == record["insertion_code"]
        )
        spec = CHI_SPECS[residue.name]
        device = torch.device("cpu")
        map_a = _alt_atom_map(residue, "A", device)
        map_b = _alt_atom_map(residue, "B", device)
        b_atoms = [
            atom for atom in residue
            if atom.altloc == "B"
            and atom.element.name != "H"
            and atom.name.strip() not in {"N", "CA", "C", "O"}
        ]
        names = [atom.name.strip() for atom in b_atoms]
        template = torch.stack([map_a[name] for name in names])
        deposited_b = torch.stack([map_b[name] for name in names])
        fixed = {name: value for name, value in map_a.items() if name not in names}

        def coordinates(delta: torch.Tensor) -> torch.Tensor:
            return torsion_to_coords(
                template, names, delta, list(spec["rotations"]), fixed
            )

        chi_a = torch.stack([
            dihedral(*(map_a[name] for name in quartet))
            for quartet in spec["dihedrals"]
        ])
        chi_b = torch.stack([
            dihedral(*(map_b[name] for name in quartet))
            for quartet in spec["dihedrals"]
        ])
        delta = wrap_angles(chi_b - chi_a)
        plus, minus = coordinates(delta), coordinates(-delta)
        plus_error = symmetry_aware_rmsd(plus, deposited_b, names, residue.name)
        minus_error = symmetry_aware_rmsd(minus, deposited_b, names, residue.name)
        if minus_error < plus_error:
            delta = -delta
        kinematic_error = float(symmetry_aware_rmsd(
            coordinates(delta), deposited_b, names, residue.name
        ))
        identity_error = float(symmetry_aware_rmsd(
            coordinates(torch.zeros_like(delta)), template, names, residue.name
        ))

        probe = torch.linspace(-0.37, 0.41, len(delta), requires_grad=True)
        probe_coordinates = coordinates(probe)
        weights = torch.linspace(
            0.1, 1.0, probe_coordinates.numel()
        ).reshape_as(probe_coordinates)
        objective = (probe_coordinates * weights).sum()
        gradient = torch.autograd.grad(objective, probe)[0]
        finite_gradient = bool(torch.isfinite(gradient).all())
        responsive_chi = int((gradient.abs() > 1e-8).sum())
        passed = (
            identity_error <= 1e-6
            and kinematic_error <= args.max_kinematic_rmsd
            and finite_gradient
            and responsive_chi == len(delta)
        )
        rows.append({
            "site": record["key"],
            "residue_name": residue.name,
            "n_chi": len(delta),
            "sidechain_heavy_atoms": len(names),
            "identity_rmsd_conventional": identity_error,
            "kinematic_B_rmsd_conventional": kinematic_error,
            "finite_gradient": finite_gradient,
            "responsive_chi": responsive_chi,
            "gradient": ";".join(f"{value:.9g}" for value in gradient.tolist()),
            "passed": passed,
        })
        atomic_csv(args.output / "site_validation.csv", rows)
        atomic_json(args.output / "stage_manifest.json", {
            "status": "running",
            "completed_sites": len(rows),
            "total_sites": len(selection["sites"]),
        })
        print(json.dumps({"site": record["key"], "passed": passed}), flush=True)

    failures = [row for row in rows if not row["passed"]]
    summary = {
        "status": "passed" if not failures else "failed",
        "sites": len(rows),
        "residue_types": sorted({row["residue_name"] for row in rows}),
        "maximum_identity_rmsd": max(row["identity_rmsd_conventional"] for row in rows),
        "maximum_kinematic_B_rmsd": max(
            row["kinematic_B_rmsd_conventional"] for row in rows
        ),
        "all_gradients_finite": all(row["finite_gradient"] for row in rows),
        "all_chi_responsive": all(
            row["responsive_chi"] == row["n_chi"] for row in rows
        ),
        "failures": failures,
    }
    atomic_json(args.output / "validation_summary.json", summary)
    atomic_json(args.output / "stage_manifest.json", {
        "status": summary["status"], "completed_sites": len(rows)
    })
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise RuntimeError(f"expanded residue validation failed at {len(failures)} sites")


if __name__ == "__main__":
    main()
