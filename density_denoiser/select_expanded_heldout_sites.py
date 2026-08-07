from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np
import torch

from experiments.probe4.core import dihedral, torsion_to_coords, wrap_angles

from .data_pipeline import _pdb_id, discover_pdbs
from .dataset import manifest_path, read_manifest
from .five_site_optimizer import _alt_atom_map
from .residue_geometry import CHI_SPECS, symmetry_aware_rmsd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit expanded residue types in untouched test proteins"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-altloc-rmsd", type=float, default=0.5)
    parser.add_argument("--max-kinematic-rmsd", type=float, default=0.1)
    parser.add_argument("--min-occupancy", type=float, default=0.1)
    args = parser.parse_args()

    pdb_paths = {_pdb_id(path): path for path in discover_pdbs(args.data_root, "test")}
    records = read_manifest(manifest_path(args.data_root, "test", "crystal"))
    rows: list[dict] = []
    for record in sorted(records, key=lambda value: value["key"]):
        if not bool(record["is_altloc"]) or record["residue_name"] not in CHI_SPECS:
            continue
        row = {
            "key": record["key"],
            "pdb_id": record["pdb_id"],
            "chain": record["chain"],
            "residue_number": record["residue_number"],
            "insertion_code": record["insertion_code"],
            "residue_name": record["residue_name"],
            "n_chi": len(CHI_SPECS[record["residue_name"]]["dihedrals"]),
            "representable": False,
            "error": "",
        }
        try:
            pdb_path = pdb_paths[record["pdb_id"]]
            structure = gemmi.read_structure(str(pdb_path))
            residue = next(
                residue for chain in structure[0] if chain.name == record["chain"]
                for residue in chain
                if residue.seqid.num == int(record["residue_number"])
                and residue.seqid.icode == record["insertion_code"]
            )
            map_a = _alt_atom_map(residue, "A", torch.device("cpu"))
            map_b = _alt_atom_map(residue, "B", torch.device("cpu"))
            b_atoms = [
                atom for atom in residue
                if atom.altloc == "B"
                and atom.element.name != "H"
                and atom.name.strip() not in {"N", "CA", "C", "O"}
            ]
            names = [atom.name.strip() for atom in b_atoms]
            if not names or any(name not in map_a or name not in map_b for name in names):
                raise ValueError("incomplete A/B heavy-atom correspondence")
            spec = CHI_SPECS[record["residue_name"]]
            required = {name for quartet in spec["dihedrals"] for name in quartet}
            if any(name not in map_a or name not in map_b for name in required):
                raise ValueError("missing chi-defining atom")
            chi_a = torch.stack([
                dihedral(*(map_a[name] for name in quartet))
                for quartet in spec["dihedrals"]
            ])
            chi_b = torch.stack([
                dihedral(*(map_b[name] for name in quartet))
                for quartet in spec["dihedrals"]
            ])
            delta = wrap_angles(chi_b - chi_a)
            template = torch.stack([map_a[name] for name in names])
            deposited_a = template
            deposited_b = torch.stack([map_b[name] for name in names])
            fixed = {name: value for name, value in map_a.items() if name not in names}

            def coordinates(candidate: torch.Tensor) -> torch.Tensor:
                return torsion_to_coords(
                    template, names, candidate, list(spec["rotations"]), fixed
                )

            plus, minus = coordinates(delta), coordinates(-delta)
            kinematic = min(
                float(symmetry_aware_rmsd(
                    plus, deposited_b, names, record["residue_name"]
                )),
                float(symmetry_aware_rmsd(
                    minus, deposited_b, names, record["residue_name"]
                )),
            )
            separation = float(symmetry_aware_rmsd(
                deposited_a, deposited_b, names, record["residue_name"]
            ))
            a_occ = float(np.median([
                atom.occ for atom in residue
                if atom.altloc == "A" and atom.name.strip() in names
            ]))
            b_occ = float(np.median([atom.occ for atom in b_atoms]))
            row.update({
                "deposited_A_occupancy": a_occ,
                "deposited_B_occupancy": b_occ,
                "deposited_A_to_B_rmsd_conventional": separation,
                "kinematic_to_deposited_B_rmsd_conventional": kinematic,
                "pair_path": record["pair_path"],
                "pdb_path": str(pdb_path),
                "representable": (
                    a_occ >= args.min_occupancy
                    and b_occ >= args.min_occupancy
                    and separation >= args.min_altloc_rmsd
                    and kinematic <= args.max_kinematic_rmsd
                ),
            })
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "candidate_audit.csv").open("w", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    eligible = [row for row in rows if row["representable"]]
    summary = {
        "criteria": {
            "source_split": "test",
            "min_altloc_rmsd_conventional": args.min_altloc_rmsd,
            "max_kinematic_rmsd_conventional": args.max_kinematic_rmsd,
            "min_A_and_B_occupancy": args.min_occupancy,
            "one_site_per_protein_for_final_selection": True,
            "denoiser_or_optimizer_metric_used": False,
        },
        "audited_sites": len(rows),
        "eligible_sites": len(eligible),
        "eligible_proteins": len({row["pdb_id"] for row in eligible}),
        "eligible_by_residue": dict(sorted(Counter(
            row["residue_name"] for row in eligible
        ).items())),
    }
    (args.output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
