from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gemmi
import numpy as np
import torch

from experiments.probe4.core import dihedral, torsion_to_coords, wrap_angles

from .data_pipeline import _pdb_id, discover_pdbs
from .dataset import manifest_path, read_manifest
from .five_site_optimizer import CHI_SPECS, _alt_atom_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select deterministic representable optimizer sites from held-out proteins"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-kinematic-rmsd", type=float, default=0.1)
    parser.add_argument("--min-altloc-rmsd", type=float, default=0.5)
    args = parser.parse_args()

    pdb_paths = {_pdb_id(path): path for path in discover_pdbs(args.data_root, "test")}
    records = [
        record for record in read_manifest(manifest_path(args.data_root, "test", "crystal"))
        if bool(record["is_altloc"]) and record["residue_name"] in CHI_SPECS
    ]
    audits = []
    for record in sorted(records, key=lambda value: value["key"]):
        result = {
            "key": record["key"], "pdb_id": record["pdb_id"],
            "residue_name": record["residue_name"], "representable": False,
        }
        try:
            structure = gemmi.read_structure(str(pdb_paths[record["pdb_id"]]))
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
                if atom.altloc == "B" and atom.name.strip() not in {"N", "CA", "C", "O"}
            ]
            names = [atom.name.strip() for atom in b_atoms]
            if not names or any(name not in map_a or name not in map_b for name in names):
                raise ValueError("incomplete A/B atom-name correspondence")
            spec = CHI_SPECS[residue.name]
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
            deposited_b = torch.stack([map_b[name] for name in names])
            fixed = {name: value for name, value in map_a.items() if name not in names}

            def coordinates(candidate: torch.Tensor) -> torch.Tensor:
                return torsion_to_coords(
                    template, names, candidate, list(spec["rotations"]), fixed
                )

            plus, minus = coordinates(delta), coordinates(-delta)
            if torch.mean((minus - deposited_b).square()) < torch.mean(
                (plus - deposited_b).square()
            ):
                delta = -delta
            kinematic = coordinates(delta)
            rmsd = float(torch.sqrt(torch.mean((kinematic - deposited_b).square())))
            deposited_a = torch.stack([map_a[name] for name in names])
            altloc_rmsd = float(torch.sqrt(torch.mean((deposited_a - deposited_b).square())))
            a_occ = float(np.median([
                atom.occ for atom in residue
                if atom.altloc == "A" and atom.name.strip() in names
            ]))
            b_occ = float(np.median([atom.occ for atom in b_atoms]))
            result.update({
                "chain": record["chain"],
                "residue_number": record["residue_number"],
                "insertion_code": record["insertion_code"],
                "kinematic_to_deposited_B_rmsd": rmsd,
                "deposited_A_to_B_rmsd": altloc_rmsd,
                "deposited_A_occupancy": a_occ,
                "deposited_B_occupancy": b_occ,
                "pair_path": record["pair_path"],
                "pdb_path": str(pdb_paths[record["pdb_id"]]),
                "representable": (
                    rmsd <= args.max_kinematic_rmsd
                    and altloc_rmsd >= args.min_altloc_rmsd
                    and a_occ >= 0.1 and b_occ >= 0.1
                ),
                "error": "",
            })
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        audits.append(result)

    quotas = {"MET": 2, "ARG": 2, "ASP": 1}
    selected, used_proteins = [], set()
    for residue_name, count in quotas.items():
        eligible = sorted([
            row for row in audits
            if row["residue_name"] == residue_name and row["representable"]
        ], key=lambda row: (
            -row["deposited_A_to_B_rmsd"],
            row["kinematic_to_deposited_B_rmsd"], row["key"],
        ))
        for row in eligible:
            if row["pdb_id"] in used_proteins:
                continue
            selected.append(row)
            used_proteins.add(row["pdb_id"])
            if sum(value["residue_name"] == residue_name for value in selected) == count:
                break
    if len(selected) != sum(quotas.values()):
        raise RuntimeError(f"could not satisfy held-out quotas; selected {selected}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "candidate_audit.csv").open("w", newline="") as handle:
        fieldnames = sorted({key for row in audits for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audits)
    selection = {
        "selection_rule": (
            "largest-altloc-separation representable 2 MET, 2 ARG, and 1 ASP sites "
            "from distinct untouched test PDB IDs; kinematic RMSD <= threshold and "
            "no denoiser metric used"
        ),
        "max_kinematic_rmsd": args.max_kinematic_rmsd,
        "min_altloc_rmsd": args.min_altloc_rmsd,
        "sites": selected,
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True))
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
