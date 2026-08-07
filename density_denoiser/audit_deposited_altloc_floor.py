from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import gemmi
import numpy as np
import torch

from experiments.probe4.core import dihedral, wrap_angles

from .audit_five_site_endpoints import classify_rotamer
from .clash_environment import (
    EnvironmentAtom,
    compatible_spatial_metrics,
    normalized_altloc,
)
from .data_pipeline import _pdb_id, discover_pdbs
from .dataset import manifest_path, read_manifest
from .five_site_optimizer import _alt_atom_map
from .residue_geometry import AUDIT_RULE_VERSION, CHI_SPECS


BACKBONE_NAMES = {"N", "CA", "C", "O"}


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        if rows:
            fieldnames = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def deterministic_site_priority(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _find_residue(structure: gemmi.Structure, record: dict) -> gemmi.Residue:
    return next(
        residue
        for chain in structure[0]
        if chain.name == record["chain"]
        for residue in chain
        if residue.seqid.num == int(record["residue_number"])
        and residue.seqid.icode == record["insertion_code"]
    )


def eligible_pair(
    structure: gemmi.Structure, record: dict, minimum_occupancy: float
) -> tuple[bool, str]:
    try:
        residue = _find_residue(structure, record)
        map_a = _alt_atom_map(residue, "A", torch.device("cpu"))
        map_b = _alt_atom_map(residue, "B", torch.device("cpu"))
        names = [
            atom.name.strip()
            for atom in residue
            if normalized_altloc(atom.altloc) == "B"
            and atom.element.name != "H"
            and atom.name.strip() not in BACKBONE_NAMES
        ]
        required = {
            name
            for quartet in CHI_SPECS[residue.name]["dihedrals"]
            for name in quartet
        }
        if not names:
            return False, "no_B_sidechain_atoms"
        if any(name not in map_a or name not in map_b for name in set(names) | required):
            return False, "incomplete_A_B_correspondence"
        a_occupancies = [
            float(atom.occ)
            for atom in residue
            if normalized_altloc(atom.altloc) == "A"
            and atom.name.strip() in names
        ]
        b_occupancies = [
            float(atom.occ)
            for atom in residue
            if normalized_altloc(atom.altloc) == "B"
            and atom.name.strip() in names
        ]
        if (
            not a_occupancies
            or not b_occupancies
            or np.median(a_occupancies) < minimum_occupancy
            or np.median(b_occupancies) < minimum_occupancy
        ):
            return False, "occupancy_below_threshold"
        return True, ""
    except Exception as error:
        return False, f"{type(error).__name__}:{error}"


def choose_stratified_sites(
    records: list[dict],
    pdb_paths: dict[str, Path],
    maximum_per_residue: int,
    minimum_occupancy: float,
) -> tuple[list[dict], Counter]:
    candidates: dict[str, list[dict]] = defaultdict(list)
    exclusions: Counter = Counter()
    structures: dict[str, gemmi.Structure] = {}
    seen_keys = set()
    for record in records:
        if (
            not bool(record["is_altloc"])
            or record["residue_name"] not in CHI_SPECS
            or record["key"] in seen_keys
        ):
            continue
        seen_keys.add(record["key"])
        pdb_id = record["pdb_id"]
        try:
            if pdb_id not in structures:
                structures[pdb_id] = gemmi.read_structure(str(pdb_paths[pdb_id]))
            structure = structures[pdb_id]
        except Exception as error:
            exclusions[f"structure:{type(error).__name__}"] += 1
            continue
        eligible, reason = eligible_pair(structure, record, minimum_occupancy)
        if not eligible:
            exclusions[reason] += 1
            continue
        candidates[record["residue_name"]].append(record)

    selected = []
    for residue_name, rows in sorted(candidates.items()):
        used_proteins = set()
        for row in sorted(rows, key=lambda item: deterministic_site_priority(item["key"])):
            if row["pdb_id"] in used_proteins:
                continue
            selected.append(row)
            used_proteins.add(row["pdb_id"])
            if len(used_proteins) == maximum_per_residue:
                break
    return sorted(selected, key=lambda row: row["key"]), exclusions


def _actual_chi(
    candidate_map: dict[str, torch.Tensor], residue_name: str
) -> list[float]:
    return [
        math.degrees(float(wrap_angles(
            dihedral(*(candidate_map[name] for name in quartet)) - torch.pi
        )))
        for quartet in CHI_SPECS[residue_name]["dihedrals"]
    ]


def _environment_atom(
    chain: gemmi.Chain,
    residue: gemmi.Residue,
    atom: gemmi.Atom,
    xyz: tuple[float, float, float],
    prefix: str = "",
) -> EnvironmentAtom:
    return EnvironmentAtom(
        xyz=xyz,
        label=(
            f"{prefix}{chain.name}:{residue.name}{residue.seqid.num}:"
            f"{atom.name.strip()}"
        ),
        residue_group=(
            f"{prefix}{chain.name}:{residue.name}{residue.seqid.num}:"
            f"{residue.seqid.icode}"
        ),
        altloc=normalized_altloc(atom.altloc),
        occupancy=float(atom.occ),
        is_water=residue.name in {"HOH", "WAT", "DOD"},
    )


def audit_pair(
    structure: gemmi.Structure, record: dict, clash_cutoff: float
) -> list[dict[str, object]]:
    residue = _find_residue(structure, record)
    map_a = _alt_atom_map(residue, "A", torch.device("cpu"))
    map_b = _alt_atom_map(residue, "B", torch.device("cpu"))
    names = [
        atom.name.strip()
        for atom in residue
        if normalized_altloc(atom.altloc) == "B"
        and atom.element.name != "H"
        and atom.name.strip() not in BACKBONE_NAMES
    ]
    all_heavy_atoms = [
        (chain, candidate_residue, atom)
        for chain in structure[0]
        for candidate_residue in chain
        for atom in candidate_residue
        if atom.element.name != "H"
    ]
    direct = [
        _environment_atom(
            chain, candidate_residue, atom, tuple(atom.pos.tolist())
        )
        for chain, candidate_residue, atom in all_heavy_atoms
        if not (
            chain.name == record["chain"]
            and candidate_residue.seqid.num == int(record["residue_number"])
            and candidate_residue.seqid.icode == record["insertion_code"]
        )
    ]
    sampling_center = np.concatenate([
        np.asarray([map_a[name].tolist() for name in names]),
        np.asarray([map_b[name].tolist() for name in names]),
    ]).mean(axis=0)
    cell = structure.cell
    spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
    symmetry = []
    for operation_index, operation in enumerate(spacegroup.operations()):
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                for tz in (-1, 0, 1):
                    if operation_index == 0 and tx == ty == tz == 0:
                        continue
                    prefix = f"sym{operation_index}[{tx},{ty},{tz}]/"
                    for chain, candidate_residue, atom in all_heavy_atoms:
                        transformed = operation.apply_to_xyz(
                            cell.fractionalize(atom.pos).tolist()
                        )
                        position = cell.orthogonalize(gemmi.Fractional(
                            transformed[0] + tx,
                            transformed[1] + ty,
                            transformed[2] + tz,
                        ))
                        xyz = np.asarray(position.tolist())
                        if np.linalg.norm(xyz - sampling_center) <= 8.0:
                            symmetry.append(_environment_atom(
                                chain,
                                candidate_residue,
                                atom,
                                tuple(xyz.tolist()),
                                prefix,
                            ))
    rows = []
    for altloc, candidate_map in (("A", map_a), ("B", map_b)):
        xyz = np.asarray([candidate_map[name].tolist() for name in names])
        angles = _actual_chi(candidate_map, residue.name)
        state, deviations, widths, canonical = classify_rotamer(
            residue.name, angles
        )
        direct_result = compatible_spatial_metrics(
            xyz, direct, clash_cutoff, altloc
        )
        symmetry_result = compatible_spatial_metrics(
            xyz, symmetry, clash_cutoff, altloc
        )
        rows.append({
            "site": record["key"],
            "pdb_id": record["pdb_id"],
            "chain": record["chain"],
            "residue_number": int(record["residue_number"]),
            "insertion_code": record["insertion_code"],
            "residue_name": residue.name,
            "control": f"deposited_{altloc}",
            "chi_degrees": ";".join(f"{value:.4f}" for value in angles),
            "nearest_rotamer": state,
            "rotamer_deviation_degrees": ";".join(
                f"{value:.4f}" for value in deviations
            ),
            "rotamer_allowed_width_degrees": ";".join(
                f"{value:.4f}" for value in widths
            ),
            "rotamer_pass": canonical,
            "minimum_direct_distance": direct_result["minimum_distance"],
            "closest_direct_atom": direct_result["closest_atom"],
            "direct_clash_pass": direct_result["no_clash"],
            "minimum_symmetry_distance": symmetry_result["minimum_distance"],
            "closest_symmetry_atom": symmetry_result["closest_atom"],
            "symmetry_clash_pass": symmetry_result["no_clash"],
            "all_geometry_gates_pass": (
                canonical
                and bool(direct_result["no_clash"])
                and bool(symmetry_result["no_clash"])
            ),
        })
    return rows


def _bool(value: object) -> bool:
    return str(value).lower() == "true"


def summarize_by_residue(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for residue_name in sorted({str(row["residue_name"]) for row in rows}):
        subset = [row for row in rows if row["residue_name"] == residue_name]
        sites = sorted({str(row["site"]) for row in subset})
        pair_pass = sum(
            all(
                _bool(row["all_geometry_gates_pass"])
                for row in subset
                if row["site"] == site
            )
            for site in sites
        )
        passed = sum(_bool(row["all_geometry_gates_pass"]) for row in subset)
        output.append({
            "residue_name": residue_name,
            "deposited_pairs": len(sites),
            "deposited_conformers": len(subset),
            "rotamer_rejected_conformers": sum(
                not _bool(row["rotamer_pass"]) for row in subset
            ),
            "direct_clash_rejected_conformers": sum(
                not _bool(row["direct_clash_pass"]) for row in subset
            ),
            "symmetry_clash_rejected_conformers": sum(
                not _bool(row["symmetry_clash_pass"]) for row in subset
            ),
            "any_geometry_rejected_conformers": len(subset) - passed,
            "conformer_false_rejection_rate": (len(subset) - passed) / len(subset),
            "pairs_with_both_conformers_passing": pair_pass,
            "pair_false_rejection_rate": (len(sites) - pair_pass) / len(sites),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate deposited-control geometry rejection across test altlocs."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-per-residue", type=int, default=10)
    parser.add_argument("--minimum-occupancy", type=float, default=0.10)
    parser.add_argument("--clash-cutoff", type=float, default=2.0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    records = read_manifest(manifest_path(args.data_root, "test", "crystal"))
    pdb_paths = {
        _pdb_id(path): path for path in discover_pdbs(args.data_root, "test")
    }
    selected, exclusions = choose_stratified_sites(
        records,
        pdb_paths,
        args.maximum_per_residue,
        args.minimum_occupancy,
    )
    selection_rows = [{
        "key": row["key"],
        "pdb_id": row["pdb_id"],
        "chain": row["chain"],
        "residue_number": row["residue_number"],
        "insertion_code": row["insertion_code"],
        "residue_name": row["residue_name"],
        "priority_sha256": deterministic_site_priority(row["key"]),
    } for row in selected]
    _atomic_csv(args.output / "selected_sites.csv", selection_rows)
    _atomic_json(args.output / "run_config.json", {
        "audit_rule": AUDIT_RULE_VERSION,
        "split": "test",
        "selection": (
            "deterministic SHA256 priority, up to one site per protein per "
            "residue type, A/B complete, both occupancies above threshold"
        ),
        "maximum_per_residue": args.maximum_per_residue,
        "minimum_occupancy": args.minimum_occupancy,
        "clash_cutoff": args.clash_cutoff,
        "tmol_included": False,
        "selection_exclusions": dict(sorted(exclusions.items())),
    })

    audit_rows = []
    errors = []
    structures: dict[str, gemmi.Structure] = {}
    for index, record in enumerate(selected, start=1):
        try:
            if record["pdb_id"] not in structures:
                structures[record["pdb_id"]] = gemmi.read_structure(
                    str(pdb_paths[record["pdb_id"]])
                )
            structure = structures[record["pdb_id"]]
            audit_rows.extend(audit_pair(structure, record, args.clash_cutoff))
        except Exception as error:
            errors.append({
                "site": record["key"],
                "error": f"{type(error).__name__}: {error}",
            })
        _atomic_csv(args.output / "deposited_control_geometry_audit.csv", audit_rows)
        _atomic_csv(args.output / "errors.csv", errors)
        _atomic_json(args.output / "progress.json", {
            "completed_sites": index,
            "total_sites": len(selected),
            "audited_conformers": len(audit_rows),
            "errors": len(errors),
        })
        print(json.dumps({
            "completed": index,
            "total": len(selected),
            "site": record["key"],
        }), flush=True)

    by_residue = summarize_by_residue(audit_rows)
    _atomic_csv(args.output / "false_rejection_by_residue.csv", by_residue)
    passed = sum(_bool(row["all_geometry_gates_pass"]) for row in audit_rows)
    summary = {
        "audit_rule": AUDIT_RULE_VERSION,
        "selected_sites": len(selected),
        "audited_sites": len({row["site"] for row in audit_rows}),
        "audited_conformers": len(audit_rows),
        "errors": len(errors),
        "all_geometry_passed_conformers": passed,
        "conformer_false_rejection_rate": (
            (len(audit_rows) - passed) / len(audit_rows) if audit_rows else None
        ),
        "tmol_exempt": True,
        "tmol_exemption_reason": (
            "each deposited conformer is its own matched reference and therefore "
            "passes the tmol margin gate at exactly zero by construction"
        ),
    }
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
