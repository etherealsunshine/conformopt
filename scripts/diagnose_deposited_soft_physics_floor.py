from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np
import torch

from density_denoiser.clash_environment import (
    OPTIMIZER_PHYSICS_ENVIRONMENT_RULE,
    normalized_altloc,
)
from density_denoiser.five_site_optimizer import (
    _alt_atom_map,
    _selected_protein_heavy_atoms,
)
from density_denoiser.residue_geometry import (
    AUDIT_RULE_VERSION,
    CHI_SPECS,
    canonical_centers_radians,
    canonical_width_degrees,
)
from experiments.probe4.core import dihedral, wrap_angles


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def residue_key(chain: gemmi.Chain, residue: gemmi.Residue) -> str:
    return f"{chain.name}:{residue.seqid.num}:{residue.seqid.icode}"


def atom_label(
    chain: gemmi.Chain,
    residue: gemmi.Residue,
    atom: gemmi.Atom,
    prefix: str = "",
) -> str:
    altloc = normalized_altloc(atom.altloc) or "."
    return (
        f"{prefix}{chain.name}:{residue.seqid.num}:{residue.seqid.icode}:"
        f"{residue.name}:{atom.name.strip()}:{altloc}:occ={float(atom.occ):.3f}"
    )


def partition_records(
    records: list[dict[str, object]],
) -> tuple[list[dict], dict]:
    invariant: list[dict] = []
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    water_state_occupancies: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record["altloc"]:
            grouped[str(record["group"])][str(record["altloc"])].append(record)
            if record["is_water"]:
                group = str(record["group"])
                altloc = str(record["altloc"])
                water_state_occupancies[group][altloc] = max(
                    float(record["weight"]),
                    water_state_occupancies[group].get(altloc, 0.0),
                )
        else:
            invariant.append(record)
    for group, occupancies in water_state_occupancies.items():
        if sum(occupancies.values()) < 1.0 - 1e-6:
            grouped[group]["absent"] = []
    return invariant, grouped


def contact_term(
    candidate: torch.Tensor,
    candidate_names: list[str],
    invariant: list[dict],
    grouped: dict[str, dict[str, list[dict]]],
    threshold: float,
    term: str,
    hard_threshold: float,
    barrier_buffer: float,
    barrier_scale: float,
    mask_cb_ca: bool,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []

    def pair_penalty(distance: float, weight: float) -> tuple[float, float, float]:
        base = max(threshold - distance, 0.0) ** 2
        barrier = barrier_scale * (
            max(hard_threshold + barrier_buffer - distance, 0.0)
            / barrier_buffer
        ) ** 4
        return weight * (base + barrier), weight * base, weight * barrier

    total = 0.0
    candidate_np = candidate.detach().cpu().numpy()
    for moving_index, moving_name in enumerate(candidate_names):
        for record in invariant:
            if (
                mask_cb_ca
                and moving_name == "CB"
                and bool(record["same_target"])
                and record["atom_name"] == "CA"
            ):
                continue
            distance = float(np.linalg.norm(candidate_np[moving_index] - record["xyz"]))
            contribution, base, barrier = pair_penalty(
                distance, float(record["weight"])
            )
            total += contribution
            if contribution > 0.0:
                rows.append(
                    {
                        "term": term,
                        "environment_group": "invariant",
                        "selected_altloc": "",
                        "moving_atom": moving_name,
                        "environment_atom": record["label"],
                        "distance_A": distance,
                        "environment_weight": record["weight"],
                        "base_contribution": base,
                        "barrier_contribution": barrier,
                        "raw_contribution": contribution,
                    }
                )

    for group, states in grouped.items():
        state_values: list[tuple[float, str, list[dict[str, object]]]] = []
        for altloc, records in states.items():
            state_total = 0.0
            state_rows: list[dict[str, object]] = []
            for moving_index, moving_name in enumerate(candidate_names):
                for record in records:
                    distance = float(
                        np.linalg.norm(candidate_np[moving_index] - record["xyz"])
                    )
                    contribution, base, barrier = pair_penalty(distance, 1.0)
                    state_total += contribution
                    if contribution > 0.0:
                        state_rows.append(
                            {
                                "term": term,
                                "environment_group": group,
                                "selected_altloc": altloc,
                                "moving_atom": moving_name,
                                "environment_atom": record["label"],
                                "distance_A": distance,
                                "environment_weight": 1.0,
                                "base_contribution": base,
                                "barrier_contribution": barrier,
                                "raw_contribution": contribution,
                            }
                        )
            state_values.append((state_total, altloc, state_rows))
        if state_values:
            state_total, _altloc, state_rows = min(state_values, key=lambda item: item[0])
            total += state_total
            rows.extend(state_rows)
    return total, rows


def build_records(
    structure: gemmi.Structure,
    target_chain: str,
    target_number: int,
    insertion_code: str,
    names: list[str],
    ca_position: np.ndarray,
    environment_radius: float,
) -> tuple[list[dict], list[dict]]:
    heavy_atoms = _selected_protein_heavy_atoms(structure)
    direct_records: list[dict] = []
    for chain, residue, atom in heavy_atoms:
        atom_name = atom.name.strip()
        same_target = (
            chain.name == target_chain
            and residue.seqid.num == target_number
            and residue.seqid.icode == insertion_code
        )
        if same_target and atom_name in names:
            continue
        xyz = np.asarray(atom.pos.tolist(), dtype=np.float64)
        if np.linalg.norm(xyz - ca_position) > environment_radius:
            continue
        is_water = residue.name in {"HOH", "WAT", "DOD"}
        direct_records.append(
            {
                "group": residue_key(chain, residue),
                "altloc": normalized_altloc(atom.altloc),
                "is_water": is_water,
                "same_target": same_target,
                "atom_name": atom_name,
                "xyz": xyz,
                "weight": float(atom.occ) if is_water else 1.0,
                "label": atom_label(chain, residue, atom),
            }
        )

    cell = structure.cell
    spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
    symmetry_records: list[dict] = []
    for operation_index, operation in enumerate(spacegroup.operations()):
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                for tz in (-1, 0, 1):
                    if operation_index == 0 and tx == ty == tz == 0:
                        continue
                    prefix = f"sym{operation_index}[{tx},{ty},{tz}]/"
                    for chain, residue, atom in heavy_atoms:
                        transformed = operation.apply_to_xyz(
                            cell.fractionalize(atom.pos).tolist()
                        )
                        position = cell.orthogonalize(
                            gemmi.Fractional(
                                transformed[0] + tx,
                                transformed[1] + ty,
                                transformed[2] + tz,
                            )
                        )
                        xyz = np.asarray(position.tolist(), dtype=np.float64)
                        if np.linalg.norm(xyz - ca_position) > environment_radius:
                            continue
                        is_water = residue.name in {"HOH", "WAT", "DOD"}
                        symmetry_records.append(
                            {
                                "group": prefix + residue_key(chain, residue),
                                "altloc": normalized_altloc(atom.altloc),
                                "is_water": is_water,
                                "same_target": False,
                                "atom_name": atom.name.strip(),
                                "xyz": xyz,
                                "weight": float(atom.occ) if is_water else 1.0,
                                "label": atom_label(chain, residue, atom, prefix),
                            }
                        )
    return direct_records, symmetry_records


def evaluate_site(
    record: dict,
    panel: str,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], list[dict]]:
    structure = gemmi.read_structure(record["pdb_path"])
    chain = next(chain for chain in structure[0] if chain.name == record["chain"])
    residue = next(
        residue
        for residue in chain
        if residue.seqid.num == int(record["residue_number"])
        and residue.seqid.icode == record["insertion_code"]
    )
    device = torch.device("cpu")
    map_a = _alt_atom_map(residue, "A", device)
    map_b = _alt_atom_map(residue, "B", device)
    b_atoms = [
        atom
        for atom in residue
        if normalized_altloc(atom.altloc) == "B"
        and atom.element.name != "H"
        and atom.name.strip() not in {"N", "CA", "C", "O"}
    ]
    names = [atom.name.strip() for atom in b_atoms]
    if not names or any(name not in map_a or name not in map_b for name in names):
        raise RuntimeError(f"incomplete A/B target atoms at {record['key']}")
    fixed_lookup = {name: value for name, value in map_a.items() if name not in names}
    max_sidechain_radius = max(
        float(torch.linalg.vector_norm(value - map_a["CA"]))
        for value in map_a.values()
    )
    environment_radius = max_sidechain_radius + max(
        args.vdw_threshold, args.symmetry_threshold
    ) + 1.0
    direct_records, symmetry_records = build_records(
        structure,
        record["chain"],
        int(record["residue_number"]),
        record["insertion_code"],
        names,
        map_a["CA"].numpy(),
        environment_radius,
    )
    direct_invariant, direct_grouped = partition_records(direct_records)
    symmetry_invariant, symmetry_grouped = partition_records(symmetry_records)

    term_rows: list[dict] = []
    contact_rows: list[dict] = []
    rotamer_rows: list[dict] = []
    spec = CHI_SPECS[residue.name]
    for conformer, atom_map in (("A", map_a), ("B", map_b)):
        candidate = torch.stack([atom_map[name] for name in names])
        vdw, vdw_rows = contact_term(
            candidate,
            names,
            direct_invariant,
            direct_grouped,
            args.vdw_threshold,
            "vdw",
            args.symmetry_hard_threshold,
            args.symmetry_barrier_buffer,
            0.0,
            True,
        )
        symmetry, symmetry_rows = contact_term(
            candidate,
            names,
            symmetry_invariant,
            symmetry_grouped,
            args.symmetry_threshold,
            "symmetry",
            args.symmetry_hard_threshold,
            args.symmetry_barrier_buffer,
            args.symmetry_barrier_scale,
            False,
        )
        lookup = dict(fixed_lookup)
        lookup.update({name: candidate[index] for index, name in enumerate(names)})
        rotamer = 0.0
        for chi_index, quartet in enumerate(spec["dihedrals"]):
            value = wrap_angles(
                dihedral(*(lookup[name] for name in quartet)) - torch.pi
            )
            centers = torch.tensor(
                canonical_centers_radians(residue.name, chi_index),
                dtype=value.dtype,
            )
            losses = 1.0 - torch.cos(value - centers)
            nearest_index = int(torch.argmin(losses))
            width = canonical_width_degrees(residue.name, chi_index)
            contribution = float(losses[nearest_index]) * (30.0 / width) ** 2
            rotamer += contribution
            rotamer_rows.append(
                {
                    "panel": panel,
                    "site": record["key"],
                    "residue": residue.name,
                    "conformer": conformer,
                    "chi_index": chi_index + 1,
                    "chi_degrees": math.degrees(float(value)),
                    "nearest_center_degrees": math.degrees(
                        float(centers[nearest_index])
                    ),
                    "allowed_width_degrees": width,
                    "raw_contribution": contribution,
                }
            )
        for row in vdw_rows + symmetry_rows:
            contact_rows.append(
                {
                    "panel": panel,
                    "site": record["key"],
                    "residue": residue.name,
                    "conformer": conformer,
                    **row,
                }
            )
        total = (
            args.lambda_vdw * vdw
            + args.lambda_rot * rotamer
            + args.lambda_clash * symmetry
        )
        term_rows.append(
            {
                "panel": panel,
                "site": record["key"],
                "residue": residue.name,
                "conformer": conformer,
                "deposited_occupancy": float(
                    np.median(
                        [
                            atom.occ
                            for atom in residue
                            if normalized_altloc(atom.altloc) == conformer
                            and atom.name.strip() in names
                        ]
                    )
                ),
                "vdw_raw": vdw,
                "rotamer_raw": rotamer,
                "symmetry_raw": symmetry,
                "vdw_weighted": args.lambda_vdw * vdw,
                "rotamer_weighted": args.lambda_rot * rotamer,
                "symmetry_weighted": args.lambda_clash * symmetry,
                "total_weighted_soft_physics": total,
                "environment_radius_A": environment_radius,
                "direct_invariant_atoms": len(direct_invariant),
                "direct_alternate_groups": len(direct_grouped),
                "symmetry_invariant_atoms": len(symmetry_invariant),
                "symmetry_alternate_groups": len(symmetry_grouped),
            }
        )
    return term_rows, contact_rows, rotamer_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", action="append", required=True)
    parser.add_argument("--panel", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vdw-threshold", type=float, default=3.0)
    parser.add_argument("--symmetry-threshold", type=float, default=2.5)
    parser.add_argument("--symmetry-hard-threshold", type=float, default=2.0)
    parser.add_argument("--symmetry-barrier-buffer", type=float, default=0.25)
    parser.add_argument("--symmetry-barrier-scale", type=float, default=0.0)
    parser.add_argument("--lambda-vdw", type=float, default=1.0)
    parser.add_argument("--lambda-rot", type=float, default=0.5)
    parser.add_argument("--lambda-clash", type=float, default=5.0)
    args = parser.parse_args()
    if len(args.selection) != len(args.panel):
        raise ValueError("--selection and --panel counts must match")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    terms: list[dict] = []
    contacts: list[dict] = []
    rotamers: list[dict] = []
    for panel, selection_path in zip(args.panel, args.selection):
        selection = json.loads(Path(selection_path).read_text())
        for record in selection["sites"]:
            site_terms, site_contacts, site_rotamers = evaluate_site(
                record, panel, args
            )
            terms.extend(site_terms)
            contacts.extend(site_contacts)
            rotamers.extend(site_rotamers)
    terms.sort(key=lambda row: (str(row["site"]), str(row["conformer"])))
    contacts.sort(
        key=lambda row: (
            str(row["site"]),
            str(row["conformer"]),
            str(row["term"]),
            -float(row["raw_contribution"]),
        )
    )
    rotamers.sort(
        key=lambda row: (
            str(row["site"]), str(row["conformer"]), int(row["chi_index"])
        )
    )
    write_csv(args.output / "soft_terms_per_conformer.csv", terms)
    write_csv(args.output / "contact_contributions.csv", contacts)
    write_csv(args.output / "rotamer_contributions.csv", rotamers)
    summary = {
        "geometry_rule": AUDIT_RULE_VERSION,
        "optimizer_physics_environment_rule": (
            OPTIMIZER_PHYSICS_ENVIRONMENT_RULE
        ),
        "sites": len({row["site"] for row in terms}),
        "deposited_conformers": len(terms),
        "vdw_threshold_A": args.vdw_threshold,
        "symmetry_soft_threshold_A": args.symmetry_threshold,
        "symmetry_hard_threshold_A": args.symmetry_hard_threshold,
        "symmetry_barrier_scale": args.symmetry_barrier_scale,
        "lambda_vdw": args.lambda_vdw,
        "lambda_rot": args.lambda_rot,
        "lambda_clash": args.lambda_clash,
        "maximum_raw_terms": {
            key: max(float(row[key]) for row in terms)
            for key in ("vdw_raw", "rotamer_raw", "symmetry_raw")
        },
        "nonzero_symmetry_conformers": sum(
            float(row["symmetry_raw"]) > 1e-12 for row in terms
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
