from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import gemmi
import numpy as np

from density_denoiser.clash_environment import normalized_altloc
from density_denoiser.five_site_optimizer import _selected_protein_heavy_atoms


WATERS = {"HOH", "WAT", "DOD"}


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def describe(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def build_groups(structure: gemmi.Structure, record: dict, site: dict) -> dict:
    target = next(
        residue
        for chain in structure[0]
        if chain.name == record["chain"]
        for residue in chain
        if residue.seqid.num == int(record["residue_number"])
        and residue.seqid.icode == record["insertion_code"]
    )
    map_a = {}
    for atom in target:
        if normalized_altloc(atom.altloc) in {"", "A"}:
            map_a[atom.name.strip()] = np.asarray(atom.pos.tolist(), dtype=float)
    names = site["atom_names"]
    ca = map_a["CA"]
    radius = max(np.linalg.norm(map_a[name] - ca) for name in names) + 4.0
    heavy = _selected_protein_heavy_atoms(structure)
    target_group = (
        f"{record['chain']}:{int(record['residue_number'])}:"
        f"{record['insertion_code']}"
    )

    rows = {"direct": [], "symmetry": []}
    for chain, residue, atom in heavy:
        same_target = (
            chain.name == record["chain"]
            and residue.seqid.num == int(record["residue_number"])
            and residue.seqid.icode == record["insertion_code"]
        )
        if same_target and atom.name.strip() in names:
            continue
        xyz = np.asarray(atom.pos.tolist(), dtype=float)
        if np.linalg.norm(xyz - ca) <= radius:
            rows["direct"].append({
                "group": (
                    f"{chain.name}:{residue.seqid.num}:{residue.seqid.icode}"
                ),
                "altloc": normalized_altloc(atom.altloc),
                "occupancy": float(atom.occ),
                "is_water": residue.name in WATERS,
                "xyz": xyz,
            })

    cell = structure.cell
    spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
    for operation_index, operation in enumerate(spacegroup.operations()):
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                for tz in (-1, 0, 1):
                    if operation_index == 0 and tx == ty == tz == 0:
                        continue
                    for chain, residue, atom in heavy:
                        transformed = operation.apply_to_xyz(
                            cell.fractionalize(atom.pos).tolist()
                        )
                        position = cell.orthogonalize(gemmi.Fractional(
                            transformed[0] + tx,
                            transformed[1] + ty,
                            transformed[2] + tz,
                        ))
                        xyz = np.asarray(position.tolist(), dtype=float)
                        if np.linalg.norm(xyz - ca) > radius:
                            continue
                        rows["symmetry"].append({
                            "group": (
                                f"sym{operation_index}[{tx},{ty},{tz}]/"
                                f"{chain.name}:{residue.seqid.num}:"
                                f"{residue.seqid.icode}"
                            ),
                            "altloc": normalized_altloc(atom.altloc),
                            "occupancy": float(atom.occ),
                            "is_water": residue.name in WATERS,
                            "xyz": xyz,
                        })

    output = {}
    for environment, records in rows.items():
        grouped: OrderedDict[str, OrderedDict[str, list[dict]]] = OrderedDict()
        for atom in records:
            if atom["altloc"]:
                grouped.setdefault(atom["group"], OrderedDict()).setdefault(
                    atom["altloc"], []
                ).append(atom)
        groups = []
        for group, states in grouped.items():
            labels = list(states)
            coordinates = [
                np.asarray([atom["xyz"] for atom in atoms], dtype=float)
                for atoms in states.values()
            ]
            first = next(iter(next(iter(states.values()))))
            if first["is_water"]:
                occupancy_sum = sum(
                    max(atom["occupancy"] for atom in atoms)
                    for atoms in states.values()
                )
                if occupancy_sum < 1.0 - 1e-6:
                    labels.append("absent")
                    coordinates.append(np.empty((0, 3), dtype=float))
            groups.append({
                "group": group,
                "labels": labels,
                "coordinates": coordinates,
                "category": (
                    "labeled_backbone"
                    if environment == "direct" and group == target_group
                    else "labeled_water"
                    if first["is_water"]
                    else "symmetry_mate"
                    if environment == "symmetry"
                    else "direct_protein"
                ),
            })
        output[environment] = groups
    return output


def state_penalty(
    candidate: np.ndarray, state: np.ndarray, threshold: float
) -> float:
    if not state.size:
        return 0.0
    distances = np.linalg.norm(
        candidate[:, None, :] - state[None, :, :], axis=-1
    )
    return float(np.square(np.maximum(threshold - distances, 0.0)).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--tmol-input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--material-difference", type=float, default=0.01)
    parser.add_argument("--near-crossover", type=float, default=0.001)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    selections = {}
    for path in args.selection:
        for record in json.loads(path.read_text())["sites"]:
            selections[record["key"]] = record
    sites = {}
    for path in args.tmol_input:
        for site in json.loads(path.read_text())["sites"]:
            sites[site["site"]] = site

    comparison_rows = []
    census_rows = []
    for site_name, site in sorted(sites.items()):
        record = selections[site_name]
        structure = gemmi.read_structure(record["pdb_path"])
        groups = build_groups(structure, record, site)
        nonzero_groups = set()
        material_groups = set()
        category_nonzero = Counter()
        category_material = Counter()
        for candidate in site["candidates"]:
            xyz = np.asarray(candidate["coordinates"], dtype=float)
            for environment, environment_groups in groups.items():
                threshold = 3.0 if environment == "direct" else 2.5
                for group_index, group in enumerate(environment_groups):
                    penalties = [
                        state_penalty(xyz, state, threshold)
                        for state in group["coordinates"]
                    ]
                    maximum = max(penalties)
                    spread = maximum - min(penalties)
                    nonzero = maximum > 0.0
                    material = spread >= args.material_difference
                    key = (environment, group_index)
                    if nonzero:
                        nonzero_groups.add(key)
                    if material:
                        material_groups.add(key)
                    order = np.argsort(penalties, kind="stable")
                    gap = (
                        float(penalties[order[1]] - penalties[order[0]])
                        if len(order) > 1 else math.nan
                    )
                    if nonzero:
                        positive_minimum = min(penalties) > 0.0
                        comparison_rows.append({
                            "site": site_name,
                            "candidate_id": candidate["candidate_id"],
                            "start": candidate["start"],
                            "conformer": candidate["conformer"],
                            "assignment": candidate["assignment"],
                            "environment": environment,
                            "group_index": group_index,
                            "group_key": group["group"],
                            "category": group["category"],
                            "state_labels": ";".join(group["labels"]),
                            "state_penalties": ";".join(
                                f"{value:.9g}" for value in penalties
                            ),
                            "winning_state": group["labels"][int(order[0])],
                            "minimum_penalty": min(penalties),
                            "maximum_penalty": maximum,
                            "state_penalty_spread": spread,
                            "winner_runnerup_gap": gap,
                            "materially_state_dependent": material,
                            "positive_winning_penalty": positive_minimum,
                            "near_crossover": (
                                material
                                and positive_minimum
                                and gap <= args.near_crossover
                            ),
                        })
        for environment, group_index in nonzero_groups:
            category_nonzero[
                groups[environment][group_index]["category"]
            ] += 1
        for environment, group_index in material_groups:
            category_material[
                groups[environment][group_index]["category"]
            ] += 1
        census_rows.append({
            "site": site_name,
            "total_altloc_groups": sum(len(value) for value in groups.values()),
            "nonzero_in_at_least_one_endpoint": len(nonzero_groups),
            "materially_state_dependent_in_at_least_one_endpoint": len(
                material_groups
            ),
            "nonzero_by_category": json.dumps(
                category_nonzero, sort_keys=True
            ),
            "material_by_category": json.dumps(
                category_material, sort_keys=True
            ),
        })

    material_rows = [
        row for row in comparison_rows
        if row["materially_state_dependent"]
    ]
    active_material_rows = [
        row for row in material_rows if row["positive_winning_penalty"]
    ]
    near_rows = [row for row in material_rows if row["near_crossover"]]
    per_site = {}
    for site in sorted(sites):
        selected = [row for row in material_rows if row["site"] == site]
        active_selected = [
            row for row in selected if row["positive_winning_penalty"]
        ]
        near = [row for row in selected if row["near_crossover"]]
        per_site[site] = {
            "material_candidate_group_comparisons": len(selected),
            "positive_minimum_material_comparisons": len(active_selected),
            "near_crossover": len(near),
            "near_crossover_rate": len(near) / len(selected) if selected else None,
            "winner_runnerup_gap": describe([
                float(row["winner_runnerup_gap"]) for row in active_selected
            ]),
        }
    summary = {
        "thresholds": {
            "direct_soft_vdw_angstrom": 3.0,
            "symmetry_soft_angstrom": 2.5,
            "material_raw_loss_difference": args.material_difference,
            "near_crossover_raw_loss_gap": args.near_crossover,
        },
        "census": census_rows,
        "material_candidate_group_comparisons": len(material_rows),
        "positive_minimum_material_comparisons": len(active_material_rows),
        "near_crossover": len(near_rows),
        "near_crossover_rate": (
            len(near_rows) / len(material_rows) if material_rows else None
        ),
        "winner_runnerup_gap": describe([
            float(row["winner_runnerup_gap"]) for row in active_material_rows
        ]),
        "near_crossover_by_site": dict(Counter(
            row["site"] for row in near_rows
        )),
        "near_crossover_by_category": dict(Counter(
            row["category"] for row in near_rows
        )),
        "per_site": per_site,
    }
    args.output.mkdir(parents=True)
    atomic_csv(args.output / "contested_group_census.csv", census_rows)
    atomic_csv(args.output / "state_penalty_comparisons.csv", comparison_rows)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
