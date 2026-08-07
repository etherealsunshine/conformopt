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

from density_denoiser.clash_environment import normalized_altloc
from density_denoiser.summarize_endpoint_audit import as_bool


WATERS = {"HOH", "WAT", "DOD"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def occupancy_signature(states: dict[str, list[dict]]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted(
        (label, round(float(np.median([atom["occupancy"] for atom in atoms])), 2))
        for label, atoms in states.items()
    ))


def state_coordinates(atoms: list[dict]) -> np.ndarray:
    return np.asarray([atom["xyz"] for atom in atoms], dtype=float)


def minimum_distance(left: np.ndarray, right: np.ndarray) -> float:
    if not left.size or not right.size:
        return math.inf
    return float(np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1).min())


def union_find_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    parent = {node: node for node in nodes}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    grouped: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        grouped[find(node)].append(node)
    return list(grouped.values())


def build_environment(
    structure: gemmi.Structure,
    record: dict,
    deposited_a: np.ndarray,
    deposited_b: np.ndarray,
) -> dict[str, dict[str, dict[str, list[dict]]]]:
    all_atoms = [
        (chain, residue, atom)
        for chain in structure[0]
        for residue in chain
        for atom in residue
        if atom.element.name != "H"
    ]
    direct_records = []
    for chain, residue, atom in all_atoms:
        if (
            chain.name == record["chain"]
            and residue.seqid.num == int(record["residue_number"])
            and residue.seqid.icode == record["insertion_code"]
        ):
            continue
        direct_records.append({
            "group": (
                f"{chain.name}:{residue.name}{residue.seqid.num}:"
                f"{residue.seqid.icode}"
            ),
            "domain": "direct",
            "altloc": normalized_altloc(atom.altloc),
            "occupancy": float(atom.occ),
            "is_water": residue.name in WATERS,
            "xyz": tuple(atom.pos.tolist()),
        })

    center = np.concatenate((deposited_a, deposited_b), axis=0).mean(axis=0)
    symmetry_records = []
    cell = structure.cell
    spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
    for operation_index, operation in enumerate(spacegroup.operations()):
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                for tz in (-1, 0, 1):
                    if operation_index == 0 and tx == ty == tz == 0:
                        continue
                    domain = f"sym{operation_index}[{tx},{ty},{tz}]"
                    for chain, residue, atom in all_atoms:
                        transformed = operation.apply_to_xyz(
                            cell.fractionalize(atom.pos).tolist()
                        )
                        position = cell.orthogonalize(gemmi.Fractional(
                            transformed[0] + tx,
                            transformed[1] + ty,
                            transformed[2] + tz,
                        ))
                        xyz = np.asarray(position.tolist())
                        if np.linalg.norm(xyz - center) > 8.0:
                            continue
                        symmetry_records.append({
                            "group": (
                                f"{domain}/{chain.name}:{residue.name}"
                                f"{residue.seqid.num}:{residue.seqid.icode}"
                            ),
                            "domain": domain,
                            "altloc": normalized_altloc(atom.altloc),
                            "occupancy": float(atom.occ),
                            "is_water": residue.name in WATERS,
                            "xyz": tuple(xyz.tolist()),
                        })

    output = {}
    for environment, rows in (
        ("direct", direct_records), ("symmetry", symmetry_records)
    ):
        groups: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            if row["altloc"]:
                groups[row["group"]][row["altloc"]].append(row)
        output[environment] = dict(groups)
    return output


def winner(
    candidate: np.ndarray,
    states: dict[str, list[dict]],
    assignment: str,
    clash_cutoff: float,
) -> tuple[str, float, float, str, bool, dict[str, float]]:
    is_water = next(iter(next(iter(states.values()))))["is_water"]
    clearances = {}
    for label, atoms in states.items():
        occupancy = float(np.median([atom["occupancy"] for atom in atoms]))
        threshold = clash_cutoff * occupancy if is_water else clash_cutoff
        clearances[label] = (
            minimum_distance(candidate, state_coordinates(atoms)) - threshold
        )
    if is_water:
        if assignment in states:
            label = assignment
            atoms = states[label]
            occupancy = float(np.median([atom["occupancy"] for atom in atoms]))
            clearance = clearances[label]
            relevant = clearance >= 0.0 and any(
                value < 0.0 for key, value in clearances.items() if key != label
            )
            return (
                label, occupancy, clearance, "audit_match", relevant, clearances
            )
        absent = max(
            0.0,
            1.0 - sum(
                float(np.median([atom["occupancy"] for atom in atoms]))
                for atoms in states.values()
            ),
        )
        relevant = any(value < 0.0 for value in clearances.values())
        return (
            "absent", absent, math.inf, "audit_omission", relevant, clearances
        )

    choices = []
    for label, atoms in states.items():
        clearance = clearances[label]
        occupancy = float(np.median([atom["occupancy"] for atom in atoms]))
        choices.append((clearance, label, occupancy))
    clearance, label, occupancy = max(choices)
    relevant = clearance >= 0.0 and any(
        value < 0.0 for key, value in clearances.items() if key != label
    )
    return (
        label, occupancy, clearance, "protein_min_clearance", relevant, clearances
    )


def infer_coupled_components(
    groups: dict[str, dict[str, list[dict]]],
    spatial_cutoff: float,
) -> tuple[dict[str, int], dict[int, list[str]]]:
    by_domain_signature: dict[
        tuple[str, tuple[tuple[str, float], ...]], list[str]
    ] = defaultdict(list)
    coordinates = {}
    for group, states in groups.items():
        first = next(iter(next(iter(states.values()))))
        domain = first["domain"]
        by_domain_signature[(domain, occupancy_signature(states))].append(group)
        coordinates[group] = np.concatenate([
            state_coordinates(atoms) for atoms in states.values()
        ], axis=0)

    member_to_component = {}
    components = {}
    next_id = 0
    for members in by_domain_signature.values():
        edges = []
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                if minimum_distance(
                    coordinates[left], coordinates[right]
                ) <= spatial_cutoff:
                    edges.append((left, right))
        for component in union_find_components(members, edges):
            if len(component) < 2:
                continue
            components[next_id] = component
            for member in component:
                member_to_component[member] = next_id
            next_id += 1
    return member_to_component, components


def summarize(rows: list[dict[str, object]], population: str) -> dict[str, object]:
    selected = [row for row in rows if row["population"] == population]
    flagged = [row for row in selected if row["unrealizable_winner_set"]]
    by_site = {}
    for site in sorted({str(row["site"]) for row in selected}):
        site_rows = [row for row in selected if row["site"] == site]
        site_flagged = [row for row in site_rows if row["unrealizable_winner_set"]]
        by_site[site] = {
            "accepted_conformers": len(site_rows),
            "unrealizable_winner_sets": len(site_flagged),
            "rate": len(site_flagged) / len(site_rows) if site_rows else None,
            "coupled_label_inconsistent": sum(
                bool(row["coupled_label_inconsistent"]) for row in site_rows
            ),
            "target_label_inconsistent": sum(
                bool(row["target_label_inconsistent"]) for row in site_rows
            ),
            "occupancy_incompatible": sum(
                bool(row["occupancy_incompatible"]) for row in site_rows
            ),
        }
    return {
        "population": population,
        "accepted_conformers": len(selected),
        "unrealizable_winner_sets": len(flagged),
        "rate": len(flagged) / len(selected) if selected else None,
        "coupled_label_inconsistent": sum(
            bool(row["coupled_label_inconsistent"]) for row in selected
        ),
        "target_label_inconsistent": sum(
            bool(row["target_label_inconsistent"]) for row in selected
        ),
        "occupancy_incompatible": sum(
            bool(row["occupancy_incompatible"]) for row in selected
        ),
        "per_site": by_site,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--tmol-input", type=Path, action="append", required=True)
    parser.add_argument("--conformer-table", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clash-cutoff", type=float, default=2.0)
    parser.add_argument("--coupling-distance", type=float, default=4.0)
    parser.add_argument("--occupancy-slack", type=float, default=0.02)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    selections = {}
    for path in args.selection:
        for record in json.loads(path.read_text())["sites"]:
            selections[record["key"]] = record
    tmol_sites = {}
    for path in args.tmol_input:
        for site in json.loads(path.read_text())["sites"]:
            tmol_sites[site["site"]] = site
    conformers = {}
    for path in args.conformer_table:
        for row in read_csv(path):
            conformers[row["candidate_id"]] = row
    if set(tmol_sites) != set(selections):
        raise ValueError("selection and tmol site sets differ")

    winner_rows = []
    candidate_rows = []
    for site_name, site in sorted(tmol_sites.items()):
        record = selections[site_name]
        structure = gemmi.read_structure(record["pdb_path"])
        deposited_a = np.asarray(site["A"], dtype=float)
        deposited_b = np.asarray(site["B"], dtype=float)
        environments = build_environment(
            structure, record, deposited_a, deposited_b
        )
        target = next(
            residue
            for chain in structure[0]
            if chain.name == record["chain"]
            for residue in chain
            if residue.seqid.num == int(record["residue_number"])
            and residue.seqid.icode == record["insertion_code"]
        )
        target_states: dict[str, list[dict]] = defaultdict(list)
        for atom in target:
            altloc = normalized_altloc(atom.altloc)
            if altloc:
                target_states[altloc].append({
                    "occupancy": float(atom.occ),
                    "xyz": tuple(atom.pos.tolist()),
                })
        target_signature = occupancy_signature(target_states)
        target_occupancies = dict(target_signature)

        candidates = []
        for item in site["candidates"]:
            row = conformers[item["candidate_id"]]
            if not (
                as_bool(row["no_direct_clash"])
                and as_bool(row["no_symmetry_clash"])
            ):
                continue
            candidates.append({
                "population": "accepted_endpoint",
                "candidate_id": item["candidate_id"],
                "assignment": item["assignment"],
                "occupancy": float(row["occupancy"]),
                "coordinates": np.asarray(item["coordinates"], dtype=float),
            })
        for assignment, coordinates in (("A", deposited_a), ("B", deposited_b)):
            candidates.append({
                "population": "deposited_control",
                "candidate_id": f"{site_name}_deposited_{assignment}",
                "assignment": assignment,
                "occupancy": target_occupancies[assignment],
                "coordinates": coordinates,
            })

        combined_groups = {}
        component_lookup = {}
        components = {}
        component_offset = 0
        for environment, groups in environments.items():
            local_lookup, local_components = infer_coupled_components(
                groups, args.coupling_distance
            )
            for group, component in local_lookup.items():
                component_lookup[(environment, group)] = component + component_offset
            for component, members in local_components.items():
                components[component + component_offset] = [
                    (environment, member) for member in members
                ]
            component_offset += len(local_components)
            for group, states in groups.items():
                combined_groups[(environment, group)] = states

        for candidate in candidates:
            winners = {}
            relevant_groups = {}
            occupancy_incompatible = False
            target_label_inconsistent = False
            for (environment, group), states in combined_groups.items():
                (
                    selected,
                    state_occupancy,
                    clearance,
                    mode,
                    decision_relevant,
                    state_clearances,
                ) = winner(
                    candidate["coordinates"],
                    states,
                    candidate["assignment"],
                    args.clash_cutoff,
                )
                signature = occupancy_signature(states)
                target_coupled = (
                    signature == target_signature
                    and minimum_distance(
                        candidate["coordinates"],
                        np.concatenate([
                            state_coordinates(atoms) for atoms in states.values()
                        ], axis=0),
                    ) <= args.coupling_distance
                )
                state_occupancy_incompatible = decision_relevant and (
                    state_occupancy + args.occupancy_slack
                    < candidate["occupancy"]
                )
                state_target_inconsistent = (
                    decision_relevant
                    and
                    target_coupled
                    and candidate["assignment"] in {"A", "B"}
                    and selected != candidate["assignment"]
                )
                occupancy_incompatible |= state_occupancy_incompatible
                target_label_inconsistent |= state_target_inconsistent
                winners[(environment, group)] = selected
                relevant_groups[(environment, group)] = decision_relevant
                winner_rows.append({
                    "population": candidate["population"],
                    "site": site_name,
                    "candidate_id": candidate["candidate_id"],
                    "candidate_assignment": candidate["assignment"],
                    "candidate_occupancy": candidate["occupancy"],
                    "environment": environment,
                    "neighbor_group": group,
                    "is_water": next(iter(next(iter(states.values()))))["is_water"],
                    "occupancy_signature": json.dumps(signature),
                    "inferred_coupled_component": component_lookup.get(
                        (environment, group), ""
                    ),
                    "target_coupled": target_coupled,
                    "winning_state": selected,
                    "winning_state_occupancy": state_occupancy,
                    "winning_clearance": clearance,
                    "selection_mode": mode,
                    "state_clearances": json.dumps(state_clearances, sort_keys=True),
                    "decision_relevant": decision_relevant,
                    "occupancy_incompatible": state_occupancy_incompatible,
                    "target_label_inconsistent": state_target_inconsistent,
                })

            coupled_label_inconsistent = False
            for members in components.values():
                labels = {
                    winners[member] for member in members
                    if winners[member] != "absent"
                    and relevant_groups[member]
                }
                if len(labels) > 1:
                    coupled_label_inconsistent = True
                    break
            candidate_rows.append({
                "population": candidate["population"],
                "site": site_name,
                "candidate_id": candidate["candidate_id"],
                "assignment": candidate["assignment"],
                "occupancy": candidate["occupancy"],
                "altloc_neighbor_groups": len(combined_groups),
                "decision_relevant_groups": sum(relevant_groups.values()),
                "decision_relevant_direct_groups": sum(
                    relevant for (environment, _group), relevant
                    in relevant_groups.items() if environment == "direct"
                ),
                "decision_relevant_symmetry_groups": sum(
                    relevant for (environment, _group), relevant
                    in relevant_groups.items() if environment == "symmetry"
                ),
                "inferred_coupled_components": len(components),
                "coupled_label_inconsistent": coupled_label_inconsistent,
                "target_label_inconsistent": target_label_inconsistent,
                "occupancy_incompatible": occupancy_incompatible,
                "unrealizable_winner_set": (
                    coupled_label_inconsistent
                    or target_label_inconsistent
                    or occupancy_incompatible
                ),
            })

    summary = {
        "definitions": {
            "accepted_endpoint": (
                "active frozen endpoint conformer passing direct and symmetry "
                "clash gates; rotamer and tmol are not conditioning filters"
            ),
            "coupled_component": (
                "two or more altloc neighbor groups in the same direct or "
                "symmetry-image domain, with identical label/occupancy "
                "signature rounded to 0.01 and any atom pair within 4.0 A"
            ),
            "occupancy_incompatible": (
                "winning state occupancy plus 0.02 is below candidate occupancy; "
                "the slack covers two-decimal PDB occupancy rounding"
            ),
            "water_semantics": (
                "hard audit matching-label inclusion; a missing/incompatible "
                "labeled water is logged as absent, not as a min-selected state"
            ),
        },
        "accepted_endpoint": summarize(candidate_rows, "accepted_endpoint"),
        "deposited_control": summarize(candidate_rows, "deposited_control"),
    }
    args.output.mkdir(parents=True)
    atomic_csv(args.output / "neighbor_state_winners.csv", winner_rows)
    atomic_csv(args.output / "candidate_realizability.csv", candidate_rows)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
