from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import gemmi
import numpy as np
import torch

from experiments.probe4.core import dihedral, torsion_to_coords, wrap_angles

from .clash_environment import (
    EnvironmentAtom,
    compatible_spatial_metrics,
    normalized_altloc,
)
from .five_site_optimizer import _alt_atom_map
from .residue_geometry import (
    AUDIT_RULE_VERSION,
    CHI_SPECS,
    canonical_centers_degrees,
    classify_rotamer_degrees,
    symmetry_aware_rmsd,
)


STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

EXPECTED_HEAVY_ATOMS = {
    "ALA": {"N", "CA", "C", "O", "CB"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "ASP": {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    "CYS": {"N", "CA", "C", "O", "CB", "SG"},
    "GLN": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "GLY": {"N", "CA", "C", "O"},
    "HIS": {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    "LEU": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    "LYS": {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    "PHE": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"N", "CA", "C", "O", "CB", "CG", "CD"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TRP": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "TYR": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
}

CONFORMER_MATCHING_VERSION = "optimal_one_to_one_AB_rmsd_v1"


def canonical_centers(resname: str, index: int) -> list[tuple[str, float]]:
    return canonical_centers_degrees(resname, index)


def classify_rotamer(
    resname: str, angles: list[float]
) -> tuple[str, list[float], list[float], bool]:
    return classify_rotamer_degrees(resname, angles)


def optimal_ab_assignments(
    occupancies: list[float],
    rmsd_a: list[float],
    rmsd_b: list[float],
    *,
    active_occupancy: float,
    found_occupancy: float,
    rmsd_cutoff: float,
) -> list[str]:
    """Assign at most one eligible conformer to each deposited A/B state.

    The Hungarian objective first maximizes the number of valid matches and
    then minimizes their total RMSD. Conformers must exceed the reporting
    occupancy threshold and satisfy the same strict RMSD cutoff used by the
    recovery gate. Active conformers not selected for A or B remain ``other``.
    """
    if not (len(occupancies) == len(rmsd_a) == len(rmsd_b)):
        raise ValueError("occupancy and RMSD vectors must have equal length")

    assignments = [
        "inactive" if occupancy <= active_occupancy else "other"
        for occupancy in occupancies
    ]
    eligible = [
        index
        for index, occupancy in enumerate(occupancies)
        if occupancy > found_occupancy
    ]
    if not eligible:
        return assignments

    # With exactly two deposited states, exhaustive distinct-pair evaluation is
    # the dependency-free equivalent of a 2 x K Hungarian assignment.
    valid_a = [index for index in eligible if rmsd_a[index] < rmsd_cutoff]
    valid_b = [index for index in eligible if rmsd_b[index] < rmsd_cutoff]
    pairs = [
        (rmsd_a[index_a] + rmsd_b[index_b], index_a, index_b)
        for index_a in valid_a
        for index_b in valid_b
        if index_a != index_b
    ]
    if pairs:
        _total, index_a, index_b = min(pairs)
        assignments[index_a] = "A"
        assignments[index_b] = "B"
        return assignments

    single_matches = [
        (rmsd_a[index], index, "A") for index in valid_a
    ] + [
        (rmsd_b[index], index, "B") for index in valid_b
    ]
    if single_matches:
        _distance, index, state = min(single_matches)
        assignments[index] = state
    return assignments


def merge_then_assign_conformers(
    occupancies: list[float],
    rmsd_a: list[float],
    rmsd_b: list[float],
    pairwise_rmsd: list[list[float]],
    *,
    active_occupancy: float,
    found_occupancy: float,
    rmsd_cutoff: float,
    merge_rmsd_threshold: float,
) -> dict[str, list]:
    """Cluster near-duplicate active slots, sum occupancy, then assign A/B.

    Clusters are single-linkage connected components under the strict merge
    threshold. A preliminary one-to-one assignment protects distinct A and B
    anchors from being merged into the same component. Anchored clusters keep
    the anchor geometry; unanchored clusters use an occupancy-weighted medoid.
    Reported occupancy is the sum over all cluster members.
    """
    size = len(occupancies)
    if not (
        len(rmsd_a) == len(rmsd_b) == len(pairwise_rmsd) == size
        and all(len(row) == size for row in pairwise_rmsd)
    ):
        raise ValueError("occupancy and RMSD inputs must describe the same slots")

    active = [
        index
        for index, occupancy in enumerate(occupancies)
        if occupancy > active_occupancy
    ]
    preliminary_assignments = optimal_ab_assignments(
        occupancies,
        rmsd_a,
        rmsd_b,
        active_occupancy=active_occupancy,
        found_occupancy=found_occupancy,
        rmsd_cutoff=rmsd_cutoff,
    )
    parent = list(range(size))
    protected_state = [
        assignment if assignment in {"A", "B"} else None
        for assignment in preliminary_assignments
    ]

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        state_left = protected_state[root_left]
        state_right = protected_state[root_right]
        if (
            state_left is not None
            and state_right is not None
            and state_left != state_right
        ):
            return
        new_root, old_root = min(root_left, root_right), max(root_left, root_right)
        parent[old_root] = new_root
        protected_state[new_root] = state_left or state_right

    if merge_rmsd_threshold > 0.0:
        for offset, left in enumerate(active):
            for right in active[offset + 1:]:
                if pairwise_rmsd[left][right] < merge_rmsd_threshold:
                    union(left, right)

    grouped: dict[int, list[int]] = {}
    for index in active:
        grouped.setdefault(find(index), []).append(index)
    clusters = sorted(grouped.values(), key=min)

    representatives, cluster_occupancies = [], []
    for members in clusters:
        anchors = [
            index
            for index in members
            if preliminary_assignments[index] in {"A", "B"}
        ]
        if anchors:
            representative = anchors[0]
        else:
            representative = min(
                members,
                key=lambda candidate: (
                    sum(
                        occupancies[other] * pairwise_rmsd[candidate][other]
                        for other in members
                    ),
                    candidate,
                ),
            )
        representatives.append(representative)
        cluster_occupancies.append(sum(occupancies[index] for index in members))

    cluster_assignments = optimal_ab_assignments(
        cluster_occupancies,
        [rmsd_a[index] for index in representatives],
        [rmsd_b[index] for index in representatives],
        active_occupancy=0.0,
        found_occupancy=found_occupancy,
        rmsd_cutoff=rmsd_cutoff,
    )

    assignments = [
        "inactive" if occupancy <= active_occupancy else "other"
        for occupancy in occupancies
    ]
    member_cluster = [-1] * size
    member_cluster_occupancy = [0.0] * size
    member_is_representative = [False] * size
    member_cluster_assignment = [
        "inactive" if occupancy <= active_occupancy else "other"
        for occupancy in occupancies
    ]
    for cluster_index, members in enumerate(clusters):
        representative = representatives[cluster_index]
        state = cluster_assignments[cluster_index]
        for index in members:
            member_cluster[index] = cluster_index
            member_cluster_occupancy[index] = cluster_occupancies[cluster_index]
            member_cluster_assignment[index] = state
            if index != representative and state in {"A", "B"}:
                assignments[index] = "merged_duplicate"
        member_is_representative[representative] = True
        assignments[representative] = state

    return {
        "assignments": assignments,
        "cluster_index": member_cluster,
        "cluster_occupancy": member_cluster_occupancy,
        "cluster_representative": member_is_representative,
        "cluster_assignment": member_cluster_assignment,
        "clusters": clusters,
        "representatives": representatives,
        "cluster_occupancies": cluster_occupancies,
        "cluster_assignments": cluster_assignments,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tmol_segment(
    path: Path, structure: gemmi.Structure, chain_name: str, residue_number: int,
    half_window: int = 12,
    environment_coordinates: np.ndarray | None = None,
    target_altloc: str | None = None,
) -> None:
    chain = next(chain for chain in structure[0] if chain.name == chain_name)
    residues = list(chain)
    center_index = next(
        index for index, residue in enumerate(residues) if residue.seqid.num == residue_number
    )
    def complete_conformer(residue: gemmi.Residue) -> dict[str, gemmi.Atom] | None:
        if residue.name not in EXPECTED_HEAVY_ATOMS:
            return None
        blank = {}
        alternate: dict[str, dict[str, gemmi.Atom]] = {}
        for atom in residue:
            if atom.element.name == "H":
                continue
            name = atom.name.strip()
            alt = "" if atom.altloc in ("\x00", " ") else atom.altloc
            if not alt:
                blank[name] = atom
            else:
                alternate.setdefault(alt, {})[name] = atom
        choices = []
        for alt, atoms in alternate.items():
            combined = {**blank, **atoms}
            completeness = len(EXPECTED_HEAVY_ATOMS[residue.name] & set(combined))
            occupancy = sum(atom.occ for atom in atoms.values())
            choices.append((completeness, occupancy, alt, combined, atoms))
        if not choices:
            choices.append((
                len(EXPECTED_HEAVY_ATOMS[residue.name] & set(blank)),
                0.0,
                "",
                blank,
                blank,
            ))
        if residue.seqid.num == residue_number and target_altloc is not None:
            matching = [item for item in choices if item[2] == target_altloc]
            chosen_item = max(
                matching or choices, key=lambda item: (item[0], item[1])
            )
        elif environment_coordinates is not None and len(choices) > 1:
            def clearance(item) -> float:
                coordinates = np.asarray([
                    atom.pos.tolist() for atom in item[4].values()
                ])
                distances = np.linalg.norm(
                    environment_coordinates[:, None, :]
                    - coordinates[None, :, :],
                    axis=-1,
                )
                return float(distances.min())

            chosen_item = max(
                choices, key=lambda item: (item[0], clearance(item), item[1])
            )
        else:
            chosen_item = max(choices, key=lambda item: (item[0], item[1]))
        chosen = chosen_item[3]
        if not EXPECTED_HEAVY_ATOMS[residue.name].issubset(chosen):
            return None
        return chosen

    # Keep the largest chemically complete contiguous segment around the target.
    # Stopping at an incomplete neighbor is preferable to silently deleting atoms
    # from a residue that tmol will interpret as chemically present.
    selected_indices = [center_index]
    for direction in (-1, 1):
        for offset in range(1, half_window + 1):
            index = center_index + direction * offset
            if index < 0 or index >= len(residues):
                break
            if complete_conformer(residues[index]) is None:
                break
            selected_indices.append(index)
    selected = [residues[index] for index in sorted(selected_indices)]
    lines, serial = [], 1
    for residue in selected:
        chosen = complete_conformer(residue)
        if chosen is None:
            raise RuntimeError(f"incomplete residue escaped segment filtering: {residue}")
        for name, atom in chosen.items():
            xyz = atom.pos
            element = atom.element.name.upper().rjust(2)
            lines.append(
                f"ATOM  {serial:5d} {name:^4s} {residue.name:>3s} {chain_name:1s}"
                f"{residue.seqid.num:4d}    {xyz.x:8.3f}{xyz.y:8.3f}{xyz.z:8.3f}"
                f"{1.00:6.2f}{atom.b_iso:6.2f}          {element:>2s}"
            )
            serial += 1
    lines.extend(("TER", "END"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conventional-RMSD and physical audit of saved five-site ensembles"
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="denoised")
    parser.add_argument("--active-occupancy", type=float, default=0.05)
    parser.add_argument("--found-occupancy", type=float, default=0.10)
    parser.add_argument("--rmsd-cutoff", type=float, default=1.0)
    parser.add_argument(
        "--merge-rmsd-threshold",
        type=float,
        default=0.0,
        help=(
            "single-linkage RMSD threshold for summing near-duplicate slot "
            "occupancies before one-to-one assignment; 0 disables merging"
        ),
    )
    parser.add_argument("--occupancy-tolerance", type=float, default=0.20)
    parser.add_argument("--clash-cutoff", type=float, default=2.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    visualization = args.output / "visualization"
    visualization.mkdir(exist_ok=True)
    selection = json.loads(args.selection.read_text())
    ensemble_rows, conformer_rows, control_rows = [], [], []
    tmol_inputs = {"sites": []}
    generator = torch.Generator().manual_seed(91)

    for site_index, record in enumerate(selection["sites"]):
        structure = gemmi.read_structure(record["pdb_path"])
        chain = next(chain for chain in structure[0] if chain.name == record["chain"])
        residue = next(
            residue for residue in chain
            if residue.seqid.num == int(record["residue_number"])
            and residue.seqid.icode == record["insertion_code"]
        )
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
        spec = CHI_SPECS[residue.name]
        template = torch.stack([map_a[name] for name in names])
        deposited_b = torch.stack([map_b[name] for name in names])
        fixed = {name: value for name, value in map_a.items() if name not in names}

        def coordinates(delta: torch.Tensor) -> torch.Tensor:
            return torsion_to_coords(
                template, names, delta, list(spec["rotations"]), fixed
            )

        chi_a_internal = torch.stack([
            dihedral(*(map_a[name] for name in quartet)) for quartet in spec["dihedrals"]
        ])
        chi_b_internal = torch.stack([
            dihedral(*(map_b[name] for name in quartet)) for quartet in spec["dihedrals"]
        ])
        true_delta = wrap_angles(chi_b_internal - chi_a_internal)
        plus, minus = coordinates(true_delta), coordinates(-true_delta)
        if symmetry_aware_rmsd(
            minus, deposited_b, names, residue.name
        ) < symmetry_aware_rmsd(plus, deposited_b, names, residue.name):
            true_delta = -true_delta
        kinematic_a = coordinates(torch.zeros(len(spec["rotations"]))).detach()
        kinematic_b = coordinates(true_delta).detach()

        def conventional_rmsd(candidate: torch.Tensor, reference: torch.Tensor) -> float:
            return float(symmetry_aware_rmsd(
                candidate, reference, names, residue.name
            ))

        def actual_chi(candidate: torch.Tensor) -> list[float]:
            lookup = dict(fixed)
            lookup.update({name: candidate[index] for index, name in enumerate(names)})
            return [
                math.degrees(float(wrap_angles(
                    dihedral(*(lookup[name] for name in quartet)) - torch.pi
                )))
                for quartet in spec["dihedrals"]
            ]

        all_heavy_atoms = [
            (candidate_chain, candidate_residue, atom)
            for candidate_chain in structure[0]
            for candidate_residue in candidate_chain
            for atom in candidate_residue if atom.element.name != "H"
        ]

        cell = structure.cell
        spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
        sampling_center = torch.cat((kinematic_a, kinematic_b)).mean(dim=0).numpy()

        direct_environment_records = []
        for candidate_chain, candidate_residue, atom in all_heavy_atoms:
            if (
                candidate_chain.name == record["chain"]
                and candidate_residue.seqid.num == int(record["residue_number"])
                and candidate_residue.seqid.icode == record["insertion_code"]
            ):
                continue
            direct_environment_records.append(EnvironmentAtom(
                xyz=tuple(atom.pos.tolist()),
                label=(
                    f"{candidate_chain.name}:{candidate_residue.name}"
                    f"{candidate_residue.seqid.num}:{atom.name.strip()}"
                ),
                residue_group=(
                    f"{candidate_chain.name}:{candidate_residue.name}"
                    f"{candidate_residue.seqid.num}:{candidate_residue.seqid.icode}"
                ),
                altloc=normalized_altloc(atom.altloc),
                occupancy=float(atom.occ),
                is_water=candidate_residue.name in {"HOH", "WAT", "DOD"},
            ))

        symmetry_environment_records = []
        for operation_index, operation in enumerate(spacegroup.operations()):
            for tx in (-1, 0, 1):
                for ty in (-1, 0, 1):
                    for tz in (-1, 0, 1):
                        if operation_index == 0 and tx == ty == tz == 0:
                            continue
                        for candidate_chain, candidate_residue, atom in all_heavy_atoms:
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
                                symmetry_environment_records.append(EnvironmentAtom(
                                    xyz=tuple(xyz.tolist()),
                                    label=(
                                        f"sym{operation_index}[{tx},{ty},{tz}]/"
                                        f"{candidate_chain.name}:{candidate_residue.name}"
                                        f"{candidate_residue.seqid.num}:{atom.name.strip()}"
                                    ),
                                    residue_group=(
                                        f"sym{operation_index}[{tx},{ty},{tz}]/"
                                        f"{candidate_chain.name}:{candidate_residue.name}"
                                        f"{candidate_residue.seqid.num}:"
                                        f"{candidate_residue.seqid.icode}"
                                    ),
                                    altloc=normalized_altloc(atom.altloc),
                                    occupancy=float(atom.occ),
                                    is_water=(
                                        candidate_residue.name in {"HOH", "WAT", "DOD"}
                                    ),
                                ))

        def spatial_metrics(
            candidate: torch.Tensor, candidate_altloc: str | None
        ) -> dict:
            xyz = candidate.detach().numpy()
            direct = compatible_spatial_metrics(
                xyz,
                direct_environment_records,
                args.clash_cutoff,
                candidate_altloc,
            )
            symmetry = compatible_spatial_metrics(
                xyz,
                symmetry_environment_records,
                args.clash_cutoff,
                candidate_altloc,
            )
            result = {
                "min_direct_distance": direct["minimum_distance"],
                "closest_direct_atom": direct["closest_atom"],
                "min_direct_clearance": direct["minimum_clearance"],
                "no_direct_clash": direct["no_clash"],
                "min_symmetry_distance": symmetry["minimum_distance"],
                "closest_symmetry_atom": symmetry["closest_atom"],
                "min_symmetry_clearance": symmetry["minimum_clearance"],
                "no_symmetry_clash": symmetry["no_clash"],
            }
            return result

        for control_label, control_candidate in (
            ("deposited_A", template), ("deposited_B", deposited_b)
        ):
            control_angles = actual_chi(control_candidate)
            (
                control_state,
                control_deviations,
                control_widths,
                control_canonical,
            ) = classify_rotamer(residue.name, control_angles)
            control_altloc = control_label.removeprefix("deposited_")
            control_rows.append({
                "site": record["key"], "control": control_label,
                "chi_degrees": ";".join(f"{value:.4f}" for value in control_angles),
                "nearest_rotamer": control_state,
                "rotamer_deviation_degrees": ";".join(
                    f"{value:.4f}" for value in control_deviations
                ),
                "rotamer_allowed_width_degrees": ";".join(
                    f"{value:.4f}" for value in control_widths
                ),
                "rotamer_within_allowed_width": control_canonical,
                "canonical_like_30deg": control_canonical,
                **spatial_metrics(control_candidate, control_altloc),
            })

        result_path = (
            args.results_root / record["key"] / args.target
            / f"{record['key']}_starts.csv"
        )
        result_rows = list(csv.DictReader(result_path.open()))
        site_candidates = []
        for row in result_rows:
            occupancies = [float(value) for value in row["occupancies"].split(";")]
            chi_rows = [
                [float(value) for value in values.split(";")]
                for values in row["final_chi_radians"].split("|")
            ]
            candidates = [coordinates(torch.tensor(values)).detach() for values in chi_rows]
            rmsd_a = [conventional_rmsd(value, kinematic_a) for value in candidates]
            rmsd_b = [conventional_rmsd(value, kinematic_b) for value in candidates]
            pairwise_rmsd = [
                [
                    0.0 if left == right else conventional_rmsd(
                        candidates[left], candidates[right]
                    )
                    for right in range(len(candidates))
                ]
                for left in range(len(candidates))
            ]
            merged_assignment = merge_then_assign_conformers(
                occupancies,
                rmsd_a,
                rmsd_b,
                pairwise_rmsd,
                active_occupancy=args.active_occupancy,
                found_occupancy=args.found_occupancy,
                rmsd_cutoff=args.rmsd_cutoff,
                merge_rmsd_threshold=args.merge_rmsd_threshold,
            )
            assignments = merged_assignment["assignments"]
            predicted_a = sum(
                occupancy
                for occupancy, label in zip(
                    merged_assignment["cluster_occupancies"],
                    merged_assignment["cluster_assignments"],
                )
                if label == "A"
            )
            predicted_b = sum(
                occupancy
                for occupancy, label in zip(
                    merged_assignment["cluster_occupancies"],
                    merged_assignment["cluster_assignments"],
                )
                if label == "B"
            )
            found_a = "A" in merged_assignment["cluster_assignments"]
            found_b = "B" in merged_assignment["cluster_assignments"]
            occupancy_accurate = (
                abs(predicted_a - float(row["target_A_occupancy"])) <= args.occupancy_tolerance
                and abs(predicted_b - float(row["target_B_occupancy"])) <= args.occupancy_tolerance
            )
            active_indices = [
                index for index, occupancy in enumerate(occupancies)
                if occupancy > args.active_occupancy
            ]
            active_geometry_valid = []
            for conformer_index in active_indices:
                candidate = candidates[conformer_index]
                angles = actual_chi(candidate)
                state, deviations, widths, canonical = classify_rotamer(
                    residue.name, angles
                )
                assignment = assignments[conformer_index]
                environment_assignment = merged_assignment[
                    "cluster_assignment"
                ][conformer_index]
                spatial = spatial_metrics(
                    candidate,
                    (
                        environment_assignment
                        if environment_assignment in {"A", "B"}
                        else None
                    ),
                )
                no_clash = (
                    spatial["no_direct_clash"] and spatial["no_symmetry_clash"]
                )
                candidate_id = f"{record['key']}__{int(row['start']):03d}__{conformer_index}"
                conformer_rows.append({
                    "candidate_id": candidate_id,
                    "site": record["key"], "start": int(row["start"]),
                    "conformer": conformer_index, "occupancy": occupancies[conformer_index],
                    "assignment": assignments[conformer_index],
                    "merge_cluster": merged_assignment[
                        "cluster_index"
                    ][conformer_index],
                    "merge_cluster_occupancy": merged_assignment[
                        "cluster_occupancy"
                    ][conformer_index],
                    "merge_cluster_representative": merged_assignment[
                        "cluster_representative"
                    ][conformer_index],
                    "merge_cluster_assignment": environment_assignment,
                    "assigned_occupancy": (
                        merged_assignment["cluster_occupancy"][conformer_index]
                        if merged_assignment["cluster_representative"][
                            conformer_index
                        ]
                        else 0.0
                    ),
                    "rmsd_to_A_conventional": rmsd_a[conformer_index],
                    "rmsd_to_B_conventional": rmsd_b[conformer_index],
                    "chi_degrees": ";".join(f"{value:.4f}" for value in angles),
                    "nearest_rotamer": state,
                    "rotamer_deviation_degrees": ";".join(
                        f"{value:.4f}" for value in deviations
                    ),
                    "rotamer_allowed_width_degrees": ";".join(
                        f"{value:.4f}" for value in widths
                    ),
                    "rotamer_within_allowed_width": canonical,
                    "canonical_like_30deg": canonical,
                    **spatial,
                    "no_sub2A_clash": no_clash,
                    "geometry_physical_valid": no_clash and canonical,
                })
                active_geometry_valid.append(no_clash and canonical)
                site_candidates.append({
                    "candidate_id": candidate_id,
                    "start": int(row["start"]), "conformer": conformer_index,
                    "assignment": assignment,
                    "merge_cluster": merged_assignment[
                        "cluster_index"
                    ][conformer_index],
                    "merge_cluster_occupancy": merged_assignment[
                        "cluster_occupancy"
                    ][conformer_index],
                    "merge_cluster_representative": merged_assignment[
                        "cluster_representative"
                    ][conformer_index],
                    "merge_cluster_assignment": environment_assignment,
                    "coordinates": candidate.tolist(),
                })
            geometry_valid = bool(active_geometry_valid) and all(active_geometry_valid)
            geometric_success = found_a and found_b and occupancy_accurate
            ensemble_rows.append({
                "site": record["key"], "start": int(row["start"]),
                "target": args.target,
                "found_A_conventional": found_a, "found_B_conventional": found_b,
                "both_found_conventional": found_a and found_b,
                "predicted_A_occupancy": predicted_a,
                "predicted_B_occupancy": predicted_b,
                "target_A_occupancy": row["target_A_occupancy"],
                "target_B_occupancy": row["target_B_occupancy"],
                "occupancy_accurate": occupancy_accurate,
                "geometric_occupancy_success": geometric_success,
                "active_conformers": len(active_indices),
                "all_active_geometry_physical_valid": geometry_valid,
                "joint_geometry_success": geometric_success and geometry_valid,
            })

        random_rotamers = []
        base_angles = torch.tensor([math.radians(value) for value in actual_chi(template)])
        signs = []
        for chi_index in range(len(spec["dihedrals"])):
            probe = torch.zeros(len(spec["dihedrals"])); probe[chi_index] = 0.01
            moved = torch.tensor([math.radians(value) for value in actual_chi(coordinates(probe))])
            signs.append(
                1.0 if float(wrap_angles(moved[chi_index] - base_angles[chi_index])) > 0 else -1.0
            )
        for _ in range(10):
            desired = []
            for chi_index in range(len(spec["dihedrals"])):
                centers = canonical_centers(residue.name, chi_index)
                chosen = int(torch.randint(len(centers), (1,), generator=generator))
                desired.append(math.radians(centers[chosen][1]))
            delta = torch.tensor(signs) * wrap_angles(torch.tensor(desired) - base_angles)
            random_rotamers.append(coordinates(delta).detach().tolist())

        base_name = f"{record['key']}_base.pdb"
        write_tmol_segment(
            visualization / base_name, structure, record["chain"], int(record["residue_number"])
        )
        base_name_a = f"{record['key']}_base_A_frozen.pdb"
        base_name_b = f"{record['key']}_base_B_frozen.pdb"
        write_tmol_segment(
            visualization / base_name_a,
            structure,
            record["chain"],
            int(record["residue_number"]),
            environment_coordinates=template.detach().numpy(),
            target_altloc="A",
        )
        write_tmol_segment(
            visualization / base_name_b,
            structure,
            record["chain"],
            int(record["residue_number"]),
            environment_coordinates=deposited_b.detach().numpy(),
            target_altloc="B",
        )
        tmol_inputs["sites"].append({
            "site": record["key"], "chain": record["chain"],
            "residue_number": int(record["residue_number"]),
            "residue_type": residue.name, "atom_names": names,
            "base_pdb": f"visualization/{base_name}",
            "base_pdb_A": f"visualization/{base_name_a}",
            "base_pdb_B": f"visualization/{base_name_b}",
            "tmol_environment_rule": "frozen_matched_deposited_minstate_v1",
            "A": template.tolist(), "B": deposited_b.tolist(),
            "random_rotamers": random_rotamers,
            "candidates": site_candidates,
        })
        # Persist each completed protein so an interruption never discards
        # previously reconstructed ensembles or staged tmol candidates.
        write_csv(args.output / "ensemble_geometry_audit.csv", ensemble_rows)
        write_csv(args.output / "active_conformer_geometry_audit.csv", conformer_rows)
        write_csv(args.output / "deposited_control_geometry_audit.csv", control_rows)
        (args.output / "tmol_inputs.json").write_text(json.dumps(tmol_inputs) + "\n")
        print(json.dumps({
            "completed_site": site_index + 1, "total_sites": len(selection["sites"]),
            "site": record["key"], "ensembles": len(result_rows),
            "active_conformers": len(site_candidates),
        }), flush=True)

    write_csv(args.output / "ensemble_geometry_audit.csv", ensemble_rows)
    write_csv(args.output / "active_conformer_geometry_audit.csv", conformer_rows)
    write_csv(args.output / "deposited_control_geometry_audit.csv", control_rows)
    (args.output / "tmol_inputs.json").write_text(json.dumps(tmol_inputs) + "\n")
    summary = {
        "audit_rule_version": AUDIT_RULE_VERSION,
        "conformer_matching_version": CONFORMER_MATCHING_VERSION,
        "merge_rmsd_threshold_A": args.merge_rmsd_threshold,
        "status": "geometry_complete",
        "ensembles": len(ensemble_rows),
        "active_conformers": len(conformer_rows),
        "both_found_conventional": sum(row["both_found_conventional"] for row in ensemble_rows),
        "geometric_occupancy_success": sum(
            row["geometric_occupancy_success"] for row in ensemble_rows
        ),
        "geometry_physical_valid_ensembles": sum(
            row["all_active_geometry_physical_valid"] for row in ensemble_rows
        ),
        "joint_geometry_success": sum(row["joint_geometry_success"] for row in ensemble_rows),
        "sub2A_direct_active_conformers": sum(
            not row["no_direct_clash"] for row in conformer_rows
        ),
        "sub2A_symmetry_active_conformers": sum(
            not row["no_symmetry_clash"] for row in conformer_rows
        ),
        "noncanonical_active_conformers": sum(
            not row["canonical_like_30deg"] for row in conformer_rows
        ),
        "noncanonical_deposited_controls": sum(
            not row["canonical_like_30deg"] for row in control_rows
        ),
        "criteria": {
            "conventional_rmsd_cutoff_A": args.rmsd_cutoff,
            "occupancy_tolerance": args.occupancy_tolerance,
            "active_occupancy": args.active_occupancy,
            "found_occupancy": args.found_occupancy,
            "merge_rmsd_threshold_A": args.merge_rmsd_threshold,
            "merge_method": (
                "A/B-protected single-linkage connected components; protected "
                "anchor or occupancy-weighted medoid geometry; summed cluster "
                "occupancy"
            ),
            "conformer_matching": (
                "Hungarian optimal one-to-one A/B assignment; maximize valid "
                "matches then minimize total conventional RMSD"
            ),
            "clash_cutoff_A": args.clash_cutoff,
            "rotamer_gate": "shared residue/chi centers and per-chi widths",
            "tmol_gate": "no worse than matched deposited conformer",
            "altloc_clash_partition": (
                "per-neighbor min over protein altloc states; labeled waters "
                "match assigned A/B; unlabeled waters use occupancy-scaled cutoff"
            ),
        },
    }
    (args.output / "geometry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
