from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


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
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    first = b - a
    second = c - b
    third = d - c
    normal_one = np.cross(first, second)
    normal_two = np.cross(second, third)
    normal_one /= np.linalg.norm(normal_one)
    normal_two /= np.linalg.norm(normal_two)
    middle = second / np.linalg.norm(second)
    # Match gemmi/tmol's protein-torsion sign convention.
    return -math.degrees(
        math.atan2(np.dot(np.cross(normal_one, middle), normal_two),
                   np.dot(normal_one, normal_two))
    )


def backbone_angles(path: Path, chain: str, residue_number: int) -> tuple[float, float]:
    atoms: dict[tuple[str, int, str], np.ndarray] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        key = (line[21].strip(), int(line[22:26]), line[12:16].strip())
        atoms[key] = np.asarray(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
    phi = (
        dihedral(
            atoms[(chain, residue_number - 1, "C")],
            atoms[(chain, residue_number, "N")],
            atoms[(chain, residue_number, "CA")],
            atoms[(chain, residue_number, "C")],
        )
        if (chain, residue_number - 1, "C") in atoms
        else -60.0
    )
    psi = (
        dihedral(
            atoms[(chain, residue_number, "N")],
            atoms[(chain, residue_number, "CA")],
            atoms[(chain, residue_number, "C")],
            atoms[(chain, residue_number + 1, "N")],
        )
        if (chain, residue_number + 1, "N") in atoms
        else 60.0
    )
    return phi, psi


def circular_difference(values: np.ndarray, center: np.ndarray) -> np.ndarray:
    return (values - center + 180.0) % 360.0 - 180.0


def periodic_difference(
    values: np.ndarray, center: np.ndarray, periods: np.ndarray
) -> np.ndarray:
    return (values - center + periods / 2.0) % periods - periods / 2.0


def backbone_index(angle: float, start: float, step: float, size: int) -> int:
    return int(round((angle - start) / step)) % size


def classify(
    chis: list[float],
    widths: list[float],
    residue: str,
    phi: float,
    psi: float,
    library: object,
    probability_threshold: float,
    sigma_threshold: float,
) -> dict[str, object]:
    data = library.rotameric_data
    probabilities = data.rotamer_probabilities.detach().cpu().numpy()
    means = data.rotamer_means.detach().cpu().numpy()
    stdvs = data.rotamer_stdvs.detach().cpu().numpy()
    states = data.rotamers.detach().cpu().numpy()
    starts = data.backbone_dihedral_start.detach().cpu().numpy()
    steps = data.backbone_dihedral_step.detach().cpu().numpy()
    phi_indices = (
        [backbone_index(phi, starts[0], steps[0], probabilities.shape[1])]
        if math.isfinite(phi)
        else list(range(probabilities.shape[1]))
    )
    psi_indices = (
        [backbone_index(psi, starts[1], steps[1], probabilities.shape[2])]
        if math.isfinite(psi)
        else list(range(probabilities.shape[2]))
    )
    symmetry_periods = {
        "ARG": {3: 180.0},
        "ASN": {1: 180.0},
        "ASP": {1: 180.0},
        "GLN": {2: 180.0},
        "GLU": {2: 180.0},
        "PHE": {1: 180.0},
        "TYR": {1: 180.0},
    }
    periods = np.asarray(
        [symmetry_periods.get(residue, {}).get(index, 360.0)
         for index in range(len(chis))],
        dtype=float,
    )
    best_record = None
    best_covering_probability = 0.0
    covering_states = 0
    nearest_qualifying_record = None
    for phi_index in phi_indices:
        for psi_index in psi_indices:
            grid_means = means[:, phi_index, psi_index, : len(chis)]
            grid_stdvs = stdvs[:, phi_index, psi_index, : len(chis)]
            grid_probabilities = probabilities[:, phi_index, psi_index]
            differences = periodic_difference(
                np.asarray(chis, dtype=float)[None, :],
                grid_means,
                periods[None, :],
            )
            z = np.abs(differences) / np.maximum(grid_stdvs, 1e-6)
            covered = np.all(
                np.abs(differences) <= np.asarray(widths, dtype=float)[None, :],
                axis=1,
            )
            if np.any(covered):
                covering_states += int(covered.sum())
                best_covering_probability = max(
                    best_covering_probability,
                    float(grid_probabilities[covered].max()),
                )
            qualifying = grid_probabilities >= probability_threshold
            if np.any(qualifying):
                qualifying_indices = np.flatnonzero(qualifying)
                qualifying_distances = np.abs(differences[qualifying])
                qualifying_maximum = qualifying_distances.max(axis=1)
                local = int(np.argmin(qualifying_maximum))
                state_index = int(qualifying_indices[local])
                distance_record = (
                    float(qualifying_maximum[local]),
                    state_index,
                    phi_index,
                    psi_index,
                    float(grid_probabilities[state_index]),
                    qualifying_distances[local],
                )
                if (
                    nearest_qualifying_record is None
                    or distance_record[0] < nearest_qualifying_record[0]
                ):
                    nearest_qualifying_record = distance_record
            objective = np.square(z).sum(axis=1)
            state = int(np.argmin(objective))
            record = (
                float(objective[state]),
                state,
                phi_index,
                psi_index,
                float(grid_probabilities[state]),
                float(z[state].max()),
            )
            if best_record is None or record[0] < best_record[0]:
                best_record = record
    assert best_record is not None
    _objective, best, phi_index, psi_index, probability, max_z = best_record
    supported = best_covering_probability >= probability_threshold
    if nearest_qualifying_record is None:
        qualifying_max = math.nan
        qualifying_state = ""
        qualifying_probability = math.nan
        qualifying_distances = np.full(len(chis), math.nan)
    else:
        (
            qualifying_max,
            qualifying_state_index,
            _qualifying_phi,
            _qualifying_psi,
            qualifying_probability,
            qualifying_distances,
        ) = nearest_qualifying_record
        qualifying_state = ";".join(
            str(int(value)) for value in states[qualifying_state_index]
        )
    return {
        "phi_degrees": phi,
        "psi_degrees": psi,
        "missing_backbone_angles_marginalized": (
            not math.isfinite(phi) or not math.isfinite(psi)
        ),
        "phi_grid_index": phi_index,
        "psi_grid_index": psi_index,
        "nearest_joint_state": ";".join(str(int(value)) for value in states[best]),
        "nearest_state_probability": probability,
        "nearest_state_max_abs_z": max_z,
        "production_widths_degrees": ";".join(str(value) for value in widths),
        "covering_joint_states": covering_states,
        "best_covering_joint_state_probability": best_covering_probability,
        "nearest_qualifying_joint_state": qualifying_state,
        "nearest_qualifying_state_probability": qualifying_probability,
        "nearest_qualifying_per_chi_distance_degrees": ";".join(
            str(float(value)) for value in qualifying_distances
        ),
        "nearest_qualifying_max_chi_distance_degrees": qualifying_max,
        "probability_pass": supported,
        "deviation_pass": covering_states > 0,
        "independent_library_pass": supported,
        "gate_library_disagreement": not supported,
    }


def summarize(rows: list[dict[str, object]], population: str) -> list[dict[str, object]]:
    result = []
    for residue in sorted({str(row["residue"]) for row in rows}):
        selected = [row for row in rows if row["residue"] == residue]
        disagreements = sum(
            bool(row["gate_library_disagreement"]) for row in selected
        )
        result.append(
            {
                "population": population,
                "residue": residue,
                "conformers": len(selected),
                "independent_library_rejections": disagreements,
                "disagreement_rate": disagreements / len(selected),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformer-table", type=Path, action="append", required=True)
    parser.add_argument("--tmol-input", type=Path, action="append", required=True)
    parser.add_argument("--deposited-chi-table", type=Path, required=True)
    parser.add_argument("--dunbrack-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probability-threshold", type=float, default=0.003)
    parser.add_argument("--sigma-threshold", type=float, default=3.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary

    database = DunbrackRotamerLibrary.from_file(str(args.dunbrack_bin))
    table_for_residue = {
        item.residue_name: item.dun_table_name for item in database.dun_lookup
    }
    libraries = {
        item.table_name: item
        for item in (
            *database.rotameric_libraries,
            *database.semi_rotameric_libraries,
        )
    }

    sites = {}
    for path in args.tmol_input:
        root = path.parent
        for site in json.loads(path.read_text())["sites"]:
            sites[site["site"]] = (root, site)

    backbone = {}
    for site_name, (root, site) in sites.items():
        base_key = "base_pdb_A" if "base_pdb_A" in site else "base_pdb"
        backbone[site_name] = backbone_angles(
            root / site[base_key],
            site["chain"],
            int(site["residue_number"]),
        )

    endpoint_rows = []
    for path in args.conformer_table:
        for row in read_csv(path):
            if not as_bool(row["rotamer_within_allowed_width"]):
                continue
            _root, site = sites[row["site"]]
            residue = site["residue_type"]
            chis = [float(value) for value in row["chi_degrees"].split(";")]
            widths = [
                float(value)
                for value in row["rotamer_allowed_width_degrees"].split(";")
            ]
            phi, psi = backbone[row["site"]]
            endpoint_rows.append(
                {
                    "population": "current_endpoint_gate_pass",
                    "candidate_id": row["candidate_id"],
                    "site": row["site"],
                    "residue": residue,
                    "start": int(row["start"]),
                    "conformer": int(row["conformer"]),
                    "occupancy": float(row["occupancy"]),
                    "chi_degrees": row["chi_degrees"],
                    **classify(
                        chis,
                        widths,
                        residue,
                        phi,
                        psi,
                        libraries[table_for_residue[residue]],
                        args.probability_threshold,
                        args.sigma_threshold,
                    ),
                }
            )

    deposited_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.deposited_chi_table):
        deposited_groups[(row["site"], row["residue"], row["conformer"])].append(row)
    deposited_rows = []
    for (site_name, residue, conformer), rows in sorted(deposited_groups.items()):
        rows.sort(key=lambda row: int(row["chi_index"]))
        chis = [float(row["chi_degrees"]) for row in rows]
        widths = [float(row["allowed_width_degrees"]) for row in rows]
        phi, psi = backbone[site_name]
        deposited_rows.append(
            {
                "population": "deposited_control",
                "candidate_id": f"{site_name}_deposited_{conformer}",
                "site": site_name,
                "residue": residue,
                "start": "",
                "conformer": conformer,
                "occupancy": "",
                "chi_degrees": ";".join(str(value) for value in chis),
                **classify(
                    chis,
                    widths,
                    residue,
                    phi,
                    psi,
                    libraries[table_for_residue[residue]],
                    args.probability_threshold,
                    args.sigma_threshold,
                ),
            }
        )

    endpoint_summary = summarize(endpoint_rows, "current_endpoint_gate_pass")
    deposited_summary = summarize(deposited_rows, "deposited_control")
    args.output.mkdir(parents=True)
    atomic_csv(args.output / "endpoint_joint_library.csv", endpoint_rows)
    atomic_csv(args.output / "deposited_joint_library.csv", deposited_rows)
    atomic_csv(
        args.output / "disagreement_by_residue.csv",
        [*endpoint_summary, *deposited_summary],
    )
    atomic_json(
        args.output / "summary.json",
        {
            "library": "tmol bundled backbone-dependent Dunbrack joint states",
            "backbone_grid": "nearest 10-degree phi/psi bin",
            "classifier": (
                "at least one backbone-dependent joint state has probability "
                ">= threshold and covers every chi within the production "
                "gate's own per-chi widths"
            ),
            "probability_threshold": args.probability_threshold,
            "sigma_threshold": (
                "reported for nearest-state attribution only; not used in "
                "the disagreement classifier"
            ),
            "endpoint_population": (
                "all current active endpoint conformers passing the production "
                "rotamer gate; no sampling"
            ),
            "endpoint_summary": endpoint_summary,
            "deposited_control_summary": deposited_summary,
        },
    )
    print(json.dumps(
        {
            "endpoint_summary": endpoint_summary,
            "deposited_control_summary": deposited_summary,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
