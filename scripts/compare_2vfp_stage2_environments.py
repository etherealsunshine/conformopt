from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np
import torch

from density_denoiser.clash_environment import (
    SoftEnvironmentRecord,
    normalized_altloc,
    partition_soft_environment,
    soft_clash_penalty,
)
from density_denoiser.five_site_optimizer import _alt_atom_map
from density_denoiser.residue_geometry import CHI_SPECS


PROTEIN_NAMES = {
    "ALA", "ASN", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
WATER_NAMES = {"HOH", "WAT", "DOD"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def stale_selected_atoms(structure: gemmi.Structure) -> list[tuple]:
    """July-23 behavior: protein only, one preferred state per residue."""
    selected = []
    for chain in structure[0]:
        for residue in chain:
            if residue.name not in CHI_SPECS and residue.name not in PROTEIN_NAMES:
                continue
            blank: dict[str, gemmi.Atom] = {}
            alternate: dict[str, dict[str, gemmi.Atom]] = {}
            for atom in residue:
                if atom.element.name == "H":
                    continue
                name = atom.name.strip()
                altloc = normalized_altloc(atom.altloc)
                if altloc:
                    alternate.setdefault(altloc, {})[name] = atom
                else:
                    blank[name] = atom
            if alternate:
                _, atoms = max(
                    alternate.items(),
                    key=lambda item: (
                        len(item[1]),
                        sum(float(atom.occ) for atom in item[1].values()),
                    ),
                )
                chosen = {**blank, **atoms}
            else:
                chosen = blank
            selected.extend((chain, residue, atom) for atom in chosen.values())
    return selected


def all_soft_atoms(structure: gemmi.Structure) -> list[tuple]:
    result = []
    for chain in structure[0]:
        for residue in chain:
            if (
                residue.name not in CHI_SPECS
                and residue.name not in PROTEIN_NAMES
                and residue.name not in WATER_NAMES
            ):
                continue
            result.extend(
                (chain, residue, atom)
                for atom in residue
                if atom.element.name != "H"
            )
    return result


def direct_records(
    atoms: list[tuple],
    record: dict,
    names: list[str],
    ca_position: np.ndarray,
    environment_radius: float,
) -> list[SoftEnvironmentRecord]:
    result = []
    for chain, residue, atom in atoms:
        same_target = (
            chain.name == record["chain"]
            and residue.seqid.num == int(record["residue_number"])
            and residue.seqid.icode == record["insertion_code"]
        )
        if same_target and atom.name.strip() in names:
            continue
        xyz = np.asarray(atom.pos.tolist(), dtype=float)
        if np.linalg.norm(xyz - ca_position) > environment_radius:
            continue
        result.append(
            SoftEnvironmentRecord(
                xyz=tuple(xyz.tolist()),
                group_key=f"{chain.name}:{residue.seqid.num}:{residue.seqid.icode}",
                atom_name=atom.name.strip(),
                altloc=normalized_altloc(atom.altloc),
                occupancy=float(atom.occ),
                is_water=residue.name in WATER_NAMES,
            )
        )
    return result


def partition_stale(
    records: list[SoftEnvironmentRecord],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]], list]:
    xyz = torch.tensor(
        [record.xyz for record in records], dtype=torch.float32, device=device
    )
    return (
        xyz,
        torch.ones(len(records), dtype=torch.float32, device=device),
        [],
        records,
    )


def partition_water_invariant_v1(
    records: list[SoftEnvironmentRecord],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[list[torch.Tensor]], list]:
    invariant = [
        record for record in records if not record.altloc or record.is_water
    ]
    grouped: dict[str, dict[str, list[SoftEnvironmentRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if record.altloc and not record.is_water:
            grouped[record.group_key][record.altloc].append(record)
    invariant_xyz = torch.tensor(
        [record.xyz for record in invariant], dtype=torch.float32, device=device
    )
    invariant_weights = torch.tensor(
        [record.occupancy if record.is_water else 1.0 for record in invariant],
        dtype=torch.float32,
        device=device,
    )
    alternate_states = [
        [
            torch.tensor(
                [record.xyz for record in state],
                dtype=torch.float32,
                device=device,
            )
            for state in states.values()
        ]
        for states in grouped.values()
    ]
    return invariant_xyz, invariant_weights, alternate_states, invariant


def pair_mask(
    names: list[str],
    invariant: list[SoftEnvironmentRecord],
    record: dict,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones((len(names), len(invariant)), dtype=torch.bool, device=device)
    target_key = (
        f"{record['chain']}:{record['residue_number']}:{record['insertion_code']}"
    )
    for moving_index, moving_name in enumerate(names):
        for environment_index, environment_record in enumerate(invariant):
            if (
                environment_record.group_key == target_key
                and moving_name == "CB"
                and environment_record.atom_name == "CA"
            ):
                mask[moving_index, environment_index] = False
    return mask


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def load_candidates(
    audit_root: Path, site: str
) -> tuple[dict[str, dict], list[dict], set[int]]:
    payload = json.loads((audit_root / "tmol_inputs.json").read_text())
    site_payload = next(row for row in payload["sites"] if row["site"] == site)
    candidates = {
        row["candidate_id"]: row for row in site_payload["candidates"]
    }
    active = [
        row
        for row in read_csv(audit_root / "active_conformer_geometry_audit.csv")
        if row["site"] == site
    ]
    recovered = {
        int(row["start"])
        for row in read_csv(audit_root / "ensemble_geometry_audit.csv")
        if row["site"] == site and as_bool(row["both_found_conventional"])
    }
    return candidates, active, recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--stale-audit", type=Path, required=True)
    parser.add_argument("--stale-starts", type=Path, required=True)
    parser.add_argument("--fresh-audit", type=Path, required=True)
    parser.add_argument("--fresh-starts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site", default="2VFP_A_TYR417")
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    record = next(row for row in selection["sites"] if row["key"] == args.site)
    structure = gemmi.read_structure(record["pdb_path"])
    device = torch.device("cpu")
    chain = next(chain for chain in structure[0] if chain.name == record["chain"])
    residue = next(
        residue
        for residue in chain
        if residue.seqid.num == int(record["residue_number"])
        and residue.seqid.icode == record["insertion_code"]
    )
    map_a = _alt_atom_map(residue, "A", device)
    names = [
        atom.name.strip()
        for atom in residue
        if atom.altloc == "B"
        and atom.element.name != "H"
        and atom.name.strip() not in {"N", "CA", "C", "O"}
    ]
    ca_position = map_a["CA"].numpy()
    maximum_radius = max(
        float(torch.linalg.vector_norm(value - map_a["CA"])) for value in map_a.values()
    )
    environment_radius = maximum_radius + 4.0

    stale_records = direct_records(
        stale_selected_atoms(structure),
        record,
        names,
        ca_position,
        environment_radius,
    )
    all_records = direct_records(
        all_soft_atoms(structure),
        record,
        names,
        ca_position,
        environment_radius,
    )
    partitions = {
        "july23_protein_preferred_state": partition_stale(stale_records, device),
        "current_water_invariant_v1": partition_water_invariant_v1(
            all_records, device
        ),
        "water_minstate_v2": partition_soft_environment(all_records, device),
    }

    rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    recovered_by_run: dict[str, set[int]] = {}
    run_configs: dict[str, dict[str, object]] = {}
    for run_label, audit_root, starts_path in (
        ("stale_july23", args.stale_audit, args.stale_starts),
        ("current_completed", args.fresh_audit, args.fresh_starts),
    ):
        config = json.loads((starts_path.parent.parent / "run_config.json").read_text())
        run_configs[run_label] = {
            key: config.get(key)
            for key in (
                "physics_refinement_steps",
                "physics_refinement_lr_scale",
                "lambda_vdw",
                "lambda_rot",
                "lambda_clash",
                "vdw_threshold",
                "clash_threshold",
                "symmetry_hard_threshold",
                "symmetry_barrier_buffer",
                "symmetry_barrier_scale",
            )
        }
        candidates, active, recovered = load_candidates(audit_root, args.site)
        recovered_by_run[run_label] = recovered
        active_by_start: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in active:
            active_by_start[int(row["start"])].append(row)
        recorded = {
            int(row["start"]): float(row["final_vdw_loss"])
            for row in read_csv(starts_path)
        }
        for start, conformer_rows in sorted(active_by_start.items()):
            values: dict[str, float] = {}
            for rule, (xyz, weights, alternate_states, invariant) in partitions.items():
                mask = pair_mask(names, invariant, record, device)
                values[rule] = sum(
                    float(
                        soft_clash_penalty(
                            torch.tensor(
                                candidates[row["candidate_id"]]["coordinates"],
                                dtype=torch.float32,
                                device=device,
                            ),
                            xyz,
                            weights,
                            alternate_states,
                            3.0,
                            mask,
                        )
                    )
                    for row in conformer_rows
                )
            rows.append(
                {
                    "run": run_label,
                    "start": start,
                    "active_conformers": len(conformer_rows),
                    "recorded_final_vdw": recorded[start],
                    **values,
                }
            )
        run_rows = [row for row in rows if row["run"] == run_label]
        columns = (
            "recorded_final_vdw",
            "july23_protein_preferred_state",
            "current_water_invariant_v1",
            "water_minstate_v2",
        )
        recovered_rows = [
            row for row in run_rows if int(row["start"]) in recovered
        ]
        summaries[run_label] = {
            "all_50": {
                column: float(np.median([float(row[column]) for row in run_rows]))
                for column in columns
            },
            "both_found": {
                "starts": len(recovered_rows),
                **{
                    column: float(
                        np.median([float(row[column]) for row in recovered_rows])
                    )
                    for column in columns
                },
            },
        }

    common_starts = (
        recovered_by_run["stale_july23"]
        & recovered_by_run["current_completed"]
    )
    common_summary = {}
    for run_label in ("stale_july23", "current_completed"):
        common_rows = [
            row
            for row in rows
            if row["run"] == run_label and int(row["start"]) in common_starts
        ]
        common_summary[run_label] = {
            column: float(
                np.median([float(row[column]) for row in common_rows])
            )
            for column in (
                "recorded_final_vdw",
                "july23_protein_preferred_state",
                "current_water_invariant_v1",
                "water_minstate_v2",
            )
        }

    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per_start_vdw_environment_rescore.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "site": args.site,
        "environment_radius_A": environment_radius,
        "stale_direct_atoms": len(stale_records),
        "all_state_direct_atoms": len(all_records),
        "all_state_water_atoms": sum(record.is_water for record in all_records),
        "median_raw_vdw": summaries,
        "common_both_found_starts": sorted(common_starts),
        "common_both_found_median_raw_vdw": common_summary,
        "saved_stage2_config": run_configs,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
