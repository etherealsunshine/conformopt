from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import gemmi
import numpy as np
import torch

from .clash_environment import (
    SoftEnvironmentRecord,
    normalized_altloc,
    partition_soft_environment,
    soft_clash_penalty,
)
from .five_site_optimizer import _alt_atom_map, _selected_protein_heavy_atoms


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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


def identify_relevant_failed_starts(
    ensemble_rows: list[dict[str, str]],
    conformer_rows: list[dict[str, str]],
) -> list[int]:
    recovered = {
        int(row["start"])
        for row in ensemble_rows
        if _as_bool(row["geometric_occupancy_success"])
    }
    hard_failed = {
        int(row["start"])
        for row in conformer_rows
        if not _as_bool(row["no_symmetry_clash"])
    }
    return sorted(recovered & hard_failed)


def build_optimizer_symmetry_environment(
    structure: gemmi.Structure,
    record: dict,
    vdw_threshold: float,
    soft_symmetry_threshold: float,
    device: torch.device,
):
    chain = next(chain for chain in structure[0] if chain.name == record["chain"])
    residue = next(
        residue
        for residue in chain
        if residue.seqid.num == int(record["residue_number"])
        and residue.seqid.icode == record["insertion_code"]
    )
    map_a = _alt_atom_map(residue, "A", device)
    ca_position = map_a["CA"].detach().cpu().numpy()
    max_sidechain_radius = max(
        float(torch.linalg.vector_norm(value - map_a["CA"]).detach().cpu())
        for value in map_a.values()
    )
    environment_radius = (
        max_sidechain_radius
        + max(vdw_threshold, soft_symmetry_threshold)
        + 1.0
    )
    heavy_atoms = _selected_protein_heavy_atoms(structure)
    cell = structure.cell
    spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
    symmetry_records = []
    for operation_index, operation in enumerate(spacegroup.operations()):
        for tx in (-1, 0, 1):
            for ty in (-1, 0, 1):
                for tz in (-1, 0, 1):
                    if operation_index == 0 and tx == ty == tz == 0:
                        continue
                    for context_chain, context_residue, atom in heavy_atoms:
                        transformed = operation.apply_to_xyz(
                            cell.fractionalize(atom.pos).tolist()
                        )
                        position = cell.orthogonalize(gemmi.Fractional(
                            transformed[0] + tx,
                            transformed[1] + ty,
                            transformed[2] + tz,
                        ))
                        xyz = np.asarray(position.tolist())
                        if np.linalg.norm(xyz - ca_position) <= environment_radius:
                            symmetry_records.append(SoftEnvironmentRecord(
                                xyz=tuple(xyz.tolist()),
                                group_key=(
                                    f"sym{operation_index}[{tx},{ty},{tz}]/"
                                    f"{context_chain.name}:"
                                    f"{context_residue.seqid.num}:"
                                    f"{context_residue.seqid.icode}"
                                ),
                                atom_name=atom.name.strip(),
                                altloc=normalized_altloc(atom.altloc),
                                occupancy=float(atom.occ),
                                is_water=(
                                    context_residue.name in {"HOH", "WAT", "DOD"}
                                ),
                            ))
    environment = partition_soft_environment(symmetry_records, device)
    return (*environment[:3], environment_radius)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage-2 soft symmetry loss at hard-failed endpoints."
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--site", default="3GMI_A_GLU5")
    parser.add_argument("--soft-threshold", type=float, default=2.5)
    parser.add_argument("--hard-threshold", type=float, default=2.0)
    parser.add_argument("--vdw-threshold", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    record = next(row for row in selection["sites"] if row["key"] == args.site)
    structure = gemmi.read_structure(record["pdb_path"])
    device = torch.device("cpu")
    invariant, weights, alternate_states, radius = (
        build_optimizer_symmetry_environment(
            structure,
            record,
            args.vdw_threshold,
            args.soft_threshold,
            device,
        )
    )
    with (args.audit_root / "ensemble_geometry_audit.csv").open(newline="") as handle:
        ensemble_rows = [
            row for row in csv.DictReader(handle) if row["site"] == args.site
        ]
    with (args.audit_root / "active_conformer_geometry_audit.csv").open(
        newline=""
    ) as handle:
        conformer_rows = [
            row for row in csv.DictReader(handle) if row["site"] == args.site
        ]
    tmol_inputs = json.loads((args.audit_root / "tmol_inputs.json").read_text())
    site_input = next(row for row in tmol_inputs["sites"] if row["site"] == args.site)
    candidates = {row["candidate_id"]: row for row in site_input["candidates"]}
    relevant_starts = identify_relevant_failed_starts(ensemble_rows, conformer_rows)

    detail_rows = []
    endpoint_rows = []
    for start in relevant_starts:
        active_rows = [
            row for row in conformer_rows if int(row["start"]) == start
        ]
        losses = []
        for row in active_rows:
            candidate = torch.tensor(
                candidates[row["candidate_id"]]["coordinates"],
                dtype=torch.float32,
                device=device,
            )
            loss = float(soft_clash_penalty(
                candidate,
                invariant,
                weights,
                alternate_states,
                args.soft_threshold,
            ))
            losses.append(loss)
            detail_rows.append({
                "site": args.site,
                "start": start,
                "candidate_id": row["candidate_id"],
                "assignment": row["assignment"],
                "occupancy": float(row["occupancy"]),
                "hard_symmetry_pass": _as_bool(row["no_symmetry_clash"]),
                "hard_minimum_distance": float(row["min_symmetry_distance"]),
                "hard_minimum_clearance": float(row["min_symmetry_clearance"]),
                "soft_symmetry_loss": loss,
            })
        hard_failed_rows = [
            row for row in active_rows if not _as_bool(row["no_symmetry_clash"])
        ]
        endpoint_rows.append({
            "site": args.site,
            "start": start,
            "active_conformers": len(active_rows),
            "hard_failed_conformers": len(hard_failed_rows),
            "hard_minimum_distance": min(
                float(row["min_symmetry_distance"]) for row in active_rows
            ),
            "stage2_soft_symmetry_loss": sum(losses),
            "lambda_clash_weighted_loss": 5.0 * sum(losses),
        })

    if len(endpoint_rows) != 11:
        raise ValueError(
            f"expected 11 recovered+occupancy endpoints with hard symmetry failure, "
            f"found {len(endpoint_rows)}"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    _atomic_csv(args.output / "endpoint_symmetry_loss.csv", endpoint_rows)
    _atomic_csv(args.output / "conformer_symmetry_loss.csv", detail_rows)
    summary = {
        "site": args.site,
        "endpoints": len(endpoint_rows),
        "soft_symmetry_cutoff_angstrom": args.soft_threshold,
        "hard_symmetry_gate_angstrom": args.hard_threshold,
        "optimizer_environment_radius_angstrom": radius,
        "symmetry_invariant_atoms": int(invariant.shape[0]),
        "symmetry_alternate_residue_groups": len(alternate_states),
        "zero_loss_endpoints": sum(
            math.isclose(float(row["stage2_soft_symmetry_loss"]), 0.0, abs_tol=1e-12)
            for row in endpoint_rows
        ),
        "minimum_endpoint_loss": min(
            float(row["stage2_soft_symmetry_loss"]) for row in endpoint_rows
        ),
        "maximum_endpoint_loss": max(
            float(row["stage2_soft_symmetry_loss"]) for row in endpoint_rows
        ),
    }
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
