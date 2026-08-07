from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Callable

import gemmi
import numpy as np
import torch

from density_denoiser.data_pipeline import _grid_coordinates, _sidechain_atoms
from density_denoiser.five_site_optimizer import _alt_atom_map, _normalize
from density_denoiser.residue_geometry import CHI_SPECS
from experiments.probe4.core import torsion_to_coords, wrap_angles


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        if rows:
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
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def parse_vector(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split(";")], dtype=float)


def parse_chi(value: str) -> np.ndarray:
    return np.asarray(
        [[float(item) for item in row.split(";")] for row in value.split("|")],
        dtype=float,
    )


def locate_shard(
    site: str, baseline_root: Path, replacement_root: Path, replacement: bool
) -> Path:
    root = replacement_root if replacement else baseline_root
    candidates = sorted((root / "shards").glob(f"**/{site}"))
    candidates = [path for path in candidates if path.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one shard for {site} under {root}, found {candidates}"
        )
    return candidates[0]


def select_pair(
    occupancies: np.ndarray,
    assignments: list[str],
    rmsd_a: np.ndarray,
    rmsd_b: np.ndarray,
    minimum_occupancy: float = 0.10,
) -> dict[str, int] | None:
    selected = {}
    for assignment, distances in (("A", rmsd_a), ("B", rmsd_b)):
        candidates = [
            index
            for index in range(len(occupancies))
            if assignments[index] == assignment
            and occupancies[index] > minimum_occupancy
        ]
        if not candidates:
            return None
        selected[assignment] = min(candidates, key=lambda index: distances[index])
    return selected


def site_density_components(
    record: dict,
    config: dict,
    shard: Path,
    device: torch.device,
) -> tuple[torch.Tensor, Callable, Callable]:
    structure = gemmi.read_structure(record["pdb_path"])
    pair = np.load(record["pair_path"], allow_pickle=False)
    metadata = json.loads(str(pair["metadata"].item()))
    residue = next(
        residue
        for chain in structure[0]
        if chain.name == record["chain"]
        for residue in chain
        if residue.seqid.num == int(record["residue_number"])
        and residue.seqid.icode == record["insertion_code"]
    )
    map_a = _alt_atom_map(residue, "A", device)
    map_b = _alt_atom_map(residue, "B", device)
    b_atoms = [
        atom
        for atom in residue
        if atom.altloc == "B"
        and atom.element.name != "H"
        and atom.name.strip() not in {"N", "CA", "C", "O"}
    ]
    names = [atom.name.strip() for atom in b_atoms]
    if not names or any(name not in map_a or name not in map_b for name in names):
        raise RuntimeError(f"incomplete A/B sidechain atoms at {record['key']}")
    spec = CHI_SPECS[residue.name]
    template = torch.stack([map_a[name] for name in names])
    fixed_lookup = {name: value for name, value in map_a.items() if name not in names}

    def coordinates_from_chi(chi: torch.Tensor) -> torch.Tensor:
        return torsion_to_coords(
            template,
            names,
            wrap_angles(chi),
            list(spec["rotations"]),
            fixed_lookup,
        )

    center = np.asarray(metadata["center"], dtype=np.float32)
    patch_center = np.asarray(
        metadata.get("patch_center_crystal", metadata["center"]),
        dtype=np.float32,
    )
    coordinates = _grid_coordinates(
        patch_center,
        int(config["patch_size"]),
        float(config["spacing"]),
        None,
    )
    radial_mask = np.linalg.norm(coordinates - center, axis=-1) <= float(
        config["grid_radius"]
    )
    selected_grid = torch.tensor(
        coordinates[radial_mask], dtype=torch.float32, device=device
    )

    raw_occ_b = np.asarray([atom.occ for atom in b_atoms], dtype=np.float32)
    occ_b = float(np.median(raw_occ_b))
    variable_sigma2 = torch.tensor(
        [
            max(float(atom.b_iso) / (8.0 * math.pi**2), 0.04)
            for atom in b_atoms
        ],
        dtype=torch.float32,
        device=device,
    )
    variable_weights = torch.tensor(
        [
            atom.element.atomic_number * atom.occ / max(occ_b, 1e-6)
            for atom in b_atoms
        ],
        dtype=torch.float32,
        device=device,
    )
    shared_atoms = [
        atom
        for atom in _sidechain_atoms(residue)
        if atom.altloc in ("\x00", " ", "") and atom.name.strip() not in names
    ]
    shared_xyz = torch.tensor(
        [atom.pos.tolist() for atom in shared_atoms],
        dtype=torch.float32,
        device=device,
    )
    shared_sigma2 = torch.tensor(
        [
            max(float(atom.b_iso) / (8.0 * math.pi**2), 0.04)
            for atom in shared_atoms
        ],
        dtype=torch.float32,
        device=device,
    )
    shared_weights = torch.tensor(
        [atom.element.atomic_number * atom.occ for atom in shared_atoms],
        dtype=torch.float32,
        device=device,
    )

    def atom_density(
        xyz: torch.Tensor, sigma2: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        if not len(xyz):
            return torch.zeros(len(selected_grid), device=device)
        distance2 = (
            selected_grid[:, None, :] - xyz[None, :, :]
        ).square().sum(dim=-1)
        normalization = (2.0 * math.pi * sigma2).pow(-1.5)
        return (
            weights[None, :]
            * normalization[None, :]
            * torch.exp(-distance2 / (2.0 * sigma2[None, :]))
        ).sum(dim=1)

    fixed_density = atom_density(shared_xyz, shared_sigma2, shared_weights)
    target = torch.tensor(
        np.load(shard / f"{record['key']}_optimizer_synthetic_vector.npy"),
        dtype=torch.float32,
        device=device,
    )

    def components(chis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conformer_density = torch.stack(
            [
                atom_density(
                    coordinates_from_chi(row), variable_sigma2, variable_weights
                )
                for row in chis
            ]
        )
        return fixed_density, conformer_density

    return target, components, coordinates_from_chi


def density_derivatives(
    fixed: torch.Tensor,
    contributions: torch.Tensor,
    target: torch.Tensor,
    occupancies: np.ndarray,
) -> dict[str, np.ndarray | float]:
    # Production components are float32. Derivatives are evaluated in float64
    # after those exact components have been constructed, avoiding cancellation
    # in the small endpoint gradients and Hessian diagonals.
    fixed64 = fixed.double()
    contributions64 = contributions.double()
    target64 = target.double()
    occ = torch.tensor(occupancies, dtype=torch.float64, device=fixed.device)

    def render_occ(values: torch.Tensor) -> torch.Tensor:
        return _normalize(
            fixed64 + (values[:, None] * contributions64).sum(dim=0)
        )

    def loss_occ(values: torch.Tensor) -> torch.Tensor:
        return (render_occ(values) - target64).square().mean()

    logits = occ.clamp_min(1e-12).log()

    def render_logit(values: torch.Tensor) -> torch.Tensor:
        return render_occ(torch.softmax(values, dim=0))

    def loss_logit(values: torch.Tensor) -> torch.Tensor:
        return (render_logit(values) - target64).square().mean()

    occ_var = occ.detach().requires_grad_(True)
    occ_loss = loss_occ(occ_var)
    occ_grad = torch.autograd.grad(occ_loss, occ_var)[0]
    occ_hessian = torch.autograd.functional.hessian(loss_occ, occ)
    logit_var = logits.detach().requires_grad_(True)
    logit_loss = loss_logit(logit_var)
    logit_grad = torch.autograd.grad(logit_loss, logit_var)[0]
    logit_hessian = torch.autograd.functional.hessian(loss_logit, logits)

    fisher_occ = []
    fisher_logit = []
    for index in range(len(occ)):
        direction = torch.zeros_like(occ)
        direction[index] = 1.0
        _, tangent_occ = torch.autograd.functional.jvp(
            render_occ, (occ,), (direction,)
        )
        _, tangent_logit = torch.autograd.functional.jvp(
            render_logit, (logits,), (direction,)
        )
        fisher_occ.append(2.0 * tangent_occ.square().mean())
        fisher_logit.append(2.0 * tangent_logit.square().mean())

    with torch.no_grad():
        production_loss = (
            _normalize(
                fixed + (
                    torch.tensor(
                        occupancies, dtype=torch.float32, device=fixed.device
                    )[:, None]
                    * contributions
                ).sum(dim=0)
            )
            - target
        ).square().mean()
    return {
        "density_loss_float32": float(production_loss.cpu()),
        "density_loss_float64": float(occ_loss.detach().cpu()),
        "grad_occ": occ_grad.detach().cpu().numpy(),
        "hessian_occ": occ_hessian.diag().detach().cpu().numpy(),
        "fisher_occ": torch.stack(fisher_occ).detach().cpu().numpy(),
        "grad_logit": logit_grad.detach().cpu().numpy(),
        "hessian_logit": logit_hessian.diag().detach().cpu().numpy(),
        "fisher_logit": torch.stack(fisher_logit).detach().cpu().numpy(),
    }


def summarize_derivatives(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "abs_grad_occ",
        "grad_occ_squared",
        "hessian_occ_diag",
        "abs_hessian_occ_diag",
        "fisher_occ_diag",
        "abs_grad_logit",
        "grad_logit_squared",
        "hessian_logit_diag",
        "abs_hessian_logit_diag",
        "fisher_logit_diag",
    )
    output = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[("ALL", str(row["population"]))].append(row)
        groups[(str(row["site"]), str(row["population"]))].append(row)
    for (site, population), selected in sorted(groups.items()):
        summary = {
            "site": site,
            "population": population,
            "active_conformers": len(selected),
        }
        for field in fields:
            for key, value in describe(
                [float(row[field]) for row in selected]
            ).items():
                summary[f"{field}_{key}"] = value
        output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    provenance = read_csv(args.composite_root / "site_rule_provenance.csv")
    if args.site:
        requested = set(args.site)
        provenance = [row for row in provenance if row["site"] in requested]
        missing = requested - {row["site"] for row in provenance}
        if missing:
            raise RuntimeError(f"sites absent from composite provenance: {sorted(missing)}")
    slot_rows: list[dict[str, object]] = []
    derivative_rows: list[dict[str, object]] = []
    start_rows: list[dict[str, object]] = []
    config_rows = []
    density_loss_deltas = []

    for site_index, provenance_row in enumerate(provenance):
        site = provenance_row["site"]
        replacement = provenance_row["replacement_site"].lower() == "true"
        shard = locate_shard(
            site, args.baseline_root, args.replacement_root, replacement
        )
        config = json.loads((shard / "run_config.json").read_text())
        selection = json.loads(Path(config["selection"]).read_text())
        record = next(item for item in selection["sites"] if item["key"] == site)
        starts = read_csv(shard / "synthetic" / f"{site}_starts.csv")
        target, component_builder, _coordinate_builder = site_density_components(
            record, config, shard, device
        )
        config_rows.append(
            {
                "site": site,
                "shard": str(shard),
                "run_config": str(shard / "run_config.json"),
                "K": config["K"],
                "active_mask": config["nontrivial_occupancy"],
                "target": ";".join(config["target"]),
                "lambda_vdw": config["lambda_vdw"],
                "lambda_rot": config["lambda_rot"],
                "lambda_clash": config["lambda_clash"],
                "diversity_keys": ";".join(
                    sorted(key for key in config if "divers" in key.lower())
                ),
                "starts_columns_with_diversity": ";".join(
                    sorted(
                        key
                        for key in starts[0]
                        if "divers" in key.lower()
                        or "dissim" in key.lower()
                        or "repuls" in key.lower()
                    )
                ),
            }
        )
        for row in starts:
            start = int(row["start"])
            occupancies = parse_vector(row["occupancies"])
            chis = torch.tensor(
                parse_chi(row["final_chi_radians"]),
                dtype=torch.float32,
                device=device,
            )
            assignments = row["assignments"].split(";")
            rmsd_a = parse_vector(row["rmsd_to_A"])
            rmsd_b = parse_vector(row["rmsd_to_B"])
            pair = select_pair(occupancies, assignments, rmsd_a, rmsd_b)
            selected = set(pair.values()) if pair else set()
            fixed, contributions = component_builder(chis)
            derivatives = density_derivatives(
                fixed, contributions, target, occupancies
            )
            density_loss_delta = (
                float(derivatives["density_loss_float32"])
                - float(row["final_density_loss"])
            )
            density_loss_deltas.append(density_loss_delta)
            active = occupancies > float(config["nontrivial_occupancy"])
            submask_mass = float(occupancies[~active].sum())
            matched_mass = float(occupancies[list(selected)].sum()) if selected else 0.0
            target_sum = float(row["target_A_occupancy"]) + float(
                row["target_B_occupancy"]
            )
            extra_active_mass = float(
                sum(
                    occupancies[index]
                    for index in range(len(occupancies))
                    if active[index] and index not in selected
                )
            )
            start_rows.append(
                {
                    "site": site,
                    "start": start,
                    "pair_complete": pair is not None,
                    "selected_A_slot": pair["A"] if pair else -1,
                    "selected_B_slot": pair["B"] if pair else -1,
                    "matched_occupancy_sum": matched_mass,
                    "target_A_occupancy": float(row["target_A_occupancy"]),
                    "target_B_occupancy": float(row["target_B_occupancy"]),
                    "target_AB_occupancy_sum": target_sum,
                    "matched_minus_target_AB_sum": matched_mass - target_sum,
                    "submask_occupancy_mass": submask_mass,
                    "extra_active_occupancy_mass": extra_active_mass,
                    "all_slot_occupancy_sum": float(occupancies.sum()),
                    "active_slots": int(active.sum()),
                    "background_or_absent_slot_exists": False,
                    "density_loss_saved": float(row["final_density_loss"]),
                    "density_loss_recomputed_float32": derivatives[
                        "density_loss_float32"
                    ],
                    "density_loss_recompute_delta": density_loss_delta,
                }
            )
            for index in range(len(occupancies)):
                if index in selected:
                    population = "matched"
                    role = (
                        "matched_A" if pair and index == pair["A"] else "matched_B"
                    )
                elif active[index]:
                    population = "extra"
                    role = "extra_active"
                else:
                    population = "inactive"
                    role = "below_active_mask"
                slot = {
                    "site": site,
                    "start": start,
                    "slot": index,
                    "occupancy": float(occupancies[index]),
                    "active": bool(active[index]),
                    "below_active_mask": not bool(active[index]),
                    "assignment": assignments[index],
                    "rmsd_to_A": float(rmsd_a[index]),
                    "rmsd_to_B": float(rmsd_b[index]),
                    "selected_assigned_pair": index in selected,
                    "population": population,
                    "slot_role": role,
                    "submask_occupancy_mass": submask_mass,
                    "matched_occupancy_sum": matched_mass,
                    "target_AB_occupancy_sum": target_sum,
                    "matched_minus_target_AB_sum": matched_mass - target_sum,
                    "background_or_absent_slot_exists": False,
                }
                slot_rows.append(slot)
                if active[index]:
                    grad_occ = float(derivatives["grad_occ"][index])
                    hessian_occ = float(derivatives["hessian_occ"][index])
                    grad_logit = float(derivatives["grad_logit"][index])
                    hessian_logit = float(derivatives["hessian_logit"][index])
                    derivative_rows.append(
                        {
                            **slot,
                            "density_loss": derivatives[
                                "density_loss_float32"
                            ],
                            "grad_occ": grad_occ,
                            "abs_grad_occ": abs(grad_occ),
                            "grad_occ_squared": grad_occ**2,
                            "hessian_occ_diag": hessian_occ,
                            "abs_hessian_occ_diag": abs(hessian_occ),
                            "fisher_occ_diag": float(
                                derivatives["fisher_occ"][index]
                            ),
                            "grad_logit": grad_logit,
                            "abs_grad_logit": abs(grad_logit),
                            "grad_logit_squared": grad_logit**2,
                            "hessian_logit_diag": hessian_logit,
                            "abs_hessian_logit_diag": abs(hessian_logit),
                            "fisher_logit_diag": float(
                                derivatives["fisher_logit"][index]
                            ),
                        }
                    )
        print(
            json.dumps(
                {
                    "site": site,
                    "completed_sites": site_index + 1,
                    "starts": len(starts),
                }
            ),
            flush=True,
        )

    derivative_summary = summarize_derivatives(derivative_rows)
    paired_starts = [row for row in start_rows if row["pair_complete"]]
    two_vfp = [row for row in start_rows if row["site"] == "2VFP_A_TYR417"]
    source_path = Path(__file__).resolve().parents[1] / (
        "density_denoiser/five_site_optimizer.py"
    )
    summary = {
        "kind": "saved-endpoint-only diagnostic; no optimization or production changes",
        "device": str(device),
        "sites": len(provenance),
        "starts": len(start_rows),
        "slots": len(slot_rows),
        "active_conformers": len(derivative_rows),
        "matched_conformers": sum(
            row["population"] == "matched" for row in derivative_rows
        ),
        "extra_active_conformers": sum(
            row["population"] == "extra" for row in derivative_rows
        ),
        "active_mask_strictly_greater_than": 0.05,
        "derivative_definition": {
            "post_softmax_occupancy": (
                "ambient partial derivative holding the other three occupancies "
                "fixed; this direction is off the simplex"
            ),
            "logit": "exact derivative through the K=4 softmax Jacobian",
            "hessian": "exact autograd Hessian diagonal with endpoint coordinates fixed",
            "fisher": (
                "positive Gauss-Newton/Fisher diagonal 2*mean((d rendered/d parameter)^2)"
            ),
        },
        "density_loss_reproduction_delta": describe(density_loss_deltas),
        "paired_start_matched_minus_target_AB_sum": describe(
            [
                float(row["matched_minus_target_AB_sum"])
                for row in paired_starts
            ]
        ),
        "all_start_submask_occupancy_mass": describe(
            [float(row["submask_occupancy_mass"]) for row in start_rows]
        ),
        "two_vfp": {
            "starts": len(two_vfp),
            "paired_starts": sum(row["pair_complete"] for row in two_vfp),
            "submask_occupancy_mass": describe(
                [float(row["submask_occupancy_mass"]) for row in two_vfp]
            ),
            "matched_occupancy_sum_paired": describe(
                [
                    float(row["matched_occupancy_sum"])
                    for row in two_vfp
                    if row["pair_complete"]
                ]
            ),
            "matched_minus_target_AB_sum_paired": describe(
                [
                    float(row["matched_minus_target_AB_sum"])
                    for row in two_vfp
                    if row["pair_complete"]
                ]
            ),
        },
        "diversity_provenance": {
            "present_in_frozen_objective": False,
            "weight": 0.0,
            "functional_form": "none",
            "conformer_pairs": [],
            "recorded_endpoint_value": "no column; term absent",
            "gradient_to_extra_slot_occupancy": 0.0,
            "gradient_to_extra_slot_position": 0.0,
            "verified_config_diversity_keys": sorted(
                {
                    row["diversity_keys"]
                    for row in config_rows
                    if row["diversity_keys"]
                }
            ),
            "verified_endpoint_diversity_columns": sorted(
                {
                    row["starts_columns_with_diversity"]
                    for row in config_rows
                    if row["starts_columns_with_diversity"]
                }
            ),
        },
        "conformer_count_selection": {
            "fixed_K": 4,
            "candidate_slots_are_all_conformers": True,
            "background_or_absent_candidate_slot": False,
            "complexity_or_sparsity_penalty": False,
            "reported_active_mask": "occupancy > 0.05",
            "physics_occupancy_use": (
                "hard occupancy > 0.05 inclusion; every included slot is charged "
                "equally, with no smooth occupancy scaling"
            ),
        },
        "five_site_optimizer_sha256_current_local_source": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "frozen_five_site_optimizer_sha256": sorted(
            {row["five_site_optimizer_sha256"] for row in provenance}
        ),
    }

    args.output.mkdir(parents=True)
    atomic_csv(args.output / "k4_slot_occupancies.csv", slot_rows)
    atomic_csv(args.output / "k4_start_occupancy_summary.csv", start_rows)
    atomic_csv(args.output / "active_conformer_density_derivatives.csv", derivative_rows)
    atomic_csv(args.output / "density_derivative_summary.csv", derivative_summary)
    atomic_csv(args.output / "source_config_provenance.csv", config_rows)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
