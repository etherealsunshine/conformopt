"""Read-only residual-versus-canonical-rotamer diagnostic for frozen v3.

The diagnostic reconstructs every saved K=4 endpoint, renders target and
endpoint densities in the optimizer's native additive density space, and asks
whether any member of the exact production canonical-state pool can explain
the remaining residual.  It never invokes an optimizer or changes an endpoint.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import gemmi
import numpy as np
import torch

from density_denoiser.data_pipeline import _grid_coordinates
from density_denoiser.five_site_optimizer import (
    _unique_canonical_centers_radians,
)
from density_denoiser.residue_geometry import CHI_SPECS, symmetry_aware_rmsd
from experiments.probe4.core import dihedral, torsion_to_coords, wrap_angles
from scripts.diagnose_frozen_v3_occupancy_pooling import (
    ACTIVE_THRESHOLD,
    FOUND_THRESHOLD,
    atomic_csv,
    atomic_json,
    conventional_rmsd,
    describe,
    identify_single_recovery,
    load_optimizer_rows,
    load_v3_payload,
    parse_vector,
    raw_slot_rows,
)


TAIL_SITES = {"1ZV8_E_ASN1", "2VFP_A_TYR417", "5Z8H_A_MET730",
              "7UO8_A_GLN53", "4C16_A_MET258"}
RMSD_CUTOFF = 1.0


def parse_matrix(value: str) -> np.ndarray:
    return np.asarray(
        [[float(item) for item in row.split(";")] for row in value.split("|")],
        dtype=np.float64,
    )


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = np.asarray(left, dtype=np.float64) - float(np.mean(left))
    right_centered = np.asarray(right, dtype=np.float64) - float(np.mean(right))
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    return (
        float(np.dot(left_centered, right_centered) / denominator)
        if denominator > 1e-15 else 0.0
    )


def scalar_density_fit(
    residual: np.ndarray, density: np.ndarray
) -> dict[str, float]:
    """Fit one non-negative occupancy in raw additive density space."""
    residual = np.asarray(residual, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    density_norm2 = float(np.dot(density, density))
    residual_norm2 = float(np.dot(residual, residual))
    unconstrained = (
        float(np.dot(density, residual) / density_norm2)
        if density_norm2 > 1e-15 else 0.0
    )
    occupancy = max(unconstrained, 0.0)
    fitted_residual = residual - occupancy * density
    fitted_sse = float(np.dot(fitted_residual, fitted_residual))
    denominator = float(np.linalg.norm(density) * np.linalg.norm(residual))
    return {
        "pearson_correlation": pearson_correlation(density, residual),
        "cosine_similarity": (
            float(np.dot(density, residual) / denominator)
            if denominator > 1e-15 else 0.0
        ),
        "unconstrained_fitted_occupancy": unconstrained,
        "fitted_occupancy_nnls": occupancy,
        "baseline_sse": residual_norm2,
        "fitted_sse": fitted_sse,
        "sse_reduction": residual_norm2 - fitted_sse,
        "fraction_sse_reduced": (
            (residual_norm2 - fitted_sse) / residual_norm2
            if residual_norm2 > 1e-15 else 0.0
        ),
    }


def select_best_fit(fits: list[dict[str, float]]) -> int:
    """Select the density producing the largest non-negative LS improvement."""
    if not fits:
        raise ValueError("cannot select from an empty fit collection")
    return max(
        range(len(fits)),
        key=lambda index: (
            fits[index]["sse_reduction"],
            fits[index]["pearson_correlation"],
            -index,
        ),
    )


def canonical_physical_states(
    residue_type: str, n_chi: int
) -> list[tuple[float, ...]]:
    centers = [
        _unique_canonical_centers_radians(residue_type, chi_index)
        for chi_index in range(n_chi)
    ]
    return [
        tuple(float(value) for value in state)
        for state in itertools.product(*centers)
    ]


def build_site_geometry(
    site: dict[str, object], target_path: Path
) -> dict[str, object]:
    audit_root = Path(str(site["_audit_root"]))
    maps: dict[str, dict[str, torch.Tensor]] = {}
    residue_b = None
    for state, field in (("A", "base_pdb_A"), ("B", "base_pdb_B")):
        structure = gemmi.read_structure(str(audit_root / str(site[field])))
        residue = next(
            residue
            for chain in structure[0] if chain.name == site["chain"]
            for residue in chain
            if residue.seqid.num == int(site["residue_number"])
        )
        if state == "B":
            residue_b = residue
        maps[state] = {
            atom.name.strip(): torch.tensor(
                atom.pos.tolist(), dtype=torch.float32
            )
            for atom in residue if atom.element.name != "H"
        }
    assert residue_b is not None

    names = list(site["atom_names"])
    residue_type = str(site["residue_type"])
    spec = CHI_SPECS[residue_type]
    map_a, map_b = maps["A"], maps["B"]
    template = torch.stack([map_a[name] for name in names])
    fixed_lookup = {
        name: value for name, value in map_a.items() if name not in names
    }

    def from_delta(delta: torch.Tensor) -> torch.Tensor:
        return torsion_to_coords(
            template,
            names,
            wrap_angles(delta),
            list(spec["rotations"]),
            fixed_lookup,
        )

    def physical_chi(candidate: torch.Tensor) -> torch.Tensor:
        lookup = dict(fixed_lookup)
        lookup.update({
            name: candidate[index] for index, name in enumerate(names)
        })
        return torch.stack([
            wrap_angles(
                dihedral(*(lookup[name] for name in quartet)) - torch.pi
            )
            for quartet in spec["dihedrals"]
        ])

    zero = torch.zeros(len(spec["rotations"]), dtype=torch.float32)
    deposited_a = from_delta(zero)
    base_physical = physical_chi(deposited_a).detach()
    direction = []
    for chi_index in range(len(spec["rotations"])):
        probe = zero.clone()
        probe[chi_index] = 0.01
        moved = physical_chi(from_delta(probe)).detach()
        sign = float(torch.sign(wrap_angles(
            moved[chi_index] - base_physical[chi_index]
        )))
        direction.append(sign or 1.0)
    delta_direction = torch.tensor(direction, dtype=torch.float32)

    def from_physical(desired: tuple[float, ...]) -> torch.Tensor:
        desired_tensor = torch.tensor(desired, dtype=torch.float32)
        delta = delta_direction * wrap_angles(desired_tensor - base_physical)
        return from_delta(delta)

    chi_a = torch.stack([
        dihedral(*(map_a[name] for name in quartet))
        for quartet in spec["dihedrals"]
    ])
    chi_b = torch.stack([
        dihedral(*(map_b[name] for name in quartet))
        for quartet in spec["dihedrals"]
    ])
    true_delta = wrap_angles(chi_b - chi_a)
    deposited_b_reference = torch.stack([map_b[name] for name in names])
    plus, minus = from_delta(true_delta), from_delta(-true_delta)
    if symmetry_aware_rmsd(
        minus, deposited_b_reference, names, residue_type
    ) < symmetry_aware_rmsd(
        plus, deposited_b_reference, names, residue_type
    ):
        true_delta = -true_delta
    deposited_b = from_delta(true_delta)

    target_npz = np.load(target_path, allow_pickle=False)
    metadata = json.loads(str(target_npz["metadata"].item()))
    mask = np.asarray(target_npz["radial_mask"], dtype=bool)
    center = np.asarray(metadata["center"], dtype=np.float32)
    grid = _grid_coordinates(center, 32, 0.5, None)[mask].reshape(-1, 3)
    atom_lookup = {atom.name.strip(): atom for atom in residue_b}
    sigma2 = np.asarray([
        max(float(atom_lookup[name].b_iso) / (8.0 * math.pi**2), 0.04)
        for name in names
    ], dtype=np.float64)
    weights = np.asarray([
        float(atom_lookup[name].element.atomic_number) for name in names
    ], dtype=np.float64)
    normalization = np.power(2.0 * math.pi * sigma2, -1.5)

    def render(xyz: np.ndarray | torch.Tensor) -> np.ndarray:
        array = (
            xyz.detach().cpu().numpy()
            if isinstance(xyz, torch.Tensor) else np.asarray(xyz)
        )
        distance2 = np.square(
            grid[:, None, :] - array.astype(np.float64)[None, :, :]
        ).sum(axis=-1)
        return (
            weights[None, :]
            * normalization[None, :]
            * np.exp(-distance2 / (2.0 * sigma2[None, :]))
        ).sum(axis=1)

    states = canonical_physical_states(residue_type, len(spec["rotations"]))
    rotamer_coordinates = [
        from_physical(state).detach().cpu().numpy().astype(np.float64)
        for state in states
    ]
    return {
        "names": names,
        "residue_type": residue_type,
        "from_delta": from_delta,
        "deposited_A": deposited_a.detach().cpu().numpy().astype(np.float64),
        "deposited_B": deposited_b.detach().cpu().numpy().astype(np.float64),
        "render": render,
        "canonical_states": states,
        "rotamer_coordinates": rotamer_coordinates,
        "rotamer_densities": np.asarray([
            render(xyz) for xyz in rotamer_coordinates
        ]),
        "mask_voxels": int(mask.sum()),
    }


def fraction(rows: list[dict[str, object]], field: str) -> float:
    return (
        sum(bool(row[field]) for row in rows) / len(rows)
        if rows else 0.0
    )


def summarize_group(rows: list[dict[str, object]]) -> dict[str, object]:
    actionable = [
        row for row in rows
        if bool(row["best_rotamer_within_1A_of_missed"])
        and float(row["best_rotamer_pearson_correlation"]) >= 0.2
        and float(row["best_rotamer_fitted_occupancy_nnls"]) >= 0.05
    ]
    return {
        "starts": len(rows),
        "deposited_missed_ceiling_pearson": describe(
            float(row["deposited_missed_pearson_correlation"]) for row in rows
        ),
        "deposited_missed_ceiling_cosine": describe(
            float(row["deposited_missed_cosine_similarity"]) for row in rows
        ),
        "deposited_missed_ceiling_fitted_occupancy": describe(
            float(row["deposited_missed_fitted_occupancy_nnls"])
            for row in rows
        ),
        "best_rotamer_pearson": describe(
            float(row["best_rotamer_pearson_correlation"]) for row in rows
        ),
        "best_rotamer_fitted_occupancy": describe(
            float(row["best_rotamer_fitted_occupancy_nnls"]) for row in rows
        ),
        "best_rotamer_rmsd_to_missed_A": describe(
            float(row["best_rotamer_rmsd_to_missed_A"]) for row in rows
        ),
        "best_rotamer_within_1A_of_missed_count": sum(
            bool(row["best_rotamer_within_1A_of_missed"]) for row in rows
        ),
        "best_rotamer_within_1A_of_missed_fraction": fraction(
            rows, "best_rotamer_within_1A_of_missed"
        ),
        "actionable_canonical_fit_definition": (
            "best rotamer within 1.0 A of missed, Pearson r>=0.2, "
            "and fitted occupancy>=0.05"
        ),
        "actionable_canonical_fit_count": len(actionable),
        "actionable_canonical_fit_fraction": (
            len(actionable) / len(rows) if rows else 0.0
        ),
        "ceiling_pearson_at_least_0p2_fraction": (
            sum(
                float(row["deposited_missed_pearson_correlation"]) >= 0.2
                for row in rows
            ) / len(rows) if rows else 0.0
        ),
        "ceiling_pearson_at_least_0p5_fraction": (
            sum(
                float(row["deposited_missed_pearson_correlation"]) >= 0.5
                for row in rows
            ) / len(rows) if rows else 0.0
        ),
    }


def report_markdown(summary: dict[str, object]) -> str:
    overall = summary["cohorts"]["all_single_recovery"]
    ceiling = overall["deposited_missed_ceiling_pearson"]
    best = overall["best_rotamer_pearson"]
    lines = [
        "# Frozen-v3 raw-residual canonical-rotamer diagnostic",
        "",
        "## Deposited missed-conformer ceiling",
        "",
        (
            f"Across {overall['starts']} historical single-recovery starts, "
            f"the deposited missed conformer has median raw-residual Pearson "
            f"correlation **{ceiling['median']:.4f}** "
            f"(IQR {ceiling['q25']:.4f}–{ceiling['q75']:.4f})."
        ),
        "",
        (
            f"The best canonical rotamer has median Pearson correlation "
            f"{best['median']:.4f}; "
            f"{overall['best_rotamer_within_1A_of_missed_count']}/"
            f"{overall['starts']} are within 1.0 Å of the missed conformer."
        ),
        "",
        "All densities are native, pre-z-score additive densities on the saved "
        "Stage-1 radial mask. Best fits minimize raw-space SSE with a single "
        "non-negative occupancy; Pearson correlation is reported separately.",
        "",
        "## Per-site summary",
        "",
        "| site | starts | missed ceiling median r | best rotamer median r | "
        "best within 1 Å | low-q active slot available |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for site, record in summary["sites"].items():
        ceiling_median = record["deposited_missed_ceiling_pearson"].get(
            "median"
        )
        best_median = record["best_rotamer_pearson"].get("median")
        lines.append(
            f"| {site} | {record['starts']} | "
            f"{ceiling_median:.4f} | " if ceiling_median is not None else
            f"| {site} | {record['starts']} | — | "
        )
        lines[-1] += (
            f"{best_median:.4f} | " if best_median is not None else "— | "
        )
        lines[-1] += (
            f"{record['best_rotamer_within_1A_of_missed_count']} | "
            f"{record['starts_with_active_slot_below_0p10']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    metric_root = (
        args.frozen_root / "analysis"
        / "metric_v3_protected_merge_sweep" / "0p5"
    )
    optimizer_rows, target_paths = load_optimizer_rows(
        args.baseline_root, args.replacement_root
    )
    active_by_slot, coordinates, sites, _, _ = load_v3_payload(metric_root)
    slots_by_start = raw_slot_rows(optimizer_rows)
    single_by_site: dict[str, list[tuple[tuple[str, int], dict[str, object]]]] = (
        defaultdict(list)
    )
    counts: dict[str, int] = defaultdict(int)
    for key, row in sorted(optimizer_rows.items()):
        identity = identify_single_recovery(row)
        if identity is None:
            continue
        single_by_site[key[0]].append((key, identity))
        counts[str(identity["recovery_rank"])] += 1
    if counts != {"major_only": 142, "minor_only": 45}:
        raise RuntimeError(f"single-recovery guard failed: {dict(counts)}")

    per_start: list[dict[str, object]] = []
    per_rotamer: list[dict[str, object]] = []
    target_validation: dict[str, dict[str, float]] = {}
    site_order = sorted(
        single_by_site,
        key=lambda site: (site not in TAIL_SITES, site),
    )
    reconstruction_rmsds = []
    for site_name in site_order:
        site = sites[site_name]
        geometry = build_site_geometry(site, target_paths[site_name])
        names = list(geometry["names"])
        residue_type = str(geometry["residue_type"])
        deposited = {
            state: np.asarray(geometry[f"deposited_{state}"])
            for state in ("A", "B")
        }
        render = geometry["render"]
        site_optimizer = optimizer_rows[single_by_site[site_name][0][0]]
        target_a = float(site_optimizer["target_A_occupancy"])
        target_b = float(site_optimizer["target_B_occupancy"])
        target = (
            target_a * render(deposited["A"])
            + target_b * render(deposited["B"])
        )
        saved_target = np.asarray(np.load(
            target_paths[site_name].parent
            / f"{site_name}_optimizer_synthetic_vector.npy"
        ), dtype=np.float64)
        normalized_target = (
            target - target.mean()
        ) / max(target.std(ddof=1), 1e-6)
        validation_residual = normalized_target - saved_target
        target_validation[site_name] = {
            "relative_l2": float(
                np.linalg.norm(validation_residual)
                / max(np.linalg.norm(saved_target), 1e-15)
            ),
            "pearson_correlation": pearson_correlation(
                normalized_target, saved_target
            ),
            "max_absolute_error": float(
                np.max(np.abs(validation_residual))
            ),
        }
        rotamer_densities = np.asarray(geometry["rotamer_densities"])

        for key, identity in single_by_site[site_name]:
            optimizer = optimizer_rows[key]
            endpoint_chi = parse_matrix(optimizer["final_chi_radians"])
            occupancies = np.asarray(
                parse_vector(optimizer["occupancies"]), dtype=np.float64
            )
            if endpoint_chi.shape[0] != len(occupancies):
                raise ValueError(f"endpoint shape mismatch for {key}")
            endpoint_coordinates = [
                geometry["from_delta"](
                    torch.tensor(row, dtype=torch.float32)
                ).detach().cpu().numpy().astype(np.float64)
                for row in endpoint_chi
            ]
            rendered = sum(
                occupancy * render(xyz)
                for occupancy, xyz in zip(occupancies, endpoint_coordinates)
            )
            residual = target - rendered

            for slot, xyz in enumerate(endpoint_coordinates):
                saved_key = (site_name, key[1], slot)
                if saved_key in coordinates:
                    reconstruction_rmsds.append(conventional_rmsd(
                        xyz, coordinates[saved_key], names, residue_type
                    ))

            missed = str(identity["missed_state"])
            recovered = str(identity["recovered_state"])
            rotamer_fits = [
                scalar_density_fit(residual, density)
                for density in rotamer_densities
            ]
            best_index = select_best_fit(rotamer_fits)
            deposited_fit = scalar_density_fit(
                residual, render(deposited[missed])
            )
            best_xyz = geometry["rotamer_coordinates"][best_index]
            best_rmsd_missed = conventional_rmsd(
                best_xyz, deposited[missed], names, residue_type
            )
            best_rmsd_recovered = conventional_rmsd(
                best_xyz, deposited[recovered], names, residue_type
            )
            low_active = [
                slot for slot, occupancy in enumerate(occupancies)
                if ACTIVE_THRESHOLD < occupancy < FOUND_THRESHOLD
            ]
            start_record = {
                "site": site_name,
                "start": key[1],
                "residue_type": residue_type,
                "tail_site": site_name in TAIL_SITES,
                **identity,
                "canonical_rotamers_enumerated": len(rotamer_fits),
                "raw_residual_norm": float(np.linalg.norm(residual)),
                "raw_residual_sum": float(np.sum(residual)),
                "best_rotamer_index": best_index,
                "best_rotamer_physical_chi_degrees": ";".join(
                    f"{math.degrees(value):.9g}"
                    for value in geometry["canonical_states"][best_index]
                ),
                **{
                    f"best_rotamer_{field}": value
                    for field, value in rotamer_fits[best_index].items()
                },
                "best_rotamer_rmsd_to_missed_A": best_rmsd_missed,
                "best_rotamer_rmsd_to_recovered_A": best_rmsd_recovered,
                "best_rotamer_within_1A_of_missed": (
                    best_rmsd_missed < RMSD_CUTOFF
                ),
                **{
                    f"deposited_missed_{field}": value
                    for field, value in deposited_fit.items()
                },
                "active_slots_below_0p10": len(low_active),
                "active_slot_indices_below_0p10": ";".join(
                    str(slot) for slot in low_active
                ),
                "has_active_slot_below_0p10": bool(low_active),
            }
            per_start.append(start_record)
            for rotamer_index, (
                state, xyz, fit
            ) in enumerate(zip(
                geometry["canonical_states"],
                geometry["rotamer_coordinates"],
                rotamer_fits,
            )):
                per_rotamer.append({
                    "site": site_name,
                    "start": key[1],
                    "recovery_rank": identity["recovery_rank"],
                    "missed_state": missed,
                    "recovered_state": recovered,
                    "rotamer_index": rotamer_index,
                    "physical_chi_degrees": ";".join(
                        f"{math.degrees(value):.9g}" for value in state
                    ),
                    **fit,
                    "rmsd_to_missed_A": conventional_rmsd(
                        xyz, deposited[missed], names, residue_type
                    ),
                    "rmsd_to_recovered_A": conventional_rmsd(
                        xyz, deposited[recovered], names, residue_type
                    ),
                    "within_1A_of_missed": conventional_rmsd(
                        xyz, deposited[missed], names, residue_type
                    ) < RMSD_CUTOFF,
                    "selected_best_fit": rotamer_index == best_index,
                })

        # Per-site checkpoint, with the five tail sites processed first.
        atomic_csv(args.output / "per_start.csv", per_start)
        atomic_csv(args.output / "per_rotamer.csv", per_rotamer)
        atomic_json(args.output / "progress.json", {
            "completed_sites": site_order[:site_order.index(site_name) + 1],
            "starts_complete": len(per_start),
            "rotamer_rows_complete": len(per_rotamer),
        })

    if len(per_start) != 187:
        raise RuntimeError(f"expected 187 per-start rows, found {len(per_start)}")
    per_start.sort(key=lambda row: (str(row["site"]), int(row["start"])))
    per_rotamer.sort(key=lambda row: (
        str(row["site"]), int(row["start"]), int(row["rotamer_index"])
    ))
    atomic_csv(args.output / "per_start.csv", per_start)
    atomic_csv(args.output / "per_rotamer.csv", per_rotamer)

    summary: dict[str, object] = {
        "provenance": {
            "frozen_root": str(args.frozen_root),
            "baseline_root": str(args.baseline_root),
            "replacement_root": str(args.replacement_root),
            "metric": "qfit-synth20-merge050-one-to-one-tmol044-v3",
            "endpoint_source": "saved frozen-v3 optimizer CSV rows",
            "optimizer_runs": 0,
            "density_space": "native additive density before z-scoring",
            "mask": "saved Stage-1 radial mask",
            "rotamer_pool": (
                "Cartesian pool of physically unique residue-specific marginal "
                "canonical chi centers used by production"
            ),
            "fit": (
                "single-column raw-density least squares with occupancy q>=0; "
                "best minimizes SSE, Pearson reported separately"
            ),
        },
        "guards": {
            "major_only_missed_minor": counts["major_only"],
            "minor_only_missed_major": counts["minor_only"],
            "total_single_recovery": len(per_start),
            "maximum_active_coordinate_reconstruction_rmsd_A": (
                max(reconstruction_rmsds) if reconstruction_rmsds else None
            ),
            "maximum_target_reconstruction_relative_l2": max(
                row["relative_l2"] for row in target_validation.values()
            ),
            "minimum_target_reconstruction_pearson_correlation": min(
                row["pearson_correlation"]
                for row in target_validation.values()
            ),
            "maximum_target_reconstruction_absolute_error": max(
                row["max_absolute_error"]
                for row in target_validation.values()
            ),
        },
        "cohorts": {
            "all_single_recovery": summarize_group(per_start),
            "missed_minor": summarize_group([
                row for row in per_start if row["recovery_rank"] == "major_only"
            ]),
            "missed_major": summarize_group([
                row for row in per_start if row["recovery_rank"] == "minor_only"
            ]),
            "tail_sites": summarize_group([
                row for row in per_start if bool(row["tail_site"])
            ]),
        },
        "sites": {},
    }
    site_rows: list[dict[str, object]] = []
    all_site_names = sorted({key[0] for key in optimizer_rows})
    for site_name in all_site_names:
        rows = [row for row in per_start if row["site"] == site_name]
        record = summarize_group(rows)
        record.update({
            "tail_site": site_name in TAIL_SITES,
            "missed_minor_starts": sum(
                row["recovery_rank"] == "major_only" for row in rows
            ),
            "missed_major_starts": sum(
                row["recovery_rank"] == "minor_only" for row in rows
            ),
            "starts_with_active_slot_below_0p10": sum(
                bool(row["has_active_slot_below_0p10"]) for row in rows
            ),
            "active_slots_below_0p10_total": sum(
                int(row["active_slots_below_0p10"]) for row in rows
            ),
        })
        summary["sites"][site_name] = record
        flat_record = {
            "site": site_name,
            "tail_site": record["tail_site"],
            "starts": record["starts"],
            "missed_minor_starts": record["missed_minor_starts"],
            "missed_major_starts": record["missed_major_starts"],
            "deposited_missed_ceiling_pearson_median": (
                record["deposited_missed_ceiling_pearson"].get("median")
            ),
            "deposited_missed_ceiling_fitted_occupancy_median": (
                record["deposited_missed_ceiling_fitted_occupancy"].get(
                    "median"
                )
            ),
            "best_rotamer_pearson_median": (
                record["best_rotamer_pearson"].get("median")
            ),
            "best_rotamer_fitted_occupancy_median": (
                record["best_rotamer_fitted_occupancy"].get("median")
            ),
            "best_rotamer_rmsd_to_missed_median_A": (
                record["best_rotamer_rmsd_to_missed_A"].get("median")
            ),
            "best_rotamer_within_1A_of_missed_count": (
                record["best_rotamer_within_1A_of_missed_count"]
            ),
            "actionable_canonical_fit_count": (
                record["actionable_canonical_fit_count"]
            ),
            "starts_with_active_slot_below_0p10": (
                record["starts_with_active_slot_below_0p10"]
            ),
            "active_slots_below_0p10_total": (
                record["active_slots_below_0p10_total"]
            ),
        }
        site_rows.append(flat_record)
    atomic_csv(args.output / "per_site.csv", site_rows)
    atomic_json(args.output / "summary.json", summary)
    atomic_text(args.output / "report.md", report_markdown(summary))
    atomic_json(args.output / "progress.json", {
        "status": "complete",
        "completed_sites": site_order,
        "starts_complete": len(per_start),
        "rotamer_rows_complete": len(per_rotamer),
    })


if __name__ == "__main__":
    main()
