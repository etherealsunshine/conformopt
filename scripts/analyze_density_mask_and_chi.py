"""Read-only density-mask variance and deposited A/B torsion diagnostics."""

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
import torch

from density_denoiser.data_pipeline import _grid_coordinates, _sidechain_atoms
from density_denoiser.five_site_optimizer import _alt_atom_map
from density_denoiser.residue_geometry import CHI_SPECS
from experiments.probe4.core import dihedral, torsion_to_coords, wrap_angles


DIFFERENCE_FRACTION = 0.10
ATOM_UNION_RADIUS_A = 1.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def parse_matrix(value: str) -> np.ndarray:
    return np.asarray(
        [
            [float(item) for item in row.split(";")]
            for row in value.split("|")
        ],
        dtype=np.float64,
    )


def parse_vector(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split(";")], dtype=float)


def wrap_degrees(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    return float(np.dot(left_centered, right_centered) / denominator)


def relative_density_separation(
    target: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    target_z = (target - target.mean()) / max(target.std(ddof=1), 1e-6)
    candidate_z = (
        candidate - candidate.mean()
    ) / max(candidate.std(ddof=1), 1e-6)
    zscore_mse = float(np.mean((candidate_z - target_z) ** 2))
    zscore_scale = float(np.mean(target_z**2))
    raw_mse = float(np.mean((candidate - target) ** 2))
    raw_scale = float(np.mean(target**2))
    denominator = float(np.dot(candidate, candidate))
    fitted_scale = (
        float(np.dot(candidate, target)) / denominator
        if denominator > 0 else 0.0
    )
    scaled_mse = float(np.mean((fitted_scale * candidate - target) ** 2))
    return {
        "zscore_mse": zscore_mse,
        "zscore_loss_scale": zscore_scale,
        "zscore_relative_separation": zscore_mse / zscore_scale,
        "raw_mse": raw_mse,
        "raw_loss_scale": raw_scale,
        "raw_relative_separation": raw_mse / raw_scale,
        "fitted_global_scale": fitted_scale,
        "scale_fitted_raw_mse": scaled_mse,
        "scale_fitted_raw_loss_scale": raw_scale,
        "scale_fitted_raw_relative_separation": scaled_mse / raw_scale,
    }


def variance_decomposition(
    fixed: np.ndarray,
    sidechain: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    fixed_selected = fixed[mask]
    sidechain_selected = sidechain[mask]
    total_selected = fixed_selected + sidechain_selected
    fixed_var = float(np.var(fixed_selected))
    sidechain_var = float(np.var(sidechain_selected))
    covariance = float(np.mean(
        (fixed_selected - fixed_selected.mean())
        * (sidechain_selected - sidechain_selected.mean())
    ))
    cross_term = 2.0 * covariance
    total_var = float(np.var(total_selected))
    reconstructed = fixed_var + sidechain_var + cross_term
    if not math.isclose(total_var, reconstructed, rel_tol=1e-7, abs_tol=1e-8):
        raise RuntimeError(
            f"variance decomposition failed: {total_var} != {reconstructed}"
        )
    return {
        "n_voxels": int(mask.sum()),
        "fixed_variance": fixed_var,
        "sidechain_variance": sidechain_var,
        "cross_term_2cov": cross_term,
        "total_variance": total_var,
        "sidechain_pure_fraction": sidechain_var / total_var,
        "sidechain_attributable_fraction": (
            sidechain_var + cross_term
        ) / total_var,
        "fixed_fraction": fixed_var / total_var,
    }


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    selections: dict[str, dict] = {}
    for path in args.selection:
        for record in json.loads(path.read_text())["sites"]:
            selections[record["key"]] = record
    provenance = {
        row["site"]: row for row in read_csv(args.source_provenance)
    }
    probe_fixed = [
        row for row in read_csv(args.probe)
        if row["occupancy_mode"] == "fixed_minor"
    ]
    probe_by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in probe_fixed:
        probe_by_site[row["site"]].append(row)

    device = torch.device("cpu")
    variance_rows: list[dict[str, object]] = []
    chi_rows: list[dict[str, object]] = []
    endpoint_rows: list[dict[str, object]] = []

    for site, record in sorted(selections.items()):
        source = provenance[site]
        shard = Path(source["shard"])
        config = json.loads(Path(source["run_config"]).read_text())
        structure = gemmi.read_structure(record["pdb_path"])
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
            atom for atom in residue
            if atom.altloc == "B"
            and atom.element.name != "H"
            and atom.name.strip() not in {"N", "CA", "C", "O"}
        ]
        names = [atom.name.strip() for atom in b_atoms]
        if not names or any(name not in map_a or name not in map_b for name in names):
            raise RuntimeError(f"incomplete A/B sidechain atoms at {site}")

        spec = CHI_SPECS[residue.name]
        chi_a = torch.stack([
            dihedral(*(map_a[name] for name in quartet))
            for quartet in spec["dihedrals"]
        ])
        chi_b = torch.stack([
            dihedral(*(map_b[name] for name in quartet))
            for quartet in spec["dihedrals"]
        ])
        chi_a_deg = np.degrees(chi_a.cpu().numpy())
        chi_b_deg = np.degrees(chi_b.cpu().numpy())
        signed_delta = np.asarray([
            wrap_degrees(float(right - left))
            for left, right in zip(chi_a_deg, chi_b_deg)
        ])
        absolute_delta = np.abs(signed_delta)
        largest_index = int(np.argmax(absolute_delta))

        raw_occ_b = np.asarray([atom.occ for atom in b_atoms], dtype=np.float32)
        occ_b = float(np.median(raw_occ_b))
        a_atoms = [
            atom for atom in residue
            if atom.altloc == "A" and atom.name.strip() in names
        ]
        occ_a = float(np.median([atom.occ for atom in a_atoms]))
        occ_total = max(occ_a + occ_b, 1e-6)
        relative_a = occ_a / occ_total
        relative_b = occ_b / occ_total

        pair = np.load(record["pair_path"], allow_pickle=False)
        metadata = json.loads(str(pair["metadata"].item()))
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
        radial_mask = (
            np.linalg.norm(coordinates - center, axis=-1)
            <= float(config["grid_radius"])
        )
        selected_grid = torch.tensor(
            coordinates[radial_mask], dtype=torch.float32, device=device
        )

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
            atom for atom in _sidechain_atoms(residue)
            if atom.altloc in ("\x00", " ", "") and atom.name.strip() not in names
        ]
        shared_xyz = torch.tensor(
            [atom.pos.tolist() for atom in shared_atoms],
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 3)
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

        xyz_a = torch.stack([map_a[name] for name in names])
        xyz_b = torch.stack([map_b[name] for name in names])
        fixed = atom_density(
            shared_xyz, shared_sigma2, shared_weights
        ).cpu().numpy().astype(np.float64)
        density_a = atom_density(
            xyz_a, variable_sigma2, variable_weights
        ).cpu().numpy().astype(np.float64)
        density_b = atom_density(
            xyz_b, variable_sigma2, variable_weights
        ).cpu().numpy().astype(np.float64)
        sidechain = relative_a * density_a + relative_b * density_b
        target_raw = fixed + sidechain

        difference = np.abs(density_a - density_b)
        difference_mask = difference > DIFFERENCE_FRACTION * difference.max()
        atom_union_xyz = torch.cat([xyz_a, xyz_b, shared_xyz], dim=0)
        union_mask = (
            torch.cdist(selected_grid, atom_union_xyz).min(dim=1).values
            <= ATOM_UNION_RADIUS_A
        ).cpu().numpy()
        moving_union_mask = (
            torch.cdist(
                selected_grid, torch.cat([xyz_a, xyz_b], dim=0)
            ).min(dim=1).values
            <= ATOM_UNION_RADIUS_A
        ).cpu().numpy()

        masks = {
            "full": np.ones(len(fixed), dtype=bool),
            "difference": difference_mask,
            "atom_union": union_mask,
            "moving_atom_union": moving_union_mask,
        }
        decompositions = {
            "full": variance_decomposition(
                fixed, sidechain, masks["full"]
            ),
            "difference": variance_decomposition(
                fixed, sidechain, masks["difference"]
            ),
            "atom_union": variance_decomposition(
                fixed, sidechain, masks["atom_union"]
            ),
            "moving_atom_union": variance_decomposition(
                fixed, sidechain, masks["moving_atom_union"]
            ),
        }
        major = "A" if occ_a >= occ_b else "B"
        major_density = density_a if major == "A" else density_b
        collapsed_raw = fixed + major_density
        collapsed_correlation = correlation(target_raw, collapsed_raw)
        deposited_separation = relative_density_separation(
            target_raw, collapsed_raw
        )
        variance_row: dict[str, object] = {
            "site": site,
            "residue_type": residue.name,
            "grid_radius_A": float(config["grid_radius"]),
            "spacing_A": float(config["spacing"]),
            "fixed_atom_count": len(shared_atoms),
            "moving_atom_count": len(names),
            "difference_fraction_of_max": DIFFERENCE_FRACTION,
            "atom_union_radius_A": ATOM_UNION_RADIUS_A,
            "deposited_major": major,
            "deposited_major_collapsed_correlation": collapsed_correlation,
            "correlation_drop_correct_to_major_collapsed": (
                1.0 - collapsed_correlation
            ),
            **{
                f"deposited_major_collapse_{key}": value
                for key, value in deposited_separation.items()
            },
        }
        for mask_name, values in decompositions.items():
            for key, value in values.items():
                variance_row[f"{mask_name}_{key}"] = value
            selected = masks[mask_name]
            mask_correlation = correlation(
                target_raw[selected], collapsed_raw[selected]
            )
            variance_row[
                f"{mask_name}_major_collapsed_correlation"
            ] = mask_correlation
            variance_row[
                f"{mask_name}_correlation_drop_correct_to_major_collapsed"
            ] = 1.0 - mask_correlation
        variance_rows.append(variance_row)

        eligible = probe_by_site.get(site, [])
        chi_row: dict[str, object] = {
            "site": site,
            "residue_type": residue.name,
            "n_chi": len(absolute_delta),
            "chi_A_degrees": ";".join(f"{value:.6f}" for value in chi_a_deg),
            "chi_B_degrees": ";".join(f"{value:.6f}" for value in chi_b_deg),
            "signed_A_to_B_delta_degrees": ";".join(
                f"{value:.6f}" for value in signed_delta
            ),
            "absolute_A_to_B_delta_degrees": ";".join(
                f"{value:.6f}" for value in absolute_delta
            ),
            "largest_difference_chi_index_1based": largest_index + 1,
            "largest_difference_degrees": float(absolute_delta[largest_index]),
            "largest_difference_is_terminal": (
                largest_index == len(absolute_delta) - 1
            ),
            "probe_eligible": len(eligible),
            "probe_recovered": sum(
                row["recovered_minor_lt_1A"].lower() == "true"
                for row in eligible
            ),
        }
        chi_row["probe_recovery_rate"] = (
            chi_row["probe_recovered"] / chi_row["probe_eligible"]
            if chi_row["probe_eligible"] else ""
        )
        chi_rows.append(chi_row)

        if eligible:
            starts_path = (
                shard / "synthetic" / f"{site}_starts.csv"
            )
            starts = {
                int(row["start"]): row for row in read_csv(starts_path)
            }
            template = xyz_a
            fixed_lookup = {
                name: value for name, value in map_a.items() if name not in names
            }

            def coordinates_from_chi(values: torch.Tensor) -> torch.Tensor:
                return torsion_to_coords(
                    template,
                    names,
                    wrap_angles(values),
                    list(spec["rotations"]),
                    fixed_lookup,
                )

            for probe_row in eligible:
                start = int(probe_row["start"])
                endpoint = starts[start]
                chis = torch.tensor(
                    parse_matrix(endpoint["final_chi_radians"]),
                    dtype=torch.float32,
                    device=device,
                )
                occupancies = parse_vector(endpoint["occupancies"])
                endpoint_sidechain = np.zeros_like(fixed)
                for occupancy, values in zip(occupancies, chis):
                    endpoint_sidechain += float(occupancy) * atom_density(
                        coordinates_from_chi(values),
                        variable_sigma2,
                        variable_weights,
                    ).cpu().numpy()
                endpoint_correlation = correlation(
                    target_raw, fixed + endpoint_sidechain
                )
                endpoint_separation = relative_density_separation(
                    target_raw, fixed + endpoint_sidechain
                )
                endpoint_rows.append({
                    "site": site,
                    "start": start,
                    "minor_state": probe_row["minor_state"],
                    "major_state": probe_row["major_state"],
                    "endpoint_correlation": endpoint_correlation,
                    "correct_AB_correlation": 1.0,
                    "correlation_drop_to_correct": 1.0 - endpoint_correlation,
                    "deposited_major_collapsed_correlation": (
                        collapsed_correlation
                    ),
                    **endpoint_separation,
                })

    terminal = [
        row for row in chi_rows
        if row["probe_eligible"] and row["largest_difference_is_terminal"]
    ]
    nonterminal = [
        row for row in chi_rows
        if row["probe_eligible"] and not row["largest_difference_is_terminal"]
    ]

    endpoint_summary_rows = []
    for site in sorted({row["site"] for row in endpoint_rows}):
        selected = [row for row in endpoint_rows if row["site"] == site]
        endpoint_summary_rows.append({
            "site": site,
            "n_major_only_endpoints": len(selected),
            "endpoint_correlation_median": median([
                row["endpoint_correlation"] for row in selected
            ]),
            "endpoint_correlation_min": min(
                row["endpoint_correlation"] for row in selected
            ),
            "endpoint_correlation_max": max(
                row["endpoint_correlation"] for row in selected
            ),
            "correlation_drop_to_correct_median": median([
                row["correlation_drop_to_correct"] for row in selected
            ]),
        })

    args.output.mkdir(parents=True)
    atomic_csv(args.output / "mask_variance_by_site.csv", variance_rows)
    atomic_csv(args.output / "deposited_chi_differences.csv", chi_rows)
    atomic_csv(args.output / "major_only_endpoint_correlations.csv", endpoint_rows)
    atomic_csv(
        args.output / "major_only_endpoint_correlation_by_site.csv",
        endpoint_summary_rows,
    )
    atomic_json(args.output / "summary.json", {
        "diagnostic_only": True,
        "production_changed": False,
        "metric_changed": False,
        "grid_radius_A": sorted(set(
            float(row["grid_radius_A"]) for row in variance_rows
        )),
        "difference_mask": (
            f"abs(rho_A-rho_B) > {DIFFERENCE_FRACTION} * max"
        ),
        "atom_union_mask": (
            f"within {ATOM_UNION_RADIUS_A} A of deposited A/B sidechain atoms"
        ),
        "full_mask_sidechain_attributable_fraction": describe([
            float(row["full_sidechain_attributable_fraction"])
            for row in variance_rows
        ]),
        "difference_mask_sidechain_attributable_fraction": describe([
            float(row["difference_sidechain_attributable_fraction"])
            for row in variance_rows
        ]),
        "atom_union_sidechain_attributable_fraction": describe([
            float(row["atom_union_sidechain_attributable_fraction"])
            for row in variance_rows
        ]),
        "correct_to_major_collapsed_correlation_drop": describe([
            float(row["correlation_drop_correct_to_major_collapsed"])
            for row in variance_rows
        ]),
        "difference_mask_correct_to_major_collapsed_correlation_drop": describe([
            float(
                row[
                    "difference_"
                    "correlation_drop_correct_to_major_collapsed"
                ]
            )
            for row in variance_rows
        ]),
        "atom_union_correct_to_major_collapsed_correlation_drop": describe([
            float(
                row[
                    "atom_union_"
                    "correlation_drop_correct_to_major_collapsed"
                ]
            )
            for row in variance_rows
        ]),
        "major_only_endpoint_correlation_drop": describe([
            float(row["correlation_drop_to_correct"]) for row in endpoint_rows
        ]),
        "deposited_major_collapse_relative_separation": {
            metric: describe([
                float(row[f"deposited_major_collapse_{metric}"])
                for row in variance_rows
            ])
            for metric in (
                "zscore_relative_separation",
                "raw_relative_separation",
                "scale_fitted_raw_relative_separation",
            )
        },
        "major_only_endpoint_relative_separation": {
            metric: describe([
                float(row[metric]) for row in endpoint_rows
            ])
            for metric in (
                "zscore_relative_separation",
                "raw_relative_separation",
                "scale_fitted_raw_relative_separation",
            )
        },
        "probe_terminal_largest_difference": {
            "sites": len(terminal),
            "eligible": sum(int(row["probe_eligible"]) for row in terminal),
            "recovered": sum(int(row["probe_recovered"]) for row in terminal),
        },
        "probe_nonterminal_largest_difference": {
            "sites": len(nonterminal),
            "eligible": sum(int(row["probe_eligible"]) for row in nonterminal),
            "recovered": sum(int(row["probe_recovered"]) for row in nonterminal),
        },
    })


if __name__ == "__main__":
    main()
