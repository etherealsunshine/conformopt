"""Measure occupancy starvation timing and post-threshold slot travel."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np
import torch

from density_denoiser.residue_geometry import CHI_SPECS, symmetry_aware_rmsd
from experiments.probe4.core import torsion_to_coords


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, newline="", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else float("nan")


def load_site_metadata(audit_roots: list[Path]) -> dict[str, dict[str, object]]:
    result = {}
    for root in audit_roots:
        payload = json.loads((root / "tmol_inputs.json").read_text())
        for site in payload["sites"]:
            base_path = root / site["base_pdb_A"]
            structure = gemmi.read_structure(str(base_path))
            chain = next(
                chain for chain in structure[0] if chain.name == site["chain"]
            )
            residue = next(
                residue for residue in chain
                if residue.seqid.num == int(site["residue_number"])
            )
            lookup = {
                atom.name.strip(): torch.tensor(
                    atom.pos.tolist(), dtype=torch.float32
                )
                for atom in residue
                if atom.element.name != "H"
            }
            names = list(site["atom_names"])
            missing = set(names) - set(lookup)
            if missing:
                raise ValueError(f"{site['site']} base PDB missing {sorted(missing)}")
            result[site["site"]] = {
                "names": names,
                "resname": site["residue_type"],
                "template": torch.tensor(site["A"], dtype=torch.float32),
                "fixed_lookup": {
                    name: xyz for name, xyz in lookup.items() if name not in names
                },
            }
    if len(result) != 20:
        raise ValueError(f"expected 20 sites, found {len(result)}")
    return result


def coordinates(chi: np.ndarray, metadata: dict[str, object]) -> torch.Tensor:
    return torsion_to_coords(
        metadata["template"],
        metadata["names"],
        torch.tensor(chi, dtype=torch.float32),
        list(CHI_SPECS[metadata["resname"]]["rotations"]),
        metadata["fixed_lookup"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    detail_rows = []
    for fixed in (100, 200, 300):
        arm = args.sweep_root / f"fixed_{fixed}"
        audit_roots = [
            arm / "audit" / "original5",
            arm / "audit" / "expanded15",
        ]
        metadata = load_site_metadata(audit_roots)
        for panel in ("original5", "expanded15"):
            for shard in sorted((arm / "shards" / panel).iterdir()):
                if not shard.is_dir():
                    continue
                starts_path = (
                    shard / "synthetic" / f"{shard.name}_starts.csv"
                )
                for row in read_csv(starts_path):
                    start = int(row["start"])
                    trace_path = (
                        shard / "trajectories"
                        / f"synthetic_start_{start:03d}.npz"
                    )
                    trace = np.load(trace_path)
                    occupancies = trace["occupancies"]
                    chis = trace["chi_radians"]
                    steps = trace["step"]
                    phases = trace["phase"]
                    site_info = metadata[shard.name]
                    final_chi = chis[-1]
                    final_coordinates = [
                        coordinates(final_chi[slot], site_info)
                        for slot in range(final_chi.shape[0])
                    ]
                    for slot in range(occupancies.shape[1]):
                        output: dict[str, object] = {
                            "arm": f"fixed_{fixed}",
                            "site": shard.name,
                            "start": start,
                            "slot": slot,
                            "final_occupancy": float(occupancies[-1, slot]),
                        }
                        for threshold, label in ((0.05, "0p05"), (0.02, "0p02")):
                            indices = np.flatnonzero(
                                occupancies[:, slot] < threshold
                            )
                            if len(indices):
                                index = int(indices[0])
                                crossing_coordinates = coordinates(
                                    chis[index, slot], site_info
                                )
                                travel = float(symmetry_aware_rmsd(
                                    crossing_coordinates,
                                    final_coordinates[slot],
                                    site_info["names"],
                                    site_info["resname"],
                                ))
                                output[f"first_below_{label}_step"] = int(
                                    steps[index]
                                )
                                output[f"first_below_{label}_phase"] = int(
                                    phases[index]
                                )
                                output[f"rmsd_after_first_below_{label}_A"] = travel
                            else:
                                output[f"first_below_{label}_step"] = -1
                                output[f"first_below_{label}_phase"] = -1
                                output[f"rmsd_after_first_below_{label}_A"] = (
                                    float("nan")
                                )
                        detail_rows.append(output)

    summary_rows = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(str(row["arm"]), str(row["site"]))].append(row)
    for (arm, site), rows in sorted(grouped.items()):
        summary: dict[str, object] = {
            "arm": arm,
            "site": site,
            "slots": len(rows),
        }
        for label in ("0p05", "0p02"):
            crossed = [
                row for row in rows
                if int(row[f"first_below_{label}_step"]) >= 0
            ]
            steps = [
                float(row[f"first_below_{label}_step"]) for row in crossed
            ]
            travel = [
                float(row[f"rmsd_after_first_below_{label}_A"])
                for row in crossed
            ]
            summary[f"slots_below_{label}"] = len(crossed)
            summary[f"first_below_{label}_step_median"] = percentile(
                steps, 0.5
            )
            summary[f"post_threshold_rmsd_median_A_{label}"] = percentile(
                travel, 0.5
            )
            summary[f"post_threshold_rmsd_q25_A_{label}"] = percentile(
                travel, 0.25
            )
            summary[f"post_threshold_rmsd_q75_A_{label}"] = percentile(
                travel, 0.75
            )
        summary_rows.append(summary)

    atomic_csv(args.output / "slot_starvation_and_travel.csv", detail_rows)
    atomic_csv(args.output / "starvation_and_travel_per_site.csv", summary_rows)
    print(json.dumps({
        "control_trace_available": False,
        "limitation": (
            "Frozen control endpoints contain no per-step occupancy/chi trace; "
            "mechanism timing is measurable only in fixed 100/200/300 arms."
        ),
        "slots": len(detail_rows),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
