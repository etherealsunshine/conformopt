"""Run the five-site endpoint tmol audit on a CUDA host.

The runner is deliberately restartable: it writes the energy table and progress
manifest atomically after every completed protein/site.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import tmol
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit")

    device = torch.device("cuda")
    inputs = json.loads((args.input_root / "tmol_inputs.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    energy_path = args.output / "tmol_energies.csv"
    rows = read_existing_rows(energy_path) if args.resume else []
    completed_sites = {row["site"] for row in rows}
    calibration_rows: list[dict[str, Any]] = []

    for site_index, site in enumerate(inputs["sites"]):
        if not args.calibrate_only and site["site"] in completed_sites:
            print(f"already complete: {site['site']}", flush=True)
            continue

        base_pdb = args.input_root / site["base_pdb"]
        base_pose = tmol.pose_stack_from_pdb(str(base_pdb), device=device)
        scorer = tmol.beta2016_score_function(device).render_whole_pose_scoring_module(
            base_pose
        )
        base_lines = base_pdb.read_text().splitlines()
        candidate_path = Path("/tmp") / f"{site['site']}_tmol_candidate.pdb"
        energy_cache: dict[tuple[float, ...], float] = {}

        def energy(candidate: list[list[float]]) -> float:
            cache_key = tuple(round(value, 3) for atom in candidate for value in atom)
            if cache_key in energy_cache:
                return energy_cache[cache_key]
            replacements = dict(zip(site["atom_names"], candidate))
            candidate_lines = []
            for line in base_lines:
                if (
                    line.startswith("ATOM")
                    and line[21].strip() == site["chain"]
                    and int(line[22:26]) == int(site["residue_number"])
                    and line[12:16].strip() in replacements
                ):
                    x, y, z = replacements[line[12:16].strip()]
                    line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                candidate_lines.append(line)
            candidate_path.write_text("\n".join(candidate_lines) + "\n")
            # Reparse every conformation so tmol rebuilds hydrogen coordinates.
            candidate_pose = tmol.pose_stack_from_pdb(str(candidate_path), device=device)
            value = float(scorer(candidate_pose.coords).sum().detach().cpu())
            energy_cache[cache_key] = value
            return value

        energy_a = energy(site["A"])
        energy_b = energy(site["B"])
        if args.calibrate_only:
            random_energy = energy(site["random_rotamers"][0])
            calibration_rows.append(
                {
                    "site": site["site"],
                    "tmol_A": energy_a,
                    "tmol_B": energy_b,
                    "random_0": random_energy,
                    "finite": all(
                        torch.isfinite(torch.tensor(value)).item()
                        for value in (energy_a, energy_b, random_energy)
                    ),
                }
            )
            print(f"calibrated {site_index + 1}/{len(inputs['sites'])}: {site['site']}", flush=True)
            continue

        random_energies = [energy(candidate) for candidate in site["random_rotamers"]]
        site_rows = []
        for candidate in site["candidates"]:
            site_rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "site": site["site"],
                    "start": candidate["start"],
                    "conformer": candidate["conformer"],
                    "tmol_energy": energy(candidate["coordinates"]),
                    "tmol_A": energy_a,
                    "tmol_B": energy_b,
                    "random_mean": sum(random_energies) / len(random_energies),
                    "random_min": min(random_energies),
                    "random_max": max(random_energies),
                }
            )
        rows.extend(site_rows)
        atomic_rows(energy_path, rows)
        atomic_json(
            args.output / "tmol_progress.json",
            {
                "completed_sites": len({row["site"] for row in rows}),
                "total_sites": len(inputs["sites"]),
                "latest_site": site["site"],
                "endpoints_scored": len(rows),
                "tmol_version": tmol.__version__,
            },
        )
        print(
            f"scored {site_index + 1}/{len(inputs['sites'])}: "
            f"{site['site']} ({len(rows)} endpoints total)",
            flush=True,
        )

    if args.calibrate_only:
        if not calibration_rows or not all(row["finite"] for row in calibration_rows):
            raise RuntimeError("tmol calibration produced a missing or non-finite score")
        atomic_json(
            args.output / "tmol_calibration.json",
            {
                "status": "passed",
                "device": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "tmol_version": tmol.__version__,
                "sites": calibration_rows,
            },
        )
        return

    atomic_json(
        args.output / "tmol_manifest.json",
        {
            "status": "complete",
            "tmol_version": tmol.__version__,
            "active_conformers_scored": len(rows),
            "sites": len({row["site"] for row in rows}),
        },
    )


if __name__ == "__main__":
    main()
