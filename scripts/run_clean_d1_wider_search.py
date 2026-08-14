#!/usr/bin/env python3
"""Recovery-blind wider PDB search for medium/large backbone alternates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from build_backbone_altloc_site_list import scan_file
from clean_d1_benchmark import atomic_json, atomic_csv, screen_one, site_key


def fetch(url: str, destination: Path) -> str:
    if destination.exists() and destination.stat().st_size > 0:
        return "cached"
    request = Request(url, headers={"User-Agent": "qfitonsteroids-clean-d1/1.0"})
    with urlopen(request, timeout=90) as response:
        payload = response.read()
    temporary = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return "downloaded"


def rcsb_ids(count: int) -> list[str]:
    import urllib.error

    query = {
        "query": {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "range", "value": {"from": 0.8, "to": 2.0},
        }},
        "request_options": {"paginate": {"start": 0, "rows": min(count * 3, 10000)}},
        "return_type": "entry",
    }
    request = Request(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "qfitonsteroids-clean-d1/1.0"},
    )
    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode(errors="replace")) from error
    return [str(item["identifier"]).upper() for item in result.get("result_set", [])]


def local_ids() -> set[str]:
    answer = set()
    for split in ("train", "test"):
        for path in Path(f"/home/dev/qfit_unet_data/{split}").glob("*.pdb"):
            answer.add(path.stem.upper())
    return answer


def acquire_one(root: Path, pdb_id: str) -> dict[str, object]:
    source = root / "source" / f"{pdb_id.lower()}.pdb"
    sf = root / "cache" / "wider" / "structure_factors" / f"{pdb_id}-sf.cif"
    try:
        fetch(f"https://files.rcsb.org/download/{pdb_id}.pdb", source)
        fetch(f"https://files.rcsb.org/download/{pdb_id}-sf.cif", sf)
        return {"pdb_id": pdb_id, "status": "downloaded", "source": str(source), "sf": str(sf)}
    except Exception as error:  # acquisition errors remain separate from site failures
        return {"pdb_id": pdb_id, "status": "error", "error_type": type(error).__name__, "error": repr(error)}


def convert_one(root: Path, row: dict[str, object]) -> dict[str, object]:
    pdb_id = str(row["pdb_id"])
    source = root / "source" / f"{pdb_id.lower()}.pdb"
    sf = root / "cache" / "wider" / "structure_factors" / f"{pdb_id}-sf.cif"
    mtz = root / "cache" / "wider" / "mtz" / f"{pdb_id}.mtz"
    try:
        code = (
            "from density_denoiser.data_pipeline import convert_sf_cif_to_mtz; "
            "import sys; convert_sf_cif_to_mtz(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))"
        )
        completed = subprocess.run(
            ["/home/dev/qfit_unet_data/.venv/bin/python", "-c",
             "from pathlib import Path; " + code, str(sf), str(source), str(mtz)],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-2000:])
        metadata = {"conversion_stdout": completed.stdout[-2000:]}
        return {**row, "status": "complete", "mtz": str(mtz), **metadata}
    except Exception as error:
        return {**row, "status": "error", "error_type": type(error).__name__, "error": repr(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structures", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = args.output / "data"
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["CLEAN_D1_WIDER_ROOT"] = str(root)
    existing = local_ids()
    queried = rcsb_ids(args.structures)
    ids = [pdb_id for pdb_id in queried if pdb_id not in existing][:args.structures]
    atomic_json(args.output / "run_config.json", {
        "operation": "recovery-blind wider PDB search",
        "rcsb_resolution_query_A": [0.8, 2.0],
        "requested_external_structures": args.structures,
        "excluded_original_panel_pool": "all PDB files already in local train/test",
        "target_bins": ["medium 0.8-1.5 A", "large >=1.5 A"],
        "selection_does_not_read_recovery": True,
        "all_nine_gates_applied_in_screen_one": True,
    })
    atomic_json(args.output / "candidate_query.json", {"queried": len(queried), "selected_external_ids": ids})
    downloads = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(acquire_one, root, pdb_id) for pdb_id in ids]
        for future in as_completed(futures):
            downloads.append(future.result())
    atomic_json(args.output / "acquisition.json", {"rows": downloads, "complete": sum(r["status"] == "downloaded" for r in downloads)})
    complete_ids = {r["pdb_id"] for r in downloads if r["status"] == "downloaded"}
    conversions = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        futures = [executor.submit(convert_one, root, r) for r in downloads if r["pdb_id"] in complete_ids]
        for future in as_completed(futures):
            conversions.append(future.result())
    atomic_json(args.output / "conversion.json", {"rows": conversions, "complete": sum(r["status"] == "complete" for r in conversions)})
    conversion_ids = {r["pdb_id"] for r in conversions if r["status"] == "complete"}
    all_candidates = []
    for pdb_id in sorted(conversion_ids):
        all_candidates.extend(scan_file(root / "source" / f"{pdb_id.lower()}.pdb", root, "wider"))
    target_candidates = [row for row in all_candidates if
                        float(row["max_backbone_deviation"]) >= 0.8 and
                        min(float(row["occupancy_a"]), float(row["occupancy_b"])) >= 0.25]
    target_candidates.sort(key=lambda row: (float(row["max_backbone_deviation"]), site_key(row)))
    atomic_csv(args.output / "all_candidate_rows.csv", all_candidates)
    atomic_csv(args.output / "target_candidate_rows.csv", target_candidates)
    atomic_json(args.output / "candidate_counts.json", {
        "external_structures_selected": len(ids),
        "structures_with_reflections": len(conversion_ids),
        "candidate_rows_scanned": len(all_candidates),
        "target_candidate_rows_screened": len(target_candidates),
        "target_prefilter": "max single-backbone-atom displacement >= 0.8 A and minor occupancy >= 0.25; exact central RMSD and all nine gates are measured below",
    })
    rows = []
    for index, candidate in enumerate(target_candidates, start=1):
        rows.append(screen_one(candidate, args.output / "scratch", args.device))
        atomic_json(args.output / "progress.json", {"status": "running", "screened": index, "total": len(target_candidates)})
        atomic_csv(args.output / "per_site.csv", rows)
    atomic_json(args.output / "summary.json", {
        "status": "complete", "candidate_rows_scanned": len(all_candidates),
        "target_candidate_rows_screened": len(target_candidates), "rows": rows,
    })
    atomic_json(args.output / "progress.json", {"status": "complete", "screened": len(rows), "total": len(target_candidates)})


if __name__ == "__main__":
    main()
