#!/usr/bin/env python3
"""Run remaining Zenodo signature-panel sites as independent analysis shards.

The per-site analysis is unchanged; this wrapper only parallelizes independent
sites and atomically checkpoints the aggregate result after each shard.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_sites(panel_root: Path) -> list[dict[str, str]]:
    with (panel_root / "selected_sites.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def label(row: dict[str, str]) -> str:
    return f"{row['pdb_id']}_{row['chain']}_{row['resname']}{row['residue_number']}"


def result_from_shard(shard_root: Path, site: str) -> dict[str, object] | None:
    final = shard_root / "zenodo_signature_full_analysis.json"
    progress = shard_root / "progress.json"
    for path in (final, progress):
        if not path.is_file():
            continue
        try:
            obj = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        rows = obj.get("rows", [])
        if rows:
            row = rows[0]
            if row.get("site") == site:
                return row
    return None


def run_one(row: dict[str, str], args) -> dict[str, object]:
    site = label(row)
    shard_root = args.output_root / "shards" / site
    shard_root.mkdir(parents=True, exist_ok=True)
    existing = result_from_shard(shard_root, site)
    if existing is not None and existing.get("status") != "failed":
        return existing

    command = [
        sys.executable,
        "-X",
        "faulthandler",
        str(args.analysis_script),
        "--panel-root",
        str(args.panel_root),
        "--execution-root",
        str(args.execution_root),
        "--qfit-phenix-root",
        str(args.qfit_phenix_root),
        "--aprime-phenix-root",
        str(args.aprime_phenix_root),
        "--output-root",
        str(shard_root),
        "--device",
        args.device,
        "--site",
        site,
    ]
    log_path = shard_root / "controller.log"
    environment = os.environ.copy()
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=args.analysis_script.parent.parent,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    except Exception as exc:
        return {"site": site, "status": "failed", "error": repr(exc)}

    result = result_from_shard(shard_root, site)
    if result is not None:
        return result
    return {
        "site": site,
        "status": "failed",
        "error": f"shard exited {completed.returncode} without a result",
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--qfit-phenix-root", type=Path, required=True)
    parser.add_argument("--aprime-phenix-root", type=Path, required=True)
    parser.add_argument("--analysis-script", type=Path, required=True)
    parser.add_argument("--base-output-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    sites = load_sites(args.panel_root)
    by_site: dict[str, dict[str, object]] = {}
    if args.base_output_root:
        base_progress = args.base_output_root / "progress.json"
        if base_progress.is_file():
            base_rows = read_json(base_progress).get("rows", [])
            by_site.update({row["site"]: row for row in base_rows})

    all_labels = {label(row) for row in sites}
    by_site = {
        site: row for site, row in by_site.items()
        if site in all_labels and row.get("status") != "failed"
    }
    remaining = [row for row in sites if label(row) not in by_site]
    atomic_json(args.output_root / "progress.json", {
        "status": "running",
        "completed_sites": len(by_site),
        "total_sites": len(sites),
        "workers": args.workers,
        "rows": [by_site[key] for key in sorted(by_site)],
    })

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, row, args): label(row) for row in remaining}
        for future in concurrent.futures.as_completed(futures):
            site = futures[future]
            try:
                by_site[site] = future.result()
            except Exception as exc:
                by_site[site] = {"site": site, "status": "failed", "error": repr(exc)}
            ordered = [by_site[key] for key in sorted(by_site)]
            atomic_json(args.output_root / "progress.json", {
                "status": "running",
                "completed_sites": len(ordered),
                "total_sites": len(sites),
                "workers": args.workers,
                "rows": ordered,
            })

    ordered = [by_site[key] for key in sorted(by_site)]
    atomic_json(args.output_root / "zenodo_signature_full_analysis.json", {
        "status": "complete",
        "rows": ordered,
    })
    atomic_json(args.output_root / "progress.json", {
        "status": "complete",
        "completed_sites": len(ordered),
        "total_sites": len(sites),
        "workers": args.workers,
        "rows": ordered,
    })
    print(json.dumps({
        "status": "complete",
        "completed_sites": len(ordered),
        "failed_sites": [row["site"] for row in ordered if row.get("status") == "failed"],
    }, indent=2))


if __name__ == "__main__":
    main()
