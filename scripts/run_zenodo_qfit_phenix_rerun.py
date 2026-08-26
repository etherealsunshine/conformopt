#!/usr/bin/env python3
"""Checkpointed Phenix-only rerun for an existing Zenodo qFit A-prime panel.

The A-prime endpoints are treated as immutable inputs.  This runner only
repeats the missing Phenix stage with an explicitly selected Phenix executable
and writes a separate, versioned output tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def site_inputs(execution_root: Path, selected: set[str] | None) -> list[Path]:
    sites = [
        path for path in execution_root.iterdir()
        if path.is_dir() and (path / "phenix_input.pdb").is_file()
    ]
    sites.sort(key=lambda path: path.name)
    if selected:
        sites = [path for path in sites if path.name in selected]
    if not sites:
        raise RuntimeError(f"no existing phenix_input.pdb files found under {execution_root}")
    return sites


def run_site(site: Path, panel_root: Path, output_root: Path,
             phenix_bin: Path, timeout_s: int,
             generate_rfree_flags: bool) -> dict[str, object]:
    label = site.name
    site_output = output_root / label
    site_output.mkdir(parents=True, exist_ok=True)
    status_path = site_output / "status.json"
    existing = None
    if status_path.is_file():
        try:
            existing = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            existing = None
    if existing and existing.get("status") == "complete":
        return existing

    pdb_id = label.split("_", 1)[0].lower()
    mtz = panel_root / "inputs" / "map_mtz" / f"{pdb_id}.mtz"
    phenix_input = site / "phenix_input.pdb"
    command = [
        str(phenix_bin),
        str(phenix_input),
        str(mtz),
        f"output.prefix={site_output / 'refined'}",
        "strategy=individual_sites+individual_adp",
    ]
    if generate_rfree_flags:
        command.append("xray_data.r_free_flags.generate=True")
    started = time.perf_counter()
    running = {
        "status": "running",
        "site": label,
        "started_at": utc_now(),
        "command": command,
        "phenix_bin": str(phenix_bin),
        "input_endpoint": str(phenix_input),
        "map_mtz": str(mtz),
        "timeout_s": timeout_s,
        "generate_rfree_flags": generate_rfree_flags,
    }
    atomic_json(status_path, running)
    log_path = site_output / "phenix.log"
    try:
        if not phenix_bin.is_file() or not os.access(phenix_bin, os.X_OK):
            raise FileNotFoundError(f"Phenix executable is not executable: {phenix_bin}")
        if not phenix_input.is_file():
            raise FileNotFoundError(f"missing endpoint input: {phenix_input}")
        if not mtz.is_file():
            raise FileNotFoundError(f"missing map MTZ: {mtz}")
        env = os.environ.copy()
        env["PATH"] = str(phenix_bin.parent) + os.pathsep + env.get("PATH", "")
        env.setdefault("PHENIX_ROOT", str(phenix_bin.parent.parent))
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=site_output,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
        result = {
            **running,
            "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": int(completed.returncode),
            "finished_at": utc_now(),
            "elapsed_s": time.perf_counter() - started,
            "log": str(log_path),
        }
    except subprocess.TimeoutExpired as exc:
        result = {
            **running,
            "status": "timeout",
            "finished_at": utc_now(),
            "elapsed_s": time.perf_counter() - started,
            "timeout_s": timeout_s,
            "error": repr(exc),
            "log": str(log_path),
        }
    except Exception as exc:  # checkpoint the site and allow remaining sites to run
        result = {
            **running,
            "status": "failed",
            "finished_at": utc_now(),
            "elapsed_s": time.perf_counter() - started,
            "error": repr(exc),
            "log": str(log_path),
        }
    atomic_json(status_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--phenix-bin", type=Path,
        default=Path("/home/dev/qfit_unet_data/phenix-2.2-6143/bin/phenix.refine"),
    )
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--generate-rfree-flags", action="store_true")
    parser.add_argument("--site", action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and not args.resume:
        raise FileExistsError(
            f"output root already exists; pass --resume explicitly: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.site) if args.site else None
    sites = site_inputs(args.execution_root, selected)
    manifest = {
        "status": "running",
        "created_at": utc_now(),
        "execution_root": str(args.execution_root),
        "panel_root": str(args.panel_root),
        "output_root": str(args.output_root),
        "phenix_bin": str(args.phenix_bin),
        "timeout_s": args.timeout_s,
        "strategy": "individual_sites+individual_adp",
        "generate_rfree_flags": args.generate_rfree_flags,
        "sites": [site.name for site in sites],
        "source_optimizer_parameters": {
            "inner_nfev": 8,
            "outer_updates": 6,
            "chi_nfev": 20,
            "clash_weight": 33.0051,
        },
    }
    atomic_json(args.output_root / "run_config.json", manifest)

    results: list[dict[str, object]] = []
    for site in sites:
        result = run_site(site, args.panel_root, args.output_root,
                          args.phenix_bin, args.timeout_s,
                          args.generate_rfree_flags)
        results.append(result)
        atomic_json(args.output_root / "progress.json", {
            "status": "running",
            "completed_sites": len(results),
            "total_sites": len(sites),
            "results": results,
        })
    atomic_json(args.output_root / "progress.json", {
        "status": "complete",
        "completed_sites": len(results),
        "total_sites": len(sites),
        "results": results,
    })
    manifest["status"] = "complete"
    manifest["finished_at"] = utc_now()
    atomic_json(args.output_root / "run_config.json", manifest)


if __name__ == "__main__":
    sys.exit(main())
