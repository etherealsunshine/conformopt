#!/usr/bin/env python3
"""Run the direct qFit -> Phenix arm for the Zenodo signature panel.

The published qFit models are immutable inputs.  Each site first uses the
deposited FreeR flags in the panel MTZ.  If Phenix rejects those flags because
they do not cover the complete Fobs array, the site is retried in a separate
subdirectory with Phenix-generated FreeR flags and that distinction is
recorded in the checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
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


def run_attempt(site_label: str, pdb_path: Path, mtz_path: Path,
                output_dir: Path, phenix_bin: Path, timeout_s: int,
                generate_rfree_flags: bool) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "phenix.log"
    command = [
        str(phenix_bin), str(pdb_path), str(mtz_path),
        f"output.prefix={output_dir / 'refined'}",
        "strategy=individual_sites+individual_adp",
    ]
    if generate_rfree_flags:
        command.append("xray_data.r_free_flags.generate=True")
    started = time.perf_counter()
    env = os.environ.copy()
    env["PATH"] = str(phenix_bin.parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("PHENIX_ROOT", str(phenix_bin.parent.parent))
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(
                command, cwd=output_dir, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=timeout_s, check=False,
            )
        status = "complete" if completed.returncode == 0 else "failed"
        return {
            "site": site_label, "status": status,
            "returncode": int(completed.returncode),
            "generate_rfree_flags": generate_rfree_flags,
            "command": command, "log": str(log_path),
            "elapsed_s": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "site": site_label, "status": "timeout",
            "generate_rfree_flags": generate_rfree_flags,
            "command": command, "log": str(log_path),
            "elapsed_s": time.perf_counter() - started,
            "error": repr(exc),
        }
    except Exception as exc:
        return {
            "site": site_label, "status": "failed",
            "generate_rfree_flags": generate_rfree_flags,
            "command": command, "log": str(log_path),
            "elapsed_s": time.perf_counter() - started,
            "error": repr(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--phenix-bin", type=Path,
        default=Path("/home/dev/qfit_unet_data/phenix-2.2-6143/bin/phenix.refine"),
    )
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--site", action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.output_root.exists() and not args.resume:
        raise FileExistsError(
            f"output root already exists; pass --resume explicitly: {args.output_root}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = set(args.site) if args.site else None
    rows = list(csv.DictReader((args.panel_root / "selected_sites.csv").open()))
    rows = [row for row in rows if not selected or (
        f"{row['pdb_id']}_{row['chain']}_{row['resname']}{row['residue_number']}"
        in selected
    )]
    if not rows:
        raise RuntimeError("no selected sites found")
    manifest = {
        "status": "running", "created_at": utc_now(),
        "panel_root": str(args.panel_root), "output_root": str(args.output_root),
        "phenix_bin": str(args.phenix_bin), "timeout_s": args.timeout_s,
        "strategy": "individual_sites+individual_adp",
        "fallback": "retry with xray_data.r_free_flags.generate=True only after a native FreeR failure",
        "sites": [f"{r['pdb_id']}_{r['chain']}_{r['resname']}{r['residue_number']}" for r in rows],
    }
    atomic_json(args.output_root / "run_config.json", manifest)
    results: list[dict[str, object]] = []
    for row in rows:
        label = f"{row['pdb_id']}_{row['chain']}_{row['resname']}{row['residue_number']}"
        site_dir = args.output_root / label
        status_path = site_dir / "status.json"
        if args.resume and status_path.is_file():
            try:
                existing = json.loads(status_path.read_text())
            except json.JSONDecodeError:
                existing = None
            if existing and existing.get("status") == "complete":
                results.append(existing)
                continue
        pdb_path = args.panel_root / "inputs" / "qfit" / f"{row['pdb_id']}_qFit.pdb"
        mtz_path = args.panel_root / "inputs" / "map_mtz" / f"{row['pdb_id'].lower()}.mtz"
        native = run_attempt(label, pdb_path, mtz_path, site_dir / "native",
                             args.phenix_bin, args.timeout_s, False)
        attempts = [native]
        final = native
        if native.get("status") != "complete":
            generated = run_attempt(label, pdb_path, mtz_path, site_dir / "generated_rfree",
                                    args.phenix_bin, args.timeout_s, True)
            attempts.append(generated)
            final = generated
        result = {
            "site": label, "status": final.get("status"),
            "selected_attempt": "generated_rfree" if final is not native else "native",
            "attempts": attempts,
            "input_qfit_pdb": str(pdb_path), "map_mtz": str(mtz_path),
            "finished_at": utc_now(),
        }
        atomic_json(status_path, result)
        results.append(result)
        atomic_json(args.output_root / "progress.json", {
            "status": "running", "completed_sites": len(results),
            "total_sites": len(rows), "results": results,
        })
    atomic_json(args.output_root / "progress.json", {
        "status": "complete", "completed_sites": len(results),
        "total_sites": len(rows), "results": results,
    })
    manifest["status"] = "complete"
    manifest["finished_at"] = utc_now()
    atomic_json(args.output_root / "run_config.json", manifest)


if __name__ == "__main__":
    main()
