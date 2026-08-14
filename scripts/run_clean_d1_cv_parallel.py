#!/usr/bin/env python3
"""Serial and spawned five-fold timing check for clean-D1 guarded CV."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import qfit  # noqa: F401  # keep qFit/CCTBX imports before CUDA Torch
import numpy as np
import torch

from clean_d1_benchmark import site_key
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential
from run_d1_slot_coordination import build_specs, worker


def gpu_memory_mb() -> int:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return max(int(line.strip()) for line in output.splitlines() if line.strip())
    except Exception:
        return -1


def make_spec_template(root: Path, site: tuple[str, str, int], start: Path,
                       flip_root: Path, device: str,
                       inner_nfev: int, outer_updates: int) -> dict[str, object]:
    specs = build_specs(
        root / "specs", flip_root, site=site, mask_scope="window",
        rama_floor=0.02, start_pdb=start, b_factor_mode="single_conformer",
        device=device, occupancy_scheme="mirror", mirror_eta=0.001,
        inner_nfev=inner_nfev, outer_updates=outer_updates,
    )
    spec = next(item for item in specs if item["label"] == "D_null_axis2_30deg")
    return spec


def configure_fold(template: dict[str, object], output: Path, fold: int,
                   treatment: str, train: np.ndarray) -> dict[str, object]:
    spec = copy.deepcopy(template)
    spec["output"] = str(output)
    spec["fold"] = fold
    spec["treatment"] = treatment
    spec["training_indices"] = train.tolist()
    if treatment == "dB_zero":
        spec["fixed_b_offset"] = 0.0
    return spec


def run_fold(spec: dict[str, object]) -> dict[str, object]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    started = time.perf_counter()
    result = worker(spec)
    elapsed = time.perf_counter() - started
    return {
        "fold": int(spec["fold"]), "treatment": str(spec["treatment"]),
        "elapsed_s": elapsed,
        "result": result,
    }


def compact(result: dict[str, object]) -> dict[str, object]:
    value = result["result"]
    return {
        "fold": result["fold"], "treatment": result["treatment"],
        "elapsed_s": result["elapsed_s"],
        "rss": value["final_rss"], "occupancies": value["final_occupancies"],
        "b_offset_A2": value["final_b_offset_A2"],
        "slot_rmsds": value["slot_rmsds"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--start", type=Path, required=True)
    ap.add_argument("--flip-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--inner-nfev", type=int, default=8)
    ap.add_argument("--outer-updates", type=int, default=6)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    site_row = next(row for row in manifest if site_key(row) == args.site)
    site = (str(site_row["pdb_id"]), str(site_row["chain"]), int(site_row["resnum"]))
    base = APrimeSequential(
        args.output / "base", 80, 6, *site, renderer_backend="torch",
        residual_scale_mode="none", map_scaler_structure="full",
        mask_scope="window", device=args.device, start_pdb=args.start,
        b_factor_mode="single_conformer",
    )
    folds = blocked_splits(base.base)
    treatments = ("dB_fitted", "dB_zero")
    serial_rows = []
    parallel_rows = []
    peak_memory = gpu_memory_mb()
    serial_started = time.perf_counter()
    templates = []
    for fold, (train, _test, _direction) in enumerate(folds):
        template = make_spec_template(
            args.output / "templates" / f"fold_{fold}", site, args.start,
            args.flip_root, args.device, args.inner_nfev, args.outer_updates,
        )
        templates.append((template, train))
    for treatment in treatments:
        for fold, (template, train) in enumerate(templates):
            spec = configure_fold(
                template, args.output / "serial" / treatment / f"fold_{fold}" / "D_null_axis2_30deg",
                fold, treatment, train,
            )
            row = run_fold(spec)
            serial_rows.append(row)
            peak_memory = max(peak_memory, gpu_memory_mb())
    serial_wall = time.perf_counter() - serial_started

    jobs = []
    for treatment in treatments:
        for fold, (template, train) in enumerate(templates):
            spec = configure_fold(
                template, args.output / "parallel" / treatment / f"fold_{fold}" / "D_null_axis2_30deg",
                fold, treatment, train,
            )
            jobs.append(spec)

    parallel_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=5, mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(run_fold, spec): spec for spec in jobs}
        while futures:
            peak_memory = max(peak_memory, gpu_memory_mb())
            done = [future for future in futures if future.done()]
            for future in done:
                parallel_rows.append(future.result())
                del futures[future]
            if futures:
                time.sleep(1.0)
    parallel_wall = time.perf_counter() - parallel_started
    peak_memory = max(peak_memory, gpu_memory_mb())

    serial_rows.sort(key=lambda row: (row["treatment"], row["fold"]))
    parallel_rows.sort(key=lambda row: (row["treatment"], row["fold"]))
    serial_by_key = {(row["treatment"], row["fold"]): row for row in serial_rows}
    parallel_by_key = {(row["treatment"], row["fold"]): row for row in parallel_rows}
    equivalence = {}
    for key in serial_by_key:
        a = serial_by_key[key]["result"]
        b = parallel_by_key[key]["result"]
        equivalence[f"{key[0]}_fold{key[1]}"] = {
            "rss_abs_diff": abs(float(a["final_rss"]) - float(b["final_rss"])),
            "occupancy_max_abs_diff": float(np.max(np.abs(np.asarray(a["final_occupancies"]) - np.asarray(b["final_occupancies"]))),),
            "b_offset_abs_diff": abs(float(a["final_b_offset_A2"]) - float(b["final_b_offset_A2"])),
            "slot_rmsd_max_abs_diff": float(max(
                abs(a["slot_rmsds"][i][key2] - b["slot_rmsds"][i][key2])
                for i in range(2) for key2 in ("to_A", "to_B")
            )),
        }
    report = {
        "site": args.site, "device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "inner_nfev": args.inner_nfev, "outer_updates": args.outer_updates,
        "occupancy_scheme": "mirror", "mirror_eta": 0.001,
        "serial_wall_s": serial_wall, "parallel_wall_s": parallel_wall,
        "peak_gpu_memory_mb": peak_memory,
        "serial_folds": [compact(row) for row in serial_rows],
        "parallel_folds": [compact(row) for row in parallel_rows],
        "serial_vs_parallel": equivalence,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
