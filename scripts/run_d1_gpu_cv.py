#!/usr/bin/env python3
"""Run the five full-window blocked-CV folds in the CUDA/qFit environment."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


QFIT_SITE = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/site-packages"
QFIT_DYNLIB = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/lib-dynload"
QFIT_SRC = "/home/dev/workspace/external/qfit-3.0/src"
WORKSPACE = "/home/dev/workspace"
sys.path.insert(0, QFIT_SITE)
import numpy  # noqa: E402  # load qFit's NumPy ABI before CUDA Torch
sys.path.remove(QFIT_SITE)
import torch  # noqa: E402
sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]


def run(root: Path, pdb_id: str = "7T7A", chain: str = "A", resnum: int = 396) -> None:
    script = "scripts/run_d1_aprime_leakage_corrected_cv.py"
    root.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        output = root / f"split_{fold}"
        if (output / "heldout_result.json").is_file():
            continue
        sys.argv = [
            script, "--output", str(output), "--fold", str(fold),
            "--inner-nfev", "8", "--outer-updates", "6",
            "--slot2-occupancy-floor", "0.02", "--pdb-id", pdb_id,
            "--chain", chain, "--resnum", str(resnum), "--mask-scope", "window",
            "--rama-floor", "0.02", "--device", "cuda",
        ]
        runpy.run_path(script, run_name="__main__")
    sys.argv = [script, "--output", str(root), "--aggregate",
                "--pdb-id", pdb_id, "--chain", chain, "--resnum", str(resnum)]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    if len(sys.argv) not in {2, 8}:
        raise SystemExit(
            "usage: run_d1_gpu_cv.py OUTPUT [--pdb-id PDB --chain CHAIN --resnum RESNUM]"
        )
    if len(sys.argv) == 2:
        run(Path(sys.argv[1]))
    else:
        run(Path(sys.argv[1]), sys.argv[3], sys.argv[5], int(sys.argv[7]))
