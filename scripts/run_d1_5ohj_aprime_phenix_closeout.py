#!/usr/bin/env python3
"""Complete the 5OHJ qFit -> A-prime -> Phenix four-way closeout."""

from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_d1_qfit_selected_aprime_closeout import replace_central
from occupancy_selection import solve_affine_qp
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential
from run_d1_clash_audit import endpoint_line


SITE = "5OHJ_A_SER540"
PDB_ID, CHAIN, RESNUM = "5OHJ", "A", 540
QFIT_ROOT = Path("/home/dev/qfit_unet_data/qfit_audit/d1_pipeline_panel_selection_v1/qfit_miqp")
NEUTRAL_ROOT = Path("/home/dev/qfit_unet_data/qfit_audit/d1_pipeline_panel_selection_v1/neutral_starts")
APRIME_RUN = Path("/home/dev/qfit_unet_data/qfit_audit/d1_qfit_selected_aprime_refinement_v1")
APRIME_SITE = APRIME_RUN / SITE
MTZ = Path("/home/dev/qfit_unet_data/cache/train/mtz/5OHJ.mtz")
PHENIX = Path("/home/dev/qfit_unet_data/phenix-2.2-6143/bin/phenix.refine")
CLASHSCORE = Path("/home/dev/qfit_unet_data/phenix-2.2-6143/bin/phenix.clashscore")
PHENIX_ROOT = PHENIX.parent.parent
MONOMER_CANDIDATES = (
    PHENIX_ROOT / "lib" / "python3.11" / "site-packages" / "chem_data" / "chemical_components",
    PHENIX_ROOT / "lib" / "python3.11" / "site-packages" / "mmtbx" / "chemical_components",
    PHENIX_ROOT / "chem_data" / "chemical_components",
)
MONOMER_ROOT = next((p for p in MONOMER_CANDIDATES if p.is_dir()), MONOMER_CANDIDATES[0])
OUTPUT = Path(os.environ.get(
    "D1_5OHJ_CLOSEOUT_OUTPUT",
    "/home/dev/qfit_unet_data/qfit_audit/d1_5ohj_aprime_phenix_closeout_v1",
))


def phenix_nproc() -> int:
    """Return the pod's CPU quota, rather than the host's visible CPU count."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return int(os.environ.get("D1_PHENIX_NPROC", "12"))


def window_residues(runner: APrimeSequential) -> set[tuple[str, str]]:
    return {
        (str(runner.base.full_structure.chain[int(i)]).strip(),
         str(int(runner.base.full_structure.resi[int(i)])))
        for i in runner.window.selection
    }


def write_pair_pdb(neutral: Path, runner: APrimeSequential, slots: np.ndarray,
                   occupancies: np.ndarray, output: Path) -> None:
    templates = [line for line in neutral.read_text().splitlines()
                 if line.startswith(("ATOM  ", "HETATM"))
                 and (line[21].strip(), line[22:26].strip()) in window_residues(runner)]
    if len(templates) != len(runner.window.selection):
        raise RuntimeError(f"window template count {len(templates)} != {len(runner.window.selection)}")
    residues = window_residues(runner)
    kept = []
    for line in neutral.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and (line[21].strip(), line[22:26].strip()) in residues:
            continue
        if line.startswith(("MODEL", "ENDMDL", "END")):
            continue
        kept.append(line)
    for altloc, coords, occ in zip(("A", "B"), slots, occupancies):
        kept.extend(endpoint_line(t, xyz, altloc=altloc, occupancy=float(occ))
                    for t, xyz in zip(templates, coords))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(kept + ["END", ""]) )


def read_pair_pdb(path: Path, runner: APrimeSequential) -> np.ndarray:
    wanted = {(str(runner.base.full_structure.chain[int(i)]).strip(),
               str(int(runner.base.full_structure.resi[int(i)])),
               str(runner.base.full_structure.name[int(i)]).strip())
              for i in runner.window.selection}
    by_key: dict[tuple[str, str, str, str], list[np.ndarray]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        key0 = (line[21].strip(), line[22:26].strip(), line[12:16].strip())
        if key0 in wanted:
            key = (*key0, line[16].strip())
            by_key[key].append(np.asarray([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float))
    slots = []
    for altloc in ("A", "B"):
        coords = []
        for i in runner.window.selection:
            key0 = (str(runner.base.full_structure.chain[int(i)]).strip(),
                    str(int(runner.base.full_structure.resi[int(i)])).strip(),
                    str(runner.base.full_structure.name[int(i)]).strip())
            values = by_key.get((*key0, altloc)) or by_key.get((*key0, ""))
            if not values:
                raise RuntimeError(f"missing {altloc} {key0} in {path}")
            coords.append(values[0])
        slots.append(np.asarray(coords, dtype=float))
    return np.stack(slots)


def score_pair(runner: APrimeSequential, pair: np.ndarray) -> dict[str, object]:
    models = runner.base.model_density_batch(pair, slots=np.asarray((0, 1)), b_offset=0.0)
    folds = []
    for fold, (train, test, direction) in enumerate(blocked_splits(runner.base)):
        weights, intercept, train_rss = solve_affine_qp(runner.base.target[train], models[:, train])
        residual = runner.base.target[test] - weights @ models[:, test] - intercept
        folds.append({
            "fold": fold,
            "heldout_rss": float(np.square(residual).sum()),
            "train_rss": float(train_rss),
            "weights": np.asarray(weights).tolist(),
            "intercept": float(intercept),
            "heldout_voxels": int(len(test)),
            "direction": np.asarray(direction).tolist(),
        })
    values = np.asarray([r["heldout_rss"] for r in folds], dtype=float)
    return {"folds": folds, "mean": float(values.mean()), "median": float(np.median(values))}


def refine(input_pdb: Path, label: str) -> tuple[Path, list[str]]:
    out = OUTPUT / f"phenix_{label}"
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / "refined"
    log = out / "refine.log"
    command = [str(PHENIX), str(input_pdb), str(MTZ), f"output.prefix={prefix}",
               "strategy=individual_sites+individual_adp", "main.number_of_macro_cycles=5",
               f"nproc={phenix_nproc()}"]
    with log.open("w") as handle:
        proc = subprocess.run(command, cwd=out, stdout=handle, stderr=subprocess.STDOUT, check=False)
    outputs = sorted(out.glob("refined_*.pdb"))
    if proc.returncode or not outputs:
        raise RuntimeError(f"phenix.refine {label} failed rc={proc.returncode}: {log}")
    return outputs[-1], command


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    qroot = QFIT_ROOT / SITE
    qresult = json.loads((qroot / "result.json").read_text())
    qz = np.load(qroot / "selected.npz", allow_pickle=False)
    selected = np.asarray(qz["selected_indices"], dtype=int)
    if len(selected) != 1:
        raise RuntimeError(f"expected one qFit alternate, got {selected.tolist()}")
    neutral = Path(qresult["start_pdb"])
    runner = APrimeSequential(OUTPUT / "scoring_context", 1, 1, PDB_ID, CHAIN, RESNUM,
                              renderer_backend="torch", residual_scale_mode="none",
                              map_scaler_structure="full", mask_scope="window",
                              device=os.environ.get("D1_DEVICE", "cpu"),
                              start_pdb=neutral, b_factor_mode="single_conformer", density_atom_scope="all")
    candidates = np.asarray(qz["candidate_coordinates"], dtype=float)
    names = [str(x) for x in qz["candidate_atom_names"].tolist()]
    qfit_pair = np.stack((runner.initial,
                          replace_central(runner.initial, runner, candidates[int(selected[0])], names)))
    qfit_occ = np.asarray([max(1.0 - float(qz["occupancies"][selected].sum()), 0.0),
                           float(qz["occupancies"][selected[0]])])
    with np.load(APRIME_SITE / "geometry_outer_500.npz") as saved:
        aprime_pair = np.stack((saved["slot1_window"], saved["slot2_window"]))
    with np.load(APRIME_SITE / "resume_state.npz") as saved:
        aprime_occ = np.asarray(saved["occupancy_weights"], dtype=float)
    qfit_pdb = OUTPUT / "qfit_selected.pdb"
    aprime_pdb = OUTPUT / "aprime_endpoint.pdb"
    write_pair_pdb(neutral, runner, qfit_pair, qfit_occ, qfit_pdb)
    write_pair_pdb(neutral, runner, aprime_pair, aprime_occ, aprime_pdb)
    report = {"status": "running", "site": SITE, "inputs": {"qfit_pdb": str(qfit_pdb), "aprime_pdb": str(aprime_pdb), "aprime_source": str(APRIME_SITE / "geometry_outer_500.npz")}, "models": {}}
    report["models"]["qfit_raw"] = {"heldout": score_pair(runner, qfit_pair), "pdb": str(qfit_pdb)}
    report["models"]["aprime"] = {"heldout": score_pair(runner, aprime_pair), "pdb": str(aprime_pdb)}
    (OUTPUT / "summary_in_progress.json").write_text(json.dumps(report, indent=2) + "\n")
    qfit_refined, qfit_cmd = refine(qfit_pdb, "qfit")
    aprime_refined, aprime_cmd = refine(aprime_pdb, "aprime")
    qfit_phenix_pair = read_pair_pdb(qfit_refined, runner)
    aprime_phenix_pair = read_pair_pdb(aprime_refined, runner)
    report["models"]["qfit_phenix"] = {"heldout": score_pair(runner, qfit_phenix_pair), "pdb": str(qfit_refined), "command": qfit_cmd}
    report["models"]["aprime_phenix"] = {"heldout": score_pair(runner, aprime_phenix_pair), "pdb": str(aprime_refined), "command": aprime_cmd}
    base = report["models"]["qfit_raw"]["heldout"]["folds"]
    for label in ("qfit_phenix", "aprime", "aprime_phenix"):
        report.setdefault("paired_difference_vs_qfit_raw", {})[label] = [
            float(a["heldout_rss"] - b["heldout_rss"])
            for a, b in zip(report["models"][label]["heldout"]["folds"], base)
        ]
    report["status"] = "complete"
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "models": {k: {"mean": v["heldout"]["mean"], "pdb": v["pdb"]} for k, v in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
