#!/usr/bin/env python3
"""Filtered clash audit for the 5OHJ A-prime/Phenix closeout."""

from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

from analyze_phenix_clash_audit import (
    Connectivity,
    load_json_log,
    read_pdb_atoms,
    serious_contacts,
)


ROOT = Path(os.environ.get(
    "D1_5OHJ_CLOSEOUT_OUTPUT",
    "/home/dev/qfit_unet_data/qfit_audit/d1_5ohj_aprime_phenix_closeout_v1",
))
PHENIX_ROOT = Path("/home/dev/qfit_unet_data/phenix-2.2-6143")
CLASHSCORE = PHENIX_ROOT / "bin" / "phenix.clashscore"
MONOMER_CANDIDATES = (
    PHENIX_ROOT / "lib" / "python3.11" / "site-packages" / "chem_data" / "chemical_components",
    PHENIX_ROOT / "lib" / "python3.11" / "site-packages" / "mmtbx" / "chemical_components",
    PHENIX_ROOT / "chem_data" / "chemical_components",
)
MONOMER_ROOT = next((p for p in MONOMER_CANDIDATES if p.is_dir()), MONOMER_CANDIDATES[0])


def compact(contact: dict) -> dict[str, object]:
    def label(x: dict) -> str:
        return f"{x['chain']}:{x['resname']}{x['resseq']}:{x['name']}" + (f"@{x['altloc']}" if x['altloc'] else "")
    return {
        "a": label(contact["a"]),
        "b": label(contact["b"]),
        "overlap_A": contact["overlap"],
        "graph_distance": contact["graph_distance"],
    }


def audit(label: str, pdb: Path, window_labels: set[tuple[str, str, str, str]]) -> dict[str, object]:
    log = ROOT / "clash_logs" / f"{label}.json.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(CLASHSCORE), str(pdb), "json=True", "keep_hydrogens=False", "condensed_probe=True"]
    with log.open("w") as handle:
        proc = subprocess.run(command, cwd=log.parent, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if proc.returncode:
        raise RuntimeError(f"phenix.clashscore failed for {label}: rc={proc.returncode}")
    obj = load_json_log(log)
    contacts = serious_contacts(obj, Connectivity(pdb, MONOMER_ROOT))

    def inside(contact: dict, side: str) -> bool:
        atom = contact[side]
        return (atom["chain"], atom["resseq"], atom["icode"], atom["resname"], atom["name"]) in window_labels

    internal = [c for c in contacts if inside(c, "a") and inside(c, "b")]
    neighbour = [c for c in contacts if inside(c, "a") != inside(c, "b")]
    summary = obj["summary_results"][""]
    raw_count = int(summary["num_clashes"])
    raw_score = float(summary["clashscore"])
    return {
        "pdb": str(pdb),
        "raw_phenix_clashes": raw_count,
        "raw_phenix_clashscore": raw_score,
        "filtered_clashes": len(contacts),
        "filtered_clashscore": raw_score * len(contacts) / raw_count if raw_count else 0.0,
        "internal_window_clashes": len(internal),
        "window_neighbour_clashes": len(neighbour),
        "worst_internal": [compact(c) for c in internal[:10]],
        "worst_neighbour": [compact(c) for c in neighbour[:10]],
        "protocol": "Phenix/Probe condensed_probe=True, keep_hydrogens=False; CCP4 monomer CIF plus peptide C-N connectivity; remove graph-distance 1-2 and 1-3 pairs",
    }


def main() -> None:
    summary = json.loads((ROOT / "summary.json").read_text())
    pdbs = {label: Path(model["pdb"]) for label, model in summary["models"].items()}
    reference = next(iter(pdbs.values()))
    atoms = read_pdb_atoms(reference)
    window_resnums = {str(i) for i in range(537, 544)}
    window_labels = {
        (a["chain"], a["resseq"], a["icode"], a["resname"], a["name"])
        for a in atoms if a["chain"] == "A" and a["resseq"] in window_resnums
    }
    result = {"status": "running", "site": "5OHJ_A_SER540", "window": "A:537-543", "models": {}}
    (ROOT / "clash_summary_in_progress.json").write_text(json.dumps(result, indent=2) + "\n")
    for label, pdb in pdbs.items():
        result["models"][label] = audit(label, pdb, window_labels)
        (ROOT / "clash_summary_in_progress.json").write_text(json.dumps(result, indent=2) + "\n")
    result["status"] = "complete"
    (ROOT / "clash_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
