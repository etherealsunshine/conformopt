#!/usr/bin/env python3
"""D3 audit: ANISOU prevalence versus deposited X-ray resolution."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import tempfile
from pathlib import Path


AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "MSE",
}
FLOAT_RE = r"([-+]?\d+(?:\.\d+)?)"
MAIN_CHAIN_CB = {"N", "CA", "C", "O", "CB"}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp = Path(handle.name)
    tmp.replace(path)


def resolution(text: str) -> float | None:
    for pattern in (
        r"RESOLUTION\s+RANGE\s+HIGH\s+\(ANGSTROMS\)\s*:\s*" + FLOAT_RE,
        r"REMARK\s+2\s+RESOLUTION\.\s+" + FLOAT_RE + r"\s+ANGSTROMS",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1))
    return None


def classify(path: Path) -> dict[str, object] | None:
    text = path.read_text(errors="replace")
    value = resolution(text)
    if value is None or not (1.0 <= value <= 2.0):
        return None
    total = 0
    protein = 0
    main_chain_cb = 0
    other_protein = 0
    atom_names: dict[str, int] = {}
    for line in text.splitlines():
        if not line.startswith("ANISOU"):
            continue
        total += 1
        atom_name = line[12:16].strip()
        resname = line[17:20].strip().upper()
        record = line[0:6].strip()
        atom_names[atom_name] = atom_names.get(atom_name, 0) + 1
        if record == "ATOM" and resname in AMINO_ACIDS:
            protein += 1
            if atom_name in MAIN_CHAIN_CB:
                main_chain_cb += 1
            else:
                other_protein += 1
    if total == 0:
        category = "none"
    elif protein == 0:
        category = "only waters/ions/ligands"
    else:
        category = "protein main-chain+CB"
    return {
        "pdb_id": path.stem.upper(),
        "source_path": str(path),
        "resolution_A": value,
        "anisou_records": total,
        "protein_anisou_records": protein,
        "protein_main_chain_cb_records": main_chain_cb,
        "protein_other_atom_records": other_protein,
        "anisou_atom_names": atom_names,
        "category": category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    candidates = []
    for split in ("train", "test"):
        for path in sorted((args.data_root / split).glob("*.pdb")):
            record = classify(path)
            if record is not None:
                record["split"] = split
                candidates.append(record)
    if len(candidates) < args.sample_size:
        raise RuntimeError(f"only {len(candidates)} qualifying entries, need {args.sample_size}")
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(candidates, args.sample_size), key=lambda row: row["pdb_id"])

    bins = []
    lower = 1.0
    while lower < 2.0:
        upper = round(lower + 0.2, 1)
        in_bin = [row for row in sample if lower <= row["resolution_A"] < upper or (upper == 2.0 and row["resolution_A"] == upper)]
        bins.append({
            "low_A": lower,
            "high_A": upper,
            "n": len(in_bin),
            "category_counts": {category: sum(row["category"] == category for row in in_bin) for category in ("none", "only waters/ions/ligands", "protein main-chain+CB")},
        })
        lower = upper
    thresholds = {
        str(threshold): {
            "n_at_or_better": sum(row["resolution_A"] <= threshold for row in sample),
            "fraction_at_or_better": sum(row["resolution_A"] <= threshold for row in sample) / len(sample),
        }
        for threshold in (1.45, 1.50, 1.55)
    }
    summary = {
        "seed": args.seed,
        "qualifying_population": len(candidates),
        "sample_size": len(sample),
        "resolution_range_A": [1.0, 2.0],
        "category_counts": {category: sum(row["category"] == category for row in sample) for category in ("none", "only waters/ions/ligands", "protein main-chain+CB")},
        "resolution_bins_A": bins,
        "threshold_comparison": thresholds,
        "classification_note": "protein main-chain+CB means at least one ANISOU on an ATOM record for a standard amino-acid residue; protein_other_atom_records is retained to expose whether the ANISOU extends beyond N/CA/C/O/CB.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in sample for key in row if key != "anisou_atom_names"})
    with (args.output_dir / "sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: value for key, value in row.items() if key != "anisou_atom_names"} for row in sample)
    atomic_json(args.output_dir / "sample_full.json", sample)
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
