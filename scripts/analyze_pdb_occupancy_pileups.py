#!/usr/bin/env python3
"""Measure deposited protein minor-altloc occupancy pileups.

The primary set is the Keedy et al. 2018 PTP1B deposition series plus three
other PDB entries whose RCSB titles explicitly identify qFit multiconformer
models (5IVD, 5IVI, and 6NI9).  Controls are deterministic X-ray structures
selected from the RCSB search API, screened to remove qFit-labelled entries,
and required to contain at least one protein residue with multiple deposited
altlocs.

This is a deposited-coordinate assay.  It cannot recover occupancies that
were relaxed by a later refinement step, and the control screen is provenance
by metadata rather than a proof that qFit was never used privately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

try:
    import gemmi
except ImportError as exc:  # pragma: no cover - runtime dependency on pod
    raise SystemExit("gemmi is required; run this on the qfit audit pod") from exc


KEEDY_PTP1B = (
    "6B90 6B8E 6B8T 6B8X 6B8Z 6BAI 6B95 "
    "5QDE 5QDF 5QDG 5QDH 5QDI 5QDJ 5QDK 5QDL 5QDM 5QDN 5QDO 5QDP "
    "5QDQ 5QDR 5QDS 5QDT 5QDU 5QDV 5QDW 5QDX 5QDY 5QDZ 5QE0 5QE1 "
    "5QE2 5QE3 5QE4 5QE5 5QE6 5QE7 5QE8 5QE9 5QEA 5QEB 5QEC 5QED "
    "5QEE 5QEF 5QEG 5QEH 5QEI 5QEJ 5QEK 5QEL 5QEM 5QEN 5QEO 5QEP "
    "5QEQ 5QER 5QES 5QET 5QEU 5QEV 5QEW 5QEX 5QEY 5QEZ 5QF0 5QF1 "
    "5QF2 5QF3 5QF4 5QF5 5QF6 5QF7 5QF8 5QF9 5QFA 5QFB 5QFC 5QFD "
    "5QFE 5QFF 5QFG 5QFH 5QFI 5QFJ 5QFK 5QFL 5QFM 5QFN 5QFO 5QFP "
    "5QFQ 5QFR 5QFS 5QFT 5QFU 5QFV 5QFW 5QFX 5QFY 5QFZ 5QG0 5QG1 "
    "5QG2 5QG3 5QG4 5QG5 5QG6 5QG7 5QG8 5QG9 5QGA 5QGB 5QGC 5QGD "
    "5QGE 5QGF"
).split()

OTHER_QFIT = ["5IVD", "5IVI", "6NI9"]
QFIT_IDS = sorted(set(KEEDY_PTP1B + OTHER_QFIT))
TARGETS = (0.20, 0.25, 0.33, 0.50)
AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
}


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "qfitonsteroids-occupancy-audit/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def download_cif(pdb_id: str, coordinate_dir: Path) -> Path:
    path = coordinate_dir / f"{pdb_id.lower()}.cif"
    if not path.exists():
        path.write_bytes(fetch_bytes(f"https://files.rcsb.org/download/{pdb_id}.cif"))
    return path


def cif_text_metadata(path: Path) -> str:
    block = gemmi.cif.read(str(path)).sole_block()
    values = []
    for key in ("_struct.title", "_citation.title", "_audit_author.name"):
        try:
            value = block.find_value(key)
        except Exception:
            value = ""
        if value:
            values.append(str(value))
    return " ".join(values)


def extract_minor_occupancies(path: Path) -> tuple[list[float], int]:
    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        return [], 0
    occupancies: list[float] = []
    residue_count = 0
    for model in [structure[0]]:
        for chain in model:
            for residue in chain:
                if residue.name.strip() not in AMINO_ACIDS:
                    continue
                groups: dict[str, list[float]] = {}
                for atom in residue:
                    altloc = str(atom.altloc).strip()
                    if not altloc or atom.occ <= 0.0:
                        continue
                    groups.setdefault(altloc, []).append(float(atom.occ))
                groups = {key: values for key, values in groups.items() if len(values) >= 2}
                if len(groups) < 2:
                    continue
                residue_count += 1
                representative = sorted(
                    (float(np.median(values)) for values in groups.values()), reverse=True
                )
                occupancies.extend(representative[1:])
    return occupancies, residue_count


def entry_record(pdb_id: str, path: Path, group: str) -> dict[str, Any]:
    occupancies, residues = extract_minor_occupancies(path)
    return {
        "pdb_id": pdb_id,
        "group": group,
        "metadata_sha256": hashlib.sha256(cif_text_metadata(path).encode()).hexdigest(),
        "n_multiconformer_residues": residues,
        "n_minor_occupancies": len(occupancies),
        "minor_occupancies": occupancies,
    }


def rcsb_candidate_ids(rows: int = 2000) -> list[str]:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text", "parameters": {
                    "attribute": "rcsb_entry_info.experimental_method",
                    "operator": "exact_match", "value": "X-RAY DIFFRACTION",
                }},
                {"type": "terminal", "service": "attribute", "parameters": {
                    "attribute": "rcsb_accession_info.initial_release_date",
                    "operator": "range", "value": {"from": "2016-01-01", "to": "2024-12-31", "include_lower": True, "include_upper": True},
                }},
                {"type": "terminal", "service": "attribute", "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "range", "value": {"from": 1.0, "to": 2.5, "include_lower": True, "include_upper": True},
                }},
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": rows}},
    }
    request = Request(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "qfitonsteroids-occupancy-audit/1.0"},
    )
    payload = json.loads(urlopen(request, timeout=60).read())
    return [str(result["identifier"]).upper() for result in payload.get("result_set", [])]


def select_controls(coordinate_dir: Path, count: int, seed: int) -> tuple[list[dict[str, Any]], list[str]]:
    candidates = rcsb_candidate_ids()
    rng = random.Random(seed)
    rng.shuffle(candidates)
    controls: list[dict[str, Any]] = []
    rejected_qfit: list[str] = []
    qfit_set = set(QFIT_IDS)
    for pdb_id in candidates:
        if pdb_id in qfit_set:
            continue
        try:
            path = download_cif(pdb_id, coordinate_dir)
            metadata = cif_text_metadata(path).lower()
            if "qfit" in metadata:
                rejected_qfit.append(pdb_id)
                continue
            occupancies, residues = extract_minor_occupancies(path)
        except (HTTPError, URLError, OSError, RuntimeError, ValueError):
            continue
        if residues == 0 or not occupancies:
            continue
        controls.append({
            "pdb_id": pdb_id,
            "group": "non-qFit-control",
            "metadata_sha256": hashlib.sha256(metadata.encode()).hexdigest(),
            "n_multiconformer_residues": residues,
            "n_minor_occupancies": len(occupancies),
            "minor_occupancies": occupancies,
        })
        if len(controls) >= count:
            break
    if len(controls) < count:
        raise RuntimeError(f"only found {len(controls)} controls with protein altlocs; requested {count}")
    return controls, rejected_qfit


def target_counts(records: list[dict[str, Any]], target: float, tolerance: float = 0.0051) -> dict[str, Any]:
    all_values = [value for record in records for value in record["minor_occupancies"]]
    hit = sum(abs(value - target) <= tolerance for value in all_values)
    # PDB occupancies are normally written to two decimals.  The +/-0.0051
    # window tests the displayed target bin without depending on float noise.
    bins = Counter(round(value, 2) for value in all_values)
    neighbors = [bins.get(round(target + delta, 2), 0) for delta in (-0.02, -0.01, 0.01, 0.02)]
    return {
        "target": target,
        "tolerance": tolerance,
        "n_minor_occupancies": len(all_values),
        "n_hits": hit,
        "fraction": hit / len(all_values) if all_values else float("nan"),
        "target_bin_count": bins.get(round(target, 2), 0),
        "neighbor_bin_counts": neighbors,
        "neighbor_bin_mean": float(np.mean(neighbors)),
        "target_to_neighbor_ratio": (bins.get(round(target, 2), 0) / float(np.mean(neighbors))
                                      if np.mean(neighbors) > 0 else None),
        "n_entries_with_hit": sum(
            any(abs(value - target) <= tolerance for value in record["minor_occupancies"])
            for record in records
        ),
    }


def bootstrap_entry_fraction(records_a: list[dict[str, Any]], records_b: list[dict[str, Any]], target: float, seed: int, draws: int = 10000) -> dict[str, float]:
    def fractions(records: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray([
            np.mean([abs(value - target) <= 0.0051 for value in record["minor_occupancies"]])
            for record in records if record["minor_occupancies"]
        ], dtype=float)
    a, b = fractions(records_a), fractions(records_b)
    rng = np.random.default_rng(seed)
    differences = np.empty(draws)
    for index in range(draws):
        differences[index] = rng.choice(a, size=len(a), replace=True).mean() - rng.choice(b, size=len(b), replace=True).mean()
    observed = float(a.mean() - b.mean())
    return {
        "qfit_entry_fraction_mean": float(a.mean()),
        "control_entry_fraction_mean": float(b.mean()),
        "observed_difference": observed,
        "bootstrap_95pct_interval_low": float(np.quantile(differences, 0.025)),
        "bootstrap_95pct_interval_high": float(np.quantile(differences, 0.975)),
        "bootstrap_two_sided_p": float(np.mean(np.abs(differences) >= abs(observed))),
        "qfit_entries_with_minor_occ": int(len(a)),
        "control_entries_with_minor_occ": int(len(b)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coordinate-root", type=Path, required=True)
    parser.add_argument("--control-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    coordinate_dir = args.coordinate_root / "coordinates"
    coordinate_dir.mkdir(parents=True, exist_ok=True)

    qfit_records = []
    failures = []
    for pdb_id in QFIT_IDS:
        try:
            qfit_records.append(entry_record(pdb_id, download_cif(pdb_id, coordinate_dir), "qFit"))
        except (HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
            failures.append({"pdb_id": pdb_id, "error": str(exc)})
    controls, rejected_qfit = select_controls(coordinate_dir, args.control_count, args.seed)

    summaries = []
    for group, records in (("qFit", qfit_records), ("non-qFit-control", controls)):
        for target in TARGETS:
            summary = {"group": group, **target_counts(records, target)}
            if group == "qFit":
                summary["comparison"] = bootstrap_entry_fraction(qfit_records, controls, target, args.seed + int(target * 100))
            summaries.append(summary)

    result = {
        "status": "complete",
        "seed": args.seed,
        "qfit_ids": QFIT_IDS,
        "keedy_ptp1b_ids": KEEDY_PTP1B,
        "other_qfit_ids": OTHER_QFIT,
        "n_qfit_entries_downloaded": len(qfit_records),
        "n_qfit_entries_requested": len(QFIT_IDS),
        "n_control_entries": len(controls),
        "control_selection": {
            "source": "RCSB Search API: X-ray, 1.0-2.5 A, initial release 2016-2024; deterministic shuffle",
            "requirements": "at least one protein multiconformer residue with >=2 atoms per altloc",
            "qfit_exclusion": "explicit qFit IDs plus metadata containing qfit",
            "rejected_qfit_metadata_ids": rejected_qfit,
        },
        "coordinate_parser": "gemmi; first model; amino-acid residues; altloc groups with >=2 atoms; residue median occupancy; all non-major altlocs counted as minor",
        "failures": failures,
        "entry_records": qfit_records + controls,
        "target_summaries": summaries,
    }
    (args.output / "occupancy_pileup.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (args.output / "entry_records.csv").open("w", newline="") as handle:
        fields = ["pdb_id", "group", "n_multiconformer_residues", "n_minor_occupancies"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in qfit_records + controls)
    with (args.output / "target_summaries.csv").open("w", newline="") as handle:
        fields = ["group", "target", "n_minor_occupancies", "n_hits", "fraction", "target_bin_count", "neighbor_bin_counts", "neighbor_bin_mean", "target_to_neighbor_ratio", "n_entries_with_hit"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps({"qfit": len(qfit_records), "controls": len(controls), "failures": failures, "summaries": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
