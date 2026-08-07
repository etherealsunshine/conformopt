#!/usr/bin/env python3
"""Build a deposited-backbone-altloc candidate list without performance data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median

BACKBONE = ("N", "CA", "C", "O")
AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "MSE",
}
FLOAT_RE = r"([-+]?\d+(?:\.\d+)?)"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temp = Path(handle.name)
    os.replace(temp, path)


Coord = tuple[float, float, float]


def distance(a: Coord, b: Coord) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def angle_deg(a: Coord, b: Coord, c: Coord, d: Coord) -> float:
    va = [a[i] - b[i] for i in range(3)]
    vb = [c[i] - d[i] for i in range(3)]
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if na == 0 or nb == 0:
        return float("nan")
    cosine = max(-1.0, min(1.0, sum(x * y for x, y in zip(va, vb)) / (na * nb)))
    return math.degrees(math.acos(cosine))


def resolve_resolution(text: str) -> tuple[float | None, str]:
    match = re.search(
        r"RESOLUTION\s+RANGE\s+HIGH\s+\(ANGSTROMS\)\s*:\s*" + FLOAT_RE,
        text,
        re.I,
    )
    if match:
        return float(match.group(1)), "remark_3"
    match = re.search(r"REMARK\s+2\s+RESOLUTION\.\s+" + FLOAT_RE + r"\s+ANGSTROMS", text, re.I)
    if match:
        return float(match.group(1)), "remark_2"
    return None, "missing"


def is_xray(text: str) -> bool:
    return bool(re.search(r"^EXPDTA\s+.*X-RAY DIFFRACTION", text, re.I | re.M))


def sf_path(data_root: Path, pdb_id: str) -> str | None:
    for split in ("train", "test"):
        for suffix in ("-sf.cif", ".mtz", "-sf.mtz"):
            path = data_root / "cache" / split / "structure_factors" / f"{pdb_id}{suffix}"
            if path.exists():
                return str(path)
            path = data_root / "cache" / split / "mtz" / f"{pdb_id}{suffix}"
            if path.exists():
                return str(path)
    return None


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q1": None, "median": None, "q3": None, "max": None}
    ordered = sorted(values)
    q1 = ordered[(len(ordered) - 1) // 4]
    q3 = ordered[(3 * (len(ordered) - 1)) // 4]
    return {
        "min": ordered[0],
        "q1": q1,
        "median": median(ordered),
        "q3": q3,
        "max": ordered[-1],
    }


def scan_file(path: Path, data_root: Path, split: str) -> list[dict[str, object]]:
    text = path.read_text(errors="replace")
    pdb_id = path.stem.upper()
    resolution, resolution_source = resolve_resolution(text)
    sf = sf_path(data_root, pdb_id)
    # The prepared PDBs are atom-only files and often omit EXPDTA.  A deposited
    # structure-factor file is therefore the X-ray evidence for those records.
    if resolution is None or not (0.8 <= resolution <= 2.0) or (not is_xray(text) and sf is None):
        return []

    # These synchronized source files are ordinary fixed-column PDB records.
    # Parsing only the four backbone atoms avoids the much slower full Gemmi
    # structure construction while retaining the deposited coordinates and
    # occupancies needed for this metadata-only gate.
    chains: dict[str, dict[tuple[int, str], dict[str, dict[str, tuple[Coord, float]]]]] = {}
    residue_names: dict[tuple[str, tuple[int, str]], str] = {}
    for line in text.splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip()
        altloc = line[16:17].strip()
        resname = line[17:20].strip().upper()
        chain_name = line[21:22].strip()
        if atom_name not in BACKBONE or altloc not in {"A", "B"} or resname not in AMINO_ACIDS:
            continue
        try:
            resnum = int(line[22:26])
            icode = line[26:27].strip()
            coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            occupancy = float(line[54:60]) if line[54:60].strip() else 1.0
        except (ValueError, IndexError):
            continue
        key = (resnum, icode)
        residue_names.setdefault((chain_name, key), resname)
        residue = chains.setdefault(chain_name, {}).setdefault(key, {"A": {}, "B": {}})
        previous = residue[altloc].get(atom_name)
        if previous is None or occupancy > previous[1]:
            residue[altloc][atom_name] = (coord, occupancy)

    rows: list[dict[str, object]] = []
    for chain_name, chain_residues in chains.items():
        residues = list(chain_residues.items())
        for index, ((resnum, insertion_code), atoms) in enumerate(residues):
            common = [name for name in BACKBONE if name in atoms["A"] and name in atoms["B"]]
            if not common:
                continue
            deviations = {name: distance(atoms["A"][name][0], atoms["B"][name][0]) for name in common}
            max_deviation = max(deviations.values())
            if max_deviation <= 0.10:
                continue
            occ_a = median([atoms["A"][name][1] for name in common])
            occ_b = median([atoms["B"][name][1] for name in common])
            c_c = distance(atoms["A"]["C"][0], atoms["B"]["C"][0]) if "C" in common else float("nan")
            o_o = distance(atoms["A"]["O"][0], atoms["B"]["O"][0]) if "O" in common else float("nan")
            rotation = (
                angle_deg(atoms["A"]["C"][0], atoms["A"]["O"][0], atoms["B"]["C"][0], atoms["B"]["O"][0])
                if "C" in common and "O" in common
                else float("nan")
            )
            flank_distances: list[float] = []
            for flank_index in (index - 1, index + 1):
                if not (0 <= flank_index < len(residues)):
                    continue
                flank = residues[flank_index][1]
                if "CA" in flank["A"] and "CA" in flank["B"]:
                    flank_distances.append(distance(flank["A"]["CA"][0], flank["B"]["CA"][0]))
            prev_ca = flank_distances[0] if len(flank_distances) > 0 else float("nan")
            next_ca = flank_distances[1] if len(flank_distances) > 1 else float("nan")
            passes_flip = (
                math.isfinite(o_o)
                and math.isfinite(c_c)
                and math.isfinite(rotation)
                and o_o > c_c + 1.0
                and rotation >= 90.0
                and math.isfinite(prev_ca)
                and math.isfinite(next_ca)
                and prev_ca < 1.5
                and next_ca < 1.5
            )
            rows.append({
                "pdb_id": pdb_id,
                "split": split,
                "source_path": str(path),
                "chain": chain_name,
                "resnum": resnum,
                "insertion_code": insertion_code,
                "resname": residue_names[(chain_name, (resnum, insertion_code))],
                "resolution": resolution,
                "resolution_source": resolution_source,
                "is_xray": is_xray(text),
                "xray_inferred_from_structure_factors": not is_xray(text) and sf is not None,
                "structure_factors": sf is not None,
                "structure_factor_path": sf or "",
                "occupancy_a": occ_a,
                "occupancy_b": occ_b,
                "n_backbone_common": len(common),
                "max_backbone_deviation": max_deviation,
                "n_deviation": deviations.get("N", float("nan")),
                "ca_deviation": deviations.get("CA", float("nan")),
                "c_deviation": deviations.get("C", float("nan")),
                "o_deviation": deviations.get("O", float("nan")),
                "o_o_distance": o_o,
                "c_c_distance": c_c,
                "o_minus_c_distance": o_o - c_c if math.isfinite(o_o) and math.isfinite(c_c) else float("nan"),
                "carbonyl_rotation_deg": rotation,
                "prev_flanking_ca_distance": prev_ca,
                "next_flanking_ca_distance": next_ca,
                "passes_flip_filter": passes_flip,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sources = [(path, split) for split in ("train", "test") for path in sorted((args.data_root / split).glob("*.pdb"))]
    rows: list[dict[str, object]] = []
    errors = 0
    # Parsing is CPU-heavy Python work. Keep checkpoint writes in the parent,
    # while distributing independent PDBs across pod CPUs.
    worker_count = max(1, args.workers)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = executor.map(
            scan_file,
            (path for path, _ in sources),
            (args.data_root for _ in sources),
            (split for _, split in sources),
            chunksize=4,
        )
        for count, result in enumerate(futures, start=1):
            try:
                rows.extend(result)
            except Exception:
                errors += 1
            if count % 50 == 0:
                atomic_json(args.output_dir / "progress.json", {"processed": count, "total": len(sources), "rows": len(rows), "errors": errors})
    rows = [row for row in rows if row["structure_factors"]]
    rows.sort(key=lambda row: (float(row["max_backbone_deviation"]), row["pdb_id"], row["chain"], int(row["resnum"])))
    summary = {
        "source_files": len(sources),
        "parse_or_scan_errors": errors,
        "eligible_sites_with_structure_factors": len(rows),
        "flip_filter_sites": sum(bool(row["passes_flip_filter"]) for row in rows),
        "backbone_deviation_A": summarize([float(row["max_backbone_deviation"]) for row in rows]),
        "resolution_A": summarize([float(row["resolution"]) for row in rows]),
        "flip_filter_definition": "O-O > C-C + 1.0 A; C->O vector rotation >= 90 deg; both flanking CA A/B distances < 1.5 A",
        "selection_rule": "X-ray evidence from EXPDTA or deposited structure factors, 0.8 <= resolution <= 2.0 A, common A/B backbone atom deviation > 0.10 A, structure-factor file present",
        "pdbe_used": False,
    }
    atomic_csv(args.output_dir / "candidate_sites.csv", rows)
    atomic_csv(args.output_dir / "flip_filter_sites.csv", [row for row in rows if row["passes_flip_filter"]])
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "progress.json", {"processed": len(sources), "total": len(sources), "rows": len(rows), "errors": errors, "status": "complete"})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
