from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np

from .data_pipeline import _sidechain_atoms
from .dataset import manifest_path, read_manifest


METRICS = (
    "raw_local_mse",
    "denoised_local_mse",
    "raw_local_pearson",
    "denoised_local_pearson",
    "raw_top10_overlap",
    "denoised_top10_overlap",
    "local_mse_fractional_improvement",
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def occupancy_bin(value: float) -> str:
    if value < 0.10:
        return "<0.10"
    if value < 0.20:
        return "0.10-0.20"
    if value < 0.30:
        return "0.20-0.30"
    if value < 0.40:
        return "0.30-0.40"
    return ">=0.40"


def _altloc(atom: gemmi.Atom) -> str:
    return "" if atom.altloc in ("\x00", " ") else atom.altloc


def _residue_lookup(structure: gemmi.Structure) -> dict[tuple[str, int, str, str], gemmi.Residue]:
    return {
        (chain.name, residue.seqid.num, residue.seqid.icode, residue.name): residue
        for chain in structure[0]
        for residue in chain
    }


def _site_occupancies(residue: gemmi.Residue) -> dict:
    by_altloc: dict[str, list[float]] = defaultdict(list)
    for atom in _sidechain_atoms(residue):
        label = _altloc(atom)
        if label:
            by_altloc[label].append(float(atom.occ))
    occupancies = {
        label: float(np.median(values)) for label, values in by_altloc.items()
    }
    if len(occupancies) < 2:
        raise ValueError(f"expected at least two sidechain altlocs, found {occupancies}")
    values = sorted(occupancies.values())
    return {
        "altloc_count": len(values),
        "minor_occupancy": values[0],
        "major_occupancy": values[-1],
        "occupancy_sum": sum(values),
        "occupancies": ";".join(
            f"{label}:{occupancies[label]:.6g}" for label in sorted(occupancies)
        ),
    }


def _percentiles(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        f"p{percentile:02d}": float(np.percentile(values, percentile))
        for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
    }


def _group_summary(label: str, rows: list[dict]) -> dict:
    raw_corr = np.asarray([row["raw_local_pearson"] for row in rows])
    den_corr = np.asarray([row["denoised_local_pearson"] for row in rows])
    raw_peak = np.asarray([row["raw_top10_overlap"] for row in rows])
    den_peak = np.asarray([row["denoised_top10_overlap"] for row in rows])
    improvement = np.asarray([
        row["local_mse_fractional_improvement"] for row in rows
    ])
    return {
        "group": label,
        "sites": len(rows),
        "mean_raw_local_pearson": float(raw_corr.mean()),
        "mean_denoised_local_pearson": float(den_corr.mean()),
        "mean_local_pearson_delta": float((den_corr - raw_corr).mean()),
        "median_denoised_local_pearson": float(np.median(den_corr)),
        "fraction_denoised_local_pearson_below_0.80": float((den_corr < 0.80).mean()),
        "fraction_denoised_local_pearson_below_0.90": float((den_corr < 0.90).mean()),
        "mean_raw_top10_overlap": float(raw_peak.mean()),
        "mean_denoised_top10_overlap": float(den_peak.mean()),
        "mean_top10_overlap_delta": float((den_peak - raw_peak).mean()),
        "fraction_denoised_top10_overlap_below_0.50": float((den_peak < 0.50).mean()),
        "mean_local_mse_fractional_improvement": float(improvement.mean()),
    }


def _summarize(rows: list[dict]) -> dict:
    raw_corr = np.asarray([row["raw_local_pearson"] for row in rows])
    den_corr = np.asarray([row["denoised_local_pearson"] for row in rows])
    raw_peak = np.asarray([row["raw_top10_overlap"] for row in rows])
    den_peak = np.asarray([row["denoised_top10_overlap"] for row in rows])
    return {
        "status": "complete",
        "sites": len(rows),
        "proteins": len({row["pdb_id"] for row in rows}),
        "distributions": {
            key: _percentiles(rows, key) for key in METRICS
        },
        "paired_changes": {
            "local_pearson_better": int((den_corr > raw_corr).sum()),
            "local_pearson_equal": int(np.isclose(den_corr, raw_corr).sum()),
            "local_pearson_worse": int((den_corr < raw_corr).sum()),
            "top10_overlap_better": int((den_peak > raw_peak).sum()),
            "top10_overlap_equal": int(np.isclose(den_peak, raw_peak).sum()),
            "top10_overlap_worse": int((den_peak < raw_peak).sum()),
        },
        "remaining_error_tail": {
            "denoised_local_pearson_below_0.80": int((den_corr < 0.80).sum()),
            "denoised_local_pearson_below_0.90": int((den_corr < 0.90).sum()),
            "denoised_top10_overlap_below_0.50": int((den_peak < 0.50).sum()),
            "denoised_top10_overlap_below_0.75": int((den_peak < 0.75).sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratify frozen U-Net metrics across held-out altloc sites"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with args.metrics.open(newline="") as handle:
        metric_rows = [row for row in csv.DictReader(handle) if row["is_altloc"].lower() == "true"]
    record_by_key = {
        record["key"]: record
        for record in read_manifest(manifest_path(args.data_root, "test", "crystal"))
    }

    enriched_path = args.output / "altloc_metrics_enriched.csv"
    rows: list[dict] = []
    if enriched_path.exists():
        with enriched_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key in METRICS + (
                "altloc_count", "minor_occupancy", "major_occupancy", "occupancy_sum",
            ):
                row[key] = float(row[key])
    completed = {row["key"] for row in rows}
    metrics_by_protein: dict[str, list[dict]] = defaultdict(list)
    for row in metric_rows:
        if row["key"] not in completed:
            metrics_by_protein[row["pdb_id"]].append(row)

    for protein_index, pdb_id in enumerate(sorted(metrics_by_protein), start=1):
        structure = gemmi.read_structure(str(args.data_root / "test" / f"{pdb_id.lower()}.pdb"))
        residues = _residue_lookup(structure)
        for metric_row in metrics_by_protein[pdb_id]:
            record = record_by_key[metric_row["key"]]
            residue = residues[(
                record["chain"], int(record["residue_number"]),
                record["insertion_code"], record["residue_name"],
            )]
            occupancy = _site_occupancies(residue)
            row = {
                "key": metric_row["key"],
                "pdb_id": pdb_id,
                "residue_name": record["residue_name"],
                **occupancy,
                "occupancy_bin": occupancy_bin(occupancy["minor_occupancy"]),
                **{key: float(metric_row[key]) for key in METRICS},
            }
            rows.append(row)
        _atomic_csv(enriched_path, rows)
        _atomic_json(args.output / "stage_manifest.json", {
            "status": "running",
            "completed_proteins": protein_index,
            "completed_sites": len(rows),
            "total_sites": len(metric_rows),
        })
        print(json.dumps({
            "protein": pdb_id, "completed_sites": len(rows),
            "total_sites": len(metric_rows),
        }), flush=True)

    summary = _summarize(rows)
    _atomic_json(args.output / "altloc_summary.json", summary)
    for field, filename in (
        ("residue_name", "altloc_by_residue.csv"),
        ("occupancy_bin", "altloc_by_minor_occupancy.csv"),
        ("altloc_count", "altloc_by_conformer_count.csv"),
    ):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        _atomic_csv(
            args.output / filename,
            [_group_summary(label, groups[label]) for label in sorted(groups)],
        )
    _atomic_csv(
        args.output / "worst_denoised_local_pearson.csv",
        sorted(rows, key=lambda row: row["denoised_local_pearson"])[:100],
    )
    _atomic_csv(
        args.output / "largest_local_pearson_regressions.csv",
        sorted(
            rows,
            key=lambda row: row["denoised_local_pearson"] - row["raw_local_pearson"],
        )[:100],
    )
    _atomic_json(args.output / "stage_manifest.json", {"status": "complete"})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
