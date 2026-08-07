from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from .diagnose_frozen_tmol_gate import pearson_and_spearman, rmsd_bin
from .summarize_endpoint_audit import select_assigned_pair


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def summarize_assignment(rows: list[dict[str, str]]) -> dict[str, object]:
    margins = np.asarray([float(row["tmol_delta_vs_matched_AB"]) for row in rows])
    rmsds = np.asarray([
        float(row[f"rmsd_to_{row['assignment']}_conventional"]) for row in rows
    ])
    pearson = spearman = None
    if len(rows) >= 2 and np.std(margins) > 0 and np.std(rmsds) > 0:
        pearson, spearman = pearson_and_spearman(rmsds, margins)
    return {
        "conformers": len(rows),
        "tmol_pass_at_0": int((margins <= 0.0).sum()),
        "tmol_pass_rate_at_0": float((margins <= 0.0).mean()),
        "margin_min": float(margins.min()),
        "margin_q05": float(np.quantile(margins, 0.05)),
        "margin_q25": float(np.quantile(margins, 0.25)),
        "margin_median": float(np.median(margins)),
        "margin_mean": float(margins.mean()),
        "margin_q75": float(np.quantile(margins, 0.75)),
        "margin_q95": float(np.quantile(margins, 0.95)),
        "margin_max": float(margins.max()),
        "matched_rmsd_mean": float(rmsds.mean()),
        "margin_vs_rmsd_pearson": pearson if pearson is not None else "",
        "margin_vs_rmsd_spearman": spearman if spearman is not None else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break matched tmol margins down by deposited assignment."
    )
    parser.add_argument("--strict-table", type=Path, action="append", required=True)
    parser.add_argument("--site", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    wanted = set(args.site)
    source = [
        row
        for table in args.strict_table
        for row in _read_csv(table)
        if row["site"] in wanted and row["assignment"] in {"A", "B"}
        and math.isfinite(float(row["tmol_delta_vs_matched_AB"]))
    ]
    if {row["site"] for row in source} != wanted:
        raise ValueError("one or more requested sites have no finite matched rows")

    raw = []
    summary = []
    bins = []
    pair_rows = []
    pair_summary = []
    for site in sorted(wanted):
        for assignment in ("A", "B"):
            rows = [
                row for row in source
                if row["site"] == site and row["assignment"] == assignment
            ]
            if not rows:
                continue
            summary.append({
                "site": site,
                "assignment": assignment,
                **summarize_assignment(rows),
            })
            for row in rows:
                matched_rmsd = float(
                    row[f"rmsd_to_{assignment}_conventional"]
                )
                raw.append({
                    "candidate_id": row["candidate_id"],
                    "site": site,
                    "start": int(row["start"]),
                    "conformer": int(row["conformer"]),
                    "assignment": assignment,
                    "occupancy": float(row["occupancy"]),
                    "matched_rmsd": matched_rmsd,
                    "rmsd_bin": rmsd_bin(matched_rmsd),
                    "tmol_margin": float(row["tmol_delta_vs_matched_AB"]),
                    "tmol_pass_at_0": (
                        float(row["tmol_delta_vs_matched_AB"]) <= 0.0
                    ),
                })
            for label in ("<=0.1", "0.1-0.3", "0.3-0.6", "0.6-1.0"):
                values = np.asarray([
                    item["tmol_margin"] for item in raw
                    if item["site"] == site
                    and item["assignment"] == assignment
                    and item["rmsd_bin"] == label
                ])
                if len(values):
                    bins.append({
                        "site": site,
                        "assignment": assignment,
                        "rmsd_bin": label,
                        "conformers": int(len(values)),
                        "margin_mean": float(values.mean()),
                        "margin_median": float(np.median(values)),
                        "margin_q05": float(np.quantile(values, 0.05)),
                        "margin_q95": float(np.quantile(values, 0.95)),
                        "tmol_pass_at_0": int((values <= 0.0).sum()),
                        "tmol_pass_rate_at_0": float((values <= 0.0).mean()),
                    })
        site_rows = [row for row in source if row["site"] == site]
        selected_rows = []
        for start in sorted({int(row["start"]) for row in site_rows}):
            pair = select_assigned_pair([
                row for row in site_rows if int(row["start"]) == start
            ])
            if pair is None:
                continue
            margin_a = float(pair["A"]["tmol_delta_vs_matched_AB"])
            margin_b = float(pair["B"]["tmol_delta_vs_matched_AB"])
            selected_rows.append((margin_a, margin_b))
            pair_rows.append({
                "site": site,
                "start": start,
                "candidate_A": pair["A"]["candidate_id"],
                "candidate_B": pair["B"]["candidate_id"],
                "margin_A": margin_a,
                "margin_B": margin_b,
                "A_pass_at_0": margin_a <= 0.0,
                "B_pass_at_0": margin_b <= 0.0,
                "both_pass_at_0": margin_a <= 0.0 and margin_b <= 0.0,
            })
        if selected_rows:
            values = np.asarray(selected_rows)
            pass_a = values[:, 0] <= 0.0
            pass_b = values[:, 1] <= 0.0
            correlation = (
                float(np.corrcoef(values[:, 0], values[:, 1])[0, 1])
                if len(values) >= 2
                and np.std(values[:, 0]) > 0
                and np.std(values[:, 1]) > 0
                else ""
            )
            pair_summary.append({
                "site": site,
                "complete_assigned_pairs": len(values),
                "A_pass_pairs": int(pass_a.sum()),
                "B_pass_pairs": int(pass_b.sum()),
                "both_pass_pairs": int((pass_a & pass_b).sum()),
                "A_only_pass_pairs": int((pass_a & ~pass_b).sum()),
                "B_only_pass_pairs": int((~pass_a & pass_b).sum()),
                "neither_pass_pairs": int((~pass_a & ~pass_b).sum()),
                "expected_both_pass_if_independent": float(
                    len(values) * pass_a.mean() * pass_b.mean()
                ),
                "paired_margin_pearson": correlation,
            })

    args.output.mkdir(parents=True)
    _atomic_csv(args.output / "assignment_summary.csv", summary)
    _atomic_csv(args.output / "assignment_margin_vs_rmsd.csv", raw)
    _atomic_csv(args.output / "assignment_rmsd_bins.csv", bins)
    _atomic_csv(args.output / "selected_pair_margins.csv", pair_rows)
    _atomic_csv(args.output / "selected_pair_summary.csv", pair_summary)
    _atomic_json(args.output / "summary.json", {
        "sites": sorted(wanted),
        "finite_matched_conformers": len(raw),
        "tmol_pass_definition": "candidate_minus_matched_deposited <= 0.0",
        "assignment_summary": summary,
        "selected_pair_summary": pair_summary,
    })


if __name__ == "__main__":
    main()
