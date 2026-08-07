from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


METRICS = (
    "abs_grad_occ",
    "grad_occ_squared",
    "abs_hessian_occ_diag",
    "fisher_occ_diag",
    "abs_grad_logit",
    "grad_logit_squared",
    "abs_hessian_logit_diag",
    "fisher_logit_diag",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def describe(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def auc_higher_matched(
    matched: list[float], extra: list[float]
) -> Optional[float]:
    left = np.asarray(matched, dtype=float)
    right = np.asarray(extra, dtype=float)
    if not len(left) or not len(right):
        return None
    combined = np.concatenate([left, right])
    ranks = average_ranks(combined)
    rank_sum = ranks[: len(left)].sum()
    u = rank_sum - len(left) * (len(left) + 1) / 2.0
    return float(u / (len(left) * len(right)))


def population_comparison(
    rows: list[dict[str, str]], site: str
) -> list[dict[str, object]]:
    selected = rows if site == "ALL" else [row for row in rows if row["site"] == site]
    matched_rows = [row for row in selected if row["population"] == "matched"]
    extra_rows = [row for row in selected if row["population"] == "extra"]
    output = []
    for metric in METRICS:
        matched = [float(row[metric]) for row in matched_rows]
        extra = [float(row[metric]) for row in extra_rows]
        matched_desc = describe(matched)
        extra_desc = describe(extra)
        if matched and extra:
            matched_array = np.asarray(matched)
            extra_array = np.asarray(extra)
            matched_q05, matched_q95 = np.quantile(matched_array, (0.05, 0.95))
            extra_q05, extra_q95 = np.quantile(extra_array, (0.05, 0.95))
            extra_inside_matched_90 = float(
                ((extra_array >= matched_q05) & (extra_array <= matched_q95)).mean()
            )
            matched_inside_extra_90 = float(
                ((matched_array >= extra_q05) & (matched_array <= extra_q95)).mean()
            )
        else:
            extra_inside_matched_90 = math.nan
            matched_inside_extra_90 = math.nan
        output.append(
            {
                "site": site,
                "metric": metric,
                "matched_n": len(matched),
                "extra_n": len(extra),
                "auc_higher_value_indicates_matched": auc_higher_matched(
                    matched, extra
                ),
                "extra_fraction_inside_matched_q05_q95": extra_inside_matched_90,
                "matched_fraction_inside_extra_q05_q95": matched_inside_extra_90,
                **{f"matched_{key}": value for key, value in matched_desc.items()},
                **{f"extra_{key}": value for key, value in extra_desc.items()},
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    args = parser.parse_args()
    derivatives = read_csv(
        args.analysis_root / "active_conformer_density_derivatives.csv"
    )
    starts = read_csv(args.analysis_root / "k4_start_occupancy_summary.csv")

    sites = sorted({row["site"] for row in derivatives})
    comparison_rows = population_comparison(derivatives, "ALL")
    for site in sites:
        comparison_rows.extend(population_comparison(derivatives, site))

    by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in starts:
        by_site[row["site"]].append(row)
    occupancy_rows = []
    for site in sites:
        selected = by_site[site]
        paired = [row for row in selected if row["pair_complete"] == "True"]
        values = {
            "submask_mass_all_starts": [
                float(row["submask_occupancy_mass"]) for row in selected
            ],
            "extra_active_mass_all_starts": [
                float(row["extra_active_occupancy_mass"]) for row in selected
            ],
            "matched_mass_paired_starts": [
                float(row["matched_occupancy_sum"]) for row in paired
            ],
            "matched_minus_target_paired_starts": [
                float(row["matched_minus_target_AB_sum"]) for row in paired
            ],
        }
        output = {
            "site": site,
            "starts": len(selected),
            "paired_starts": len(paired),
            "paired_starts_undershooting_target_AB_sum": sum(
                float(row["matched_minus_target_AB_sum"]) < 0.0 for row in paired
            ),
        }
        for label, numbers in values.items():
            for key, value in describe(numbers).items():
                output[f"{label}_{key}"] = value
        occupancy_rows.append(output)

    overall = next(
        row
        for row in comparison_rows
        if row["site"] == "ALL" and row["metric"] == "fisher_logit_diag"
    )
    overall_occ = next(
        row
        for row in comparison_rows
        if row["site"] == "ALL" and row["metric"] == "fisher_occ_diag"
    )
    paired = [row for row in starts if row["pair_complete"] == "True"]
    summary = {
        "primary_logit_fisher_comparison": overall,
        "post_softmax_ambient_fisher_comparison": overall_occ,
        "paired_starts": len(paired),
        "paired_starts_undershooting_target_AB_sum": sum(
            float(row["matched_minus_target_AB_sum"]) < 0.0 for row in paired
        ),
        "paired_start_deficit": describe(
            [-float(row["matched_minus_target_AB_sum"]) for row in paired]
        ),
        "interpretation_rule": (
            "AUC near 0.5 and large cross-population central-range inclusion "
            "indicate substantial overlap; no threshold is selected."
        ),
    }
    atomic_csv(args.analysis_root / "density_support_separation.csv", comparison_rows)
    atomic_csv(args.analysis_root / "occupancy_accounting_by_site.csv", occupancy_rows)
    atomic_json(args.analysis_root / "interpretation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
