from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from density_denoiser.summarize_endpoint_audit import as_bool, select_assigned_pair


METRICS = (
    "fisher_occ_diag",
    "abs_hessian_occ_diag",
    "fisher_logit_diag",
    "abs_hessian_logit_diag",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def auc(matched: np.ndarray, extra: np.ndarray) -> float:
    combined = np.concatenate([matched, extra])
    ranks = average_ranks(combined)
    rank_sum = ranks[: len(matched)].sum()
    u = rank_sum - len(matched) * (len(matched) + 1) / 2.0
    return float(u / (len(matched) * len(extra)))


def describe(values: np.ndarray) -> dict[str, object]:
    return {
        "n": int(len(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--strict-table", type=Path, action="append", required=True)
    parser.add_argument("--ensemble-table", type=Path, action="append", required=True)
    parser.add_argument("--extra-table", type=Path)
    args = parser.parse_args()

    strict = [
        row for path in args.strict_table for row in read_csv(path)
    ]
    ensembles = [
        row for path in args.ensemble_table for row in read_csv(path)
    ]
    by_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in strict:
        by_start[(row["site"], int(row["start"]))].append(row)
    headline = set()
    for ensemble in ensembles:
        key = (ensemble["site"], int(ensemble["start"]))
        if not as_bool(ensemble["geometric_occupancy_success"]):
            continue
        pair = select_assigned_pair(by_start[key])
        if pair and all(
            as_bool(row["rotamer_within_allowed_width"])
            and as_bool(row["no_direct_clash"])
            and as_bool(row["no_symmetry_clash"])
            and row["assignment"] in {"A", "B"}
            and np.isfinite(float(row["tmol_delta_vs_matched_AB"]))
            and float(row["tmol_delta_vs_matched_AB"]) <= 0.44
            for row in pair.values()
        ):
            headline.add(key)

    derivatives = [
        row
        for row in read_csv(
            args.analysis_root / "active_conformer_density_derivatives.csv"
        )
        if (row["site"], int(row["start"])) in headline
    ]
    output = []
    for metric in METRICS:
        matched = np.asarray(
            [
                float(row[metric])
                for row in derivatives
                if row["population"] == "matched"
            ]
        )
        extra = np.asarray(
            [
                float(row[metric])
                for row in derivatives
                if row["population"] == "extra"
            ]
        )
        matched_q05, matched_q95 = np.quantile(matched, (0.05, 0.95))
        extra_q05, extra_q95 = np.quantile(extra, (0.05, 0.95))
        output.append(
            {
                "metric": metric,
                "auc_higher_value_indicates_matched": auc(matched, extra),
                "extra_fraction_inside_matched_q05_q95": float(
                    ((extra >= matched_q05) & (extra <= matched_q95)).mean()
                ),
                "matched_fraction_inside_extra_q05_q95": float(
                    ((matched >= extra_q05) & (matched <= extra_q95)).mean()
                ),
                **{
                    f"matched_{key}": value
                    for key, value in describe(matched).items()
                },
                **{
                    f"extra_{key}": value
                    for key, value in describe(extra).items()
                },
            }
        )
    summary = {
        "headline_starts": len(headline),
        "headline_active_conformers": len(derivatives),
        "headline_matched_conformers": sum(
            row["population"] == "matched" for row in derivatives
        ),
        "headline_extra_conformers": sum(
            row["population"] == "extra" for row in derivatives
        ),
        "comparisons": {row["metric"]: row for row in output},
    }
    separation_path = args.analysis_root / "headline_density_support_separation.csv"
    summary_path = args.analysis_root / "headline_interpretation_summary.json"
    if not separation_path.exists():
        atomic_csv(separation_path, output)
    if not summary_path.exists():
        atomic_json(summary_path, summary)
    if args.extra_table:
        extra_records = read_csv(args.extra_table)
        extra_bearing_keys = {
            (row["site"], int(row["start"]))
            for row in extra_records
            if row["headline_pass_start"] == "True"
        }
        extra_bearing = [
            row
            for row in derivatives
            if (row["site"], int(row["start"])) in extra_bearing_keys
        ]
        comparisons = {}
        for metric in METRICS:
            matched = np.asarray(
                [
                    float(row[metric])
                    for row in extra_bearing
                    if row["population"] == "matched"
                ]
            )
            extra = np.asarray(
                [
                    float(row[metric])
                    for row in extra_bearing
                    if row["population"] == "extra"
                ]
            )
            comparisons[metric] = {
                "auc_higher_value_indicates_matched": auc(matched, extra),
                "matched": describe(matched),
                "extra": describe(extra),
            }
        extra_bearing_summary = {
            "headline_extra_bearing_starts": len(extra_bearing_keys),
            "matched_conformers": sum(
                row["population"] == "matched" for row in extra_bearing
            ),
            "extra_conformers": sum(
                row["population"] == "extra" for row in extra_bearing
            ),
            "comparisons": comparisons,
        }
        extra_path = (
            args.analysis_root
            / "headline_extra_bearing_density_support_summary.json"
        )
        if extra_path.exists():
            raise FileExistsError(extra_path)
        atomic_json(extra_path, extra_bearing_summary)
        summary["extra_bearing"] = extra_bearing_summary
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
