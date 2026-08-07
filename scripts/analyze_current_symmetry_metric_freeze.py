from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(values: list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    return result[np.isfinite(result)]


def describe(values: list[float]) -> dict[str, float | int]:
    array = finite(values)
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def correlation(x: list[float], y: list[float]) -> dict[str, float | int]:
    pairs = [
        (a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)
    ]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson": float("nan"), "spearman": float("nan")}
    xx, yy = (np.asarray(values, dtype=float) for values in zip(*pairs))

    def rankdata(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
            start = end
        return ranks

    def coefficient(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "n": len(pairs),
        "pearson": coefficient(xx, yy),
        "spearman": coefficient(rankdata(xx), rankdata(yy)),
    }


def analyze_tyr417(
    label: str,
    ensemble_path: Path,
    conformer_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    site = "2VFP_A_TYR417"
    ensembles = [
        row for row in read_csv(ensemble_path)
        if row["site"] == site and as_bool(row["both_found_conventional"])
    ]
    conformers = [
        row for row in read_csv(conformer_path) if row["site"] == site
    ]
    by_start: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in conformers:
        by_start[int(row["start"])].append(row)

    rows: list[dict[str, object]] = []
    kinds: Counter[str] = Counter()
    rmsd_a: list[float] = []
    rmsd_b: list[float] = []
    for ensemble in sorted(ensembles, key=lambda row: int(row["start"])):
        start = int(ensemble["start"])
        active = by_start[start]
        assignments = Counter(row["assignment"] for row in active)
        kind = (
            f"A{assignments['A']}_B{assignments['B']}_"
            f"other{assignments['other']}_active{len(active)}"
        )
        kinds[kind] += 1
        selected = {}
        for assignment in ("A", "B"):
            candidates = [
                row for row in active
                if row["assignment"] == assignment
                and float(row["occupancy"]) > 0.10
            ]
            selected[assignment] = min(
                candidates,
                key=lambda row: float(
                    row[f"rmsd_to_{assignment}_conventional"]
                ),
            )
        selected_rmsd_a = float(selected["A"]["rmsd_to_A_conventional"])
        selected_rmsd_b = float(selected["B"]["rmsd_to_B_conventional"])
        rmsd_a.append(selected_rmsd_a)
        rmsd_b.append(selected_rmsd_b)
        predicted_a = float(ensemble["predicted_A_occupancy"])
        predicted_b = float(ensemble["predicted_B_occupancy"])
        target_a = float(ensemble["target_A_occupancy"])
        target_b = float(ensemble["target_B_occupancy"])
        rows.append(
            {
                "run": label,
                "start": start,
                "occupancy_pass": as_bool(ensemble["occupancy_accurate"]),
                "predicted_A_occupancy": predicted_a,
                "predicted_B_occupancy": predicted_b,
                "target_A_occupancy": target_a,
                "target_B_occupancy": target_b,
                "A_occupancy_error": predicted_a - target_a,
                "B_occupancy_error": predicted_b - target_b,
                "active_conformers": len(active),
                "pair_kind": kind,
                "selected_A_conformer": selected["A"]["conformer"],
                "selected_B_conformer": selected["B"]["conformer"],
                "selected_A_rmsd": selected_rmsd_a,
                "selected_B_rmsd": selected_rmsd_b,
            }
        )
    summary = {
        "recovered_starts": len(rows),
        "occupancy_passing": sum(bool(row["occupancy_pass"]) for row in rows),
        "predicted_A_occupancy": describe(
            [float(row["predicted_A_occupancy"]) for row in rows]
        ),
        "predicted_B_occupancy": describe(
            [float(row["predicted_B_occupancy"]) for row in rows]
        ),
        "A_occupancy_error": describe(
            [float(row["A_occupancy_error"]) for row in rows]
        ),
        "B_occupancy_error": describe(
            [float(row["B_occupancy_error"]) for row in rows]
        ),
        "selected_A_rmsd": describe(rmsd_a),
        "selected_B_rmsd": describe(rmsd_b),
        "pair_kinds": dict(sorted(kinds.items())),
    }
    return rows, summary


def analyze_global_rmsd(
    ensemble_paths: list[Path],
    conformer_paths: list[Path],
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    ensembles = [row for path in ensemble_paths for row in read_csv(path)]
    conformers = [row for path in conformer_paths for row in read_csv(path)]
    by_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in conformers:
        by_start[(row["site"], int(row["start"]))].append(row)

    selected_rows: list[dict[str, object]] = []
    for ensemble in ensembles:
        if not as_bool(ensemble["both_found_conventional"]):
            continue
        key = (ensemble["site"], int(ensemble["start"]))
        active = by_start[key]
        selected: dict[str, dict[str, str]] = {}
        for assignment in ("A", "B"):
            candidates = [
                row for row in active
                if row["assignment"] == assignment
                and float(row["occupancy"]) > 0.10
            ]
            if not candidates:
                break
            selected[assignment] = min(
                candidates,
                key=lambda row: float(
                    row[f"rmsd_to_{assignment}_conventional"]
                ),
            )
        if len(selected) != 2:
            continue
        selected_rows.append(
            {
                "site": ensemble["site"],
                "start": int(ensemble["start"]),
                "deposited_B_occupancy": float(
                    ensemble["target_B_occupancy"]
                ),
                "selected_A_rmsd": float(
                    selected["A"]["rmsd_to_A_conventional"]
                ),
                "selected_B_rmsd": float(
                    selected["B"]["rmsd_to_B_conventional"]
                ),
                "B_minus_A_rmsd": float(
                    selected["B"]["rmsd_to_B_conventional"]
                ) - float(selected["A"]["rmsd_to_A_conventional"]),
            }
        )

    by_site: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selected_rows:
        by_site[str(row["site"])].append(row)
    site_rows: list[dict[str, object]] = []
    for site in sorted(by_site):
        rows = by_site[site]
        a = [float(row["selected_A_rmsd"]) for row in rows]
        b = [float(row["selected_B_rmsd"]) for row in rows]
        site_rows.append(
            {
                "site": site,
                "recovered_pairs": len(rows),
                "deposited_B_occupancy": rows[0]["deposited_B_occupancy"],
                "A_rmsd_median": float(np.median(a)),
                "A_rmsd_q95": float(np.quantile(a, 0.95)),
                "B_rmsd_median": float(np.median(b)),
                "B_rmsd_q95": float(np.quantile(b, 0.95)),
                "B_minus_A_median": float(
                    np.median(np.asarray(b) - np.asarray(a))
                ),
            }
        )

    all_a = [float(row["selected_A_rmsd"]) for row in selected_rows]
    all_b = [float(row["selected_B_rmsd"]) for row in selected_rows]
    occupancies = [
        float(row["deposited_B_occupancy"]) for row in selected_rows
    ]
    site_occupancies = [
        float(row["deposited_B_occupancy"]) for row in site_rows
    ]
    site_b_medians = [float(row["B_rmsd_median"]) for row in site_rows]
    site_deltas = [float(row["B_minus_A_median"]) for row in site_rows]
    summary = {
        "selected_recovered_pairs": len(selected_rows),
        "selected_A_rmsd": describe(all_a),
        "selected_B_rmsd": describe(all_b),
        "selected_B_minus_A_rmsd": describe(
            list(np.asarray(all_b) - np.asarray(all_a))
        ),
        "endpoint_level_B_occupancy_vs_B_rmsd": correlation(
            occupancies, all_b
        ),
        "site_level_B_occupancy_vs_median_B_rmsd": correlation(
            site_occupancies, site_b_medians
        ),
        "site_level_B_occupancy_vs_median_B_minus_A": correlation(
            site_occupancies, site_deltas
        ),
    }
    return selected_rows, summary, site_rows


def analyze_8q6q(strict_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = [
        row for row in read_csv(strict_path)
        if row["site"] == "8Q6Q_B_ASP81"
        and row["assignment"] == "B"
        and math.isfinite(float(row["tmol_delta_vs_matched_AB"]))
    ]
    output = [
        {
            "candidate_id": row["candidate_id"],
            "start": int(row["start"]),
            "conformer": int(row["conformer"]),
            "occupancy": float(row["occupancy"]),
            "rmsd_to_B": float(row["rmsd_to_B_conventional"]),
            "tmol_margin": float(row["tmol_delta_vs_matched_AB"]),
            "passes_tolerance_0": float(row["tmol_delta_vs_matched_AB"]) <= 0.0,
        }
        for row in rows
    ]
    margins = [float(row["tmol_margin"]) for row in output]
    rmsds = [float(row["rmsd_to_B"]) for row in output]
    return output, {
        "B_assigned_conformers": len(output),
        "passes_tolerance_0": sum(
            bool(row["passes_tolerance_0"]) for row in output
        ),
        "pass_rate_tolerance_0": (
            sum(bool(row["passes_tolerance_0"]) for row in output) / len(output)
        ),
        "tmol_margin": describe(margins),
        "rmsd_to_B": describe(rmsds),
        "tmol_margin_vs_rmsd": correlation(rmsds, margins),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-original", type=Path, required=True)
    parser.add_argument("--current-expanded", type=Path, required=True)
    parser.add_argument("--stale-expanded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    fresh_rows, fresh_summary = analyze_tyr417(
        "fresh_current_symmetry",
        args.current_expanded / "ensemble_geometry_audit.csv",
        args.current_expanded / "active_conformer_geometry_audit.csv",
    )
    stale_rows, stale_summary = analyze_tyr417(
        "stale_symmetry_environment",
        args.stale_expanded / "ensemble_geometry_audit.csv",
        args.stale_expanded / "active_conformer_geometry_audit.csv",
    )
    write_csv(
        args.output / "2vfp_recovered_start_occupancy_and_pairs.csv",
        fresh_rows + stale_rows,
    )

    selected_rows, global_summary, site_rows = analyze_global_rmsd(
        [
            args.current_original / "ensemble_geometry_audit.csv",
            args.current_expanded / "ensemble_geometry_audit.csv",
        ],
        [
            args.current_original / "active_conformer_geometry_audit.csv",
            args.current_expanded / "active_conformer_geometry_audit.csv",
        ],
    )
    write_csv(args.output / "selected_pair_rmsd.csv", selected_rows)
    write_csv(args.output / "per_site_selected_pair_rmsd.csv", site_rows)

    q_rows, q_summary = analyze_8q6q(
        args.current_original / "active_conformer_strict_audit.csv"
    )
    write_csv(args.output / "8q6q_B_assigned_fresh_tmol_margins.csv", q_rows)

    fresh_starts = {int(row["start"]) for row in fresh_rows}
    stale_starts = {int(row["start"]) for row in stale_rows}
    summary = {
        "2VFP_A_TYR417": {
            "fresh": fresh_summary,
            "stale": stale_summary,
            "common_recovered_starts": len(fresh_starts & stale_starts),
            "fresh_only_recovered_starts": sorted(fresh_starts - stale_starts),
            "stale_only_recovered_starts": sorted(stale_starts - fresh_starts),
        },
        "global_selected_pair_rmsd": global_summary,
        "8Q6Q_B_ASP81_fresh_B_assigned_tmol": q_summary,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
