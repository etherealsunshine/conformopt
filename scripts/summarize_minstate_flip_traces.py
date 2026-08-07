from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def describe(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    candidate_group_rows = []
    group_rows = []
    trajectory_rows = []
    per_site = {}
    sites = sorted(path.name for path in args.trace_root.iterdir() if path.is_dir())
    for site in sites:
        root = args.trace_root / site
        arrays = np.load(root / "minstate_trace.npz")
        completed = int(arrays["completed_trajectories"])
        metadata = {}
        for environment in ("direct", "symmetry"):
            path = root / f"{environment}_groups.json"
            if path.exists():
                metadata[environment] = json.loads(path.read_text())

        site_group_rows = []
        for trajectory in range(completed):
            aggregate: dict[tuple[str, int], dict[str, int]] = defaultdict(
                lambda: {
                    "observations": 0,
                    "flips": 0,
                    "early_flips_0_49": 0,
                    "middle_flips_50_149": 0,
                    "late_flips_150_199": 0,
                    "final20_flips_180_199": 0,
                    "reactivation_state_changes": 0,
                }
            )
            for environment, groups in metadata.items():
                winners = arrays[f"{environment}_winners"]
                for conformer in range(winners.shape[2]):
                    for group_index, group in enumerate(groups):
                        sequence = winners[
                            trajectory, :, conformer, group_index
                        ]
                        observed_steps = np.flatnonzero(sequence >= 0)
                        if not len(observed_steps):
                            continue
                        values = sequence[observed_steps]
                        changes = values[1:] != values[:-1]
                        consecutive = np.diff(observed_steps) == 1
                        flip_steps = observed_steps[1:][changes & consecutive]
                        reactivation = int((changes & ~consecutive).sum())
                        row = {
                            "site": site,
                            "trajectory": trajectory,
                            "conformer": conformer,
                            "environment": environment,
                            "group_index": group_index,
                            "group_key": group["group_key"],
                            "category": group["category"],
                            "observed_steps": len(observed_steps),
                            "consecutive_step_flips": len(flip_steps),
                            "reactivation_state_changes": reactivation,
                            "early_flips_0_49": int((flip_steps < 50).sum()),
                            "middle_flips_50_149": int(
                                ((flip_steps >= 50) & (flip_steps < 150)).sum()
                            ),
                            "late_flips_150_199": int((flip_steps >= 150).sum()),
                            "final20_flips_180_199": int(
                                (flip_steps >= 180).sum()
                            ),
                        }
                        candidate_group_rows.append(row)
                        key = (environment, group_index)
                        for field in (
                            "observations",
                            "flips",
                            "early_flips_0_49",
                            "middle_flips_50_149",
                            "late_flips_150_199",
                            "final20_flips_180_199",
                            "reactivation_state_changes",
                        ):
                            source = (
                                "observed_steps"
                                if field == "observations"
                                else "consecutive_step_flips"
                                if field == "flips"
                                else field
                            )
                            aggregate[key][field] += int(row[source])
            current_group_rows = []
            for (environment, group_index), values in sorted(aggregate.items()):
                group = metadata[environment][group_index]
                row = {
                    "site": site,
                    "trajectory": trajectory,
                    "environment": environment,
                    "group_index": group_index,
                    "group_key": group["group_key"],
                    "category": group["category"],
                    **values,
                }
                group_rows.append(row)
                site_group_rows.append(row)
                current_group_rows.append(row)
            trajectory_rows.append({
                "site": site,
                "trajectory": trajectory,
                "groups_observed": len(current_group_rows),
                "total_flips": sum(row["flips"] for row in current_group_rows),
                "groups_with_at_least_1_flip": sum(
                    row["flips"] >= 1 for row in current_group_rows
                ),
                "groups_with_at_least_5_flips": sum(
                    row["flips"] >= 5 for row in current_group_rows
                ),
                "groups_flipping_in_last_50_steps": sum(
                    row["late_flips_150_199"] >= 1 for row in current_group_rows
                ),
                "groups_flipping_in_last_20_steps": sum(
                    row["final20_flips_180_199"] >= 1
                    for row in current_group_rows
                ),
            })
        per_site[site] = {
            "trajectories": completed,
            "group_trajectories": len(site_group_rows),
            "groups_with_at_least_1_flip": sum(
                row["flips"] >= 1 for row in site_group_rows
            ),
            "groups_with_at_least_5_flips": sum(
                row["flips"] >= 5 for row in site_group_rows
            ),
            "groups_flipping_in_last_50_steps": sum(
                row["late_flips_150_199"] >= 1 for row in site_group_rows
            ),
            "groups_flipping_in_last_20_steps": sum(
                row["final20_flips_180_199"] >= 1 for row in site_group_rows
            ),
            "flip_count_distribution": describe([
                row["flips"] for row in site_group_rows
            ]),
            "total_flips_by_window": {
                "steps_0_49": sum(
                    row["early_flips_0_49"] for row in site_group_rows
                ),
                "steps_50_149": sum(
                    row["middle_flips_50_149"] for row in site_group_rows
                ),
                "steps_150_199": sum(
                    row["late_flips_150_199"] for row in site_group_rows
                ),
                "steps_180_199": sum(
                    row["final20_flips_180_199"] for row in site_group_rows
                ),
            },
            "by_category": {
                category: {
                    "group_trajectories": len(selected),
                    "groups_with_at_least_1_flip": sum(
                        row["flips"] >= 1 for row in selected
                    ),
                    "groups_with_at_least_5_flips": sum(
                        row["flips"] >= 5 for row in selected
                    ),
                    "total_flips": sum(row["flips"] for row in selected),
                    "late_flips_150_199": sum(
                        row["late_flips_150_199"] for row in selected
                    ),
                }
                for category in sorted({
                    row["category"] for row in site_group_rows
                })
                for selected in [[
                    row for row in site_group_rows
                    if row["category"] == category
                ]]
            },
        }

    flip_counts = [row["flips"] for row in group_rows]
    summary = {
        "sites": sites,
        "per_site": per_site,
        "group_trajectories": len(group_rows),
        "groups_with_at_least_1_flip": sum(value >= 1 for value in flip_counts),
        "groups_with_at_least_5_flips": sum(value >= 5 for value in flip_counts),
        "groups_flipping_in_last_50_steps": sum(
            row["late_flips_150_199"] >= 1 for row in group_rows
        ),
        "groups_flipping_in_last_20_steps": sum(
            row["final20_flips_180_199"] >= 1 for row in group_rows
        ),
        "flip_count_distribution": describe(flip_counts),
        "flip_count_histogram": dict(sorted(Counter(flip_counts).items())),
        "total_flips_by_window": {
            "steps_0_49": sum(row["early_flips_0_49"] for row in group_rows),
            "steps_50_149": sum(row["middle_flips_50_149"] for row in group_rows),
            "steps_150_199": sum(row["late_flips_150_199"] for row in group_rows),
            "steps_180_199": sum(row["final20_flips_180_199"] for row in group_rows),
        },
    }
    args.output.mkdir(parents=True)
    atomic_csv(
        args.output / "candidate_group_flip_counts.csv", candidate_group_rows
    )
    atomic_csv(args.output / "group_trajectory_flip_counts.csv", group_rows)
    atomic_csv(args.output / "trajectory_flip_summary.csv", trajectory_rows)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
