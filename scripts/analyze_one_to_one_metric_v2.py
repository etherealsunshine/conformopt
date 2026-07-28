"""Compare frozen greedy v1 with optimal one-to-one matching v2.

This is a saved-endpoint analysis. It does not optimize conformers or alter
the production metric inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


STAGES = (
    "both_found",
    "occupancy",
    "rotamer",
    "direct_clash",
    "symmetry_clash",
    "tmol_0_44",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def select_pair(rows: list[dict[str, str]]) -> dict[str, dict[str, str]] | None:
    pair = {}
    for state in ("A", "B"):
        candidates = [
            row
            for row in rows
            if row["assignment"] == state
            and float(row.get("assigned_occupancy", row["occupancy"])) > 0.10
        ]
        if not candidates:
            return None
        pair[state] = min(
            candidates,
            key=lambda row: float(row[f"rmsd_to_{state}_conventional"]),
        )
    if pair["A"]["candidate_id"] == pair["B"]["candidate_id"]:
        raise ValueError("one conformer cannot supply both deposited states")
    return pair


def stage_flags(
    ensemble: dict[str, str],
    conformers: list[dict[str, str]],
) -> dict[str, bool]:
    both = as_bool(ensemble["both_found_conventional"])
    occupancy = both and as_bool(ensemble["occupancy_accurate"])
    pair = select_pair(conformers)
    rotamer = occupancy and pair is not None and all(
        as_bool(row.get(
            "rotamer_within_allowed_width", row["canonical_like_30deg"]
        ))
        for row in pair.values()
    )
    direct = rotamer and all(
        as_bool(row["no_direct_clash"]) for row in pair.values()
    )
    symmetry = direct and all(
        as_bool(row["no_symmetry_clash"]) for row in pair.values()
    )
    tmol = symmetry and all(
        math.isfinite(float(row["tmol_delta_vs_matched_AB"]))
        and float(row["tmol_delta_vs_matched_AB"]) <= 0.44
        for row in pair.values()
    )
    return {
        "both_found": both,
        "occupancy": occupancy,
        "rotamer": rotamer,
        "direct_clash": direct,
        "symmetry_clash": symmetry,
        "tmol_0_44": tmol,
    }


def load_version(audit_roots: list[Path]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {stage: 0 for stage in STAGES}
    )
    seen: set[tuple[str, int]] = set()
    for root in audit_roots:
        ensembles = read_csv(root / "ensemble_strict_audit.csv")
        conformers = read_csv(root / "active_conformer_strict_audit.csv")
        by_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
        for row in conformers:
            by_start[(row["site"], int(row["start"]))].append(row)
        for ensemble in ensembles:
            key = (ensemble["site"], int(ensemble["start"]))
            if key in seen:
                raise ValueError(f"duplicate ensemble: {key}")
            seen.add(key)
            flags = stage_flags(ensemble, by_start[key])
            for stage, passed in flags.items():
                counts[key[0]][stage] += int(passed)
    if len(seen) != 1_000:
        raise ValueError(f"expected 1,000 starts, found {len(seen)}")
    return dict(counts)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-audit-root", type=Path, action="append", required=True)
    parser.add_argument("--v2-audit-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v1 = load_version(args.v1_audit_root)
    v2 = load_version(args.v2_audit_root)
    if set(v1) != set(v2) or len(v1) != 20:
        raise ValueError("v1/v2 site sets do not match the 20-site panel")

    rows = []
    totals = {
        version: {stage: 0 for stage in STAGES}
        for version in ("v1", "v2", "delta")
    }
    for site in sorted(v1):
        row: dict[str, object] = {"site": site, "starts": 50}
        for stage in STAGES:
            row[f"v1_{stage}"] = v1[site][stage]
            row[f"v2_{stage}"] = v2[site][stage]
            row[f"delta_{stage}"] = v2[site][stage] - v1[site][stage]
            totals["v1"][stage] += v1[site][stage]
            totals["v2"][stage] += v2[site][stage]
            totals["delta"][stage] += row[f"delta_{stage}"]
        rows.append(row)
    write_csv(args.output / "per_site_v1_v2_cascade.csv", rows)

    total_rows = []
    for version in ("v1", "v2", "delta"):
        total_rows.append({"version": version, **totals[version]})
    write_csv(args.output / "total_v1_v2_cascade.csv", total_rows)

    summary = {
        "v1_metric_version": "qfit-synth20-assignedpair-tmol044-v1",
        "v2_metric_version": "qfit-synth20-one-to-one-assignedpair-tmol044-v2",
        "single_change": (
            "greedy independent nearest-state labels replaced by optimal "
            "one-to-one A/B assignment"
        ),
        "assignment_correction_not_optimizer_progress": True,
        "tolerance": 0.44,
        "reported_activity_threshold": 0.10,
        "totals": totals,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
