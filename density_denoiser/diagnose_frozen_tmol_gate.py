from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .residue_geometry import AUDIT_RULE_VERSION


REPLACEMENT_SITES = {
    "3A1C_B_ARG447",
    "6H59_B_ARG144",
    "3NY7_B_LYS19",
}
TMOL_RULE_VERSION = "frozen_matched_deposited_minstate_v1"


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


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


def load_composite_rows(
    baseline_root: Path, replacement_root: Path, filename: str
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for panel in ("original5", "expanded15"):
        rows.extend(
            row
            for row in _read_csv(baseline_root / panel / filename)
            if row["site"] not in REPLACEMENT_SITES
        )
    for panel in ("original2", "expanded1"):
        rows.extend(_read_csv(replacement_root / panel / filename))
    keys = [
        (row["site"], row.get("start", ""), row.get("candidate_id", ""))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate composite rows in {filename}")
    return rows


def matched_rmsd(row: dict[str, str]) -> float:
    assignment = row["assignment"]
    if assignment == "A":
        return float(row["rmsd_to_A_conventional"])
    if assignment == "B":
        return float(row["rmsd_to_B_conventional"])
    return math.nan


def rmsd_bin(value: float) -> str:
    if value <= 0.1:
        return "<=0.1"
    if value <= 0.3:
        return "0.1-0.3"
    if value <= 0.6:
        return "0.3-0.6"
    if value < 1.0:
        return "0.6-1.0"
    return ">=1.0"


def _finite_tmol(row: dict[str, str]) -> bool:
    return all(
        math.isfinite(float(row[column]))
        for column in (
            "tmol_energy",
            "tmol_reference_matched_AB",
            "tmol_delta_vs_matched_AB",
        )
    )


def summarize_margin_bins(
    matched_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    result = []
    for label in ("<=0.1", "0.1-0.3", "0.3-0.6", "0.6-1.0", ">=1.0"):
        margins = np.asarray(
            [float(row["tmol_margin"]) for row in matched_rows if row["rmsd_bin"] == label]
        )
        if not len(margins):
            continue
        result.append({
            "rmsd_bin": label,
            "conformers": int(len(margins)),
            "margin_mean": float(margins.mean()),
            "margin_std": float(margins.std(ddof=1)) if len(margins) > 1 else 0.0,
            "margin_median": float(np.median(margins)),
            "margin_q05": float(np.quantile(margins, 0.05)),
            "margin_q25": float(np.quantile(margins, 0.25)),
            "margin_q75": float(np.quantile(margins, 0.75)),
            "margin_q95": float(np.quantile(margins, 0.95)),
            "margin_positive": int((margins > 0).sum()),
            "margin_0_to_0_5": int(((margins > 0) & (margins <= 0.5)).sum()),
        })
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including deterministic tie handling."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def pearson_and_spearman(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("correlations require paired vectors of length at least two")
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(
        np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1]
    )
    return pearson, spearman


def _conformer_cascade(
    rows: list[dict[str, str]], tolerance: float
) -> dict[str, int]:
    matched = [row for row in rows if row["assignment"] in {"A", "B"}]
    finite = [row for row in matched if _finite_tmol(row)]
    rotamer = [row for row in finite if _as_bool(row["rotamer_within_allowed_width"])]
    direct = [row for row in rotamer if _as_bool(row["no_direct_clash"])]
    symmetry = [row for row in direct if _as_bool(row["no_symmetry_clash"])]
    strict = [
        row
        for row in symmetry
        if float(row["tmol_delta_vs_matched_AB"]) <= tolerance
    ]
    return {
        "active_conformers": len(rows),
        "matched_conformers": len(matched),
        "unmatched_conformers": len(rows) - len(matched),
        "nonfinite_energy_all_active": sum(
            not math.isfinite(float(row["tmol_energy"])) for row in rows
        ),
        "nonfinite_matched_conformers": len(matched) - len(finite),
        "matched_finite_conformers": len(finite),
        "matched_finite_plus_rotamer": len(rotamer),
        "plus_direct_clash_gate": len(direct),
        "plus_symmetry_clash_gate": len(symmetry),
        "plus_tmol_gate": len(strict),
        "tmol_pass_independent": sum(
            float(row["tmol_delta_vs_matched_AB"]) <= tolerance for row in finite
        ),
        "geometry_pass_independent": sum(
            _as_bool(row["geometry_physical_valid"]) for row in rows
        ),
    }


def build_per_site_cascade(
    conformers: list[dict[str, str]],
    ensembles: list[dict[str, str]],
    tolerance: float,
) -> list[dict[str, object]]:
    sites = sorted({row["site"] for row in ensembles})
    output = []
    for site in sites:
        site_conformers = [row for row in conformers if row["site"] == site]
        site_ensembles = [row for row in ensembles if row["site"] == site]
        strict_by_start: dict[int, list[bool]] = {}
        for row in site_conformers:
            start = int(row["start"])
            valid = (
                row["assignment"] in {"A", "B"}
                and _finite_tmol(row)
                and _as_bool(row["geometry_physical_valid"])
                and float(row["tmol_delta_vs_matched_AB"]) <= tolerance
            )
            strict_by_start.setdefault(start, []).append(valid)
        all_active_strict = {
            start: bool(values) and all(values)
            for start, values in strict_by_start.items()
        }
        result: dict[str, object] = {
            "site": site,
            "tmol_tolerance": tolerance,
            **_conformer_cascade(site_conformers, tolerance),
            "ensembles": len(site_ensembles),
            "both_found": sum(
                _as_bool(row["both_found_conventional"]) for row in site_ensembles
            ),
            "recovery_and_occupancy": sum(
                _as_bool(row["geometric_occupancy_success"]) for row in site_ensembles
            ),
            "all_active_strict_physical_ensembles": sum(
                all_active_strict.get(int(row["start"]), False)
                for row in site_ensembles
            ),
            "strict_joint_success": sum(
                _as_bool(row["geometric_occupancy_success"])
                and all_active_strict.get(int(row["start"]), False)
                for row in site_ensembles
            ),
        }
        output.append(result)
    return output


def _strict_counts_by_site(
    conformers: list[dict[str, str]],
    ensembles: list[dict[str, str]],
) -> dict[str, int]:
    return {
        row["site"]: int(row["strict_joint_success"])
        for row in build_per_site_cascade(conformers, ensembles, 0.0)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose matched-environment tmol as a function of RMSD."
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--intermediate-baseline-root", type=Path, required=True)
    parser.add_argument("--intermediate-replacement-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conformers = load_composite_rows(
        args.baseline_root, args.replacement_root, "active_conformer_strict_audit.csv"
    )
    ensembles = load_composite_rows(
        args.baseline_root, args.replacement_root, "ensemble_geometry_audit.csv"
    )
    if len(ensembles) != 1000 or len({row["site"] for row in ensembles}) != 20:
        raise ValueError("expected the frozen 20-site, 1000-ensemble composite")
    rules = {
        row.get("tmol_environment_rule", "")
        for row in conformers
        if row.get("tmol_environment_rule", "")
    }
    # Strict tables do not currently repeat the tmol rule column; the manifest
    # and tmol tables are the authority. Validate those tables separately.
    tmol_rows = load_composite_rows(
        args.baseline_root, args.replacement_root, "tmol_energies.csv"
    )
    rules = {row["tmol_environment_rule"] for row in tmol_rows}
    if rules != {TMOL_RULE_VERSION}:
        raise ValueError(f"unexpected tmol rules: {sorted(rules)}")

    matched_finite = []
    for row in conformers:
        if row["assignment"] not in {"A", "B"} or not _finite_tmol(row):
            continue
        rmsd = matched_rmsd(row)
        margin = float(row["tmol_delta_vs_matched_AB"])
        matched_finite.append({
            "candidate_id": row["candidate_id"],
            "site": row["site"],
            "start": int(row["start"]),
            "conformer": int(row["conformer"]),
            "assignment": row["assignment"],
            "rmsd_to_matched_deposited": rmsd,
            "tmol_margin": margin,
            "rmsd_bin": rmsd_bin(rmsd),
        })
    rmsd_values = np.asarray(
        [float(row["rmsd_to_matched_deposited"]) for row in matched_finite]
    )
    margins = np.asarray([float(row["tmol_margin"]) for row in matched_finite])
    pearson, spearman = pearson_and_spearman(rmsd_values, margins)

    cascade = []
    for tolerance in (0.0, 0.5):
        cascade.extend(build_per_site_cascade(conformers, ensembles, tolerance))

    intermediate_conformers = load_composite_rows(
        args.intermediate_baseline_root,
        args.intermediate_replacement_root,
        "active_conformer_strict_audit.csv",
    )
    intermediate_ensembles = load_composite_rows(
        args.intermediate_baseline_root,
        args.intermediate_replacement_root,
        "ensemble_geometry_audit.csv",
    )
    current_site_strict = _strict_counts_by_site(conformers, ensembles)
    intermediate_site_strict = _strict_counts_by_site(
        intermediate_conformers, intermediate_ensembles
    )
    artifact_delta = [{
        "site": site,
        "retracted_v3_strict_tolerance_0": intermediate_site_strict[site],
        "frozen_v5_strict_tolerance_0": current_site_strict[site],
        "artifact_delta": current_site_strict[site] - intermediate_site_strict[site],
    } for site in sorted(current_site_strict)]

    missing = []
    for site in sorted({row["site"] for row in conformers}):
        site_rows = [row for row in conformers if row["site"] == site]
        matched = [row for row in site_rows if row["assignment"] in {"A", "B"}]
        missing.append({
            "site": site,
            "active_conformers": len(site_rows),
            "matched_conformers": len(matched),
            "unmatched_conformers": len(site_rows) - len(matched),
            "nonfinite_energy_all_active": sum(
                not math.isfinite(float(row["tmol_energy"])) for row in site_rows
            ),
            "nonfinite_matched_conformers": sum(
                not _finite_tmol(row) for row in matched
            ),
            "finite_matched_conformers": sum(_finite_tmol(row) for row in matched),
        })

    bins = summarize_margin_bins(matched_finite)
    args.output.mkdir(parents=True, exist_ok=False)
    _atomic_csv(args.output / "matched_conformers.csv", matched_finite)
    _atomic_csv(args.output / "margin_by_rmsd_bin.csv", bins)
    _atomic_csv(args.output / "missing_counts_by_site.csv", missing)
    _atomic_csv(args.output / "per_site_cascade.csv", cascade)
    _atomic_csv(args.output / "artifact_delta_v3_to_v5.csv", artifact_delta)

    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    axis.scatter(rmsd_values, margins, s=11, alpha=0.38, linewidths=0)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axhline(0.5, color="#b23a48", linewidth=1, linestyle="--")
    axis.set(
        xlabel="RMSD to matched deposited conformer (Å)",
        ylabel="tmol margin: candidate − matched deposited",
        title="Frozen matched-environment tmol margin versus RMSD",
    )
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(args.output / "tmol_margin_vs_matched_rmsd.png", dpi=180)
    plt.close(figure)

    finite_failures = int((margins > 0.0).sum())
    totals_by_tolerance = {}
    for tolerance in (0.0, 0.5):
        rows = [row for row in cascade if row["tmol_tolerance"] == tolerance]
        totals_by_tolerance[str(tolerance)] = {
            column: int(sum(int(row[column]) for row in rows))
            for column in (
                "active_conformers",
                "matched_conformers",
                "unmatched_conformers",
                "nonfinite_matched_conformers",
                "matched_finite_conformers",
                "matched_finite_plus_rotamer",
                "plus_direct_clash_gate",
                "plus_symmetry_clash_gate",
                "plus_tmol_gate",
                "ensembles",
                "both_found",
                "recovery_and_occupancy",
                "all_active_strict_physical_ensembles",
                "strict_joint_success",
            )
        }
    summary = {
        "geometry_rule": AUDIT_RULE_VERSION,
        "tmol_rule": TMOL_RULE_VERSION,
        "tmol_tolerances": [0.0, 0.5],
        "finite_matched_conformers": len(matched_finite),
        "finite_matched_failures_at_zero": finite_failures,
        "finite_matched_failure_rate_at_zero": finite_failures / len(matched_finite),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "totals_by_tolerance": totals_by_tolerance,
        "retracted_intermediate_strict_total": sum(intermediate_site_strict.values()),
        "frozen_v5_strict_total": sum(current_site_strict.values()),
        "artifact_delta_total": (
            sum(current_site_strict.values()) - sum(intermediate_site_strict.values())
        ),
    }
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
