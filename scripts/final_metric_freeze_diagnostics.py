from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from density_denoiser.summarize_endpoint_audit import (
    as_bool,
    select_assigned_pair,
)


RMSD_BINS = (
    ("<=0.1", -math.inf, 0.1),
    ("0.1-0.3", 0.1, 0.3),
    ("0.3-0.6", 0.3, 0.6),
    ("0.6-1.0", 0.6, 1.0),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        if rows:
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
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
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


def correlations(x: list[float], y: list[float]) -> dict[str, object]:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    if (
        len(left) < 2
        or np.ptp(left) == 0.0
        or np.ptp(right) == 0.0
    ):
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(
            np.corrcoef(rankdata(left), rankdata(right))[0, 1]
        ),
    }


def matched_rmsd(row: dict[str, str]) -> float:
    return float(row[f"rmsd_to_{row['assignment']}_conventional"])


def rmsd_bin(value: float) -> str | None:
    for label, lower, upper in RMSD_BINS:
        if value <= upper and value > lower:
            return label
    return None


def tolerance_validation(
    margins: list[dict[str, str]], tolerance: float
) -> tuple[list[dict[str, object]], dict[str, object]]:
    finite = []
    for row in margins:
        if row["assignment"] not in {"A", "B"}:
            continue
        try:
            rmsd = float(row["rmsd_to_matched_deposited"])
            margin = float(row["tmol_margin_candidate_minus_matched_deposited"])
        except ValueError:
            continue
        if math.isfinite(rmsd) and math.isfinite(margin):
            finite.append((rmsd, margin))

    rows = []
    for label, _lower, _upper in RMSD_BINS:
        values = [margin for rmsd, margin in finite if rmsd_bin(rmsd) == label]
        array = np.asarray(values, dtype=float)
        passing = int((array <= tolerance).sum())
        rows.append(
            {
                "rmsd_bin": label,
                "conformers": len(values),
                "pass_at_tolerance": passing,
                "pass_rate": passing / len(values) if values else "",
                "tmol_tolerance": tolerance,
                **{
                    f"margin_{key}": value
                    for key, value in describe(values).items()
                    if key != "n"
                },
            }
        )
    reproduction = [
        margin for rmsd, margin in finite if rmsd <= 0.1
    ]
    positive = np.asarray(
        [margin for margin in reproduction if margin > 0.0], dtype=float
    )
    summary = {
        "matched_finite_conformers": len(finite),
        "tolerance": tolerance,
        "reproduction_conformers_rmsd_le_0_1": len(reproduction),
        "reproduction_pass_at_tolerance": sum(
            margin <= tolerance for margin in reproduction
        ),
        "reproduction_pass_rate": (
            sum(margin <= tolerance for margin in reproduction) / len(reproduction)
            if reproduction
            else None
        ),
        "positive_reproduction_margins": int(len(positive)),
        "positive_reproduction_q99": (
            float(np.quantile(positive, 0.99)) if len(positive) else None
        ),
        "positive_reproduction_max": (
            float(positive.max()) if len(positive) else None
        ),
        "prior_positive_q99": 0.438,
        "prior_positive_max": 0.473,
    }
    return rows, summary


def gate_failures(row: dict[str, str], tolerance: float) -> list[str]:
    failures = []
    if not as_bool(row["rotamer_within_allowed_width"]):
        failures.append("rotamer")
    if not as_bool(row["no_direct_clash"]):
        failures.append("direct_clash")
    if not as_bool(row["no_symmetry_clash"]):
        failures.append("symmetry_clash")
    if row["assignment"] not in {"A", "B"}:
        failures.append("tmol_unmatched")
    else:
        try:
            margin = float(row["tmol_delta_vs_matched_AB"])
        except ValueError:
            margin = math.nan
        if not math.isfinite(margin):
            failures.append("tmol_nonfinite")
        elif margin > tolerance:
            failures.append("tmol_margin")
    return failures


def characterize_extra_conformers(
    conformers: list[dict[str, str]],
    ensembles: list[dict[str, str]],
    sites: set[str],
    tolerance: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in conformers:
        if row["site"] in sites:
            by_key[(row["site"], int(row["start"]))].append(row)
    ensemble_lookup = {
        (row["site"], int(row["start"])): row
        for row in ensembles
        if row["site"] in sites
    }

    slot_rows = []
    start_rows = []
    site_summary: dict[str, object] = {}
    for site in sorted(sites):
        extras: list[float] = []
        causal_extras: list[float] = []
        recovered_occupancy = assigned_passes = all_passes = 0
        for start in range(50):
            active = by_key.get((site, start), [])
            ensemble = ensemble_lookup[(site, start)]
            pair = select_assigned_pair(active)
            selected_ids = {
                row["candidate_id"] for row in pair.values()
            } if pair else set()
            recovered = as_bool(ensemble["geometric_occupancy_success"])
            pair_pass = bool(pair) and all(
                not gate_failures(row, tolerance) for row in pair.values()
            )
            all_pass = bool(active) and all(
                not gate_failures(row, tolerance) for row in active
            )
            if recovered:
                recovered_occupancy += 1
                assigned_passes += pair_pass
                all_passes += all_pass
            extra_failures = []
            for row in sorted(active, key=lambda item: int(item["conformer"])):
                nearest = min(
                    float(row["rmsd_to_A_conventional"]),
                    float(row["rmsd_to_B_conventional"]),
                )
                is_extra = row["candidate_id"] not in selected_ids
                failures = gate_failures(row, tolerance)
                if is_extra:
                    extras.append(float(row["occupancy"]))
                    if recovered and pair_pass and failures:
                        causal_extras.append(float(row["occupancy"]))
                        extra_failures.extend(failures)
                slot_rows.append(
                    {
                        "site": site,
                        "start": start,
                        "active_conformers": len(active),
                        "candidate_id": row["candidate_id"],
                        "conformer": int(row["conformer"]),
                        "occupancy": float(row["occupancy"]),
                        "assignment": row["assignment"],
                        "rmsd_to_A": float(row["rmsd_to_A_conventional"]),
                        "rmsd_to_B": float(row["rmsd_to_B_conventional"]),
                        "nearest_deposited_rmsd": nearest,
                        "selected_assigned_pair": not is_extra,
                        "extra_conformer": is_extra,
                        "failed_gates_at_tolerance": ";".join(failures),
                    }
                )
            start_rows.append(
                {
                    "site": site,
                    "start": start,
                    "active_conformers": len(active),
                    "both_found": as_bool(ensemble["both_found_conventional"]),
                    "occupancy_qualified": recovered,
                    "assigned_pair_pass_at_tolerance": pair_pass,
                    "all_active_pass_at_tolerance": all_pass,
                    "lost_to_extras_only": recovered and pair_pass and not all_pass,
                    "extra_failed_gates": ";".join(sorted(set(extra_failures))),
                }
            )
        site_summary[site] = {
            "recovery_and_occupancy": recovered_occupancy,
            "assigned_pair_pass_at_tolerance": assigned_passes,
            "all_active_pass_at_tolerance": all_passes,
            "starts_lost_to_extras_only": assigned_passes - all_passes,
            "extra_active_occupancy": describe(extras),
            "causal_extra_occupancy": describe(causal_extras),
            "causal_extras_0_05_to_0_075": sum(
                0.05 <= value < 0.075 for value in causal_extras
            ),
            "causal_extras_0_05_to_0_10": sum(
                0.05 <= value < 0.10 for value in causal_extras
            ),
        }
    return slot_rows, start_rows, site_summary


def pdb_environment_summary(path: Path) -> dict[str, object]:
    atoms = []
    residues = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atoms.append(line)
        residues.append(
            (line[21].strip(), int(line[22:26]), line[26].strip(), line[17:20].strip())
        )
    unique = sorted(set(residues))
    polymer = [item for item in unique if item[3] not in {"HOH", "WAT"}]
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "atoms": len(atoms),
        "residues": len(unique),
        "polymer_residues": len(polymer),
        "residue_span": (
            f"{polymer[0][0]}:{polymer[0][1]}-{polymer[-1][0]}:{polymer[-1][1]}"
            if polymer
            else ""
        ),
    }


def five_z_eight_h(
    conformers: list[dict[str, str]],
    starts_path: Path,
    frozen_a: Path,
    frozen_b: Path,
    soft_terms_path: Path,
    contacts_path: Path,
    tolerance: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = [
        row for row in conformers
        if row["site"] == "5Z8H_A_MET730" and row["assignment"] in {"A", "B"}
    ]
    raw = []
    assignment_summary = {}
    for assignment in ("A", "B"):
        assigned = [row for row in rows if row["assignment"] == assignment]
        rmsds = [matched_rmsd(row) for row in assigned]
        margins = [float(row["tmol_delta_vs_matched_AB"]) for row in assigned]
        assignment_summary[assignment] = {
            "margins": describe(margins),
            "matched_rmsd": describe(rmsds),
            "pass_at_tolerance": sum(value <= tolerance for value in margins),
            "pass_rate_at_tolerance": (
                sum(value <= tolerance for value in margins) / len(margins)
                if margins else None
            ),
            "margin_vs_rmsd": correlations(rmsds, margins),
        }
        for row, rmsd, margin in zip(assigned, rmsds, margins):
            raw.append(
                {
                    "candidate_id": row["candidate_id"],
                    "start": int(row["start"]),
                    "conformer": int(row["conformer"]),
                    "assignment": assignment,
                    "occupancy": float(row["occupancy"]),
                    "matched_rmsd": rmsd,
                    "tmol_margin": margin,
                    "pass_at_tolerance": margin <= tolerance,
                }
            )
    first_start = read_csv(starts_path)[0]
    floors = [
        row for row in read_csv(soft_terms_path)
        if row["site"] == "5Z8H_A_MET730"
    ]
    contacts = [
        row for row in read_csv(contacts_path)
        if row["site"] == "5Z8H_A_MET730"
    ]
    summary = {
        "site": "5Z8H_A_MET730",
        "tmol_tolerance": tolerance,
        "deposited_occupancies": {
            "A": float(first_start["target_A_occupancy"]),
            "B": float(first_start["target_B_occupancy"]),
        },
        "assignment_summary": assignment_summary,
        "frozen_environments": {
            "A": pdb_environment_summary(frozen_a),
            "B": pdb_environment_summary(frozen_b),
        },
        "frozen_environments_byte_identical": (
            frozen_a.read_bytes() == frozen_b.read_bytes()
        ),
        "deposited_soft_floors": floors,
        "deposited_contact_contributions": contacts,
    }
    return raw, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite", type=Path, required=True)
    parser.add_argument("--conformer-table", type=Path, action="append", required=True)
    parser.add_argument("--ensemble-table", type=Path, action="append", required=True)
    parser.add_argument("--starts-5z8h", type=Path, required=True)
    parser.add_argument("--frozen-a-5z8h", type=Path, required=True)
    parser.add_argument("--frozen-b-5z8h", type=Path, required=True)
    parser.add_argument("--soft-terms", type=Path, required=True)
    parser.add_argument("--soft-contacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tmol-tolerance", type=float, default=0.44)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    conformers = [
        row for path in args.conformer_table for row in read_csv(path)
    ]
    ensembles = [
        row for path in args.ensemble_table for row in read_csv(path)
    ]
    if len({(row["site"], row["start"]) for row in ensembles}) != 1000:
        raise ValueError("expected exactly 1000 unique site/start ensembles")

    bins, tolerance_summary = tolerance_validation(
        read_csv(args.composite / "per_conformer_tmol_margins.csv"),
        args.tmol_tolerance,
    )
    slot_rows, start_rows, extra_summary = characterize_extra_conformers(
        conformers,
        ensembles,
        {"2V05_A_HIS168", "8Q6Q_B_ASP81"},
        args.tmol_tolerance,
    )
    five_z_raw, five_z_summary = five_z_eight_h(
        conformers,
        args.starts_5z8h,
        args.frozen_a_5z8h,
        args.frozen_b_5z8h,
        args.soft_terms,
        args.soft_contacts,
        args.tmol_tolerance,
    )

    args.output.mkdir(parents=True)
    atomic_csv(args.output / "tmol_margin_rmsd_bins.csv", bins)
    atomic_csv(args.output / "extra_active_conformers.csv", slot_rows)
    atomic_csv(args.output / "extra_active_starts.csv", start_rows)
    atomic_csv(args.output / "5z8h_assignment_margins.csv", five_z_raw)
    atomic_json(
        args.output / "summary.json",
        {
            "tmol_tolerance_validation": tolerance_summary,
            "extra_active_conformers": extra_summary,
            "5z8h": five_z_summary,
        },
    )
    print(
        json.dumps(
            {
                "tmol_tolerance_validation": tolerance_summary,
                "extra_active_conformers": extra_summary,
                "5z8h": five_z_summary,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
