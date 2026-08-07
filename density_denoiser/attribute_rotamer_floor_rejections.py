from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path


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
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def attribute_rejections(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    rejected = [
        row for row in rows
        if row["rotamer_pass"].strip().lower() != "true"
    ]
    attributions = []
    for row in rejected:
        angles = [float(value) for value in row["chi_degrees"].split(";")]
        deviations = [
            float(value) for value in row["rotamer_deviation_degrees"].split(";")
        ]
        widths = [
            float(value) for value in row["rotamer_allowed_width_degrees"].split(";")
        ]
        states = row["nearest_rotamer"].split("/")
        for index, (angle, deviation, width, state) in enumerate(
            zip(angles, deviations, widths, states), start=1
        ):
            if deviation > width:
                attributions.append({
                    "site": row["site"],
                    "control": row["control"],
                    "residue_name": row["residue_name"],
                    "failed_chi": f"chi{index}",
                    "chi_degrees": angle,
                    "nearest_state": state,
                    "deviation_degrees": deviation,
                    "allowed_width_degrees": width,
                    "excess_degrees": deviation - width,
                })
    return rejected, attributions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attribute deposited-floor rotamer rejections by chi."
    )
    parser.add_argument("--audit-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    rejected, attributions = attribute_rejections(_read_csv(args.audit_table))
    attributed_controls = {
        (row["site"], row["control"]) for row in attributions
    }
    rejected_controls = {
        (row["site"], row["control"]) for row in rejected
    }
    if attributed_controls != rejected_controls:
        raise ValueError("one or more rejected conformers has no failing chi")

    args.output.mkdir(parents=True)
    _atomic_csv(args.output / "rotamer_rejection_per_chi.csv", attributions)
    counts: dict[tuple[str, str], int] = {}
    for row in attributions:
        key = (str(row["residue_name"]), str(row["failed_chi"]))
        counts[key] = counts.get(key, 0) + 1
    summary_rows = [
        {"residue_name": key[0], "failed_chi": key[1], "rejections": value}
        for key, value in sorted(counts.items())
    ]
    _atomic_csv(args.output / "rotamer_rejection_summary.csv", summary_rows)
    _atomic_json(args.output / "summary.json", {
        "rejected_conformers": len(rejected),
        "attribution_rows": len(attributions),
        "further_width_change_made": False,
        "counts": summary_rows,
    })


if __name__ == "__main__":
    main()
