"""Plot the frozen-v3 20-site synthetic recovery cascade as a bar table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


EXPECTED_TOTALS = {
    "starts": 1000,
    "v3_both_found": 742,
    "v3_occupancy": 714,
    "v3_rotamer": 710,
    "v3_direct_clash": 710,
    "v3_symmetry_clash": 710,
    "v3_tmol_0_44": 626,
}


def load_rows(path: Path) -> tuple[list[dict[str, int | str]], dict[str, int]]:
    with path.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    total_source = [row for row in source if row["site"] == "TOTAL"]
    site_source = [row for row in source if row["site"] != "TOTAL"]
    if len(total_source) != 1 or len(site_source) != 20:
        raise RuntimeError(
            f"Expected 20 sites and one TOTAL row; found {len(site_source)} and "
            f"{len(total_source)}"
        )

    total = {key: int(total_source[0][key]) for key in EXPECTED_TOTALS}
    if total != EXPECTED_TOTALS:
        raise RuntimeError(f"Frozen-v3 cascade guard failed: {total}")

    rows: list[dict[str, int | str]] = []
    for row in site_source:
        starts = int(row["starts"])
        both = int(row["v3_both_found"])
        occupancy = int(row["v3_occupancy"])
        strict = int(row["v3_tmol_0_44"])
        if starts != 50:
            raise RuntimeError(f"Expected 50 starts at {row['site']}, found {starts}")
        if not 0 <= strict <= occupancy <= both <= starts:
            raise RuntimeError(
                f"Non-monotone displayed cascade at {row['site']}: "
                f"{both}, {occupancy}, {strict}"
            )
        rows.append(
            {
                "site": row["site"],
                "site_label": row["site"].split("_", 1)[0],
                "starts": starts,
                "both_conformers": both,
                "plus_occupancy": occupancy,
                "plus_frozen_physical_audit": strict,
            }
        )

    rows.sort(
        key=lambda row: (
            -int(row["plus_frozen_physical_audit"]),
            -int(row["plus_occupancy"]),
            -int(row["both_conformers"]),
            str(row["site_label"]),
        )
    )
    return rows, total


def write_csv(path: Path, rows: list[dict[str, int | str]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, int | str]], total: dict[str, int], output: Path) -> None:
    stage_keys = (
        "both_conformers",
        "plus_occupancy",
        "plus_frozen_physical_audit",
    )
    headings = (
        "Both conformers",
        "+ occupancy",
        "+ physical audit",
    )
    totals = (
        total["v3_both_found"],
        total["v3_occupancy"],
        total["v3_tmol_0_44"],
    )
    fills = ("#9EC7EE", "#F1C39F", "#ACD8B9")
    tracks = ("#EEF2F5", "#F2F2F2", "#EEF2F0")

    figure = plt.figure(figsize=(12.4, 12.0), facecolor="white")
    axis = figure.add_axes((0.035, 0.055, 0.94, 0.91))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    label_x = 0.0
    columns = ((0.18, 0.43), (0.46, 0.71), (0.74, 0.99))
    note_y = 0.985
    header_y = 0.945
    top_y = 0.915
    bottom_y = 0.055
    row_height = (top_y - bottom_y) / len(rows)
    bar_height = row_height * 0.72

    axis.text(
        label_x,
        note_y,
        "Frozen synthetic v3 · 20 sites · 50 starts per site · "
        "strict metric qfit-synth20-merge050-one-to-one-tmol044-v3",
        ha="left",
        va="top",
        fontsize=11,
        color="#6F7478",
    )
    axis.text(
        label_x,
        header_y,
        "Held-out site",
        ha="left",
        va="center",
        fontsize=11,
        color="#6F7478",
    )
    for heading, stage_total, (left, right) in zip(headings, totals, columns):
        axis.text(
            left,
            header_y,
            heading,
            ha="left",
            va="center",
            fontsize=11,
            color="#6F7478",
        )
        axis.text(
            right,
            header_y,
            f"{stage_total}/1000",
            ha="right",
            va="center",
            fontsize=10,
            color="#8A8E92",
        )

    for index, row in enumerate(rows):
        center_y = top_y - (index + 0.5) * row_height
        axis.plot(
            [0, 1],
            [top_y - index * row_height, top_y - index * row_height],
            color="#E8EAEC",
            linewidth=0.65,
            zorder=0,
        )
        axis.text(
            label_x,
            center_y,
            str(row["site_label"]),
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#282B2E",
        )
        for stage, fill, track, (left, right) in zip(
            stage_keys, fills, tracks, columns
        ):
            width = right - left
            value = int(row[stage])
            axis.add_patch(
                Rectangle(
                    (left, center_y - bar_height / 2),
                    width,
                    bar_height,
                    facecolor=track,
                    edgecolor="none",
                    zorder=1,
                )
            )
            if value:
                axis.add_patch(
                    Rectangle(
                        (left, center_y - bar_height / 2),
                        width * value / 50,
                        bar_height,
                        facecolor=fill,
                        edgecolor="none",
                        zorder=2,
                    )
                )
            axis.text(
                right - 0.008,
                center_y,
                str(value),
                ha="right",
                va="center",
                fontsize=11.5,
                color="#202326" if value else "#85898D",
                zorder=3,
            )

    axis.plot([0, 1], [bottom_y, bottom_y], color="#E8EAEC", linewidth=0.65)
    for left, right in columns:
        axis.text(left, 0.025, "0", ha="left", va="center", fontsize=10, color="#808489")
        axis.text(right, 0.025, "50", ha="right", va="center", fontsize=10, color="#808489")

    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    rows, total = load_rows(args.input)
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "frozen_v3_recovery_cascade.csv", rows)
    plot(rows, total, args.output_dir / "frozen_v3_recovery_cascade.png")


if __name__ == "__main__":
    main()
