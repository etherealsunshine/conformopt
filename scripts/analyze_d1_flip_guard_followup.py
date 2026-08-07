#!/usr/bin/env python3
"""Guard breakdown and deviation-matched do-nothing comparison for D1 flips."""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t

from qfit.qfit import QFitOptions, QFitRotamericResidue
from qfit.structure import Structure

from run_d1_tier_a_flips import BACKBONE_NAMES, get_sampler_xmap, source_path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def residue_missing_backbone(residue):
    return [name for name in BACKBONE_NAMES if len(residue.select("name", name)) != 1]


def residue_local_xyz(residue, name):
    selection = residue.select("name", name)
    if len(selection) != 1:
        return None
    global_index = int(selection[0])
    position = np.searchsorted(residue.selection, global_index)
    return residue.coor[position]


def boundary_cause(previous, following):
    """Classify a qFit segment boundary between consecutive chain residues."""
    if previous is None or following is None:
        return "chain_terminus"
    missing = residue_missing_backbone(previous) + residue_missing_backbone(following)
    if missing:
        return "neighbour_missing_backbone_atoms"
    number_gap = following.id[0] - previous.id[0]
    has_insertion = bool(previous.id[1] or following.id[1])
    c = residue_local_xyz(previous, "C")
    n = residue_local_xyz(following, "N")
    cn_distance = float(np.linalg.norm(c - n))
    if number_gap != 1 or has_insertion:
        if number_gap > 1:
            return "missing_residue_or_numbering_gap"
        return "insertion_code_or_numbering_discontinuity"
    if not previous.is_next_residue(following):
        return "chain_break"
    return "connected_chain_but_segment_boundary"


def guard_context(row):
    path, split = source_path(row["pdb_id"])
    structure = Structure.fromfile(path).extract("altloc", ("", "A"))
    residue_id = (int(row["resnum"]), "")
    residue = structure[row["chain"]].conformers[0][residue_id]
    options = QFitOptions()
    options.qp_solver = options.miqp_solver = "CVXPYSolver"
    qfit = QFitRotamericResidue(residue, structure, get_sampler_xmap(), options)
    segment = qfit.segment
    index = segment.find(qfit.residue.id)
    required = qfit.options.neighbor_residues_required
    chain_residues = list(structure[row["chain"]].conformers[0].residues)
    chain_index = next(i for i, candidate in enumerate(chain_residues) if candidate.id == residue.id)
    lower_missing = max(0, required - index)
    upper_available = len(segment) - index - 1
    upper_missing = max(0, required - upper_available)
    # Attribute insufficiency to the *segment boundary*, not necessarily the
    # residue adjacent to the target.  For example, a target can have two
    # connected lower neighbours but still lack a third because the segment
    # broke just beyond them.
    segment_first_chain_index = chain_index - index
    segment_last_chain_index = chain_index + upper_available
    lower_outside = (
        chain_residues[segment_first_chain_index - 1]
        if segment_first_chain_index > 0 else None
    )
    lower_inside = chain_residues[segment_first_chain_index]
    upper_inside = chain_residues[segment_last_chain_index]
    upper_outside = (
        chain_residues[segment_last_chain_index + 1]
        if segment_last_chain_index + 1 < len(chain_residues) else None
    )
    lower_cause = boundary_cause(lower_outside, lower_inside) if lower_missing else "not_guarded"
    upper_cause = boundary_cause(upper_inside, upper_outside) if upper_missing else "not_guarded"
    return {
        "site": row["site"], "source_split": split,
        "segment_length": len(segment), "central_index": index,
        "available_lower_neighbours": index,
        "available_upper_neighbours": upper_available,
        "missing_lower_neighbours": lower_missing,
        "missing_upper_neighbours": upper_missing,
        "lower_cause": lower_cause, "upper_cause": upper_cause,
        "lower_adjacent_ids": "" if lower_outside is None else str(lower_outside.id),
        "upper_adjacent_ids": "" if upper_outside is None else str(upper_outside.id),
    }


def fit_group_adjusted_ratio(rows):
    """OLS: residual/deviation ~ intercept + deviation + guard indicator."""
    deviation = np.asarray([float(row["backbone_deviation_A"]) for row in rows])
    guard = np.asarray([row["sampling_outcome"] != "generated_19" for row in rows], dtype=float)
    ratio = np.asarray([
        float(row["tier_a_min_central_backbone_rmsd_A"]) / float(row["backbone_deviation_A"])
        for row in rows
    ])
    design = np.column_stack((np.ones(len(rows)), deviation, guard))
    beta, _, _, _ = np.linalg.lstsq(design, ratio, rcond=None)
    residual = ratio - design @ beta
    dof = len(rows) - design.shape[1]
    covariance = (residual @ residual / dof) * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(covariance))
    t_value = beta[2] / se[2]
    p_value = 2.0 * student_t.sf(abs(t_value), dof)
    return {
        "guard_adjusted_ratio_difference": float(beta[2]),
        "guard_adjusted_ratio_standard_error": float(se[2]),
        "guard_adjusted_ratio_p_value": float(p_value),
        "deviation_coefficient": float(beta[1]),
        "model_residual_sd": float(np.sqrt(residual @ residual / dof)),
    }


def match_by_deviation(guard, sampled):
    guard_deviation = np.asarray([float(row["backbone_deviation_A"]) for row in guard])
    sampled_deviation = np.asarray([float(row["backbone_deviation_A"]) for row in sampled])
    cost = np.abs(guard_deviation[:, None] - sampled_deviation[None, :])
    guard_i, sampled_i = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(guard_i, sampled_i):
        g, s = guard[i], sampled[j]
        g_ratio = float(g["tier_a_min_central_backbone_rmsd_A"]) / float(g["backbone_deviation_A"])
        s_ratio = float(s["tier_a_min_central_backbone_rmsd_A"]) / float(s["backbone_deviation_A"])
        matches.append({
            "guard_site": g["site"], "sampled_site": s["site"],
            "guard_deviation_A": float(g["backbone_deviation_A"]),
            "sampled_deviation_A": float(s["backbone_deviation_A"]),
            "absolute_deviation_difference_A": abs(float(g["backbone_deviation_A"]) - float(s["backbone_deviation_A"])),
            "guard_residual_over_deviation": g_ratio,
            "sampled_residual_over_deviation": s_ratio,
            "guard_minus_sampled_ratio": g_ratio - s_ratio,
        })
    return matches


def ratio_line(rows):
    deviation = np.asarray([float(row["backbone_deviation_A"]) for row in rows])
    ratio = np.asarray([
        float(row["tier_a_min_central_backbone_rmsd_A"]) / float(row["backbone_deviation_A"])
        for row in rows
    ])
    slope, intercept = np.polyfit(deviation, ratio, 1)
    return {"intercept": float(intercept), "slope_per_A": float(slope)}


def make_figure(guard, sampled, output):
    fig, ax = plt.subplots(figsize=(6.5, 4.7))
    for rows, colour, label in ((guard, "#777777", "guard: deposited A only"),
                                (sampled, "#c44e52", "qFit: 19 candidates")):
        x = np.asarray([float(row["backbone_deviation_A"]) for row in rows])
        y = np.asarray([float(row["tier_a_min_central_backbone_rmsd_A"]) / float(row["backbone_deviation_A"]) for row in rows])
        ax.scatter(x, y, color=colour, label=label)
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, intercept + slope * grid, color=colour, alpha=.65)
    ax.set(xlabel="Deposited backbone deviation (Å)", ylabel="tier-(a) residual / deviation",
           title="D1 flips: sampling versus deposited-A-only at matched scale")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.input.open() as handle:
        rows = list(csv.DictReader(handle))
    guard = [row for row in rows if row["sampling_outcome"] != "generated_19"]
    sampled = [row for row in rows if row["sampling_outcome"] == "generated_19"]
    guard_rows = [{**row, **guard_context(row)} for row in guard]
    truncated = [row for row in rows if row["qfit_guard_accepts"] == "True" and row["strict_7_residue_window_complete"] != "True"]
    matches = match_by_deviation(guard, sampled)
    raw_guard_ratio = [float(row["tier_a_min_central_backbone_rmsd_A"]) / float(row["backbone_deviation_A"]) for row in guard]
    raw_sampled_ratio = [float(row["tier_a_min_central_backbone_rmsd_A"]) / float(row["backbone_deviation_A"]) for row in sampled]
    summary = {
        "guard_sites": len(guard), "sampled_sites": len(sampled),
        "qfit_guard_accepted_truncated_window_sites": len(truncated),
        "raw_guard_median_residual_over_deviation": float(np.median(raw_guard_ratio)),
        "raw_sampled_median_residual_over_deviation": float(np.median(raw_sampled_ratio)),
        "matched_pairs": len(matches),
        "matched_median_abs_deviation_difference_A": float(np.median([row["absolute_deviation_difference_A"] for row in matches])),
        "matched_max_abs_deviation_difference_A": float(max(row["absolute_deviation_difference_A"] for row in matches)),
        "matched_guard_median_residual_over_deviation": float(np.median([row["guard_residual_over_deviation"] for row in matches])),
        "matched_sampled_median_residual_over_deviation": float(np.median([row["sampled_residual_over_deviation"] for row in matches])),
        "matched_median_guard_minus_sampled_ratio": float(np.median([row["guard_minus_sampled_ratio"] for row in matches])),
        "strict_matched_pairs_within_0.1A": sum(row["absolute_deviation_difference_A"] <= 0.1 for row in matches),
        "guard_ratio_line": ratio_line(guard),
        "sampled_ratio_line": ratio_line(sampled),
        **fit_group_adjusted_ratio(rows),
    }
    atomic_csv(args.output / "guard_breakdown.csv", guard_rows)
    atomic_csv(args.output / "deviation_matched_pairs.csv", matches)
    atomic_csv(args.output / "truncated_window_guard_accepts.csv", truncated)
    make_figure(guard, sampled, args.output / "ratio_vs_deviation.png")
    atomic_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
