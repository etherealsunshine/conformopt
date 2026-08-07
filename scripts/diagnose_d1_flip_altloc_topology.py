#!/usr/bin/env python3
"""Check whether deposited A/B labels form a connected group around flip sites.

The earlier closure diagnostic compares coordinates selected as A and B at an
anchor.  That coordinate distance is label-swap invariant.  This script adds
the missing, label-sensitive prerequisite: every residue between a flip and an
anchor must explicitly carry complete backbone A and B records before those
labels can be treated as a single conformational group.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from diagnose_d1_flip_closure import atom_coordinate
from run_d1_reachability import BACKBONE_NAMES
from run_d1_tier_a_flips import source_path
from qfit.structure import Structure


KS = (3, 4, 5, 6, 8, 10)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def site_key(site: dict[str, object]) -> str:
    return f"{site['pdb_id']}_{site['chain']}_{site['resname']}{site['resnum']}"


def raw_backbone_altlocs(path: str) -> dict[tuple[str, int, str], dict[str, set[str]]]:
    """Return explicit raw-PDB altloc labels for each backbone atom.

    Blank-altloc atoms are deliberately recorded as blank rather than silently
    copied into A/B.  They are a nonflexible gap for purposes of label-group
    continuity, even though qFit extracts them into both conformer selections.
    """
    records: dict[tuple[str, int, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE_NAMES:
            continue
        try:
            residue_id = (line[21], int(line[22:26]), line[26].strip())
        except ValueError:
            continue
        records[residue_id][atom].add(line[16].strip())
    return records


def label_class(labels: dict[str, set[str]]) -> str:
    explicit_a_b = all({"A", "B"}.issubset(labels.get(name, set())) for name in BACKBONE_NAMES)
    if explicit_a_b:
        return "explicit_complete_A_B"
    blank_complete = all("" in labels.get(name, set()) for name in BACKBONE_NAMES)
    if blank_complete:
        return "shared_blank_backbone"
    if any("A" in labels.get(name, set()) or "B" in labels.get(name, set()) for name in BACKBONE_NAMES):
        return "partial_or_mixed_A_B"
    return "missing_or_non_A_B_backbone"


def structure_segments(site: dict[str, object]):
    path, split = source_path(str(site["pdb_id"]))
    residue_id = (int(site["resnum"]), str(site.get("insertion_code", "")))
    a_structure = Structure.fromfile(path).extract("altloc", ("", "A"))
    b_structure = Structure.fromfile(path).extract("altloc", ("", "B"))
    a_chain = a_structure[str(site["chain"])].conformers[0]
    b_chain = b_structure[str(site["chain"])].conformers[0]
    a_segment = next(segment for segment in a_chain.segments if any(residue.id == residue_id for residue in segment.residues))
    b_segment = next(segment for segment in b_chain.segments if any(residue.id == residue_id for residue in segment.residues))
    a_index = [residue.id for residue in a_segment.residues].index(residue_id)
    b_index = [residue.id for residue in b_segment.residues].index(residue_id)
    return path, split, a_segment, b_segment, a_index, b_index


def anchor_difference(a_residue, b_residue) -> float:
    a = np.asarray([atom_coordinate(a_residue, name) for name in BACKBONE_NAMES])
    b = np.asarray([atom_coordinate(b_residue, name) for name in BACKBONE_NAMES])
    return float(np.max(np.abs(a - b)))


def direction_run(classes: dict[int, str], direction: int, k: int) -> tuple[bool, list[int]]:
    offsets = list(range(0, direction * k + direction, direction))
    gaps = [offset for offset in offsets if classes.get(offset) != "explicit_complete_A_B"]
    return not gaps, gaps


def analyze(site: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    path, split, a_segment, b_segment, a_index, b_index = structure_segments(site)
    raw = raw_backbone_altlocs(path)
    classes: dict[int, str] = {}
    residue_by_offset: dict[int, str] = {}
    for offset in range(-10, 11):
        position = a_index + offset
        if 0 <= position < len(a_segment.residues):
            residue = a_segment.residues[position]
            residue_by_offset[offset] = str(residue.id)
            classes[offset] = label_class(raw.get((str(site["chain"]), int(residue.id[0]), str(residue.id[1])), {}))
    complete_minus = next((k - 1 for k in range(1, 11) if classes.get(-k) != "explicit_complete_A_B"), 10)
    complete_plus = next((k - 1 for k in range(1, 11) if classes.get(k) != "explicit_complete_A_B"), 10)
    per_k = []
    for k in KS:
        row = {"site": site_key(site), "k": k, "source_split": split,
               "minus_anchor_id": residue_by_offset.get(-k), "plus_anchor_id": residue_by_offset.get(k),
               "minus_anchor_label_class": classes.get(-k, "anchor_unavailable"),
               "plus_anchor_label_class": classes.get(k, "anchor_unavailable")}
        minus_available = a_index - k >= 0 and b_index - k >= 0
        plus_available = a_index + k < len(a_segment.residues) and b_index + k < len(b_segment.residues)
        minus_contiguous, minus_gaps = direction_run(classes, -1, k) if minus_available else (False, [])
        plus_contiguous, plus_gaps = direction_run(classes, +1, k) if plus_available else (False, [])
        row.update({"minus_anchor_available": minus_available, "plus_anchor_available": plus_available,
                    "minus_contiguous_explicit_A_B": minus_contiguous,
                    "plus_contiguous_explicit_A_B": plus_contiguous,
                    "minus_gap_offsets": ";".join(map(str, minus_gaps)),
                    "plus_gap_offsets": ";".join(map(str, plus_gaps))})
        if minus_available:
            matched = anchor_difference(a_segment.residues[a_index - k], b_segment.residues[b_index - k])
            # Algebraically identical: max |A-B| == max |B-A|.  Retain both
            # columns to make the requested swap check explicit and auditable.
            row["minus_matched_A_B_difference_A"] = matched
            row["minus_swapped_B_A_difference_A"] = matched
            row["minus_swap_reduces_difference"] = False
        if plus_available:
            matched = anchor_difference(a_segment.residues[a_index + k], b_segment.residues[b_index + k])
            row["plus_matched_A_B_difference_A"] = matched
            row["plus_swapped_B_A_difference_A"] = matched
            row["plus_swap_reduces_difference"] = False
        per_k.append(row)
    profile = "; ".join(f"{offset}:{classes.get(offset, 'unavailable')}" for offset in range(-10, 11))
    summary = {"site": site_key(site), "source_split": split, "centre_label_class": classes.get(0, "unavailable"),
               "explicit_A_B_run_from_centre_minus_residues": complete_minus,
               "explicit_A_B_run_from_centre_plus_residues": complete_plus,
               "label_topology_offsets_minus10_to_plus10": profile}
    return summary, per_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = Path("/home/dev/qfit_unet_data/qfit_audit/d6_tier1_native_bic_v1/panel_manifest.json")
    sites = [site for site in json.loads(manifest.read_text()) if site["panel"] == "flip_filter"]
    closure_csv = Path("/home/dev/qfit_unet_data/qfit_audit/d1_flip_closure_anchors33_v2/per_site.csv")
    with closure_csv.open() as handle:
        testable = {row["site"] for row in csv.DictReader(handle) if row["status"] == "complete"}
    sites = [site for site in sites if site_key(site) in testable]
    summaries, rows = [], []
    for site in sites:
        summary, site_rows = analyze(site)
        summaries.append(summary); rows.extend(site_rows)
        atomic_csv(args.output / "per_site.csv", summaries)
        atomic_csv(args.output / "per_site_per_k.csv", rows)
        atomic_json(args.output / "progress.json", {"sites_complete": len(summaries), "sites_total": len(sites), "last_site": summary["site"]})
    summary = {"status": "complete", "sites": len(summaries),
               "k3_both_directions_contiguous_explicit_A_B": int(sum(
                   next(row for row in rows if row["site"] == item["site"] and row["k"] == 3)["minus_contiguous_explicit_A_B"] and
                   next(row for row in rows if row["site"] == item["site"] and row["k"] == 3)["plus_contiguous_explicit_A_B"]
                   for item in summaries)),
               "label_swap_reduces_anchor_difference_cases": 0}
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "progress.json", {"status": "complete", **summary})


if __name__ == "__main__":
    main()
