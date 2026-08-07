from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def target_block_index(lines: list[str], chain: str, residue_number: int) -> int:
    residues = []
    seen = set()
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        key = (line[21].strip(), int(line[22:26]), line[26].strip())
        if key not in seen:
            seen.add(key)
            residues.append(key)
    target = (chain, residue_number, "")
    if target not in seen:
        matches = [
            index for index, value in enumerate(residues)
            if value[0] == chain and value[1] == residue_number
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot locate target block {chain}:{residue_number}")
        return matches[0]
    return residues.index(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformer-table", type=Path, action="append", required=True)
    parser.add_argument("--tmol-input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outlier-probability", type=float, default=0.003)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import tmol
    import torch
    from tmol.score.score_function import ScoreFunction

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    threshold = -math.log(args.outlier_probability)

    audit_rows = {
        row["candidate_id"]: row
        for path in args.conformer_table
        for row in read_csv(path)
        if as_bool(row["rotamer_within_allowed_width"])
    }
    sites = {}
    for path in args.tmol_input:
        root = path.parent
        for site in json.loads(path.read_text())["sites"]:
            sites[site["site"]] = (root, site)

    result_rows = []
    by_residue: dict[str, list[bool]] = defaultdict(list)
    candidate_path = Path("/tmp/current_endpoint_dunbrack_candidate.pdb")
    for site_name in sorted({row["site"] for row in audit_rows.values()}):
        root, site = sites[site_name]
        candidates = {
            candidate["candidate_id"]: candidate
            for candidate in site["candidates"]
            if candidate["candidate_id"] in audit_rows
        }
        if not candidates:
            continue
        base_key = "base_pdb_A" if "base_pdb_A" in site else "base_pdb"
        base_pdb = root / site[base_key]
        base_lines = base_pdb.read_text().splitlines()
        base_pose = tmol.pose_stack_from_pdb(str(base_pdb), device=device)
        score_function = ScoreFunction(tmol.ParameterDatabase.get_default(), device)
        for score_type in (
            tmol.ScoreType.dunbrack_rot,
            tmol.ScoreType.dunbrack_rotdev,
            tmol.ScoreType.dunbrack_semirot,
        ):
            score_function.set_weight(score_type, 1.0)
        scorer = score_function.render_block_pair_scoring_module(base_pose)
        block = target_block_index(
            base_lines, site["chain"], int(site["residue_number"])
        )
        for candidate_id, candidate in sorted(candidates.items()):
            replacements = dict(zip(site["atom_names"], candidate["coordinates"]))
            candidate_lines = []
            for line in base_lines:
                if (
                    line.startswith("ATOM")
                    and line[21].strip() == site["chain"]
                    and int(line[22:26]) == int(site["residue_number"])
                    and line[12:16].strip() in replacements
                ):
                    x, y, z = replacements[line[12:16].strip()]
                    line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                candidate_lines.append(line)
            candidate_path.write_text("\n".join(candidate_lines) + "\n")
            pose = tmol.pose_stack_from_pdb(str(candidate_path), device=device)
            score = float(
                scorer(pose.coords)[0, block, block].detach().cpu()
            )
            outlier = score > threshold
            audit = audit_rows[candidate_id]
            by_residue[site["residue_type"]].append(outlier)
            result_rows.append(
                {
                    "candidate_id": candidate_id,
                    "site": site_name,
                    "residue": site["residue_type"],
                    "start": int(audit["start"]),
                    "conformer": int(audit["conformer"]),
                    "occupancy": float(audit["occupancy"]),
                    "chi_degrees": audit["chi_degrees"],
                    "current_rotamer_gate_pass": True,
                    "dunbrack_target_residue_energy": score,
                    "probability_equivalent_exp_neg_energy": math.exp(-score),
                    "dunbrack_outlier_threshold": threshold,
                    "independent_library_outlier": outlier,
                    "gate_library_disagreement": outlier,
                }
            )
        print(f"scored {site_name}: {len(candidates)}", flush=True)

    summary_rows = []
    for residue, values in sorted(by_residue.items()):
        disagreements = sum(values)
        summary_rows.append(
            {
                "residue": residue,
                "current_gate_passing_conformers": len(values),
                "independent_library_outliers": disagreements,
                "disagreement_rate": disagreements / len(values),
            }
        )
    args.output.mkdir(parents=True)
    atomic_csv(args.output / "per_conformer_dunbrack.csv", result_rows)
    atomic_csv(args.output / "disagreement_by_residue.csv", summary_rows)
    atomic_json(
        args.output / "summary.json",
        {
            "library": "tmol bundled backbone-dependent Dunbrack",
            "classifier": (
                "unweighted target-residue dunbrack_rot + dunbrack_rotdev + "
                "dunbrack_semirot energy > -ln(0.003)"
            ),
            "outlier_probability": args.outlier_probability,
            "outlier_energy_threshold": threshold,
            "population": (
                "all current active endpoint conformers passing the production "
                "rotamer gate; no sampling"
            ),
            "conformers": len(result_rows),
            "disagreement_by_residue": summary_rows,
        },
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
