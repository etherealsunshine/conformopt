from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import gemmi
import numpy as np
import torch

from experiments.probe4.core import dihedral, torsion_to_coords, wrap_angles

from .data_pipeline import _sidechain_atoms
from .dataset import manifest_path, read_manifest
from .five_site_optimizer import CHI_SPECS, _alt_atom_map, _canonical_centers


MAX_ATOM_SLOTS = 16
CANDIDATE_LABELS = (
    "deposited_A_plus_B",
    "A_only",
    "B_only",
    "near_15deg",
    "near_35deg",
    "random_rotamer_1",
    "random_rotamer_2",
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
    np.save(temporary, value)
    os.replace(temporary, path)


def _site_seed(key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def _atom_alt(atom: gemmi.Atom) -> str:
    return "" if atom.altloc in ("\x00", " ") else atom.altloc


def _find_residue(structure: gemmi.Structure, record: dict) -> gemmi.Residue | None:
    for chain in structure[0]:
        if chain.name != record["chain"]:
            continue
        for residue in chain:
            if (
                residue.seqid.num == int(record["residue_number"])
                and residue.seqid.icode == record["insertion_code"]
            ):
                return residue
    return None


def _build_site(
    record: dict,
    structure: gemmi.Structure,
    seed: int,
    max_atom_slots: int = MAX_ATOM_SLOTS,
) -> dict | None:
    if record["residue_name"] not in CHI_SPECS:
        return None
    residue = _find_residue(structure, record)
    if residue is None or residue.name != record["residue_name"]:
        return None

    device = torch.device("cpu")
    map_a = _alt_atom_map(residue, "A", device)
    map_b = _alt_atom_map(residue, "B", device)
    sidechain = _sidechain_atoms(residue)
    deposited_labels = {_atom_alt(atom) for atom in sidechain} - {""}
    # This first experiment is explicitly a two-conformer A/B landscape. Sites
    # with C/D states need a K>2 label generator and cannot be silently folded
    # into the A/B native candidate.
    if deposited_labels != {"A", "B"}:
        return None
    shared_atoms = [atom for atom in sidechain if _atom_alt(atom) == ""]
    moving_names = []
    for atom in sidechain:
        name = atom.name.strip()
        if _atom_alt(atom) in {"A", "B"} and name not in moving_names:
            moving_names.append(name)
    if not moving_names:
        return None
    explicit = {
        (name, alt): next((
            atom for atom in sidechain
            if atom.name.strip() == name and _atom_alt(atom) == alt
        ), None)
        for name in moving_names for alt in ("A", "B")
    }
    if any(atom is None or float(atom.occ) < 0.05 for atom in explicit.values()):
        return None

    spec = CHI_SPECS[residue.name]
    if any(name not in map_a or name not in map_b for quartet in spec["dihedrals"] for name in quartet):
        return None
    sidechain_names = []
    for atom in sidechain:
        name = atom.name.strip()
        if name not in sidechain_names:
            sidechain_names.append(name)
    if any(name not in map_a or name not in map_b for name in sidechain_names):
        return None

    # If a proximal side-chain atom is shared, rotations upstream of it cannot
    # vary without breaking its bond. Only perturb chis whose entire downstream
    # tree is explicitly alternate.
    moving_set = set(moving_names)
    movable_chi = torch.tensor([
        all(name in moving_set for name in downstream)
        for _origin, _endpoint, downstream in spec["rotations"]
    ], dtype=torch.bool)
    if not movable_chi.any():
        return None

    fixed_a = {name: value for name, value in map_a.items() if name not in sidechain_names}
    fixed_b = {name: value for name, value in map_b.items() if name not in sidechain_names}
    template_a = torch.stack([map_a[name] for name in sidechain_names])
    template_b = torch.stack([map_b[name] for name in sidechain_names])

    def coordinates(template: torch.Tensor, fixed: dict[str, torch.Tensor], delta: torch.Tensor) -> torch.Tensor:
        return torsion_to_coords(
            template, sidechain_names, delta, list(spec["rotations"]), fixed
        )

    zero = torch.zeros(len(spec["rotations"]), dtype=torch.float32)
    generator = torch.Generator().manual_seed(_site_seed(record["key"], seed))

    def noisy(template: torch.Tensor, fixed: dict[str, torch.Tensor], degrees: float) -> torch.Tensor:
        delta = torch.randn(len(zero), generator=generator) * math.radians(degrees)
        delta = torch.where(movable_chi, delta, torch.zeros_like(delta))
        return coordinates(template, fixed, delta)

    def random_rotamer(template: torch.Tensor, fixed: dict[str, torch.Tensor]) -> torch.Tensor:
        lookup = dict(fixed)
        lookup.update({name: template[index] for index, name in enumerate(sidechain_names)})
        physical = torch.stack([
            wrap_angles(dihedral(*(lookup[name] for name in quartet)) - torch.pi)
            for quartet in spec["dihedrals"]
        ])
        signs = []
        for chi_index in range(len(zero)):
            probe = torch.zeros_like(zero)
            probe[chi_index] = 0.01
            moved = coordinates(template, fixed, probe)
            moved_lookup = dict(fixed)
            moved_lookup.update({name: moved[index] for index, name in enumerate(sidechain_names)})
            moved_chi = wrap_angles(
                dihedral(*(moved_lookup[name] for name in spec["dihedrals"][chi_index]))
                - torch.pi
            )
            response = float(wrap_angles(moved_chi - physical[chi_index]))
            signs.append(1.0 if response >= 0 else -1.0)
        desired = []
        for chi_index in range(len(zero)):
            centers = _canonical_centers(residue.name, chi_index)
            selected = int(torch.randint(len(centers), (1,), generator=generator))
            desired.append(centers[selected])
        delta = torch.tensor(signs) * wrap_angles(torch.tensor(desired) - physical)
        delta = torch.where(movable_chi, delta, torch.zeros_like(delta))
        return coordinates(template, fixed, delta)

    conformer_pairs = (
        (template_a, template_b),
        (template_a, template_a),
        (template_b, template_b),
        (noisy(template_a, fixed_a, 15.0), noisy(template_b, fixed_b, 15.0)),
        (noisy(template_a, fixed_a, 35.0), noisy(template_b, fixed_b, 35.0)),
        (random_rotamer(template_a, fixed_a), random_rotamer(template_b, fixed_b)),
        (random_rotamer(template_a, fixed_a), random_rotamer(template_b, fixed_b)),
    )

    center = torch.tensor(record["center"], dtype=torch.float32)
    positions = torch.zeros((len(CANDIDATE_LABELS), max_atom_slots, 3), dtype=torch.float32)
    sigma2 = torch.ones((len(CANDIDATE_LABELS), max_atom_slots), dtype=torch.float32)
    weights = torch.zeros((len(CANDIDATE_LABELS), max_atom_slots), dtype=torch.float32)
    atom_mask = torch.zeros((len(CANDIDATE_LABELS), max_atom_slots), dtype=torch.bool)
    name_to_index = {name: index for index, name in enumerate(sidechain_names)}

    shared_names = [atom.name.strip() for atom in shared_atoms]
    explicit_names = [name for name in moving_names if name not in set(shared_names)]
    required_slots = len(shared_atoms) + 2 * len(explicit_names)
    if required_slots > max_atom_slots:
        return None

    for candidate_index, (coords_a, coords_b) in enumerate(conformer_pairs):
        slot = 0
        for atom in shared_atoms:
            name = atom.name.strip()
            positions[candidate_index, slot] = coords_a[name_to_index[name]] - center
            sigma2[candidate_index, slot] = max(float(atom.b_iso) / (8.0 * math.pi**2), 0.04)
            weights[candidate_index, slot] = atom.element.atomic_number * float(atom.occ)
            atom_mask[candidate_index, slot] = True
            slot += 1
        for name in explicit_names:
            atom_a = explicit[(name, "A")]
            atom_b = explicit[(name, "B")]
            assert atom_a is not None and atom_b is not None
            if candidate_index == 1:  # A-only carries the total deposited occupancy.
                alternatives = ((coords_a, atom_a, float(atom_a.occ) + float(atom_b.occ)),)
            elif candidate_index == 2:  # B-only carries the total deposited occupancy.
                alternatives = ((coords_a, atom_b, float(atom_a.occ) + float(atom_b.occ)),)
            else:
                alternatives = (
                    (coords_a, atom_a, float(atom_a.occ)),
                    (coords_b, atom_b, float(atom_b.occ)),
                )
            for coords_for_alt, atom, occupancy in alternatives:
                positions[candidate_index, slot] = coords_for_alt[name_to_index[name]] - center
                sigma2[candidate_index, slot] = max(
                    float(atom.b_iso) / (8.0 * math.pi**2), 0.04
                )
                weights[candidate_index, slot] = atom.element.atomic_number * occupancy
                atom_mask[candidate_index, slot] = True
                slot += 1

    return {
        "key": record["key"],
        "pair_path": record["pair_path"],
        "residue_name": residue.name,
        "positions": positions.numpy(),
        "sigma2": sigma2.numpy(),
        "weights": weights.numpy(),
        "atom_mask": atom_mask.numpy(),
        "movable_chi_count": int(movable_chi.sum()),
    }


def _write_protein_shard(path: Path, rows: list[dict]) -> None:
    _atomic_npz(
        path,
        keys=np.asarray([row["key"] for row in rows]),
        pair_paths=np.asarray([row["pair_path"] for row in rows]),
        residue_names=np.asarray([row["residue_name"] for row in rows]),
        movable_chi_counts=np.asarray([row["movable_chi_count"] for row in rows], dtype=np.int8),
        positions=np.stack([row["positions"] for row in rows]),
        sigma2=np.stack([row["sigma2"] for row in rows]),
        weights=np.stack([row["weights"] for row in rows]),
        atom_mask=np.stack([row["atom_mask"] for row in rows]),
    )


def _compile_split(shards: list[Path], output: Path, split_name: str, skipped: Counter) -> dict:
    payloads = [np.load(path) for path in shards]
    try:
        keys = np.concatenate([item["keys"] for item in payloads])
        pair_paths = np.concatenate([item["pair_paths"] for item in payloads])
        residue_names = np.concatenate([item["residue_names"] for item in payloads])
        movable = np.concatenate([item["movable_chi_counts"] for item in payloads])
        out = output / split_name
        out.mkdir(parents=True, exist_ok=True)
        for name in ("positions", "sigma2", "weights", "atom_mask"):
            _atomic_npy(out / f"{name}.npy", np.concatenate([item[name] for item in payloads]))
        index = {
            "split": split_name,
            "candidate_labels": CANDIDATE_LABELS,
            "keys": keys.tolist(),
            "pair_paths": pair_paths.tolist(),
            "residue_names": residue_names.tolist(),
            "movable_chi_counts": movable.astype(int).tolist(),
            "sites": int(len(keys)),
            "skipped": dict(skipped),
        }
        _atomic_json(out / "index.json", index)
        return index
    finally:
        for payload in payloads:
            payload.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare compact U-Net 2.0 landscape labels")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--max-sites", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text())
    protein_groups = {
        "train": set(split["train_proteins"]),
        "validation": set(split["validation_proteins"]),
    }
    records = read_manifest(manifest_path(args.data_root, "train", "crystal"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["is_altloc"] and record["residue_name"] in CHI_SPECS:
            grouped[record["pdb_id"]].append(record)

    for split_name, proteins in protein_groups.items():
        shard_directory = args.output / "shards" / split_name
        status_directory = args.output / "status" / split_name
        shards: list[Path] = []
        skipped: Counter = Counter()
        built_sites = 0
        for pdb_id in sorted(proteins):
            pdb_records = grouped.get(pdb_id, [])
            if not pdb_records:
                continue
            shard = shard_directory / f"{pdb_id}.npz"
            status = status_directory / f"{pdb_id}.json"
            if shard.exists() and status.exists() and not args.overwrite:
                saved = json.loads(status.read_text())
                shards.append(shard)
                built_sites += int(saved["sites"])
                skipped.update(saved.get("skipped", {}))
                continue
            pdb_path = args.data_root / "train" / f"{pdb_id.lower()}.pdb"
            local_skipped: Counter = Counter()
            rows = []
            if not pdb_path.exists():
                local_skipped["missing_pdb"] += len(pdb_records)
            else:
                structure = gemmi.read_structure(str(pdb_path))
                for record in pdb_records:
                    if args.max_sites and built_sites + len(rows) >= args.max_sites:
                        break
                    try:
                        label = _build_site(record, structure, args.seed)
                    except Exception as error:
                        label = None
                        local_skipped[f"exception:{type(error).__name__}"] += 1
                    if label is None:
                        local_skipped["incomplete_or_unsupported"] += 1
                    else:
                        rows.append(label)
            if rows:
                _write_protein_shard(shard, rows)
                shards.append(shard)
            _atomic_json(status, {
                "pdb_id": pdb_id,
                "split": split_name,
                "sites": len(rows),
                "skipped": dict(local_skipped),
                "complete": True,
            })
            built_sites += len(rows)
            skipped.update(local_skipped)
            print(json.dumps({
                "pdb_id": pdb_id, "split": split_name,
                "sites": len(rows), "cumulative_sites": built_sites,
                "skipped": dict(local_skipped),
            }), flush=True)
            if args.max_sites and built_sites >= args.max_sites:
                break
        if not shards:
            raise RuntimeError(f"no landscape labels built for {split_name}")
        index = _compile_split(shards, args.output, split_name, skipped)
        print(json.dumps({"completed": split_name, **index}, default=list), flush=True)


if __name__ == "__main__":
    main()
