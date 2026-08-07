from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import gemmi
import numpy as np


BACKBONE = {"N", "CA", "C", "O", "OXT"}
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


@dataclass(frozen=True)
class Site:
    pdb_id: str
    split: str
    chain: str
    residue_number: int
    insertion_code: str
    residue_name: str
    is_altloc: bool
    center: tuple[float, float, float]

    @property
    def key(self) -> str:
        suffix = self.insertion_code.strip()
        return f"{self.pdb_id}_{self.chain}_{self.residue_name}{self.residue_number}{suffix}"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _pdb_id(path: Path) -> str:
    match = re.search(r"(?i)(?<![A-Z0-9])([0-9][A-Z0-9]{3})(?![A-Z0-9])", path.stem)
    if not match:
        match = re.search(r"(?i)([0-9][A-Z0-9]{3})", path.stem)
    if not match:
        raise ValueError(f"cannot infer four-character PDB ID from {path.name}")
    return match.group(1).upper()


def discover_pdbs(data_root: Path, split: str) -> list[Path]:
    split_root = data_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"missing split directory: {split_root}")
    return sorted({*split_root.rglob("*.pdb"), *split_root.rglob("*.ent")})


def _download(url: str, destination: Path, overwrite: bool = False) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "qfit-density-denoiser/1"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, destination)


def convert_sf_cif_to_mtz(sf_cif: Path, pdb_path: Path, mtz_path: Path) -> dict:
    document = gemmi.cif.read_file(str(sf_cif))
    blocks = gemmi.as_refln_blocks(document)
    if not blocks:
        raise ValueError("structure-factor CIF has no reflection block")
    mtz = gemmi.CifToMtz().convert_block_to_mtz(blocks[0])
    structure = gemmi.read_structure(str(pdb_path))
    mtz.cell = structure.cell
    if mtz.spacegroup is None:
        mtz.spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
    if mtz.spacegroup is None:
        raise ValueError(f"unknown space group {structure.spacegroup_hm!r}")
    mtz.update_reso()
    labels = set(mtz.column_labels())
    amplitude = next((label for label in ("FP", "F", "FOBS", "F-obs") if label in labels), None)
    if amplitude is None:
        raise ValueError(f"no observed-amplitude column in {sorted(labels)}")
    values = np.asarray(mtz.column_with_label(amplitude).array, dtype=np.float32)
    if np.count_nonzero(np.isfinite(values) & (values > 0)) < 100:
        raise ValueError("fewer than 100 usable observed amplitudes")
    mtz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = mtz_path.with_suffix(".mtz.part")
    mtz.write_to_file(str(temporary))
    os.replace(temporary, mtz_path)
    return {
        "amplitude_column": amplitude,
        "reflections": int(mtz.nreflections),
        "resolution_high": float(mtz.resolution_high()),
        "spacegroup": mtz.spacegroup.hm,
    }


def acquire_one(data_root: Path, split: str, pdb_path: Path, overwrite: bool = False) -> dict:
    pdb_id = _pdb_id(pdb_path)
    cache = data_root / "cache" / split
    sf_cif = cache / "structure_factors" / f"{pdb_id}-sf.cif"
    mtz_path = cache / "mtz" / f"{pdb_id}.mtz"
    status_path = cache / "status" / "acquisition" / f"{pdb_id}.json"
    if status_path.exists() and mtz_path.exists() and not overwrite:
        return json.loads(status_path.read_text())
    payload = {"pdb_id": pdb_id, "split": split, "pdb_path": str(pdb_path)}
    try:
        _download(f"https://files.rcsb.org/download/{pdb_id}-sf.cif", sf_cif, overwrite)
        payload.update(convert_sf_cif_to_mtz(sf_cif, pdb_path, mtz_path))
        payload.update({"status": "complete", "sf_cif": str(sf_cif), "mtz": str(mtz_path)})
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, ValueError) as error:
        payload.update({"status": "skipped", "error": f"{type(error).__name__}: {error}"})
    _atomic_json(status_path, payload)
    return payload


def acquire_split(data_root: Path, split: str, workers: int, overwrite: bool) -> list[dict]:
    pdbs = discover_pdbs(data_root, split)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(acquire_one, data_root, split, path, overwrite): path for path in pdbs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    summary = {
        "split": split,
        "pdb_files": len(pdbs),
        "complete": sum(item["status"] == "complete" for item in results),
        "skipped": sum(item["status"] != "complete" for item in results),
    }
    _atomic_json(data_root / "cache" / split / "acquisition_summary.json", summary)
    return results


def _altloc(atom: gemmi.Atom) -> str:
    return "" if atom.altloc in ("\x00", " ") else atom.altloc


def _sidechain_atoms(residue: gemmi.Residue) -> list[gemmi.Atom]:
    return [
        atom for atom in residue
        if atom.name.strip() not in BACKBONE and atom.element.name != "H"
    ]


def _representative_position(residue: gemmi.Residue, name: str) -> np.ndarray:
    atoms = [atom for atom in residue if atom.name.strip() == name]
    if not atoms:
        raise ValueError(f"{residue.name} {residue.seqid} has no {name} atom")
    shared = [atom for atom in atoms if _altloc(atom) == ""]
    selected = shared or atoms
    weights = np.asarray([max(float(atom.occ), 0.0) for atom in selected])
    if not np.any(weights):
        weights = np.ones(len(selected), dtype=np.float64)
    positions = np.asarray(
        [atom.pos.tolist() for atom in selected], dtype=np.float64
    )
    return np.average(positions, axis=0, weights=weights)


def residue_frame(structure: gemmi.Structure, site: Site) -> tuple[np.ndarray, np.ndarray]:
    """Return C-alpha origin and a crystal-to-residue right-handed rotation."""
    residue = _find_residue(structure, site)
    origin = _representative_position(residue, "CA")
    beta = _representative_position(residue, "CB")
    nitrogen = _representative_position(residue, "N")
    x_axis = beta - origin
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        raise ValueError(f"{site.key} has a degenerate CA-CB axis")
    x_axis /= x_norm
    z_axis = np.cross(nitrogen - origin, x_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        raise ValueError(f"{site.key} has a degenerate residue plane")
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    # Rows are local axes expressed in crystal Cartesian coordinates, so
    # local = rotation @ (crystal - origin).
    rotation = np.stack((x_axis, y_axis, z_axis), axis=0)
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError(f"{site.key} residue frame is not orthonormal")
    if np.linalg.det(rotation) < 0.999999:
        raise ValueError(f"{site.key} residue frame is not right-handed")
    return origin, rotation


def discover_sites(structure: gemmi.Structure, pdb_id: str, split: str,
                   negatives_per_altloc: int, seed: int) -> list[Site]:
    altloc_sites: list[Site] = []
    negatives: list[Site] = []
    for chain in structure[0]:
        for residue in chain:
            if residue.name not in STANDARD_AA or residue.name == "GLY":
                continue
            sidechain = _sidechain_atoms(residue)
            if not sidechain:
                continue
            labels = {_altloc(atom) for atom in sidechain} - {""}
            positions = np.asarray([atom.pos.tolist() for atom in sidechain], dtype=np.float64)
            center = tuple(np.mean(positions, axis=0).tolist())
            site = Site(
                pdb_id=pdb_id,
                split=split,
                chain=chain.name,
                residue_number=residue.seqid.num,
                insertion_code=residue.seqid.icode,
                residue_name=residue.name,
                is_altloc={"A", "B"}.issubset(labels),
                center=center,
            )
            if site.is_altloc:
                altloc_sites.append(site)
            elif not labels and all(atom.occ >= 0.95 and atom.b_iso > 0 for atom in sidechain):
                negatives.append(site)
    wanted = min(len(negatives), negatives_per_altloc * len(altloc_sites))
    rng = np.random.default_rng(seed + sum(ord(char) for char in pdb_id))
    selected = [] if wanted == 0 else [negatives[i] for i in rng.choice(len(negatives), wanted, replace=False)]
    return altloc_sites + selected


def _find_residue(structure: gemmi.Structure, site: Site) -> gemmi.Residue:
    for chain in structure[0]:
        if chain.name != site.chain:
            continue
        for residue in chain:
            if residue.seqid.num == site.residue_number and residue.seqid.icode == site.insertion_code:
                return residue
    raise KeyError(site.key)


def _omit_sidechain(structure: gemmi.Structure, site: Site) -> gemmi.Structure:
    omitted = structure.clone()
    for atom in _sidechain_atoms(_find_residue(omitted, site)):
        atom.occ = 0.0
    return omitted


def _amplitude_column(mtz: gemmi.Mtz) -> str:
    labels = set(mtz.column_labels())
    found = next((label for label in ("FP", "F", "FOBS", "F-obs") if label in labels), None)
    if found is None:
        raise ValueError("MTZ has no observed-amplitude column")
    return found


def _sidechain_model(structure: gemmi.Structure, site: Site) -> gemmi.Model:
    source = _find_residue(structure, site)
    # Gemmi 0.6 requires a string; newer releases require an integer.
    try:
        model = gemmi.Model(1)
    except TypeError:
        model = gemmi.Model("1")
    chain = gemmi.Chain(site.chain)
    residue = gemmi.Residue()
    residue.name = source.name
    residue.seqid = source.seqid
    for atom in _sidechain_atoms(source):
        residue.add_atom(atom.clone())
    chain.add_residue(residue)
    model.add_chain(chain)
    return model


def _calculate_fcalc(calculator: gemmi.StructureFactorCalculatorX,
                     model: gemmi.Model, miller: np.ndarray) -> np.ndarray:
    return np.asarray(
        [calculator.calculate_sf_from_model(model, hkl) for hkl in miller],
        dtype=np.complex64,
    )


def _omit_map(structure: gemmi.Structure, mtz: gemmi.Mtz, site: Site,
              map_type: str, calculator: gemmi.StructureFactorCalculatorX,
              miller: np.ndarray, full_fcalc: np.ndarray) -> gemmi.FloatGrid:
    # Structure factors are linear in atoms. Computing the full model once and
    # subtracting a tiny sidechain-only calculation is exactly equivalent to
    # recalculating the entire omitted protein for every site, but much faster.
    sidechain_fcalc = _calculate_fcalc(calculator, _sidechain_model(structure, site), miller)
    fcalc = full_fcalc - sidechain_fcalc
    fobs = np.asarray(mtz.column_with_label(_amplitude_column(mtz)).array, dtype=np.float32)
    amplitudes = np.abs(fcalc)
    valid = np.isfinite(fobs) & (fobs > 0) & np.isfinite(amplitudes) & (amplitudes > 0)
    scale = float(np.dot(amplitudes[valid], fobs[valid]) / max(np.dot(amplitudes[valid], amplitudes[valid]), 1e-12))
    multiplier = 1.0 if map_type == "omit_mfo_dfc" else 2.0
    coefficient = (multiplier * fobs - scale * amplitudes) * np.exp(1j * np.angle(fcalc))
    coefficient[~valid] = 0.0
    amplitude_label, phase_label = "DENF", "DENPHI"
    if amplitude_label not in mtz.column_labels():
        mtz.add_column(amplitude_label, "F")
        mtz.add_column(phase_label, "P")
    mtz.column_with_label(amplitude_label).array[:] = np.abs(coefficient)
    mtz.column_with_label(phase_label).array[:] = np.rad2deg(np.angle(coefficient))
    return mtz.transform_f_phi_to_map(amplitude_label, phase_label, sample_rate=3.0)


def _patch_transform(center: tuple[float, float, float] | np.ndarray, size: int,
                     spacing: float, crystal_to_local: np.ndarray | None = None) -> gemmi.Transform:
    center = np.asarray(center, dtype=np.float64)
    rotation = np.eye(3) if crystal_to_local is None else np.asarray(crystal_to_local)
    half_extent = spacing * (size - 1) / 2.0
    # crystal = rotation.T @ local + center
    matrix = spacing * rotation.T
    origin = center + rotation.T @ np.full(3, -half_extent)
    transform = gemmi.Transform()
    transform.mat.fromlist(matrix.tolist())
    transform.vec.fromlist(origin.tolist())
    return transform


def extract_patch(grid: gemmi.FloatGrid, center: tuple[float, float, float] | np.ndarray,
                  size: int, spacing: float,
                  crystal_to_local: np.ndarray | None = None) -> np.ndarray:
    patch = np.empty((size, size, size), dtype=np.float32)
    grid.interpolate_values(
        patch, _patch_transform(center, size, spacing, crystal_to_local), order=1
    )
    return patch


def _grid_coordinates(center: tuple[float, float, float] | np.ndarray, size: int,
                      spacing: float,
                      crystal_to_local: np.ndarray | None = None) -> np.ndarray:
    axis = (np.arange(size, dtype=np.float32) - (size - 1) / 2.0) * spacing
    offsets = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    rotation = np.eye(3) if crystal_to_local is None else np.asarray(crystal_to_local)
    return offsets @ rotation + np.asarray(center, dtype=np.float32)


def synthetic_patch(structure: gemmi.Structure, site: Site, size: int, spacing: float,
                    target_scope: str, patch_center: np.ndarray | None = None,
                    crystal_to_local: np.ndarray | None = None) -> np.ndarray:
    center = np.asarray(site.center) if patch_center is None else np.asarray(patch_center)
    coordinates = _grid_coordinates(center, size, spacing, crystal_to_local)
    target = np.zeros((size, size, size), dtype=np.float32)
    selected: Iterable[gemmi.Atom]
    if target_scope == "sidechain":
        selected = _sidechain_atoms(_find_residue(structure, site))
    elif target_scope == "local":
        radius = spacing * size * math.sqrt(3) / 2.0 + 3.0
        selection_center = np.asarray(site.center)
        selected = [
            atom for chain in structure[0] for residue in chain for atom in residue
            if atom.element.name != "H"
            and np.linalg.norm(np.asarray(atom.pos.tolist()) - selection_center) <= radius
        ]
    else:
        raise ValueError(f"unknown target scope {target_scope}")
    for atom in selected:
        sigma2 = max(float(atom.b_iso) / (8.0 * math.pi ** 2), 0.04)
        position = np.asarray(atom.pos.tolist(), dtype=np.float32)
        distance2 = np.square(coordinates - position).sum(axis=-1)
        normalization = (2.0 * math.pi * sigma2) ** -1.5
        target += float(atom.occ) * atom.element.atomic_number * normalization * np.exp(-distance2 / (2.0 * sigma2))
    return target


def sidechain_mask(structure: gemmi.Structure, site: Site, size: int,
                   spacing: float, radius: float = 1.0,
                   patch_center: np.ndarray | None = None,
                   crystal_to_local: np.ndarray | None = None) -> np.ndarray:
    center = np.asarray(site.center) if patch_center is None else np.asarray(patch_center)
    coordinates = _grid_coordinates(center, size, spacing, crystal_to_local)
    mask = np.zeros((size, size, size), dtype=bool)
    for atom in _sidechain_atoms(_find_residue(structure, site)):
        position = np.asarray(atom.pos.tolist(), dtype=np.float32)
        mask |= np.square(coordinates - position).sum(axis=-1) <= radius ** 2
    return mask


def normalize_patch(patch: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(patch.mean())
    standard_deviation = float(patch.std())
    return ((patch - mean) / max(standard_deviation, 1e-6)).astype(np.float32), mean, standard_deviation


def _save_pair(path: Path, input_patch: np.ndarray, target_patch: np.ndarray,
               local_mask: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(
        temporary,
        input=input_patch[None],
        target=target_patch[None],
        local_mask=local_mask[None],
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, path)


def _prepared_cache(data_root: Path, split: str, frame: str) -> Path:
    cache = data_root / "cache" / split
    if frame == "crystal":
        return cache
    if frame == "residue":
        return cache / "canonical"
    raise ValueError(f"unknown frame {frame}")


def prepare_protein(data_root: Path, split: str, pdb_path: Path, map_type: str,
                    size: int, spacing: float, negatives_per_altloc: int,
                    seed: int, overwrite: bool = False,
                    frame: str = "crystal") -> dict:
    pdb_id = _pdb_id(pdb_path)
    source_cache = data_root / "cache" / split
    output_cache = _prepared_cache(data_root, split, frame)
    status_path = output_cache / "status" / "patches" / f"{pdb_id}.json"
    mtz_path = source_cache / "mtz" / f"{pdb_id}.mtz"
    if status_path.exists() and not overwrite:
        return json.loads(status_path.read_text())
    payload = {"pdb_id": pdb_id, "split": split, "map_type": map_type, "frame": frame}
    try:
        structure = gemmi.read_structure(str(pdb_path))
        mtz = gemmi.read_mtz_file(str(mtz_path))
        sites = discover_sites(structure, pdb_id, split, negatives_per_altloc, seed)
        calculator = gemmi.StructureFactorCalculatorX(structure.cell)
        miller = mtz.make_miller_array()
        full_fcalc = _calculate_fcalc(calculator, structure[0], miller)
        records = []
        target_scope = "sidechain" if map_type == "omit_mfo_dfc" else "local"
        for site in sites:
            pair_path = output_cache / "pairs" / f"{site.key}.npz"
            site_status = output_cache / "status" / "sites" / f"{site.key}.json"
            if pair_path.exists() and site_status.exists() and not overwrite:
                records.append(json.loads(site_status.read_text()))
                continue
            if frame == "residue":
                try:
                    patch_center, rotation = residue_frame(structure, site)
                except ValueError as error:
                    skipped = {
                        **asdict(site), "key": site.key, "frame": frame,
                        "status": "skipped", "error": str(error),
                    }
                    _atomic_json(site_status, skipped)
                    print(json.dumps({
                        "site": site.key, "status": "skipped", "error": str(error)
                    }), flush=True)
                    continue
            elif frame == "crystal":
                patch_center, rotation = np.asarray(site.center), None
            else:
                raise ValueError(f"unknown frame {frame}")
            experimental_grid = _omit_map(
                structure, mtz, site, map_type, calculator, miller, full_fcalc
            )
            experimental = extract_patch(
                experimental_grid, patch_center, size, spacing, rotation
            )
            target = synthetic_patch(
                structure, site, size, spacing, target_scope,
                patch_center=patch_center, crystal_to_local=rotation,
            )
            local_mask = sidechain_mask(
                structure, site, size, spacing,
                patch_center=patch_center, crystal_to_local=rotation,
            )
            input_norm, input_mean, input_std = normalize_patch(experimental)
            target_norm, target_mean, target_std = normalize_patch(target)
            metadata = {
                **asdict(site),
                "key": site.key,
                "map_type": map_type,
                "frame": frame,
                "patch_center_crystal": np.asarray(patch_center).tolist(),
                "crystal_to_local": (
                    np.eye(3) if rotation is None else np.asarray(rotation)
                ).tolist(),
                "target_scope": target_scope,
                "patch_size": size,
                "grid_spacing": spacing,
                "input_mean": input_mean,
                "input_std": input_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "pair_path": str(pair_path),
                "status": "complete",
            }
            _save_pair(pair_path, input_norm, target_norm, local_mask, metadata)
            _atomic_json(site_status, metadata)
            records.append(metadata)
            print(json.dumps({"site": site.key, "status": "complete"}), flush=True)
        payload.update({
            "status": "complete",
            "sites": len(records),
            "altloc_sites": sum(bool(record["is_altloc"]) for record in records),
            "negative_sites": sum(not bool(record["is_altloc"]) for record in records),
        })
    except Exception as error:
        payload.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    _atomic_json(status_path, payload)
    return payload


def prepare_split(data_root: Path, split: str, map_type: str, size: int,
                  spacing: float, negatives_per_altloc: int, seed: int,
                  workers: int, overwrite: bool,
                  frame: str = "crystal") -> list[dict]:
    pdbs = discover_pdbs(data_root, split)
    results = []
    if workers <= 1:
        for path in pdbs:
            result = prepare_protein(
                data_root, split, path, map_type, size, spacing,
                negatives_per_altloc, seed, overwrite, frame,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    prepare_protein, data_root, split, path, map_type, size, spacing,
                    negatives_per_altloc, seed, overwrite, frame,
                ): path
                for path in pdbs
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    _atomic_json(_prepared_cache(data_root, split, frame) / "preparation_summary.json", {
        "split": split,
        "frame": frame,
        "proteins": len(pdbs),
        "complete": sum(item["status"] == "complete" for item in results),
        "failed": sum(item["status"] != "complete" for item in results),
        "pairs": sum(int(item.get("sites", 0)) for item in results),
    })
    return results


def compile_manifest(data_root: Path, split: str, frame: str = "crystal") -> Path:
    cache = _prepared_cache(data_root, split, frame)
    records = []
    for path in sorted((cache / "status" / "sites").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "complete" and Path(record["pair_path"]).exists():
            records.append(record)
    manifest = cache / "manifest.jsonl"
    temporary = manifest.with_suffix(".jsonl.part")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, manifest)
    _atomic_json(cache / "manifest_summary.json", {
        "split": split,
        "frame": frame,
        "pairs": len(records),
        "proteins": len({record["pdb_id"] for record in records}),
        "altloc_pairs": sum(bool(record["is_altloc"]) for record in records),
        "negative_pairs": sum(not bool(record["is_altloc"]) for record in records),
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "prepare", "all", "manifest"))
    parser.add_argument(
        "--data-root", type=Path,
        default=Path(os.environ.get("QFIT_UNET_DATA", Path.home() / "qfit_unet_data")),
    )
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--map-type", choices=("omit_mfo_dfc", "omit_2mfo_dfc"), default="omit_mfo_dfc")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--frame", choices=("crystal", "residue"), default="crystal")
    parser.add_argument("--negatives-per-altloc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    splits = ("train", "test") if args.split == "both" else (args.split,)
    for split in splits:
        if args.command in ("acquire", "all"):
            acquire_split(args.data_root, split, args.workers, args.overwrite)
        if args.command in ("prepare", "all"):
            prepare_split(
                args.data_root, split, args.map_type, args.patch_size, args.spacing,
                args.negatives_per_altloc, args.seed, args.workers, args.overwrite,
                args.frame,
            )
        if args.command in ("prepare", "all", "manifest"):
            print(compile_manifest(args.data_root, split, args.frame), flush=True)


if __name__ == "__main__":
    main()
