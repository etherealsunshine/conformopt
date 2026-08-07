"""Export frozen optimizer target patches as aligned local CCP4 maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gemmi
import numpy as np


MAP_KEYS = {
    "experimental_omit_mfo_dfc": "raw",
    "denoised": "denoised",
    "synthetic_ground_truth": "denoiser_training_target",
}


def write_local_ccp4(array: np.ndarray, path: Path, spacing: float) -> None:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3 or len(set(values.shape)) != 1:
        raise ValueError(f"expected a cubic 3D patch, got {values.shape}")
    grid = gemmi.FloatGrid(*values.shape)
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    grid.set_unit_cell(gemmi.UnitCell(
        values.shape[0] * spacing,
        values.shape[1] * spacing,
        values.shape[2] * spacing,
        90.0,
        90.0,
        90.0,
    ))
    np.asarray(grid, dtype=np.float32)[:] = values
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(path))


def write_shifted_structure(
    source: Path,
    output: Path,
    origin: np.ndarray,
    box_length: float,
) -> None:
    structure = gemmi.read_structure(str(source))
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.pos.x -= float(origin[0])
                    atom.pos.y -= float(origin[1])
                    atom.pos.z -= float(origin[2])
    structure.cell = gemmi.UnitCell(
        box_length, box_length, box_length, 90.0, 90.0, 90.0
    )
    structure.spacegroup_hm = "P 1"
    structure.write_pdb(str(output))


def export_triplet(targets: Path, output: Path, pdb: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    with np.load(targets, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {
            label: np.asarray(archive[key], dtype=np.float32)
            for label, key in MAP_KEYS.items()
        }
        radial_mask = np.asarray(archive["radial_mask"], dtype=bool)

    spacing = float(metadata.get("grid_spacing", 0.5))
    patch_size = int(metadata.get("patch_size", next(iter(arrays.values())).shape[0]))
    if any(array.shape != (patch_size, patch_size, patch_size) for array in arrays.values()):
        raise ValueError("target arrays do not match metadata patch size")
    center = np.asarray(metadata["center"], dtype=np.float64)
    half_extent = (patch_size - 1) * spacing / 2.0
    origin = center - half_extent
    box_length = patch_size * spacing
    site = str(metadata.get("short_key", metadata.get("key", targets.stem)))

    map_files: dict[str, str] = {}
    for label, array in arrays.items():
        filename = f"{site}_{label}.ccp4"
        write_local_ccp4(array, output / filename, spacing)
        map_files[label] = filename

    np.savez_compressed(
        output / f"{site}_density_triplet.npz",
        **arrays,
        radial_mask=radial_mask,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    shifted_pdb = None
    if pdb is not None:
        shifted_pdb = f"{site}_local_frame.pdb"
        write_shifted_structure(pdb, output / shifted_pdb, origin, box_length)

    manifest = {
        "site": site,
        "source_targets": str(targets),
        "source_structure": str(pdb) if pdb is not None else None,
        "map_type": metadata.get("map_type"),
        "target_scope": metadata.get("target_scope"),
        "normalization": {
            "experimental_input_mean": metadata.get("input_mean"),
            "experimental_input_std": metadata.get("input_std"),
            "synthetic_target_mean": metadata.get("target_mean"),
            "synthetic_target_std": metadata.get("target_std"),
        },
        "shape": [patch_size, patch_size, patch_size],
        "spacing_angstrom": spacing,
        "local_box_length_angstrom": box_length,
        "crystal_center_xyz": center.tolist(),
        "crystal_origin_xyz": origin.tolist(),
        "coordinate_transform": "local_xyz = crystal_xyz - crystal_origin_xyz",
        "maps": map_files,
        "shifted_structure": shifted_pdb,
        "notes": [
            "All three CCP4 maps are normalized local patches used by the frozen optimizer.",
            "The experimental input is the omit_mFo-DFc patch.",
            "The ground truth is the sidechain-only synthetic deposited A/B target used to train the U-Net.",
            "The shifted PDB uses the same local P1 coordinates as the CCP4 maps.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pdb", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_triplet(args.targets, args.output, args.pdb), indent=2))


if __name__ == "__main__":
    main()
