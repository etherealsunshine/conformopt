"""Prepare leakage-controlled SampleWorks inputs from a frozen U-Net target.

The U-Net predicts a normalized, sidechain-only local patch, whereas SampleWorks'
real-space reward renders an entire structure.  This adapter therefore adds the
local raw or denoised signal to a standardized, sidechain-omit 2mFo-DFc context
map.  Both conditions use the same context, spatial taper, and scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import gemmi
import numpy as np

from density_denoiser.data_pipeline import (
    Site,
    _amplitude_column,
    _calculate_fcalc,
    _find_residue,
    _grid_coordinates,
    _omit_map,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _site_from_metadata(metadata: dict) -> Site:
    return Site(
        pdb_id=metadata["pdb_id"],
        split=metadata["split"],
        chain=metadata["chain"],
        residue_number=int(metadata["residue_number"]),
        insertion_code=metadata.get("insertion_code", " "),
        residue_name=metadata["residue_name"],
        is_altloc=bool(metadata["is_altloc"]),
        center=tuple(float(value) for value in metadata["center"]),
    )


def _write_ccp4(grid: gemmi.FloatGrid, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header(2, True)
    ccp4.write_ccp4_map(str(path))


def _trilinear(array: np.ndarray, index: np.ndarray) -> float:
    lower = np.floor(index).astype(int)
    upper = np.minimum(lower + 1, np.asarray(array.shape) - 1)
    fraction = index - lower
    value = 0.0
    for bit_x in (0, 1):
        x = upper[0] if bit_x else lower[0]
        weight_x = fraction[0] if bit_x else 1.0 - fraction[0]
        for bit_y in (0, 1):
            y = upper[1] if bit_y else lower[1]
            weight_y = fraction[1] if bit_y else 1.0 - fraction[1]
            for bit_z in (0, 1):
                z = upper[2] if bit_z else lower[2]
                weight_z = fraction[2] if bit_z else 1.0 - fraction[2]
                value += weight_x * weight_y * weight_z * float(array[x, y, z])
    return value


def _starting_structure(structure: gemmi.Structure, site: Site) -> gemmi.Structure:
    """Return an A-only target site without leaking deposited-B coordinates."""
    result = structure.clone()
    residue = _find_residue(result, site)
    remove = []
    for index, atom in enumerate(residue):
        altloc = str(atom.altloc).strip("\x00 ")
        if altloc == "B":
            remove.append(index)
        elif altloc == "A":
            atom.altloc = "\x00"
            atom.occ = 1.0
    for index in reversed(remove):
        del residue[index]
    return result


def _hybridize(
    context_grid: gemmi.FloatGrid,
    local_patch: np.ndarray,
    center: np.ndarray,
    spacing: float,
    radius: float,
    taper: float,
    signal_scale: float,
) -> tuple[gemmi.FloatGrid, dict]:
    """Add a smoothly tapered local patch to a standardized context map."""
    output = context_grid.clone()
    array = np.asarray(output, dtype=np.float32)
    context_mean = float(array.mean())
    context_std = float(array.std())
    array[:] = (array - context_mean) / max(context_std, 1e-6)

    size = int(local_patch.shape[0])
    if local_patch.shape != (size, size, size):
        raise ValueError(f"expected cubic patch, got {local_patch.shape}")
    half = (size - 1) / 2.0
    cell = output.unit_cell
    fractional = cell.fractionalize(gemmi.Position(*center.tolist()))
    fractional_center = np.asarray([fractional.x, fractional.y, fractional.z])
    wrapped_fractional_center = fractional_center - np.floor(fractional_center)
    center_index = np.asarray(
        [
            wrapped_fractional_center[0] * output.nu,
            wrapped_fractional_center[1] * output.nv,
            wrapped_fractional_center[2] * output.nw,
        ]
    )
    voxel_upper_bound = max(
        output.nu / cell.a, output.nv / cell.b, output.nw / cell.c
    )
    index_radius = int(math.ceil((radius + spacing) * voxel_upper_bound)) + 1

    changed = 0
    weights = []
    added = []
    for du in range(-index_radius, index_radius + 1):
        u = int(round(center_index[0])) + du
        for dv in range(-index_radius, index_radius + 1):
            v = int(round(center_index[1])) + dv
            for dw in range(-index_radius, index_radius + 1):
                w = int(round(center_index[2])) + dw
                position = output.get_position(u % output.nu, v % output.nv, w % output.nw)
                position_fractional = cell.fractionalize(position)
                fractional_delta = np.asarray(
                    [position_fractional.x, position_fractional.y, position_fractional.z]
                ) - fractional_center
                fractional_delta -= np.round(fractional_delta)
                delta_position = cell.orthogonalize(gemmi.Fractional(*fractional_delta.tolist()))
                delta = np.asarray(
                    [delta_position.x, delta_position.y, delta_position.z], dtype=np.float64
                )
                distance = float(np.linalg.norm(delta))
                if distance > radius:
                    continue
                patch_index = delta / spacing + half
                if np.any(patch_index < 0) or np.any(patch_index > size - 1):
                    continue
                value = _trilinear(local_patch, patch_index)
                if taper <= 0 or distance <= radius - taper:
                    weight = 1.0
                else:
                    phase = (distance - (radius - taper)) / taper
                    weight = 0.5 * (1.0 + math.cos(math.pi * phase))
                increment = signal_scale * weight * value
                array[u % output.nu, v % output.nv, w % output.nw] += increment
                changed += 1
                weights.append(weight)
                added.append(increment)

    return output, {
        "context_mean": context_mean,
        "context_std": context_std,
        "changed_grid_points": changed,
        "mean_blend_weight": float(np.mean(weights)) if weights else 0.0,
        "added_min": float(np.min(added)) if added else 0.0,
        "added_max": float(np.max(added)) if added else 0.0,
    }


def prepare(args: argparse.Namespace) -> dict:
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    targets = np.load(args.targets, allow_pickle=False)
    metadata = json.loads(str(targets["metadata"]))
    site = _site_from_metadata(metadata)
    raw = np.asarray(targets["raw"], dtype=np.float32)
    denoised = np.asarray(targets["denoised"], dtype=np.float32)

    structure = gemmi.read_structure(str(args.structure))
    mtz = gemmi.read_mtz_file(str(args.mtz))
    calculator = gemmi.StructureFactorCalculatorX(structure.cell)
    miller = mtz.make_miller_array()
    full_fcalc = _calculate_fcalc(calculator, structure[0], miller)
    context = _omit_map(
        structure,
        mtz,
        site,
        "omit_2mfo_dfc",
        calculator,
        miller,
        full_fcalc,
    )

    center = np.asarray(metadata["center"], dtype=np.float64)
    condition_metrics = {}
    outputs = {}
    for label, patch in (("raw", raw), ("denoised", denoised)):
        grid, metrics = _hybridize(
            context,
            patch,
            center,
            float(metadata["grid_spacing"]),
            args.radius,
            args.taper,
            args.signal_scale,
        )
        path = args.output / f"{site.key}_{label}_hybrid.ccp4"
        _write_ccp4(grid, path)
        outputs[label] = str(path)
        condition_metrics[label] = metrics

    start = _starting_structure(structure, site)
    start_path = args.output / f"{site.key}_start_A_only.cif"
    truth_path = args.output / f"{site.key}_deposited_truth.cif"
    start.make_mmcif_document().write_file(str(start_path))
    structure.make_mmcif_document().write_file(str(truth_path))

    resolution = float(mtz.resolution_high())
    manifest = {
        "status": "complete",
        "site": site.key,
        "hypothesis": (
            "SampleWorks guidance against the U-Net-denoised local signal recovers "
            "the held-out B conformer more often than the same sampler against raw signal."
        ),
        "structure": str(args.structure),
        "mtz": str(args.mtz),
        "targets": str(args.targets),
        "start_structure": str(start_path),
        "truth_structure": str(truth_path),
        "maps": outputs,
        "resolution_angstrom": resolution,
        "context_map": "sidechain-omit 2mFo-DFc, globally standardized",
        "local_signal_radius_angstrom": args.radius,
        "cosine_taper_angstrom": args.taper,
        "signal_scale": args.signal_scale,
        "condition_metrics": condition_metrics,
        "input_sha256": {
            "structure": _sha256(args.structure),
            "mtz": _sha256(args.mtz),
            "targets": _sha256(args.targets),
        },
    }
    _atomic_json(args.output / "input_manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--structure", type=Path, required=True)
    result.add_argument("--mtz", type=Path, required=True)
    result.add_argument("--targets", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--radius", type=float, default=4.0)
    result.add_argument("--taper", type=float, default=0.5)
    result.add_argument("--signal-scale", type=float, default=1.0)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
