"""Calibrate SampleWorks' native real-space reward on deposited A/B controls."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import gemmi
import numpy as np
import torch

from sampleworks.utils.guidance_script_utils import get_reward_function_and_structure


BACKBONE = {"N", "CA", "C", "O", "OXT"}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _find_residue(structure: gemmi.Structure, chain_name: str, residue_number: int):
    for chain in structure[0]:
        if chain.name != chain_name:
            continue
        for residue in chain:
            if residue.seqid.num == residue_number:
                return residue
    raise KeyError(f"{chain_name}:{residue_number}")


def _single_altloc(
    truth: gemmi.Structure, chain_name: str, residue_number: int, keep: str
) -> gemmi.Structure:
    result = truth.clone()
    residue = _find_residue(result, chain_name, residue_number)
    remove = []
    for index, atom in enumerate(residue):
        altloc = str(atom.altloc).strip("\x00 ")
        if altloc and altloc != keep:
            remove.append(index)
        elif altloc == keep:
            atom.altloc = "\x00"
            atom.occ = 1.0
    for index in reversed(remove):
        del residue[index]
    return result


def _write_structure(structure: gemmi.Structure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.make_mmcif_document().write_file(str(path))


def _reward_inputs(reward, structure: dict) -> dict[str, torch.Tensor]:
    return reward.structure_to_reward_input(structure)


def _identity(atom_array) -> list[tuple[str, int, str]]:
    return list(zip(atom_array.chain_id.tolist(), atom_array.res_id.tolist(), atom_array.atom_name.tolist()))


def calibrate(args: argparse.Namespace) -> dict:
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    truth = gemmi.read_structure(str(args.truth))
    controls = {}
    for label in ("A", "B"):
        structure = _single_altloc(truth, args.chain, args.residue, label)
        path = args.output / "controls" / f"deposited_{label}.cif"
        _write_structure(structure, path)
        controls[label] = path

    device = torch.device("cuda")
    results = {}
    for condition, map_path in (("raw", args.raw_map), ("denoised", args.denoised_map)):
        reward, structure_a = get_reward_function_and_structure(
            density=map_path,
            device=device,
            em=False,
            loss_order=2,
            resolution=args.resolution,
            structure_path=controls["A"],
        )
        _, structure_b = get_reward_function_and_structure(
            density=map_path,
            device=device,
            em=False,
            loss_order=2,
            resolution=args.resolution,
            structure_path=controls["B"],
        )
        if _identity(structure_a["asym_unit"]) != _identity(structure_b["asym_unit"]):
            raise ValueError("A/B control atom identities do not match")

        input_a = _reward_inputs(reward, structure_a)
        input_b = _reward_inputs(reward, structure_b)
        coords_a = input_a["coordinates"].detach().clone().requires_grad_(True)
        loss_a = reward(coordinates=coords_a, **{k: v for k, v in input_a.items() if k != "coordinates"})
        loss_a.backward()
        gradient_a = coords_a.grad.detach()
        with torch.no_grad():
            loss_b = reward(**input_b)

        atoms = structure_a["asym_unit"]
        chain_ids = np.asarray(atoms.chain_id, dtype=str)
        chain_mask = (chain_ids == args.chain) | np.char.startswith(chain_ids, args.chain)
        site_mask_np = (
            chain_mask
            & (atoms.res_id == args.residue)
            & ~np.isin(atoms.atom_name, list(BACKBONE))
        )
        if not np.any(site_mask_np):
            available = sorted(
                set(chain_ids[np.asarray(atoms.res_id) == args.residue].tolist())
            )
            raise ValueError(
                f"no sidechain atoms selected for {args.chain}:{args.residue}; "
                f"parser chain IDs at this residue: {available}"
            )
        site_mask = torch.tensor(site_mask_np, dtype=torch.bool, device=device)
        delta = input_b["coordinates"] - input_a["coordinates"]
        site_delta = delta[:, site_mask]
        site_gradient = gradient_a[:, site_mask]
        delta_norm = torch.linalg.vector_norm(site_delta).clamp_min(1e-12)
        directional_derivative = torch.sum(site_gradient * site_delta) / delta_norm

        loss_a_value = float(loss_a.detach().cpu())
        loss_b_value = float(loss_b.detach().cpu())
        results[condition] = {
            "loss_A": loss_a_value,
            "loss_B": loss_b_value,
            "B_minus_A": loss_b_value - loss_a_value,
            "B_minus_A_ppm": 1e6 * (loss_b_value - loss_a_value) / max(abs(loss_a_value), 1e-12),
            "A_to_B_directional_derivative": float(directional_derivative.detach().cpu()),
            "A_to_B_is_downhill": bool(float(directional_derivative.detach().cpu()) < 0),
            "site_atoms": int(site_mask.sum().item()),
            "A_to_B_coordinate_displacement_norm": float(delta_norm.detach().cpu()),
        }

    denoised = results["denoised"]
    raw = results["raw"]
    gate_passed = bool(
        denoised["A_to_B_is_downhill"]
        and denoised["A_to_B_directional_derivative"]
        < raw["A_to_B_directional_derivative"]
    )
    payload = {
        "status": "complete",
        "gate_passed": gate_passed,
        "gate": (
            "At deposited A, the native SampleWorks density gradient must point toward "
            "deposited B, and must do so more strongly for denoised than raw signal."
        ),
        "chain": args.chain,
        "residue": args.residue,
        "resolution_angstrom": args.resolution,
        "maps": {"raw": str(args.raw_map), "denoised": str(args.denoised_map)},
        "controls": {key: str(value) for key, value in controls.items()},
        "results": results,
    }
    _atomic_json(args.output / "calibration.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--truth", type=Path, required=True)
    result.add_argument("--raw-map", type=Path, required=True)
    result.add_argument("--denoised-map", type=Path, required=True)
    result.add_argument("--resolution", type=float, required=True)
    result.add_argument("--chain", required=True)
    result.add_argument("--residue", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    calibrate(parser().parse_args())


if __name__ == "__main__":
    main()
