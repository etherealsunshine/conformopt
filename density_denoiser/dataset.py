from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_manifest(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def manifest_path(data_root: Path, split: str, frame: str = "crystal") -> Path:
    cache = data_root / "cache" / split
    if frame == "residue":
        cache = cache / "canonical"
    elif frame != "crystal":
        raise ValueError(f"unknown frame {frame}")
    return cache / "manifest.jsonl"


def protein_train_validation_split(records: list[dict], validation_fraction: float,
                                   seed: int) -> tuple[list[dict], list[dict]]:
    proteins = sorted({record["pdb_id"] for record in records})
    generator = random.Random(seed)
    generator.shuffle(proteins)
    validation_count = max(1, round(len(proteins) * validation_fraction)) if len(proteins) > 1 else 0
    validation = set(proteins[:validation_count])
    return (
        [record for record in records if record["pdb_id"] not in validation],
        [record for record in records if record["pdb_id"] in validation],
    )


def _translate_zero(volume: torch.Tensor, shifts: tuple[int, int, int]) -> torch.Tensor:
    translated = torch.roll(volume, shifts=shifts, dims=(1, 2, 3))
    for dimension, shift in enumerate(shifts, start=1):
        if shift > 0:
            index = [slice(None)] * 4
            index[dimension] = slice(0, shift)
            translated[tuple(index)] = 0
        elif shift < 0:
            index = [slice(None)] * 4
            index[dimension] = slice(shift, None)
            translated[tuple(index)] = 0
    return translated


def augment_pair(input_patch: torch.Tensor, target_patch: torch.Tensor,
                 local_mask: torch.Tensor, max_translation: int,
                 noise_standard_deviation: float,
                 rotation_augmentation: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if rotation_augmentation:
        # Compositions of quarter turns are proper rotations and preserve chirality.
        for axes in ((1, 2), (1, 3), (2, 3)):
            turns = random.randrange(4)
            input_patch = torch.rot90(input_patch, turns, axes)
            target_patch = torch.rot90(target_patch, turns, axes)
            local_mask = torch.rot90(local_mask, turns, axes)
    shifts = tuple(random.randint(-max_translation, max_translation) for _ in range(3))
    input_patch = _translate_zero(input_patch, shifts)
    target_patch = _translate_zero(target_patch, shifts)
    local_mask = _translate_zero(local_mask, shifts)
    if noise_standard_deviation > 0:
        input_patch = input_patch + noise_standard_deviation * torch.randn_like(input_patch)
    return input_patch, target_patch, local_mask


class DensityPairDataset(Dataset):
    def __init__(self, records: list[dict], augment: bool = False,
                 max_translation: int = 2, noise_standard_deviation: float = 0.05,
                 rotation_augmentation: bool = True):
        self.records = records
        self.augment = augment
        self.max_translation = max_translation
        self.noise_standard_deviation = noise_standard_deviation
        self.rotation_augmentation = rotation_augmentation

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with np.load(record["pair_path"]) as archive:
            input_patch = torch.from_numpy(archive["input"].astype(np.float32))
            target_patch = torch.from_numpy(archive["target"].astype(np.float32))
            if "local_mask" in archive:
                local_mask = torch.from_numpy(archive["local_mask"].astype(bool))
            else:
                local_mask = torch.ones_like(input_patch, dtype=torch.bool)
        if self.augment:
            input_patch, target_patch, local_mask = augment_pair(
                input_patch, target_patch, local_mask,
                self.max_translation, self.noise_standard_deviation,
                self.rotation_augmentation,
            )
        return {
            "input": input_patch,
            "target": target_patch,
            "local_mask": local_mask,
            "key": record["key"],
            "pdb_id": record["pdb_id"],
            "is_altloc": bool(record["is_altloc"]),
        }
