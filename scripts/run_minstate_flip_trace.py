from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import density_denoiser.five_site_optimizer as optimizer_module


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = handle.name
    os.replace(temporary, path)


class Trace:
    def __init__(self, output: Path, site: str, steps: int = 200, k: int = 4):
        self.output = output
        self.site = site
        self.steps = steps
        self.k = k
        self.trajectory = -1
        self.step = 0
        self.active = False
        self.environments: dict[int, tuple[str, list[dict]]] = {}
        self.arrays: dict[str, np.ndarray] = {}
        self.gaps: dict[str, np.ndarray] = {}

    def capture_environment(self, records: list, alternate_states: list) -> None:
        grouped: OrderedDict[str, OrderedDict[str, list]] = OrderedDict()
        for record in records:
            if not record.altloc:
                continue
            grouped.setdefault(record.group_key, OrderedDict()).setdefault(
                record.altloc, []
            ).append(record)
        environment = (
            "symmetry"
            if any(record.group_key.startswith("sym") for record in records)
            else "direct"
        )
        metadata = []
        for (group, states), tensors in zip(grouped.items(), alternate_states):
            labels = list(states)
            if len(tensors) > len(labels):
                labels.extend(["absent"] * (len(tensors) - len(labels)))
            first = next(iter(next(iter(states.values()))))
            target_group = group == self.site_target_group
            metadata.append({
                "group_index": len(metadata),
                "group_key": group,
                "state_labels": labels,
                "is_water": bool(first.is_water),
                "labeled_target_backbone": target_group,
                "category": (
                    "labeled_backbone"
                    if target_group
                    else "labeled_water"
                    if first.is_water
                    else "symmetry_mate"
                    if environment == "symmetry"
                    else "direct_protein"
                ),
            })
        self.environments[id(alternate_states)] = (
            environment, metadata
        )
        if environment not in self.arrays:
            groups = len(metadata)
            self.arrays[environment] = np.full(
                (50, self.steps, self.k, groups), -1, dtype=np.int16
            )
            self.gaps[environment] = np.full(
                (50, self.steps, self.k, groups), np.nan, dtype=np.float32
            )
            atomic_json(
                self.output / f"{environment}_groups.json", metadata
            )

    @property
    def site_target_group(self) -> str:
        parts = self.site.split("_")
        chain = parts[1]
        residue_number = "".join(char for char in parts[2] if char.isdigit())
        return f"{chain}:{int(residue_number)}: "

    def start_stage2(self) -> None:
        if self.active:
            self.checkpoint()
        self.trajectory += 1
        self.step = 0
        self.active = True

    def optimizer_step(self) -> None:
        if self.active:
            self.step += 1
            if self.step == self.steps:
                self.checkpoint()

    def record(
        self,
        candidate: torch.Tensor,
        alternate_states: list[list[torch.Tensor]],
        state_penalties: list[list[torch.Tensor]],
        conformer: int,
    ) -> None:
        if (
            not self.active
            or self.trajectory < 0
            or self.trajectory >= 50
            or self.step >= self.steps
            or id(alternate_states) not in self.environments
            or not state_penalties
        ):
            return
        environment, _metadata = self.environments[id(alternate_states)]
        winners = []
        gaps = []
        for group_index, penalties in enumerate(state_penalties):
            values = torch.stack(penalties)
            order = torch.argsort(values, stable=True)
            winners.append(order[0])
            gaps.append(
                values[order[1]] - values[order[0]]
                if len(order) > 1
                else torch.full(
                    (), torch.nan, dtype=values.dtype, device=values.device
                )
            )
        winner_values = torch.stack(winners).detach().cpu().numpy()
        gap_values = torch.stack(gaps).detach().cpu().numpy()
        for group_index in range(len(state_penalties)):
            self.arrays[environment][
                self.trajectory, self.step, conformer, group_index
            ] = int(winner_values[group_index])
            self.gaps[environment][
                self.trajectory, self.step, conformer, group_index
            ] = float(gap_values[group_index])

    def checkpoint(self) -> None:
        arrays = {}
        for environment, values in self.arrays.items():
            arrays[f"{environment}_winners"] = values
            arrays[f"{environment}_gaps"] = self.gaps[environment]
        arrays["completed_trajectories"] = np.asarray(
            min(self.trajectory + 1, 50), dtype=np.int32
        )
        atomic_npz(self.output / "minstate_trace.npz", **arrays)
        atomic_json(self.output / "trace_status.json", {
            "site": self.site,
            "completed_trajectories": min(self.trajectory + 1, 50),
            "stage2_steps": self.steps,
            "K": self.k,
        })


def parse_wrapper_args() -> tuple[Path, str, list[str]]:
    arguments = sys.argv[1:]
    output_index = arguments.index("--trace-output")
    trace_output = Path(arguments[output_index + 1])
    del arguments[output_index:output_index + 2]
    site_index = arguments.index("--trace-site")
    site = arguments[site_index + 1]
    del arguments[site_index:site_index + 2]
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    return trace_output, site, arguments


def main() -> None:
    trace_output, site, optimizer_args = parse_wrapper_args()
    if trace_output.exists():
        raise FileExistsError(trace_output)
    trace_output.mkdir(parents=True)
    trace = Trace(trace_output, site)

    original_partition = optimizer_module.partition_soft_environment
    original_soft = optimizer_module.soft_clash_penalty
    original_barrier = optimizer_module.soft_clash_barrier_penalty
    original_adam = optimizer_module.torch.optim.Adam
    previous_learning_rate = {"value": None}

    def traced_partition(records, device):
        result = original_partition(records, device)
        trace.capture_environment(records, result[2])
        return result

    def direct_penalty(
        candidate_xyz,
        invariant_xyz,
        invariant_weights,
        alternate_states,
        threshold,
        invariant_mask=None,
    ):
        result = original_soft(
            candidate_xyz,
            invariant_xyz,
            invariant_weights,
            alternate_states,
            threshold,
            invariant_mask,
        )
        state_penalties = []
        zero = torch.zeros(
            (), dtype=candidate_xyz.dtype, device=candidate_xyz.device
        )
        for states in alternate_states:
            state_penalties.append([
                torch.clamp(
                    threshold - torch.cdist(candidate_xyz, state), min=0.0
                ).square().sum()
                if state.numel() else zero
                for state in states
            ])
        caller = sys._getframe(1)
        if "index_tensor" in caller.f_locals:
            trace.record(
                candidate_xyz,
                alternate_states,
                state_penalties,
                int(caller.f_locals["index_tensor"]),
            )
        return result

    def barrier_penalty(
        candidate_xyz,
        invariant_xyz,
        invariant_weights,
        alternate_states,
        soft_threshold,
        hard_threshold,
        barrier_buffer,
        barrier_scale,
    ):
        result = original_barrier(
            candidate_xyz,
            invariant_xyz,
            invariant_weights,
            alternate_states,
            soft_threshold,
            hard_threshold,
            barrier_buffer,
            barrier_scale,
        )
        state_penalties = []
        zero = torch.zeros(
            (), dtype=candidate_xyz.dtype, device=candidate_xyz.device
        )
        for states in alternate_states:
            penalties = []
            for state in states:
                if not state.numel():
                    penalties.append(zero)
                    continue
                distances = torch.cdist(candidate_xyz, state)
                base = torch.clamp(
                    soft_threshold - distances, min=0.0
                ).square()
                barrier = barrier_scale * (
                    torch.clamp(
                        hard_threshold + barrier_buffer - distances, min=0.0
                    ) / barrier_buffer
                ).pow(4)
                penalties.append((base + barrier).sum())
            state_penalties.append(penalties)
        caller = sys._getframe(1)
        if "index_tensor" in caller.f_locals:
            trace.record(
                candidate_xyz,
                alternate_states,
                state_penalties,
                int(caller.f_locals["index_tensor"]),
            )
        return result

    def traced_adam(params, *args, **kwargs):
        learning_rate = kwargs.get("lr", args[0] if args else 1e-3)
        instance = original_adam(params, *args, **kwargs)
        learning_rate = float(learning_rate)
        four_chi = "_ARG" in site or "_LYS" in site
        stage2 = (
            abs(learning_rate - 0.1) < 1e-8
            and (
                four_chi
                and previous_learning_rate["value"] is not None
                and abs(previous_learning_rate["value"] - 0.01) < 1e-8
                or not four_chi
                and previous_learning_rate["value"] is not None
                and abs(previous_learning_rate["value"] - 1.0) < 1e-8
            )
        )
        previous_learning_rate["value"] = learning_rate
        if stage2:
            trace.start_stage2()
        original_step = instance.step

        def step(*step_args, **step_kwargs):
            result = original_step(*step_args, **step_kwargs)
            if stage2:
                trace.optimizer_step()
            return result

        instance.step = step
        return instance

    optimizer_module.partition_soft_environment = traced_partition
    optimizer_module.soft_clash_penalty = direct_penalty
    optimizer_module.soft_clash_barrier_penalty = barrier_penalty
    optimizer_module.torch.optim.Adam = traced_adam
    sys.argv = ["five_site_optimizer.py", *optimizer_args]
    try:
        optimizer_module.main()
    finally:
        trace.checkpoint()


if __name__ == "__main__":
    main()
