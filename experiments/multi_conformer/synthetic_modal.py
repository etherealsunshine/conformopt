"""K=4 differentiable multi-conformer fitting on the five 2O1K altloc sites.

The density target contains both kinematic A and B conformers. Chi angles and
softmax occupancy logits are optimized jointly with Adam; no MLP, rotamer
library, or MIQP is used. Every five starts and every completed site/ratio are
committed to a persistent Modal Volume.
"""

from __future__ import annotations

from pathlib import Path

import modal


ROOT = Path(__file__).parent
APP = modal.App("multi-conformer-direct-2o1k")
RESULTS_VOLUME = modal.Volume.from_name("qfit-probe4-results", create_if_missing=True)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-runtime-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.8.0", extra_options="--index-url https://download.pytorch.org/whl/cu128")
    .pip_install("gemmi==0.6.7", "matplotlib==3.9.4", "numpy==1.26.4")
    .add_local_file(ROOT / "probe4_core.py", remote_path="/root/probe4_core.py", copy=True)
    .add_local_file(ROOT / "data" / "2O1K.pdb", remote_path="/data/2O1K.pdb", copy=True)
)


@APP.function(image=IMAGE, gpu="L4", timeout=86_400, volumes={"/outputs": RESULTS_VOLUME})
def run_part1(config: dict) -> dict:
    import csv
    import json
    import math
    import os
    import time

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import gemmi
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from probe4_core import dihedral, seed_everything, torsion_to_coords, wrap_angles

    device = torch.device("cuda")
    out = Path("/outputs") / config["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "stage_manifest.json"
    seed_everything(config["seed"])

    def atomic_json(path: Path, value) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)

    def commit() -> None:
        RESULTS_VOLUME.commit()

    def manifest() -> dict:
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())
        return {"protein": "2O1K", "created_at": time.time(), "stages": {}}

    def mark(stage: str, status: str, **details) -> None:
        value = manifest()
        value["stages"][stage] = {"status": status, "updated_at": time.time(), **details}
        atomic_json(manifest_path, value)
        commit()

    def done(stage: str) -> bool:
        return (
            not config["force"]
            and manifest().get("stages", {}).get(stage, {}).get("status") == "complete"
        )

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    chi_specs = {
        "ARG": {
            "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
                          ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
            "rotations": (("CA", "CB", ("CG", "CD", "NE", "CZ", "NH1", "NH2")),
                          ("CB", "CG", ("CD", "NE", "CZ", "NH1", "NH2")),
                          ("CG", "CD", ("NE", "CZ", "NH1", "NH2")),
                          ("CD", "NE", ("CZ", "NH1", "NH2"))),
        },
        "MET": {
            "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"),
                          ("CB", "CG", "SD", "CE")),
            "rotations": (("CA", "CB", ("CG", "SD", "CE")),
                          ("CB", "CG", ("SD", "CE")),
                          ("CG", "SD", ("CE",))),
        },
        "ASP": {
            "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
            "rotations": (("CA", "CB", ("CG", "OD1", "OD2")),
                          ("CB", "CG", ("OD1", "OD2"))),
        },
    }

    structure = gemmi.read_structure("/data/2O1K.pdb")

    def alt_atom_map(residue, alt: str) -> dict:
        result = {}
        for atom in residue:
            atom_alt = atom.altloc if atom.altloc not in ("\x00", " ") else ""
            if atom_alt in ("", alt):
                result[atom.name.strip()] = torch.tensor(
                    atom.pos.tolist(), dtype=torch.float32, device=device
                )
        return result

    sites = []
    for chain in structure[0]:
        for residue in chain:
            if residue.name not in chi_specs:
                continue
            alts = {atom.altloc for atom in residue if atom.altloc not in ("\x00", " ", "")}
            if not {"A", "B"}.issubset(alts):
                continue
            map_a, map_b = alt_atom_map(residue, "A"), alt_atom_map(residue, "B")
            b_atoms = [
                atom for atom in residue
                if atom.altloc == "B" and atom.name.strip() not in {"N", "CA", "C", "O"}
            ]
            names = [atom.name.strip() for atom in b_atoms]
            if not b_atoms or any(name not in map_a or name not in map_b for name in names):
                continue
            spec = chi_specs[residue.name]
            chi_a = torch.stack([
                dihedral(*(map_a[name] for name in quartet)) for quartet in spec["dihedrals"]
            ])
            chi_b = torch.stack([
                dihedral(*(map_b[name] for name in quartet)) for quartet in spec["dihedrals"]
            ])
            delta = wrap_angles(chi_b - chi_a)
            template = torch.stack([map_a[name] for name in names])
            deposited_b = torch.stack([map_b[name] for name in names])
            fixed = {name: value for name, value in map_a.items() if name not in names}
            plus = torsion_to_coords(template, names, delta, list(spec["rotations"]), fixed)
            minus = torsion_to_coords(template, names, -delta, list(spec["rotations"]), fixed)
            if torch.mean((minus - deposited_b).square()) < torch.mean((plus - deposited_b).square()):
                delta = -delta
            # Normalize out the deposited altloc occupancy. Ensemble logits
            # control the total occupancy; per-atom values retain only relative
            # differences within the sidechain (normally all ones).
            raw_occupancies = torch.tensor(
                [atom.occ for atom in b_atoms], dtype=torch.float32, device=device
            )
            occupancy_scale = raw_occupancies.median().clamp_min(1e-6)
            sites.append({
                "key": f"{chain.name}_{residue.name}{residue.seqid.num}",
                "chain": chain.name,
                "number": residue.seqid.num,
                "resname": residue.name,
                "n_chi": len(spec["rotations"]),
                "rotations": list(spec["rotations"]),
                "names": names,
                "template": template,
                "deposited_b": deposited_b,
                "fixed": fixed,
                "true_delta": delta,
                "bfactors": torch.tensor(
                    [atom.b_iso for atom in b_atoms], dtype=torch.float32, device=device
                ),
                "atom_occupancies": raw_occupancies / occupancy_scale,
                "deposited_B_occupancy": float(occupancy_scale.cpu()),
            })
    if len(sites) != 5:
        raise RuntimeError(f"expected five sites, found {[site['key'] for site in sites]}")

    def coords_from_chi(site: dict, chi: torch.Tensor) -> torch.Tensor:
        return torsion_to_coords(
            site["template"], site["names"], chi, site["rotations"], site["fixed"]
        )

    def atom_density(site: dict, coordinates: torch.Tensor) -> torch.Tensor:
        sigma2 = (site["bfactors"] / (8.0 * math.pi ** 2)).clamp_min(1e-4)
        differences = site["grid"][:, None, :] - coordinates[None, :, :]
        distance2 = differences.square().sum(dim=-1)
        normalization = (2.0 * math.pi * sigma2).pow(-1.5)
        return (
            site["atom_occupancies"][None, :]
            * normalization[None, :]
            * torch.exp(-distance2 / (2.0 * sigma2[None, :]))
        ).sum(dim=1)

    radius, spacing = config["grid_radius"], config["grid_spacing"]
    axis = torch.arange(-radius, radius + spacing / 2, spacing, device=device)
    offsets = torch.cartesian_prod(axis, axis, axis)
    offsets = offsets[torch.linalg.vector_norm(offsets, dim=1) <= radius]
    for site in sites:
        site["kinematic_a"] = coords_from_chi(
            site, torch.zeros(site["n_chi"], device=device)
        ).detach()
        site["kinematic_b"] = coords_from_chi(site, site["true_delta"]).detach()
        center = torch.cat((site["kinematic_a"], site["kinematic_b"]), dim=0).mean(dim=0)
        site["grid"] = center.unsqueeze(0) + offsets
        with torch.no_grad():
            site["rho_a"] = atom_density(site, site["kinematic_a"]).detach()
            site["rho_b"] = atom_density(site, site["kinematic_b"]).detach()

    assertion_rows = []
    for site in sites:
        target = 0.5 * site["rho_a"] + 0.5 * site["rho_b"]
        loss_a_only = (site["rho_a"] - target).square().sum()
        loss_ab = (0.5 * site["rho_a"] + 0.5 * site["rho_b"] - target).square().sum()
        assertion_rows.append({
            "site": site["key"],
            "loss_A_only": float(loss_a_only.cpu()),
            "loss_A_plus_B_50_50": float(loss_ab.cpu()),
            "kinematic_to_deposited_B_rmsd": float(torch.sqrt(torch.mean(
                (site["kinematic_b"] - site["deposited_b"]).square()
            )).cpu()),
        })
        if float(loss_ab.cpu()) >= 1e-4 or float(loss_ab.cpu()) >= float(loss_a_only.cpu()):
            raise RuntimeError(f"multi-conformer target assertion failed at {site['key']}")
    write_csv(out / "pre_experiment_assertions.csv", assertion_rows)
    atomic_json(out / "run_config.json", config)
    commit()

    def evaluate_slots(
        occupancies: np.ndarray,
        rmsd_a: np.ndarray,
        rmsd_b: np.ndarray,
        target_a: float,
        target_b: float,
    ) -> dict:
        assignments = []
        for k in range(len(occupancies)):
            if occupancies[k] <= config["nontrivial_occupancy"]:
                assignments.append("inactive")
            elif rmsd_a[k] < 1.0 and rmsd_a[k] <= rmsd_b[k]:
                assignments.append("A")
            elif rmsd_b[k] < 1.0:
                assignments.append("B")
            else:
                assignments.append("other")
        a_indices = [k for k, label in enumerate(assignments) if label == "A"]
        b_indices = [k for k, label in enumerate(assignments) if label == "B"]
        predicted_a = float(occupancies[a_indices].sum()) if a_indices else 0.0
        predicted_b = float(occupancies[b_indices].sum()) if b_indices else 0.0
        has_a = any(occupancies[k] > 0.1 for k in a_indices)
        has_b = any(occupancies[k] > 0.1 for k in b_indices)
        occupancy_accurate = (
            abs(predicted_a - target_a) <= config["occupancy_tolerance"]
            and abs(predicted_b - target_b) <= config["occupancy_tolerance"]
        )
        return {
            "assignments": assignments,
            "predicted_A_occupancy": predicted_a,
            "predicted_B_occupancy": predicted_b,
            "active_conformers": int((occupancies > config["nontrivial_occupancy"]).sum()),
            "found_A": has_a,
            "found_B": has_b,
            "occupancy_accurate": occupancy_accurate,
            "ensemble_success": has_a and has_b and occupancy_accurate,
        }

    def optimize_site(site: dict, ratio: tuple[float, float], ratio_seed: int):
        target_a, target_b = ratio
        rho_target = target_a * site["rho_a"] + target_b * site["rho_b"]
        endpoints, trajectory_arrays = [], []
        for start in range(config["n_starts"]):
            generator = torch.Generator(device=device).manual_seed(
                config["seed"] + ratio_seed + start
            )
            all_chi = torch.randn(
                (config["K"], site["n_chi"]), generator=generator, device=device
            ).requires_grad_(True)
            occupancy_logits = torch.zeros(config["K"], device=device, requires_grad=True)
            optimizer = torch.optim.Adam([all_chi, occupancy_logits], lr=config["lr"])
            losses, occupancy_path, rmsd_a_path, rmsd_b_path = [], [], [], []
            for _step in range(config["n_steps"]):
                optimizer.zero_grad(set_to_none=True)
                occupancies = torch.softmax(occupancy_logits, dim=0)
                coordinates = [
                    coords_from_chi(site, wrap_angles(all_chi[k]))
                    for k in range(config["K"])
                ]
                rho_ensemble = sum(
                    occupancies[k] * atom_density(site, coordinates[k])
                    for k in range(config["K"])
                )
                loss = (rho_ensemble - rho_target).square().sum()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    all_chi.copy_(wrap_angles(all_chi))
                    occ_now = torch.softmax(occupancy_logits, dim=0)
                    coords_now = [
                        coords_from_chi(site, all_chi[k]) for k in range(config["K"])
                    ]
                    losses.append(float(loss.detach().cpu()))
                    occupancy_path.append(occ_now.detach().cpu().numpy())
                    rmsd_a_path.append([
                        float(torch.sqrt(torch.mean((xyz - site["kinematic_a"]).square())).cpu())
                        for xyz in coords_now
                    ])
                    rmsd_b_path.append([
                        float(torch.sqrt(torch.mean((xyz - site["kinematic_b"]).square())).cpu())
                        for xyz in coords_now
                    ])
            final_occ = np.asarray(occupancy_path[-1])
            final_a, final_b = np.asarray(rmsd_a_path[-1]), np.asarray(rmsd_b_path[-1])
            evaluation = evaluate_slots(final_occ, final_a, final_b, target_a, target_b)
            endpoints.append({
                "site": site["key"],
                "ratio_A": target_a,
                "ratio_B": target_b,
                "start": start,
                "final_loss": losses[-1],
                "best_loss": min(losses),
                "occupancies": ";".join(f"{x:.8g}" for x in final_occ),
                "rmsd_to_A": ";".join(f"{x:.8g}" for x in final_a),
                "rmsd_to_B": ";".join(f"{x:.8g}" for x in final_b),
                "assignments": ";".join(evaluation["assignments"]),
                **{key: value for key, value in evaluation.items() if key != "assignments"},
                "final_chi_radians": "|".join(
                    ";".join(f"{value:.8g}" for value in row)
                    for row in all_chi.detach().cpu().numpy()
                ),
            })
            trajectory_arrays.append({
                "loss": np.asarray(losses, dtype=np.float32),
                "occupancies": np.asarray(occupancy_path, dtype=np.float32),
                "rmsd_to_A": np.asarray(rmsd_a_path, dtype=np.float32),
                "rmsd_to_B": np.asarray(rmsd_b_path, dtype=np.float32),
            })
            if (start + 1) % config["checkpoint_starts"] == 0:
                return_directory = out / "2O1K" / "multi_conformer"
                return_directory.mkdir(parents=True, exist_ok=True)
                partial = return_directory / (
                    f"{site['key']}_ratio_{int(target_a*100)}_{int(target_b*100)}_partial.csv"
                )
                write_csv(partial, endpoints)
                mark(
                    f"ratio_{target_a:.1f}_{target_b:.1f}::{site['key']}",
                    "running", completed_starts=start + 1,
                )
        stacked = {
            key: np.stack([trajectory[key] for trajectory in trajectory_arrays])
            for key in trajectory_arrays[0]
        }
        return endpoints, stacked

    def write_overlay(site: dict, best: dict, path: Path) -> None:
        occupancies = [float(value) for value in best["occupancies"].split(";")]
        chi_rows = [
            [float(value) for value in row.split(";")]
            for row in best["final_chi_radians"].split("|")
        ]
        lines = [
            f"REMARK K=4 learned ensemble for {site['key']}",
            f"REMARK target A:B = {best['ratio_A']}:{best['ratio_B']}",
        ]
        serial = 1
        for k, (occupancy, chi_values) in enumerate(zip(occupancies, chi_rows)):
            chi = torch.tensor(chi_values, dtype=torch.float32, device=device)
            coordinates = coords_from_chi(site, chi).detach().cpu().numpy()
            altloc = chr(ord("A") + k)
            for atom_index, (name, xyz) in enumerate(zip(site["names"], coordinates)):
                element = "S" if name.startswith("S") else name[0]
                bfactor = float(site["bfactors"][atom_index].cpu())
                lines.append(
                    f"ATOM  {serial:5d} {name:>4s}{altloc}{site['resname']:>3s} {site['chain']}"
                    f"{site['number']:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"{occupancy:6.2f}{bfactor:6.2f}          {element:>2s}"
                )
                serial += 1
        lines.extend(("TER", "END"))
        path.write_text("\n".join(lines) + "\n")

    multi_directory = out / "2O1K" / "multi_conformer"
    multi_directory.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    successful_sites = []
    base_ratio = (0.5, 0.5)
    for site_index, site in enumerate(sites):
        stage = f"ratio_0.5_0.5::{site['key']}"
        result_path = multi_directory / f"{site['key']}_results.csv"
        trajectory_path = multi_directory / f"{site['key']}_trajectories.npz"
        if done(stage) and result_path.exists():
            endpoints = list(csv.DictReader(result_path.open()))
            trajectories = dict(np.load(trajectory_path))
        else:
            endpoints, trajectories = optimize_site(
                site, base_ratio, ratio_seed=100_000 * (site_index + 1)
            )
            write_csv(result_path, endpoints)
            np.savez_compressed(trajectory_path, **trajectories)
            mark(stage, "complete", completed_starts=len(endpoints))
        successes = sum(str(row["ensemble_success"]) == "True" or row["ensemble_success"] is True for row in endpoints)
        both_found = sum(
            (str(row["found_A"]) == "True" or row["found_A"] is True)
            and (str(row["found_B"]) == "True" or row["found_B"] is True)
            for row in endpoints
        )
        summary_rows.append({
            "site": site["key"],
            "residue": site["resname"],
            "n_chi": site["n_chi"],
            "starts": len(endpoints),
            "both_found": both_found,
            "ensemble_success": successes,
            "mean_predicted_A_occupancy": float(np.mean([
                float(row["predicted_A_occupancy"]) for row in endpoints
            ])),
            "mean_predicted_B_occupancy": float(np.mean([
                float(row["predicted_B_occupancy"]) for row in endpoints
            ])),
            "mean_active_conformers": float(np.mean([
                float(row["active_conformers"]) for row in endpoints
            ])),
        })
        if successes >= config["success_starts"]:
            successful_sites.append(site["key"])
        best = min(
            (row for row in endpoints if str(row["ensemble_success"]) == "True" or row["ensemble_success"] is True),
            key=lambda row: float(row["final_loss"]),
            default=min(endpoints, key=lambda row: float(row["final_loss"])),
        )
        write_overlay(site, best, multi_directory / f"{site['key']}_best_ensemble.pdb")

        plt.figure(figsize=(8, 5))
        mean_occupancy = trajectories["occupancies"].mean(axis=0)
        for k in range(config["K"]):
            plt.plot(mean_occupancy[:, k], label=f"slot {k + 1}")
        plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        plt.xlabel("optimization step")
        plt.ylabel("mean occupancy")
        plt.title(site["key"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(multi_directory / f"{site['key']}_occupancy_tracking.png", dpi=180)
        plt.close()
        commit()
    write_csv(out / "2O1K" / "multi_conformer_summary.csv", summary_rows)
    mark("base_ratio_summary", "complete", successful_sites=successful_sites)

    # Gate the ratio tests: run only where the 50/50 ensemble criterion passed.
    ratio_directory = out / "2O1K" / "occupancy_ratio_test"
    ratio_directory.mkdir(parents=True, exist_ok=True)
    ratio_summary = []
    for ratio_index, ratio in enumerate(((0.7, 0.3), (0.3, 0.7))):
        for site_index, site in enumerate(sites):
            if site["key"] not in successful_sites:
                continue
            stage = f"ratio_{ratio[0]:.1f}_{ratio[1]:.1f}::{site['key']}"
            result_path = ratio_directory / (
                f"{site['key']}_ratio_{int(ratio[0]*100)}_{int(ratio[1]*100)}.csv"
            )
            trajectory_path = result_path.with_suffix(".npz")
            if done(stage) and result_path.exists():
                endpoints = list(csv.DictReader(result_path.open()))
            else:
                endpoints, trajectories = optimize_site(
                    site, ratio,
                    ratio_seed=1_000_000 * (ratio_index + 1) + 100_000 * (site_index + 1),
                )
                write_csv(result_path, endpoints)
                np.savez_compressed(trajectory_path, **trajectories)
                mark(stage, "complete", completed_starts=len(endpoints))
            ratio_summary.append({
                "site": site["key"],
                "target_A": ratio[0],
                "target_B": ratio[1],
                "successes": sum(
                    str(row["ensemble_success"]) == "True" or row["ensemble_success"] is True
                    for row in endpoints
                ),
                "mean_predicted_A": float(np.mean([
                    float(row["predicted_A_occupancy"]) for row in endpoints
                ])),
                "mean_predicted_B": float(np.mean([
                    float(row["predicted_B_occupancy"]) for row in endpoints
                ])),
            })
    if ratio_summary:
        write_csv(ratio_directory / "ratio_summary.csv", ratio_summary)
    atomic_json(out / "part1_gate.json", {
        "success_threshold": config["success_starts"],
        "successful_sites": successful_sites,
        "part2_allowed": bool(successful_sites),
        "kill_criterion_triggered": not successful_sites,
    })
    mark("pipeline", "complete", successful_sites=successful_sites)
    return manifest()


@APP.local_entrypoint()
def main(
    run_name: str = "multi_conformer_multi_protein",
    n_starts: int = 50,
    n_steps: int = 500,
    lr: float = 1.0,
    force: bool = False,
):
    config = {
        "run_name": run_name,
        "n_starts": n_starts,
        "n_steps": n_steps,
        "lr": lr,
        "K": 4,
        "seed": 41,
        "grid_radius": 4.0,
        "grid_spacing": 0.5,
        "nontrivial_occupancy": 0.05,
        "occupancy_tolerance": 0.20,
        "success_starts": 30,
        "checkpoint_starts": 5,
        "force": force,
    }
    call = run_part1.spawn(config)
    print({"status": "submitted", "function_call_id": call.object_id, "run_name": run_name})
