"""tmol physical-energy audit for all 750 Probe 4b learned endpoints.

Submit fire-and-forget with:

    UV_CACHE_DIR=/private/tmp/uv-modal uvx modal run --detach probe4b_tmol_modal.py
"""

from pathlib import Path

import modal


ROOT = Path(__file__).parent
APP = modal.App("probe4b-endpoint-tmol-audit")
VOLUME = modal.Volume.from_name("qfit-probe4-results", create_if_missing=True)
TMOL_WHEEL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.40/"
    "tmol-0.1.40%2Bcu128torch2.8-cp312-cp312-linux_x86_64.whl"
)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-runtime-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.8.0", extra_options="--index-url https://download.pytorch.org/whl/cu128")
    .run_commands(f"pip install 'tmol @ {TMOL_WHEEL}'")
    .add_local_dir(
        ROOT / "probe4b_results" / "endpoint_audit",
        remote_path="/audit",
        copy=True,
    )
)


@APP.function(image=IMAGE, gpu="L4", timeout=3_600, volumes={"/outputs": VOLUME})
def score_all() -> dict:
    import csv
    import json
    import os

    import torch
    import tmol

    device = torch.device("cuda")
    inputs = json.loads(Path("/audit/tmol_inputs.json").read_text())
    output_dir = Path("/outputs/probe4b_results/endpoint_audit")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def atomic_json(path: Path, value) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)

    for site_index, site in enumerate(inputs["sites"]):
        base_pdb = f"/audit/visualization/base_chain_{site['chain']}.pdb"
        pose = tmol.pose_stack_from_pdb(base_pdb, device=device)
        scorer = tmol.beta2016_score_function(device).render_whole_pose_scoring_module(pose)
        base_lines = Path(base_pdb).read_text().splitlines()
        candidate_path = Path(f"/tmp/{site['site']}_candidate.pdb")
        energy_cache = {}

        def energy(candidate) -> float:
            # Reparse each heavy-atom candidate so tmol rebuilds sidechain
            # hydrogens at the candidate coordinates.  Moving heavy atoms in
            # an already-hydrogenated pose leaves the original hydrogens behind
            # and creates enormous, artificial bond/clash energies.
            # PDB coordinates are represented at 0.001 A precision; endpoints
            # that round to the same physical structure therefore share a
            # score while still receiving individual output rows.
            cache_key = tuple(round(value, 3) for atom in candidate for value in atom)
            if cache_key in energy_cache:
                return energy_cache[cache_key]
            replacements = dict(zip(site["atom_names"], candidate))
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
            candidate_pose = tmol.pose_stack_from_pdb(str(candidate_path), device=device)
            result = float(scorer(candidate_pose.coords).sum().detach().cpu())
            energy_cache[cache_key] = result
            return result

        controls = {
            "A": energy(site["A"]),
            "B": energy(site["B"]),
            "random_rotamers": [energy(candidate) for candidate in site["random_rotamers"]],
        }
        for experiment, candidates in site["experiments"].items():
            for start, candidate in enumerate(candidates):
                rows.append({
                    "experiment": experiment,
                    "site": site["site"],
                    "start": start,
                    "tmol_energy": energy(candidate),
                    "tmol_A": controls["A"],
                    "tmol_B": controls["B"],
                    "random_mean": sum(controls["random_rotamers"]) / len(controls["random_rotamers"]),
                    "random_min": min(controls["random_rotamers"]),
                    "random_max": max(controls["random_rotamers"]),
                })
        atomic_json(output_dir / "tmol_progress.json", {
            "completed_sites": site_index + 1,
            "total_sites": len(inputs["sites"]),
            "latest_site": site["site"],
            "tmol_version": tmol.__version__,
        })
        with (output_dir / "tmol_energies.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        VOLUME.commit()
        print(f"scored {site_index + 1}/{len(inputs['sites'])}: {site['site']}")

    summary = {
        "status": "complete",
        "tmol_version": tmol.__version__,
        "endpoints_scored": len(rows),
        "sites": len(inputs["sites"]),
    }
    atomic_json(output_dir / "tmol_manifest.json", summary)
    VOLUME.commit()
    return summary


@APP.local_entrypoint()
def main():
    call = score_all.spawn()
    print({"status": "submitted", "function_call_id": call.object_id})
