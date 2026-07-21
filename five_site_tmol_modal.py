"""Detached corrected-tmol audit for held-out five-site denoised ensembles."""

from pathlib import Path

import modal


ROOT = Path(__file__).parent
APP = modal.App("heldout-five-site-denoised-tmol-audit")
VOLUME = modal.Volume.from_name("qfit-probe4-results", create_if_missing=True)
TMOL_WHEEL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.40/"
    "tmol-0.1.40%2Bcu128torch2.8-cp312-cp312-linux_x86_64.whl"
)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-runtime-ubuntu24.04", add_python="3.12")
    .pip_install(
        "torch==2.8.0", extra_options="--index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(f"pip install 'tmol @ {TMOL_WHEEL}'")
    .add_local_dir(
        ROOT / "heldout_five_site_physical_audit",
        remote_path="/audit",
        copy=True,
    )
)


@APP.function(image=IMAGE, gpu="L4", timeout=7_200, volumes={"/outputs": VOLUME})
def score_all() -> dict:
    import csv
    import json
    import os

    import tmol
    import torch

    device = torch.device("cuda")
    inputs = json.loads(Path("/audit/tmol_inputs.json").read_text())
    output = Path("/outputs/heldout_five_site_physical_audit")
    output.mkdir(parents=True, exist_ok=True)
    rows = []

    def atomic_json(path: Path, value) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)

    def write_rows() -> None:
        temporary = output / "tmol_energies.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output / "tmol_energies.csv")

    for site_index, site in enumerate(inputs["sites"]):
        base_pdb = Path("/audit") / site["base_pdb"]
        pose = tmol.pose_stack_from_pdb(str(base_pdb), device=device)
        scorer = tmol.beta2016_score_function(device).render_whole_pose_scoring_module(pose)
        base_lines = base_pdb.read_text().splitlines()
        candidate_path = Path(f"/tmp/{site['site']}_candidate.pdb")
        energy_cache = {}

        def energy(candidate) -> float:
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
            value = float(scorer(candidate_pose.coords).sum().detach().cpu())
            energy_cache[cache_key] = value
            return value

        energy_a, energy_b = energy(site["A"]), energy(site["B"])
        random_energies = [energy(candidate) for candidate in site["random_rotamers"]]
        for candidate in site["candidates"]:
            rows.append({
                "candidate_id": candidate["candidate_id"],
                "site": site["site"],
                "start": candidate["start"],
                "conformer": candidate["conformer"],
                "tmol_energy": energy(candidate["coordinates"]),
                "tmol_A": energy_a,
                "tmol_B": energy_b,
                "random_mean": sum(random_energies) / len(random_energies),
                "random_min": min(random_energies),
                "random_max": max(random_energies),
            })
        write_rows()
        atomic_json(output / "tmol_progress.json", {
            "completed_sites": site_index + 1,
            "total_sites": len(inputs["sites"]),
            "latest_site": site["site"],
            "endpoints_scored": len(rows),
            "tmol_version": tmol.__version__,
        })
        VOLUME.commit()
        print(f"scored {site_index + 1}/{len(inputs['sites'])}: {site['site']}", flush=True)

    summary = {
        "status": "complete",
        "tmol_version": tmol.__version__,
        "active_conformers_scored": len(rows),
        "sites": len(inputs["sites"]),
    }
    atomic_json(output / "tmol_manifest.json", summary)
    VOLUME.commit()
    return summary


@APP.local_entrypoint()
def main():
    call = score_all.spawn()
    print({"status": "submitted", "function_call_id": call.object_id})
