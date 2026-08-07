"""Reconstruct and geometrically audit all Probe 4c endpoints on Modal."""

from pathlib import Path

import modal


ROOT = Path(__file__).parent
APP = modal.App("probe4c12-endpoint-geometry-audit")
VOLUME = modal.Volume.from_name("qfit-probe4-results", create_if_missing=True)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-runtime-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.8.0", extra_options="--index-url https://download.pytorch.org/whl/cu128")
    .pip_install(
        "SFcalculator-torch==0.3.3", "gemmi==0.6.7", "numpy==1.26.4", "scipy==1.15.1"
    )
    .add_local_file(ROOT / "probe4_core.py", remote_path="/root/probe4_core.py", copy=True)
    .add_local_file(
        ROOT / "probe4b_endpoint_audit.py", remote_path="/root/probe4b_endpoint_audit.py", copy=True
    )
    .add_local_dir(ROOT / "data", remote_path="/root/data", copy=True)
    .add_local_dir(ROOT / "probe4c1_results", remote_path="/root/probe4c1_results", copy=True)
    .add_local_dir(ROOT / "probe4c2_results", remote_path="/root/probe4c2_results", copy=True)
)


@APP.function(image=IMAGE, gpu="L4", timeout=3_600, volumes={"/outputs": VOLUME})
def run_audit() -> dict:
    import probe4b_endpoint_audit as audit

    audit.ROOT = Path("/root")
    audit.OUT = Path("/outputs/probe4c12_results/endpoint_audit")
    audit.ASP_ALLOWS_QUADRATURE = True
    audit.EXPERIMENTS = {
        "probe4c1_kinematic_complex": Path(
            "/root/probe4c1_results/altloc_test/trajectories.json"
        ),
        "probe4c2_A_soft_synthetic": Path(
            "/root/probe4c2_results/A_synthetic_fobs/altloc_test/trajectories.json"
        ),
        "probe4c2_B_soft_localized": Path(
            "/root/probe4c2_results/B_localized_sf/altloc_test/trajectories.json"
        ),
        "probe4c2_C_soft_realspace": Path(
            "/root/probe4c2_results/C_realspace_local/altloc_test/trajectories.json"
        ),
    }
    audit.main()
    VOLUME.commit()
    return {"status": "complete", "endpoint_rows": 1000, "experiments": 4, "sites": 5}


@APP.local_entrypoint()
def main():
    call = run_audit.spawn()
    print({"status": "submitted", "function_call_id": call.object_id})
