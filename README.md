# qFit on Steroids

Research code for learning and optimizing crystallographic side-chain ensembles from experimental electron density. The current pipeline pairs an experimental omit-map patch with a synthetic structure-density target, denoises it with a residual 3D U-Net, and then performs differentiable multi-conformer torsion optimization with a soft physical prior.

This repository is an experiment record as well as an active codebase. The Probe 2/4/4b/4c scripts and their saved results document how the project progressed from learned-energy feasibility tests to experimental-density multi-conformer fitting.

## Current production pipeline

```text
PDB + deposited structure factors
            |
            v
sidechain-omit experimental patch ──> residual 3D U-Net ──> denoised patch
                                                              |
                                                              v
                                              K=4 torsion/occupancy optimizer
                                                stage 1: density only
                                                stage 2: soft physics
                                                              |
                                                              v
                                  RMSD + occupancy + clash + rotamer + tmol audit
```

The frozen downstream optimizer uses 50 random starts per site, 500 density-only Adam steps at `lr=1.0`, then an Adam reset and 200 soft-physics steps at one tenth the learning rate. The physics loss uses `lambda_vdw=1.0`, `lambda_rot=0.5`, and `lambda_clash=5.0`.

The strict success criterion is:

- both deposited conformers recovered with conventional, symmetry-aware RMSD below 1.0 Å;
- recovered occupancy within ±0.20 of the deposited value;
- no direct or crystallographic-symmetry contact below 2.0 Å;
- every active chi angle within 30° of an allowed residue-specific rotamer center;
- tmol energy no more than 10 units above the better deposited A/B control.

## Repository map

| Path | Purpose |
|---|---|
| `density_denoiser/` | Active data, U-Net, optimizer, validation, and audit package. |
| `data/` | Small checked-in 2O1K reference PDB/CIF/MTZ files. |
| `probe2*.py` | Early direct-coordinate and tmol/SFcalculator controls. |
| `probe4*.py` | Learned-energy, localized-loss, physics, and endpoint-audit experiments. |
| `direct_optimization_modal.py` | Direct kinematic landscape control without a learned model. |
| `multi_conformer_modal.py` | Synthetic K=4 multi-conformer fitting on 2O1K. |
| `experimental_multi_conformer_modal.py` | Experimental-map multi-conformer variants. |
| `five_site_tmol_audit.py` | Restartable local/pod tmol endpoint scorer. |
| `*_results/`, `*_audit/`, `results/` | Curated experiment outputs and reports. |
| `PROBE*.md`, `ARG129_*.md` | Scientific experiment write-ups. |
| `EBT_RESEARCH_CONTEXT.md` | Original long-form scientific motivation and project context. |
| `AGENTS.md` | Detailed operational manual, complete source-file catalog, and pod layout. |

## Runtime model

Source code lives here on the Mac. GPU execution and the large dataset live in the Astera `qfit-unet` workspace pod.

```text
local:  /Users/utkarsh/qfitonsteroids
pod:    /home/dev/workspace                 # synced source
pod:    /home/dev/qfit_unet_data            # persistent datasets/results/envs
```

Never develop a separate copy of the source inside the pod. Edit locally, keep `actl pod sync qfit-unet` running, verify the synced file, and run commands remotely.

```bash
# From this local repository
actl pod status qfit-unet
actl pod sync qfit-unet

# One-shot remote test (works whenever the pod is running)
actl pod exec qfit-unet --no-forwarding -- bash -lc \
  'cd /home/dev/workspace && /home/dev/qfit_unet_data/.venv/bin/python -m pytest -q'
```

The primary Python environment is `/home/dev/qfit_unet_data/.venv`. The separate CUDA/tmol environment is `/home/dev/qfit_unet_data/.venv-tmol`.

## Dataset preparation

The persistent dataset contains 1,985 train PDB files and 99 untouched test PDB files. Acquire deposited reflection data and convert them to MTZ:

```bash
cd /home/dev/workspace
/home/dev/qfit_unet_data/.venv/bin/python -m density_denoiser.data_pipeline acquire \
  --data-root /home/dev/qfit_unet_data --split both --workers 24
```

Generate experimental/synthetic patch pairs in the crystal frame:

```bash
/home/dev/qfit_unet_data/.venv/bin/python -m density_denoiser.data_pipeline prepare \
  --data-root /home/dev/qfit_unet_data --split both \
  --map-type omit_mfo_dfc --frame crystal --workers 24 --negatives-per-altloc 4
```

Use `--frame residue` to generate canonical residue-frame patches. Preparation is restartable and records per-protein/per-site status immediately. Do not use `--overwrite` casually.

## Train and evaluate the baseline U-Net

```bash
cd /home/dev/workspace
PY=/home/dev/qfit_unet_data/.venv/bin/python

$PY -m density_denoiser.train \
  --data-root /home/dev/qfit_unet_data \
  --frame crystal --base-channels 16 --batch-size 8 \
  --epochs 100 --workers 8 --resume

$PY -m density_denoiser.evaluate \
  --data-root /home/dev/qfit_unet_data \
  --frame crystal \
  --checkpoint /home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt
```

The historical baseline split and training curve are preserved in `original_crystal_unet_split.json` and `original_crystal_unet_training_log.csv`.

## Run the frozen held-out optimizer

The current 15-site prospective launcher performs calibration, launches one checkpointed shard per site, waits for all shards, then runs the geometry, tmol, and strict-summary audits:

```bash
cd /home/dev/workspace
nohup bash density_denoiser/run_expanded_heldout_two_stage_shards.sh \
  > /home/dev/qfit_unet_data/density_denoiser/expanded_heldout_two_stage_prospective_v1/logs/controller.log \
  2>&1 < /dev/null &
```

Monitor it without disturbing the job:

```bash
ROOT=/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_two_stage_prospective_v1
cat "$ROOT/status.txt"
find "$ROOT/pids" -name '*.status' -maxdepth 1 -print
tail -n 30 "$ROOT/logs/controller.log"
nvidia-smi
```

Do not relaunch into an existing output directory. The launcher deliberately refuses to overwrite a previous prospective run.

## Tests

Run all tests on the pod because the local machine does not carry the Linux/CUDA scientific environments:

```bash
actl pod exec qfit-unet --no-forwarding -- bash -lc \
  'cd /home/dev/workspace && /home/dev/qfit_unet_data/.venv/bin/python -m pytest -q'
```

The tests cover differentiable torsion kinematics, omit-map subtraction, canonical-frame invariance, synthetic rendering, the U-Net shape contract, residue topology, residue-specific rotamer centers, and atom-label symmetry in RMSD.

## More documentation

- Read `AGENTS.md` before changing or launching anything.
- Read `density_denoiser/README.md` for the density-pair preparation conventions.
- Read `PROBE4_FULL_EXPERIMENT_REPORT.md` and `PROBE4C1_4C2_REPORT.md` for the main experimental findings.
- Read `EBT_RESEARCH_CONTEXT.md` for the original motivation, terminology, and long-range research plan.

This is research software. Saved outputs are evidence tied to exact run configurations, not interchangeable benchmark numbers.
