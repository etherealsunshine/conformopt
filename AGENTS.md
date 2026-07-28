# AGENTS.md — qFit on Steroids Operating Manual

This file is the source of truth for agents and contributors working in this repository. Read it before editing code, running experiments, moving results, or changing the pod.

After reading the operating rules here, read
[`CURRENT_HANDOFF.md`](CURRENT_HANDOFF.md). It contains the latest 20-site
denoised and synthetic results, current pod result roots, exact launch and
monitoring commands, resolved diagnostics, and the active research backlog.

## 1. Non-negotiable operating rules

1. **Source is local; runtime is remote.** Edit `/Users/utkarsh/qfitonsteroids` on the Mac. Run tests, preprocessing, training, optimization, and tmol on the `qfit-unet` pod.
2. **Never edit source inside `/home/dev/workspace`.** It is the synchronized runtime copy. A remote-only edit can be overwritten and will not be represented correctly in local Git.
3. **Use `actl pod sync`, not scp/rsync, to send source.** Verify important edits on the pod before an expensive launch.
4. **Keep large data and model outputs on the pod PVC.** They belong under `/home/dev/qfit_unet_data`, not in the synchronized source tree.
5. **Do not modify the train/test split.** Validation is selected by whole PDB ID from train. The 99-protein test directory is held out from training and early stopping.
6. **Do not open prospective recovery results while selecting sites or calibrating support.** Site eligibility may use chemistry, occupancy, resolution, representability, and deposited A/B separation, but not optimizer or denoiser performance.
7. **Checkpoint every long process.** Acquisition, patch generation, training, optimization shards, geometry audit, and tmol audit must write progress atomically during the process—not only at the end.
8. **Detached jobs must be genuinely detached.** Use `nohup ... > log 2>&1 < /dev/null &`, write the controller PID, and verify the process and first checkpoint after launch.
9. **Never overwrite a scientific run silently.** Use a new versioned output directory. Only use `--overwrite` or `--force` when the user explicitly intends to replace data.
10. **Report conventional RMSD.** The definition is `sqrt(mean_atoms(sum_xyz(delta**2)))`, with chemically equivalent terminal atom labels minimized over valid permutations. Do not use coordinate-wise RMSD (`sqrt(mean over atoms and xyz)`), which is looser by `sqrt(3)`.
11. **Recovery alone is not success.** The primary result is the strict joint metric: both A/B found, occupancy correct, no sub-2 Å direct or symmetry clash, canonical chi angles, and acceptable tmol energy.
12. **Preserve user data and unrelated changes.** Never clean, reset, or delete a result tree just because it is old or untracked.

## 2. Local and remote topology

### Local repository

```text
/Users/utkarsh/qfitonsteroids/
  .git/                         local Git metadata; excluded from actl sync
  README.md                     human entry point
  AGENTS.md                     this operating manual
  EBT_RESEARCH_CONTEXT.md       original long-form scientific context
  density_denoiser/             active package and production launchers
  data/                         small 2O1K reference inputs
  results/ and *_results/       curated local outputs
  *_modal.py                    historical/alternate Modal launchers
```

### Astera workspace

The active workspace is project `qfit-unet` in namespace `diffuse`.

```text
/home/dev/
  workspace/                    synchronized copy of this local repository
  qfit_unet_data/               persistent home-PVC data and outputs
```

The source and data trees have different lifecycles:

- `/home/dev/workspace` is replaceable and synchronized from the Mac.
- `/home/dev/qfit_unet_data` persists across normal `actl pod down` / `actl pod up` cycles as long as the PVC is kept.
- `actl pod down --delete-data` destroys the PVC and must never be used without explicit confirmation.

### Python environments on the pod

| Environment | Use |
|---|---|
| `/home/dev/qfit_unet_data/.venv/bin/python` | Gemmi, PyTorch, NumPy, SciPy, training, preprocessing, optimization, tests. |
| `/home/dev/qfit_unet_data/.venv-tmol/bin/python` | Linux CUDA tmol scoring and its compatible PyTorch stack. |
| Local `.venv-tmol/` | Historical machine-specific environment; ignored by Git and not a valid substitute for the pod tmol environment. |

Never load a checkpoint with unrestricted pickle when a weights-only path is available. Model-loading code should use `torch.load(..., weights_only=True)` or an explicitly documented compatibility fallback.

## 3. Pod data tree

Current root: `/home/dev/qfit_unet_data`.

```text
qfit_unet_data/
  train/                         1,985 source PDB files
  test/                             99 held-out source PDB files
  cache/
    train/
      structure_factors/         downloaded `PDBID-sf.cif`
      mtz/                       derived reflection MTZ files
      pairs/                     crystal-frame `.npz` density pairs
      status/
        acquisition/             one JSON per acquisition attempt
        sites/                   discovered-site records
        patches/                 per-protein patch-generation status
      canonical/
        pairs/                   residue-frame `.npz` density pairs
        status/                  canonical preparation status
    test/                        same schema as `train/`
  density_denoiser/
    model/                       original crystal-frame U-Net checkpoints/logs
    model_canonical/             first canonical-frame model
    model_canonical_150_fresh/   fresh 150-epoch canonical experiment
    model_unet2*/                landscape-auxiliary pilots/calibrations
    evaluation/                  test-set reconstruction metrics/figures
    unet2_landscape_cache*/      cached candidate landscapes and manifests
    heldout_five_site_selection_v3/
                                  frozen original five-site selection
    heldout_five_site_two_stage/ original five-site best production run
    heldout_five_site_two_stage_audit/
                                  its geometry/tmol/strict audit
    expanded_heldout_selection_15_v2_symmetry/
                                  frozen symmetry-aware 15-site selection
    expanded_heldout_residue_kinematic_validation_v1/
                                  topology and differentiability validation
    expanded_heldout_residue_support_calibration_v2/
                                  15-site density/physics calibration
    expanded_residue_backward_compat_calibration_v1/
                                  original-five regression calibration
    expanded_heldout_two_stage_prospective_v1/
                                  current 15-site prospective production run
  logs/                          acquisition/preparation controller logs/PIDs
  .venv/                         primary runtime
  .venv-tmol/                    tmol runtime
```

### Density-pair files

Each prepared `.npz` pair contains a normalized experimental input patch, a synthetic target patch, a local side-chain mask, and metadata needed to identify/reconstruct the site. Patches are normally `(1, 32, 32, 32)` at 0.5 Å spacing.

- Crystal frame: sampled on crystal axes, stored under `cache/{split}/pairs`.
- Residue frame: origin at Cα, x = Cα→Cβ, z = `(Cα→N) × (Cα→Cβ)`, y = z×x; stored under `cache/{split}/canonical/pairs`.
- `omit_mfo_dfc`: experimental difference-map input with a sidechain-only synthetic target.
- `omit_2mfo_dfc`: full local synthetic target.

Source PDBs are never modified.

### Standard run-directory schema

```text
RUN_ROOT/
  status.txt                    current controller stage
  controller.pid               detached controller PID
  calibration/                 deposited A/B density and physics gate
  logs/
    controller.log
    calibration.log
    SITE.log                    per-site optimizer output
    geometry_audit.log
    tmol_calibration.log
    tmol_audit.log
    strict_summary.log
  pids/
    SITE.pid
    SITE.status                 written complete/failed after wait
  shards/
    SITE/
      run_config.json
      stage_manifest.json
      denoised/SITE_starts.csv
      ...targets, checkpoints, summaries...
  audit/
    geometry_summary.json
    ensemble_geometry_audit.csv
    active_conformer_geometry_audit.csv
    deposited_control_geometry_audit.csv
    tmol_inputs.json
    tmol_calibration.json
    tmol_energies.csv
    tmol_progress.json
    tmol_manifest.json
    ensemble_strict_audit.csv
    active_conformer_strict_audit.csv
    strict_per_site.csv
    strict_summary.json
    visualization/
```

`status.txt=complete` means the full controller—including strict summary—finished. A dead controller with `status.txt=optimizing` or `optimizer_failed` is not complete even if some shards have results.

## 4. Scientific data flow and invariants

1. `data_pipeline acquire` discovers PDBs, downloads deposited structure factors, converts them to MTZ, and writes per-protein status.
2. `data_pipeline prepare` discovers altloc/negative sites, computes an omit map, extracts normalized experimental patches, renders synthetic targets, and writes pair records.
3. `train.py` trains the residual 3D U-Net on train proteins and early-stops using a protein-level validation split selected only from train.
4. `evaluate.py` compares raw, denoised, synthetic, identity, and Gaussian-blur baselines on untouched test proteins.
5. `five_site_optimizer.py` converts deposited side-chain geometry into differentiable chi-angle kinematics, optimizes K conformers plus occupancy logits, and saves every start.
6. `audit_five_site_endpoints.py` reconstructs endpoints and measures conventional symmetry-aware RMSD, occupancies, rotamers, direct contacts, and crystal-symmetry contacts.
7. `five_site_tmol_audit.py` reparses each endpoint so hydrogens are rebuilt, scores it against deposited and random controls, and checkpoints after every site.
8. `summarize_endpoint_audit.py` merges geometry and tmol evidence into strict joint success.

### Frozen best optimizer

Unless a new experiment explicitly changes one factor, use:

```text
target                     denoised
frame                      crystal
K                          4
starts                     50
density stage              500 steps, Adam, lr 1.0
physics stage              Adam reset, 200 steps, lr 0.1
lambda_vdw                 1.0
lambda_rot                 0.5
lambda_clash               5.0
VDW soft threshold         3.0 Å
symmetry soft threshold    2.5 Å
```

Soft physics belongs in the downstream optimizer. It does not alter the trained U-Net.

## 5. Complete active-package file catalog

### `density_denoiser/`

| File | Responsibility and usage |
|---|---|
| `__init__.py` | Package marker and short project description. |
| `README.md` | Density-pair conventions and the original preprocessing/training quick start. Keep consistent with the root README and this file. |
| `data_pipeline.py` | CLI for `acquire`, `prepare`, `all`, and `manifest`. Owns PDB discovery, RCSB structure-factor acquisition, CIF→MTZ conversion, altloc discovery, omit maps, local frames, patch extraction, synthetic rendering, masks, normalization, and atomic status files. |
| `dataset.py` | Manifest loading, leakage-safe protein train/validation split, translation/noise/cube-rotation augmentation, and `DensityPairDataset`. Residue-frame training disables arbitrary cube rotation by default. |
| `model.py` | `ConvBlock`, the shape-correct `DensityDenoiser` 3D U-Net, residual wrapper `ResidualDensityDenoiser`, and spatial-gradient helper. The residual model predicts a correction added to the input. |
| `train.py` | Baseline U-Net trainer. Uses density L2 plus spatial-gradient loss, protein-level validation, atomic last/best checkpoints, patience/min-delta early stopping, and resume support. |
| `evaluate.py` | Held-out reconstruction evaluator and visualizer. Reports L2/Pearson/local correlation and identity/Gaussian-blur baselines. |
| `audit.py` | Audits prepared patch datasets for leakage, shape/finite-value problems, normalization, and input-target statistics. Run before expensive training. |
| `benchmark.py` | GPU throughput/VRAM benchmark across batch sizes and worker counts. Use it to size a run; it does not train a model. |
| `landscape.py` | Candidate-density rendering, radial masks, candidate energies, `LandscapeDataset`, and landscape-distillation/ranking loss used by U-Net 2.0. |
| `prepare_landscape_cache.py` | Builds restartable per-protein candidate-landscape shards from the frozen split, then compiles train/validation arrays. |
| `train_unet2.py` | Fine-tunes from the original U-Net with density reconstruction plus landscape-ranking/distillation objectives and a density-regression guardrail. Saves accepted and rejected epoch diagnostics. |
| `snapshot_unet2_checkpoints.py` | Atomically snapshots live U-Net 2.0 checkpoints while a trainer is running; protects intermediate states from later replacement. |
| `residue_geometry.py` | Production chi topology for 17 non-PRO chi-bearing residue types, residue-specific canonical rotamer centers, chemically equivalent atom permutations, and symmetry-aware conventional RMSD. This is the shared geometry authority. |
| `select_heldout_sites.py` | Chooses the original five held-out MET/ARG/ASP sites under occupancy, separation, and kinematic-representability gates. |
| `select_expanded_heldout_sites.py` | Chooses the untouched 15-protein/15-residue-type panel without using recovery or density performance. Uses symmetry-aware representability. |
| `validate_expanded_residue_support.py` | Validates identity reconstruction, deposited-B kinematic reconstruction, finite gradients, and responsiveness of every chi for the expanded panel. Writes each site immediately. |
| `five_site_optimizer.py` | Main downstream optimizer and calibration engine. Supports raw/synthetic/denoised targets, K conformers, occupancies, coarse-to-fine schedules, density-only/soft-physics/two-stage modes, fixed/released occupancy tests, seeded initializations, and sequential residual experiments. Despite the historical name, shared residue topology now supports the expanded panel. |
| `audit_five_site_endpoints.py` | Main geometry/physical audit. Reconstructs every active conformer, applies atom-symmetry-aware conventional RMSD, classifies rotamers, checks direct and crystallographic-symmetry clashes, writes visualization PDBs, and creates `tmol_inputs.json`. |
| `summarize_endpoint_audit.py` | Joins geometry and tmol tables and writes conformer-, ensemble-, site-, and global strict metrics. |
| `run_heldout_five_site_shards.sh` | Historical density-only five-site launcher. Retain for regression comparisons; not the current best pipeline. |
| `run_heldout_five_site_soft_physics_shards.sh` | Historical one-stage soft-physics launcher. Demonstrated physical improvements but harmed density-basin navigation. |
| `run_heldout_five_site_two_stage_shards.sh` | Frozen original-five launcher: density stage followed by low-LR physics refinement. |
| `run_expanded_heldout_two_stage_shards.sh` | Current prospective 15-site controller. Calibrates, launches 15 independent 50-start shards, audits geometry, runs tmol, summarizes strict success, and refuses overwrite. |

## 6. Complete root source-file catalog

### Current pod-side utilities

| File | Responsibility and usage |
|---|---|
| `five_site_tmol_audit.py` | Restartable CUDA/tmol scorer used by the current pod pipeline. Use the `.venv-tmol` interpreter. |
| `inject_interactive_landscape_data.py` | Injects downsampled five-site surface arrays into a specific local interactive HTML visualization. Its absolute visualization path is session-specific; update deliberately before reuse. |
| `test_density_denoiser.py` | U-Net shape, omit subtraction, frame, rigid-rotation invariance, landscape-rendering, and physics-center tests. |
| `test_probe4_core.py` | Differentiability and geometry tests for the original Probe 4 kinematic core. |
| `test_residue_geometry.py` | Coverage, residue-specific rotamers, and equivalent-atom RMSD tests for production residue support. |

### Probe 2 and shared learned-energy core

| File | Responsibility and usage |
|---|---|
| `probe2.py` | Early Cartesian multi-start recovery of 2O1K A:ARG129 against synthetic amplitudes. |
| `probe2_sfcalculator.py` | Small CPU control using the actual SFcalculator stack. |
| `probe2_modal.py` | Modal GPU tmol/SFcalculator smoke, sweep, and torsion-sweep launcher. |
| `probe4_core.py` | Reusable learned scalar energy, residue/chi features, Rodrigues rotations, torsion kinematics, first-order refinement, dihedrals, R-factor normalization, and gradient assertions. |

### Probe 4 learned-energy experiments

| File | Responsibility and usage |
|---|---|
| `probe4_modal.py` | Original detached Modal learned-energy experiment with resumable training/evaluation stages. |
| `probe4b_oracle.py` | Scores deposited A and representable B under the three Probe 4b localized objectives before training. |
| `probe4b_endpoint_audit.py` | Reconstructs Probe 4b endpoints and measures chi/rotamer/direct/symmetry-clash geometry. |
| `probe4b_render_endpoints.py` | Produces density-backed 2D overlays of Probe 4b deposited and learned endpoints. |
| `probe4b_tmol_modal.py` | Detached Modal tmol scoring for all Probe 4b endpoints. |
| `probe4c_endpoint_audit.py` | Configures the validated Probe 4b audit machinery for Probe 4c result paths. |
| `probe4c_audit_modal.py` | Detached Modal geometry audit for Probe 4c.1/4c.2 endpoints. |
| `probe4c_tmol_modal.py` | Corrected, checkpointed Modal tmol audit for Probe 4c endpoints. |
| `probe4c_compile_results.py` | Merges Probe 4c geometry/tmol audits into the experiment result trees. |
| `probe4c12_compile_results.py` | Equivalent compiler for the corrected Probe 4c.1/4c.2 outputs. |

### Direct and multi-conformer controls

| File | Responsibility and usage |
|---|---|
| `direct_optimization_modal.py` | No-network direct chi optimization against the complex structure-factor target; establishes whether a target landscape is navigable. |
| `multi_conformer_modal.py` | Synthetic K=4 differentiable multi-conformer fitting on five 2O1K sites and occupancy-ratio controls. |
| `experimental_multi_conformer_modal.py` | Experimental 2mFo-DFc/omit/difference/kick-map K=4 fitting with differentiable cross-term and Gaussian self-overlap. |
| `five_site_tmol_modal.py` | Legacy Modal version of held-out five-site tmol scoring. The pod-side `five_site_tmol_audit.py` is preferred when the H100 pod is available. |

Modal scripts embed image, volume, and app configuration. Run them with `uvx modal run --detach FILE.py` only when the experiment explicitly targets Modal. Do not substitute Modal for the active pod pipeline without documenting the environment change.

## 7. Data, reports, and result-tree catalog

### Reference inputs

| File | Meaning |
|---|---|
| `data/2O1K.pdb` | PDB-format deposited 2O1K model used throughout early probes. |
| `data/2O1K.cif` | mmCIF structure model. |
| `data/2O1K-sf.cif` | Deposited structure factors before conversion. |
| `data/2O1K.mtz` | Converted reflection data. |
| `data/2O1K_A108_130_altlocs.pdb` | Focused segment containing altloc records. |
| `data/2O1K_A108_130_complete.pdb` | Complete local chain segment for geometry/energy scoring. |
| `data/2O1K_A108_136_complete.pdb` | Extended complete segment used by neighborhood/symmetry audits. |

### Narrative reports and fixed metadata

| File | Meaning |
|---|---|
| `EBT_RESEARCH_CONTEXT.md` | Original complete concept document covering SampleWorks, EBTs, Fobs, and the de-risking plan. |
| `PROBE4.md` | Short original Probe 4 run/resume notes. |
| `PROBE4_COMBINED_REPORT.md` | Reciprocal learned-energy run versus supervised-chi control. |
| `PROBE4_FULL_EXPERIMENT_REPORT.md` | Combined Probe 4 and 4b scientific report. |
| `PROBE4C1_4C2_REPORT.md` | Kinematic-target correction, soft-physics calibration, and endpoint audit. |
| `ARG129_COARSE_TO_FINE_REPORT.md` | A_ARG129 coarse-to-fine, decay, and Adam-reset findings. |
| `original_crystal_unet_split.json` | Frozen protein-level train/validation assignment for the first crystal-frame U-Net. |
| `original_crystal_unet_training_log.csv` | Per-epoch training/validation curve for that model. |
| `heldout_five_site_two_stage_stage1_regression.json` | Check that two-stage stage 1 reproduces the density-only optimizer. |

### Generated result directories

These are scientific records, not importable code. Preserve their `run_config.json`, `stage_manifest.json`, raw tables, and controls together.

| Directory | Experiment lineage |
|---|---|
| `probe4_results/` | Empty/local placeholder for the earliest Probe 4 run. |
| `probe4_results_download/` | Downloaded reciprocal learned-energy artifacts. |
| `probe4_supervised_results/` | Supervised chi-target control artifacts. |
| `probe4b_results/` | Three localized-loss experiments, oracle checks, training curves, landscapes, endpoint geometry/tmol audit, and visualization files. |
| `probe4c_results/` | First complex-target and soft-physics variants plus audit outputs. |
| `probe4c1_results/` | Corrected kinematic complex-target experiment. |
| `probe4c2_results/` | Corrected soft-physics experiments for synthetic, localized-SF, and real-space targets. |
| `probe4c12_results/` | Combined corrected Probe 4c.1/4c.2 endpoint audit. |
| `realspace_kinematic_1000steps/` | Long direct real-space kinematic control across five 2O1K sites. |
| `arg129_coarse_to_fine_results/` | A/B ARG129 coarse-to-fine schedules, trajectories, decay/reset tests. |
| `five_site_coarse_to_fine_decay_reset/` | Five-site synthetic coarse-to-fine decay/reset production outputs. |
| `five_site_landscape_results/` | Paired 3D real-space vs structure-factor landscape arrays and figures. |
| `bmet112_experimental_sampling_50starts/` | Early B_MET112 optimization against an experimental map. |
| `multi_conformer_multi_protein/` | Synthetic K=4 five-site and occupancy-ratio validation. |
| `experimental_multi_conformer_2o1k/` | Local calibration/config manifest for experimental K=4 fitting. |
| `heldout_five_site_physical_audit/` | Geometry/strict artifacts for the density-only held-out run. |
| `heldout_five_site_soft_physics_audit/` | One-stage soft-physics geometry/tmol/strict audit. |
| `heldout_five_site_two_stage_audit/` | Best original-five two-stage geometry/tmol/strict audit. |
| `results/` | Curated cross-experiment deliverables: Probe 2 controls, experimental map-variant comparisons, U-Net 2.0 initialization tests, and expanded-panel selection report. |

Common file meanings:

- `run_config.json`: exact inputs, hyperparameters, checkpoint, and output path.
- `stage_manifest.json`: resumable state and completed stages.
- `*_starts.csv` / `endpoints_*.csv`: one row per optimization start or recovered conformer.
- `aggregate_summary.csv` / `comparison_table.csv`: derived summaries; never discard raw rows.
- `trajectories*.npz`: dense optimization histories for plotting/diagnosis.
- `*_targets.npz`: frozen input/target patches used by an optimizer.
- `*.pdb`, `*.pml`, `*.cxc`: structural visualization deliverables.
- `tmol_inputs.json`: endpoint coordinates and controls prepared for tmol.
- `tmol_energies.csv`: raw endpoint and deposited/random-control scores.
- `strict_summary.json`: final joint-metric source of truth.

## 8. Routine commands

Run these from the local repository unless a block explicitly says it belongs in the remote shell.

### Check and enter the pod

```bash
actl pod list
actl pod status qfit-unet
actl pod exec qfit-unet
```

`actl pod exec` derives the project from the current folder unless the explicit `qfit-unet` positional name is provided. Use the explicit name when invoking it from another folder.

### Start or restore sync

```bash
cd /Users/utkarsh/qfitonsteroids
actl pod sync qfit-unet
```

This command stays in the foreground. Closing it stops local sync/SSH forwarding but does not stop the pod or properly detached remote jobs.

### Verify a changed file reached the pod

```bash
LOCAL=$(shasum -a 256 density_denoiser/five_site_optimizer.py | awk '{print $1}')
REMOTE=$(actl pod exec qfit-unet --no-forwarding -- bash -lc \
  "sha256sum /home/dev/workspace/density_denoiser/five_site_optimizer.py | awk '{print \\\$1}'")
printf 'local=%s\nremote=%s\n' "$LOCAL" "$REMOTE"
```

### Run tests

```bash
actl pod exec qfit-unet --no-forwarding -- bash -lc \
  'cd /home/dev/workspace && /home/dev/qfit_unet_data/.venv/bin/python -m pytest -q'
```

Run focused tests while iterating, then the full suite before production:

```bash
/home/dev/qfit_unet_data/.venv/bin/python -m pytest -q \
  test_density_denoiser.py test_probe4_core.py test_residue_geometry.py
```

### Audit prepared data

In the pod shell:

```bash
cd /home/dev/workspace
PY=/home/dev/qfit_unet_data/.venv/bin/python
$PY -m density_denoiser.data_pipeline manifest \
  --data-root /home/dev/qfit_unet_data --split both --frame crystal
$PY -m density_denoiser.audit \
  --data-root /home/dev/qfit_unet_data --frame crystal
```

Always inspect command help before assuming a flag:

```bash
$PY -m density_denoiser.data_pipeline --help
$PY -m density_denoiser.train --help
$PY -m density_denoiser.five_site_optimizer --help
```

### Launch a detached controller

In the pod shell:

```bash
cd /home/dev/workspace
OUT=/home/dev/qfit_unet_data/density_denoiser/NAME_OF_NEW_RUN
mkdir -p "$OUT/logs"
nohup env OUTPUT="$OUT" bash density_denoiser/run_expanded_heldout_two_stage_shards.sh \
  > "$OUT/logs/controller.log" 2>&1 < /dev/null &
echo $! > "$OUT/controller.pid"
```

Immediately verify:

```bash
ps -p "$(cat "$OUT/controller.pid")" -o pid,etime,%cpu,%mem,stat,cmd
cat "$OUT/status.txt"
tail -n 30 "$OUT/logs/controller.log"
nvidia-smi
```

### Monitor without relaunching

```bash
OUT=/home/dev/qfit_unet_data/density_denoiser/NAME_OF_RUN
cat "$OUT/status.txt"
find "$OUT/pids" -maxdepth 1 -name '*.status' -print -exec cat {} \;
for f in "$OUT"/logs/*.log; do printf '\n== %s ==\n' "$f"; tail -n 3 "$f"; done
```

Never infer completion solely from GPU utilization. Confirm the controller is terminal, status is `complete`, all expected shards exist, and `strict_summary.json` parses.

### Resume tmol only

If geometry completed and tmol was interrupted:

```bash
cd /home/dev/workspace
OUT=/home/dev/qfit_unet_data/density_denoiser/NAME_OF_RUN
/home/dev/qfit_unet_data/.venv-tmol/bin/python five_site_tmol_audit.py \
  --input-root "$OUT/audit" --output "$OUT/audit" --resume
/home/dev/qfit_unet_data/.venv/bin/python \
  -m density_denoiser.summarize_endpoint_audit --audit-root "$OUT/audit"
```

### Stop the pod safely

```bash
actl pod down qfit-unet
```

This frees the GPU and keeps the home PVC. It terminates jobs still running in the pod, so first check controllers and obtain user approval if work is active. Never add `--delete-data` unless the user explicitly requests permanent PVC destruction.

## 9. Change protocol

For every scientific code change:

1. State the hypothesis and the single factor being changed.
2. Inspect existing user changes and avoid unrelated cleanup.
3. Edit locally.
4. Add or update a focused deterministic test.
5. Start/confirm `actl pod sync` and compare the important file checksum.
6. Run focused tests on the pod, then the full relevant suite.
7. Run deposited A/B and synthetic/oracle calibration before opening expensive recovery results.
8. Use a new output directory containing a complete `run_config.json`.
9. Detach, checkpoint, and verify startup.
10. At completion, audit every endpoint and report raw counts as `successes / starts`, per site and in aggregate.
11. Distinguish measured facts, derived metrics, and scientific interpretation.

If a new residue is added, update `CHI_SPECS`, allowed centers, atom-equivalence rules, tests, selection validation, deposited A/B calibration, and the endpoint audit together. PRO is intentionally unsupported by the current open-chain kinematics because ring closure requires a different parameterization.

## 10. Git policy

- The repository uses branch `main`.
- Commit source, documentation, small reference inputs, frozen selections, run configurations, concise tables, and final reports.
- Do not commit virtual environments, caches, PID/log noise, secrets, or multi-gigabyte raw datasets/checkpoints.
- Large remote artifacts remain on the PVC. Pull only the curated evidence needed for a report.
- Before committing, inspect `git status --short`, `git diff --check`, and the exact staged diff.
- Do not rewrite history or discard uncommitted work without explicit authorization.
- A result commit must include enough metadata to identify the checkpoint, selection, frame, target, optimizer settings, seed, and audit thresholds.

## 11. Current prospective run

As of 2026-07-21, the frozen expanded-panel run is:

```text
/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_two_stage_prospective_v1
```

It uses the original crystal-frame checkpoint:

```text
/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt
```

and the frozen selection:

```text
/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json
```

The 15 sites are ASN 1ZV8:E:1, CYS 6Y4G:B:260, GLN 7UO8:A:53, GLU 3GMI:A:5, HIS 2V05:A:168, ILE 8FBE:B:92, LEU 7T7A:A:396, LYS 3NY7:B:19, PHE 5KWB:A:591, SER 3K8W:A:337, THR 4MKM:A:77, TRP 5DBA:A:325, TYR 2VFP:A:417, VAL 8DJ2:A:893, and the MET bridge 5Z8H:A:730.

Do not change, restart, or overwrite that run while it is active. Monitor it and let the controller proceed through all audit stages.
