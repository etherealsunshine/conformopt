# Exact Numbers for the Talk — Audited 2026-07-21

This version separates the held-out 20-site production result from earlier 2O1K controls. Every numerical claim below is either directly present in a saved run artifact or derived arithmetically from those artifacts.

## Bottom line

- The held-out experiment contains **20 proteins, 20 altloc sites, 17 unique residue types, and 1,000 random-start ensembles**.
- **227/1,000 starts (22.7%)** pass the recorded strict joint criterion.
- **9/20 sites (45%)**, not 10/20, have at least one strict success.
- **18/20 sites (90%)** have at least one start that recovers both deposited conformers. This is the defensible site-level discovery number; “87%” was not exact.
- The original five contribute **55/250** strict successes; the prospective 15 contribute **172/750**.
- The production 20-site runs did **not** use coarse-to-fine optimization. They used 500 full-resolution density-only steps followed by 200 low-learning-rate soft-physics steps.
- Coarse-to-fine was validated in separate 2O1K synthetic controls and used in later 3A1C initialization/sequential tests, but it was not part of the reported 227/1,000 production result.

## Evidence used

Primary evidence:

- Original-five audit: `artifacts/heldout_five_site_two_stage_audit/strict_summary.json` and its conformer/ensemble CSV files.
- Prospective-15 audit on the Astera PVC: `/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_two_stage_prospective_v1/audit/strict_summary.json` and its conformer/ensemble CSV files.
- Saved per-site production configurations in the corresponding `shards/*/run_config.json` trees.
- Frozen launchers: `density_denoiser/run_heldout_five_site_two_stage_shards.sh` and `density_denoiser/run_expanded_heldout_two_stage_shards.sh`.
- Optimizer implementation: the ordinary production branch of `density_denoiser/five_site_optimizer.py`.
- Original denoiser checkpoint, split, and training log: pod checkpoint `density_denoiser/model/denoiser_best.pt` plus `artifacts/metadata/original_crystal_unet_split.json` and `artifacts/metadata/original_crystal_unet_training_log.csv`.
- Separate synthetic controls: `artifacts/arg129_coarse_to_fine_results`, `artifacts/five_site_coarse_to_fine_decay_reset`, and `artifacts/multi_conformer_multi_protein/2O1K`.

## Exact production methodology

### Data and denoiser

- The model used for all 20 held-out sites is the original **crystal-frame residual 3D U-Net**, checkpoint `denoiser_best.pt`.
- Input: a normalized **experimental sidechain-omit mFo-DFc patch**.
- Training target: a normalized **synthetic sidechain-density patch** rendered from the deposited model.
- Patch shape: `1 × 32 × 32 × 32`; grid spacing: **0.5 Å**.
- Training manifest: **403,875 patches** from a protein-level split of **1,607 training proteins and 178 validation proteins** (1,785 proteins represented in the split).
- Untouched test manifest: **17,879 patches from 99 test PDB files**. The 20 reported proteins were selected from this test subset.
- Architecture: residual wrapper around a four-level 3D U-Net, base width **32**, with **10,739,873 trainable/state-dict parameters**. The previous “~2M” statement was wrong.
- Training objective:

  `density MSE + 0.1 × spatial-gradient MSE`

- Saved training settings: batch size 32, AdamW, initial lr 0.001, weight decay 0.0001, cosine schedule, 100 requested/completed epochs, seed 20260717.
- Best saved checkpoint: **epoch 95**, validation L2 **0.0586105**, validation Pearson correlation **0.9696364**. The final epoch was 99; the best checkpoint, not the last checkpoint, was used.
- The checkpoint can and should be loaded with `weights_only=True` while allowlisting the checkpoint's known `pathlib.PosixPath` metadata type.

### Differentiable ensemble model

- Sidechains are parameterized by residue-specific chi torsions and reconstructed by forward kinematics.
- Each start contains **K=4 slots**.
- Slot occupancies are a softmax over four logits and therefore sum to one.
- Moving-atom density is a Gaussian atom model with per-atom variance

  `sigma^2 = max(B / (8*pi^2), 0.04)`.

- Atomic weights use atomic number and deposited atom occupancy. The rendered and target density vectors are normalized before MSE is computed.
- Optimization is restricted to the spherical **4 Å radial mask** inside the 32-cube patch.
- There is **no separate quadratic slot-overlap penalty** in the production loss. The earlier document incorrectly listed one.

### Production optimization schedule used for all 20 sites

For every original-five site and every prospective-15 site:

1. Initialize four chi vectors from random normal values and all occupancy logits to zero.
2. Run **500 full-resolution density-only Adam steps at lr=1.0**:

   `L_density = mean((rendered_density - denoised_target)^2)`

3. Reset Adam.
4. Run **200 full-resolution soft-physics refinement steps at lr=0.1**:

   `L = L_density + 1.0 L_vdw + 0.5 L_rotamer + 5.0 L_symmetry`

5. Wrap chi angles after every step.
6. Repeat for **50 independently seeded starts per site**, with global seed 41.

The soft terms are:

- Direct VDW: squared hinge penalty below 3.0 Å, excluding the moving CB–CA covalent pair.
- Rotamer prior: minimum `1 - cos(angle - center)` over residue- and chi-specific allowed centers.
- Symmetry VDW: squared hinge penalty below 2.5 Å against generated crystallographic symmetry mates.

All 20 saved run configurations agree on K, steps, learning rates, physics weights, seed, target, frame, patch size, spacing, and occupancy thresholds. The prospective 15-site controller first ran the deposited-A/kinematic-B physics calibration and then launched one shard per site.

### Direct answer: did all 20 use coarse-to-fine?

**No.** The 20-site production result used full-resolution density for all 500 stage-1 steps.

The code does construct 4 Å-, 2 Å-, and unblurred target vectors, but the ordinary production loop never iterates over that blur schedule. The explicit schedule

`4 Å / lr 1.0 / 100 steps -> 2 Å / lr 0.1 / 100 -> full / lr 0.01 / 100`

appears in the specialized 3A1C initialization and sequential-testing branches. It was also used in earlier 2O1K synthetic controls. Therefore, do not describe the 227/1,000 result as coarse-to-fine unless the 20-site panel is rerun with a newly implemented production schedule.

## Strict success definition

An active slot is one with occupancy **>0.05**. A deposited state is “found” only if a matching slot has occupancy **>0.10**.

An ensemble is a recorded strict joint success when all of the following hold:

1. Deposited A and deposited B are each found with **conventional, symmetry-aware RMSD <1.0 Å**. Conventional RMSD is `sqrt(mean_atoms(sum_xyz(error^2)))`.
2. Occupancy assigned to A and occupancy assigned to B are each within **±0.20** of the deposited values. Occupancy is summed across every slot assigned to that state.
3. Every active slot has no direct or crystallographic-symmetry interatomic distance below **2.0 Å**.
4. Every chi of every active slot is within **30°** of one of the implemented residue-specific canonical centers.
5. Every active slot has finite tmol energy no more than **10 units above the lower-energy deposited A/B reference** for that site.

Important caveat: the recorded strict metric does **not** require exactly two active conformers. K is fixed at four, and strict ensembles may contain two, three, or four active slots. Exact model-selection results are reported separately below.

## Exact per-site results

| Protein | Site | Residue | n-chi | Both found | Both + occupancy | Strict |
|---|---|---:|---:|---:|---:|---:|
| 3A1C | B:ARG447 | ARG | 4 | 0/50 | 0/50 | **0/50** |
| 4C16 | A:MET258 | MET | 3 | 13/50 | 9/50 | **5/50** |
| 6H59 | B:ARG144 | ARG | 4 | 8/50 | 7/50 | **6/50** |
| 7F72 | A:MET103 | MET | 3 | 38/50 | 37/50 | **33/50** |
| 8Q6Q | B:ASP81 | ASP | 2 | 12/50 | 11/50 | **11/50** |
| 1ZV8 | E:ASN1 | ASN | 2 | 6/50 | 0/50 | **0/50** |
| 2V05 | A:HIS168 | HIS | 2 | 0/50 | 0/50 | **0/50** |
| 2VFP | A:TYR417 | TYR | 2 | 9/50 | 0/50 | **0/50** |
| 3GMI | A:GLU5 | GLU | 3 | 38/50 | 29/50 | **0/50** |
| 3K8W | A:SER337 | SER | 1 | 50/50 | 50/50 | **0/50** |
| 3NY7 | B:LYS19 | LYS | 4 | 4/50 | 2/50 | **0/50** |
| 4MKM | A:THR77 | THR | 1 | 50/50 | 50/50 | **50/50** |
| 5DBA | A:TRP325 | TRP | 2 | 38/50 | 34/50 | **0/50** |
| 5KWB | A:PHE591 | PHE | 2 | 36/50 | 35/50 | **34/50** |
| 5Z8H | A:MET730 | MET | 3 | 11/50 | 11/50 | **11/50** |
| 6Y4G | B:CYS260 | CYS | 1 | 50/50 | 50/50 | **50/50** |
| 7T7A | A:LEU396 | LEU | 2 | 28/50 | 28/50 | **27/50** |
| 7UO8 | A:GLN53 | GLN | 3 | 12/50 | 0/50 | **0/50** |
| 8DJ2 | A:VAL893 | VAL | 1 | 50/50 | 0/50 | **0/50** |
| 8FBE | B:ILE92 | ILE | 2 | 44/50 | 16/50 | **0/50** |
| **Original 5** |  |  |  | **71/250** | **64/250** | **55/250** |
| **Prospective 15** |  |  |  | **426/750** | **305/750** | **172/750** |
| **Combined 20** |  |  |  | **497/1,000** | **369/1,000** | **227/1,000** |

The 17 unique residue types are ARG, MET, ASP, ASN, HIS, TYR, GLU, SER, LYS, TRP, PHE, CYS, LEU, GLN, VAL, ILE, and THR.

## Defensible aggregate statements

### Start-level

- Both conformers found: **497/1,000 (49.7%)**.
- Both found and occupancy within ±0.20: **369/1,000 (36.9%)**.
- Full recorded strict success: **227/1,000 (22.7%)**.

### Site-level

- At least one both-found start: **18/20 sites (90%)**. Only 3A1C ARG447 and 2V05 HIS168 have 0/50 both-found.
- At least one strict start: **9/20 sites (45%)**.
- More than 25/50 strict starts: **5/20 sites (25%)** — THR77, CYS260, PHE591, LEU396, and MET103.
- Perfect 50/50 strict starts: **2/20 sites (10%)** — THR77 and CYS260.
- Zero strict starts: **11/20 sites (55%)**, not 10/20.

Do not mix the 49.7% start-level discovery fraction with the 90% site-level discovery fraction. They answer different questions.

## Why strict success fails

Counts below are sequential. “Occupancy failures” means both states were found but the ±0.20 occupancy gate failed. “Physical failures” means both states and occupancy passed, but at least one active slot failed rotamer, clash, or tmol validation.

| Site | Not both found | Occupancy failures after both found | Physical failures after occupancy | Strict |
|---|---:|---:|---:|---:|
| 3A1C ARG447 | 50 | 0 | 0 | 0 |
| 4C16 MET258 | 37 | 4 | 4 | 5 |
| 6H59 ARG144 | 42 | 1 | 1 | 6 |
| 7F72 MET103 | 12 | 1 | 4 | 33 |
| 8Q6Q ASP81 | 38 | 1 | 0 | 11 |
| 1ZV8 ASN1 | 44 | 6 | 0 | 0 |
| 2V05 HIS168 | 50 | 0 | 0 | 0 |
| 2VFP TYR417 | 41 | 9 | 0 | 0 |
| 3GMI GLU5 | 12 | 9 | 29 | 0 |
| 3K8W SER337 | 0 | 0 | 50 | 0 |
| 3NY7 LYS19 | 46 | 2 | 2 | 0 |
| 4MKM THR77 | 0 | 0 | 0 | 50 |
| 5DBA TRP325 | 12 | 4 | 34 | 0 |
| 5KWB PHE591 | 14 | 1 | 1 | 34 |
| 5Z8H MET730 | 39 | 0 | 0 | 11 |
| 6Y4G CYS260 | 0 | 0 | 0 | 50 |
| 7T7A LEU396 | 22 | 0 | 1 | 27 |
| 7UO8 GLN53 | 38 | 12 | 0 | 0 |
| 8DJ2 VAL893 | 0 | 50 | 0 | 0 |
| 8FBE ILE92 | 6 | 28 | 16 | 0 |
| **Combined** | **503** | **128** | **142** | **227** |

For the zero-strict prospective sites, the audited physical causes among ensembles that reached the physical gate were:

- GLU5: 29 physical failures; all 29 clash, with one also noncanonical and tmol-invalid.
- SER337: all 50 fail the implemented canonical-angle check and otherwise pass clash/tmol checks.
- LYS19: both occupancy-valid ensembles fail tmol only.
- TRP325: all 34 are noncanonical and clash; 25 are also tmol-invalid.
- ILE92: all 16 clash and are tmol-invalid.

These labels overlap. They are not additive independent causes. For example, a TRP ensemble may simultaneously be noncanonical, clashing, and high energy.

The old statement “only 3/20 are genuine recovery failures” was too strong. Exactly two sites have zero both-found starts, but several other sites are strongly recovery-limited. Likewise, calling the SER and VAL fixes “trivial” and projecting a 32% strict rate was a counterfactual, not a measured result; those projections should not be presented as data.

## Physical audit

| Panel | Active conformers | Canonical | Clash-free | tmol-valid | Strictly physical | Median tmol delta |
|---|---:|---:|---:|---:|---:|---:|
| Original 5 | 709 | 659 (93.0%) | 672 (94.8%) | 616 (86.9%) | 562 (79.3%) | 0.819 |
| Prospective 15 | 2,289 | 1,936 (84.6%) | 1,851 (80.9%) | 1,911 (83.5%) | 1,467 (64.1%) | 1.208 |
| Combined 20 | 2,998 | 2,595 (86.6%) | 2,523 (84.2%) | 2,527 (84.3%) | 2,029 (67.7%) | 1.022 |

“Strictly physical” here is a conformer-level intersection of canonical, clash-free, and tmol-valid. It is not the ensemble strict success rate.

For the original five, 55 of the 64 starts that already passed recovery plus occupancy also passed all physical checks: **55/64 = 85.9%**. That conditional pass-through should not be described as the overall physical validity of all recovered or all active endpoints.

## Exact model-selection caveat

The recorded strict criterion permits extra active slots. Among the 227 strict ensembles:

| Active slots | Strict ensembles |
|---:|---:|
| Exactly 2 | 127 |
| Exactly 3 | 81 |
| Exactly 4 | 19 |

Therefore:

- Recorded strict, any active count: **227/1,000 (22.7%)**.
- Strict plus exact K=2 model selection: **127/1,000 (12.7%)**.

Sensitivity to occupancy tolerance:

| Occupancy tolerance | Strict, any active count | Strict + exactly 2 active |
|---:|---:|---:|
| ±0.20 | 227/1,000 | 127/1,000 |
| ±0.10 | 139/1,000 | 68/1,000 |
| ±0.05 | 53/1,000 | 27/1,000 |

This distinction is essential if the talk claims recovery of the **correct number** of conformers.

## What the original-five ablation really shows

All three rows below use the held-out original-five panel and the denoised target; they are not raw experimental-map versus synthetic-map rows.

| Downstream optimizer | Strict success |
|---|---:|
| Density-only | 9/250 (3.6%) |
| Soft physics throughout | 25/250 (10.0%) |
| Density first, then soft physics | **55/250 (22.0%)** |

The stage-1 regression artifact checks all 250 starts and reports zero differences in chi, occupancy, and density loss between the density-only baseline and stage 1 of the staged run. This supports the interpretation that the improvement comes from adding a separate low-lr physics polish without changing the exploration stage.

## Coarse-to-fine evidence — separate synthetic controls

The A_ARG129 2O1K synthetic control did use the explicit blur/lr/reset schedule:

| Control | RMSD-B <0.50 Å |
|---|---:|
| Full-resolution baseline, 200 steps at lr 1.0 | 5/50 |
| 4 Å -> 2 Å -> full, 100 steps each, lr 1.0 -> 0.1 -> 0.01, Adam reset | 43/50 |

A later five-site synthetic coarse-to-fine run produced **46/50** for A_ARG129. These are strong navigation controls, but they are not the schedule or outcome of the held-out 20-site production run.

## Synthetic multi-conformer control — do not call it strict physical success

The separate 2O1K synthetic K=4 experiment reports:

- Both deposited states found: **153/250 (61.2%)**.
- Both found plus occupancy criterion: **139/250 (55.6%)**.

Those are recovery/occupancy metrics, not the later tmol/rotamer/clash strict audit. Also, 2O1K was present in the denoiser training split, so this is an integration upper bound rather than held-out generalization. It should not be placed in the same “strict success” column as the held-out 55/250 or 227/1,000 results.

For A_MET112 specifically, the synthetic ensemble experiment reported 49/50 both-found and 46/50 recovery-plus-occupancy success. Its occupancy-ratio controls reported 45/50 at target 70:30 and 49/50 at target 30:70. The earlier “4.1 percentage points” summary is a synthetic-control statistic, not a held-out 20-site occupancy error.

## Runtime — measured, not estimated

For the prospective 15-site run on the H100 pod:

- Per-site shards began at approximately 17:16 UTC.
- The fastest site finished at 17:41, about 25 minutes later.
- The slowest site finished at 18:00, about 44 minutes later.
- Geometry, tmol, and summary completed at 18:01:43.
- Total controller wall time was about **46 minutes** because 15 site shards ran concurrently.

Thus “2–3 seconds per start” and “2 minutes per site” are not supported by this production run. A rough observed equivalent is approximately **30–53 seconds per start within a site shard**, although startup and site complexity differ and starts were not benchmarked in isolation.

## Presentation-safe headline slide

Use:

```text
20 untouched test proteins
20 altloc sites across 17 residue types
1,000 random-start ensembles

497/1,000 recover both deposited conformers
369/1,000 also match deposited occupancies within ±0.20
227/1,000 pass the recorded geometry + occupancy + physical audit

18/20 sites recover both conformers at least once
9/20 sites achieve at least one strict success
2/20 sites are strict in all 50 starts
```

If exact model selection matters, add:

```text
127/1,000 pass strict validation with exactly two active conformers
```

## Presentation-safe interpretation

The supported conclusion is:

> A crystal-frame residual U-Net followed by continuous K=4 torsion/occupancy optimization can recover both deposited states on 18 of 20 untouched test sites. Across all starts, 49.7% recover both states and 22.7% pass the recorded recovery, occupancy, rotamer, clash, and tmol audit. Performance is highly site-dependent, and exact two-conformer model selection reduces the aggregate to 12.7%.

Avoid claiming:

- that all 20 production sites used coarse-to-fine;
- 10/20 sites with strict success;
- 12 residue types;
- an exact 87% discovery rate;
- that 139/250 synthetic recovery/occupancy successes are strict physical successes;
- that every strict ensemble has exactly two active conformers;
- that 50 starts guarantee success from the pooled 22.7% rate;
- that SER/VAL fixes are trivial or would certainly raise strict success to 32%;
- a direct qFit advantage before a matched benchmark;
- 2–3 seconds per start for the production pipeline.

The pooled 22.7% is not a site-independent Bernoulli probability: 11 sites have 0/50 strict success, while two have 50/50. Therefore the old calculation claiming a 99.998% chance of at least one success from 50 starts is statistically invalid for a new site.

