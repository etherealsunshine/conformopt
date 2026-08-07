# Current Project Handoff

**Updated:** 2026-07-29
**Purpose:** This is the first document a new agent/chat should read after
`AGENTS.md`. It records the current scientific state, the exact result trees,
the commands used to operate the Astera pod, and the next experiments under
discussion.

## 0. Latest diagnostic: residual minor-conformer probe

### 2026-07-29 containing-mask sweep complete

The corrected containing-mask sweep is complete under frozen metric
`qfit-synth20-merge050-one-to-one-tmol044-v3`.

| Arm | Found | Strict | Raw minor / major misses |
|---|---:|---:|---:|
| Frozen control | 742 | 626 | 142 / 45 |
| Containing, uniform | 741 | 622 | 139 / 63 |
| Containing, variance weighted | 690 | 591 | 161 / 66 |

Neither arm improved the intended minor-conformer endpoint. The uniform mask
was aggregate-neutral but traded three fewer minor misses for eighteen more
major misses. Variance weighting was harmful.

The first compiled per-site comparison accidentally loaded the stale
unprotected merge audit for its control rows. Arm totals were correct, but the
nested control cascade, control audit-derived diagnostics, and all per-site
deltas were not. Corrected output:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_containing_mask_sweep_v1/
analysis/frozen_v3_comparison_v2_corrected
```

Full result, weight-concentration table, and bug scope:

```text
results/containing_mask_sweep_v1_report.md
```

### 2026-07-29 corrected containing-mask experiment

The production-mask read found exactly two deposited atoms outside the 4.0 A
sphere: 3A1C ARG447 NH2 in A/minor at radius 5.0215 A and B/major at
4.9397 A. Despite losing 33.61% and 31.45% of their native conformer density,
3A1C recovers its minor state in 49/50 starts. Across the 19 ranked sites,
minor outside-density fraction is not positively associated with minor
failure (Pearson -0.0777; -0.1244 against major-only misses).

The prior reachable-mask contradiction came from using only discrete
canonical-center tuples. Twenty-five deposited atoms are more than 1 A from
that union, but only 2/40 deposited conformers fail the production marginal
rotamer gate. The atom sets and padding were identical; broad continuous
gate windows, not mainly genuine off-rotamer geometry, explain the mismatch.

Corrected Arms F/G use a 1 A-padded union of canonical-center positions and
deposited A/B atoms. Both preflight at zero reachable and deposited atoms
outside, with A+B best at 20/20 sites. Arm F contains 133--4,982 voxels/site
and has median deposited major-collapse correlation loss 0.1168, versus
0.0988 for production. Arm G's variance weights use canonical states only.

The detached two-arm experiment is active:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_containing_mask_sweep_v1
```

Controller PID `62240`; Arm F runs first, followed automatically by Arm G and
their frozen-v3 geometry/tmol audits. The frozen control is reused.

Full diagnostic and live-run provenance:

```text
results/production_mask_coverage_diagnostic_v1.md
```

### 2026-07-29 density-mask sweep preflight

The requested normalization diagnostic did not justify replacing z-scored
MSE before the mask sweep. Median relative major-collapse separation at
deposited coordinates was 0.1976 under z-scored MSE, 0.2160 under native MSE,
and 0.1529 after a fitted global scale. Across the 129 actual major-only
endpoints the ordering reversed: 0.0731, 0.0605, and 0.0584. Native MSE is
therefore not a consistently larger lever.

The mask-containment preflight did stop the launch. The 3.0 A and 2.5 A
spheres exclude 1,154 and 1,424 enumerated reachable atom positions,
respectively, and exclude 20 and 38 deposited A/B atoms. The proposed 1.0 A
reachable-volume mask contains every enumerated canonical position but misses
25 deposited atoms. Even the production 4.0 A sphere excludes 599 enumerated
reachable positions and two deposited atoms at 3A1C. All arms retained A+B as
the best synthetic calibration at 20/20 sites, but any recovery delta would
still be confounded by unscored conformer escape. No optimization arm was
launched and the frozen metric was untouched.

Full report:

```text
results/density_mask_sweep_preflight_v1.md
```

Pod preflight root:

```text
/home/dev/qfit_unet_data/density_denoiser/density_mask_sweep_preflight_v1
```

### 2026-07-29 density-mask and torsion read

The production 4.0 A radial mask contains 2,176 voxels. In the frozen
synthetic 20-site panel, `fixed_density` contains zero atoms at every site:
it represents only shared/unlabeled atoms of the target residue's sidechain,
not the neighboring protein. Thus `Var(fixed)=0`, the fixed/sidechain cross
term is zero, and sidechain slots account for 100% of masked variance. The
proposed unchangeable-fixed-density dilution mechanism is absent.

The broad mask nevertheless compresses A/B discrimination. Replacing the
correct deposited A/B ensemble with the deposited major conformer alone
changes correlation by median 0.0988 over the full mask, versus 0.1829 over
voxels where `|rho_A-rho_B| > 0.1 max` and 0.1787 within 1 A of the deposited
A/B sidechain atoms. The 129 saved major-only endpoint ensembles have median
full-mask correlation 0.9635 with the correct target.

The per-chi read rejects a simple terminal-chi reachability explanation.
3GMI's largest deposited A/B change is chi1 (129.0 degrees) despite 8/9 probe
recovery; 7T7A changes both chi1/chi2 by 99.3/114.5 degrees and recovers 4/5.
The low sites require multiple large rotations: 1ZV8 changes chi1/chi2 by
170.2/173.0 degrees, and 7UO8 changes chi1/chi2/chi3 by
118.3/142.3/56.2 degrees. Across eligible starts, terminal-largest sites
recover 25/97 versus 13/32 for nonterminal-largest sites, but this is strongly
site-confounded.

No optimization or metric change was made. Full report:

```text
results/density_mask_chi_diagnostic_v1.md
```

The residual probe is complete for all 129 frozen-v3 starts that recovered
only the occupancy-major state. Frozen endpoints were subtracted from the
synthetic target and one fresh production-initialized slot was optimized
against the residual.

| Probe occupancy | Minor recovered | Rate |
|---|---:|---:|
| Fixed to deposited minor | 38 / 129 | 29.5% |
| Free sigmoid | 16 / 129 | 12.4% |

The production renderer z-score normalizes the full fixed-plus-weighted-slot
density after summation. The probe instead optimized against normalized
target minus normalized endpoint using a separately normalized slot
footprint. That is not an additive missing-conformer target. Therefore the
competition hypothesis is **neither supported nor refuted**, and 38/129 is a
lower bound from a mismatched probe rather than an estimate.

The actual slot occupancies are a softmax and always sum to one; only their
ratios are represented. A common logit shift is an exact no-op, so absolute
sidechain occupancy is not identifiable in this parameterization. The
reported matched A+B deficit is mass held by unmatched/sub-mask slots.

A replay of all 26 close-separation fixed-probe successes measured median
initial-to-final slot travel of 1.696 Å (IQR 1.299–2.418): zero traveled under
0.5 Å and only two under 1.0 Å. Close-site composition contributes to the
separation anti-correlation, but the successes are not merely trivial
threshold crossings. No production or metric change was made.

Full report:

```text
results/residual_minor_probe_v1_report.md
results/residual_probe_normalization_followup_v1.md
```

Authoritative combined pod root:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_remaining_residual_minor_probe_v1
```

## 0a. Previous model experiment: initialization sweep

The 20-site synthetic initialization sweep is complete under the unchanged
frozen metric `qfit-synth20-merge050-one-to-one-tmol044-v3`.

| Initialization | Found | Strict | Minor / major misses |
|---|---:|---:|---:|
| Frozen control | 742 | 626 | 142 / 45 |
| Canonical, deposited-A-free | 720 | 608 | 169 / 57 |
| Canonical + deposited-A anchor | 719 | 615 | 183 / 42 |
| Deposited-A cloud, 120 degrees | 739 | 632 | 149 / 54 |

No arm improved the primary minor-conformer endpoint. Canonical
stratification was harmful. The wider cloud gained six strict starts but lost
three recoveries and worsened both minor and major misses; its gain was
site-heterogeneous rather than evidence for the proposed mechanism.

Full report:

```text
results/initialization_sweep_v1_report.md
```

Authoritative pod root:

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_initialization_sweep_v1
```

## 1. Read this correctly

There are two different 20-site benchmarks. Do not mix their numbers:

1. **Denoised experimental omit-map targets** test the complete U-Net plus
   optimizer pipeline.
2. **Synthetic targets** are an optimizer oracle/control. They test whether the
   downstream search can recover the deposited ensemble when the target is
   clean and exactly representable.

The current worktree is intentionally dirty and includes a source/results
reorganization. Many historical root files have moved under `experiments/`,
`docs/`, and `scripts/`; Git may show their old paths as deleted and their new
paths as untracked. Do not clean, reset, restore, or mass-delete anything.

## 2. Current scientific pipeline

```text
PDB + deposited structure factors
              |
              v
sidechain-omit mFo-DFc patch
              |
              v
original crystal-frame residual 3D U-Net
              |
              v
denoised side-chain-density patch
              |
              v
K=4 residue-specific chi/occupancy optimization
  Stage 1: density-only
  Stage 2: reset Adam and add soft physics
              |
              v
conventional RMSD + occupancy + rotamer + clash + symmetry + tmol audit
```

The production model remains:

```text
/home/dev/qfit_unet_data/density_denoiser/model/denoiser_best.pt
```

It is the original **crystal-frame** U-Net. The canonical-frame model and U-Net
2.0 landscape-loss pilots were informative experiments but did not replace it.
The recorded baseline held-out reconstruction metrics include validation
L2 `0.0586105` and Pearson `0.9696364`. These are voxel reconstruction metrics,
not conformer-recovery metrics.

The U-Net input is a normalized experimental sidechain-omit `mFo-DFc` patch.
The target is a synthetic side-chain density patch. The simple omit procedure
removes the target sidechain from the model and recomputes map coefficients;
it does not add noise.

## 3. Current optimizer and strict metric

Common settings:

```text
representation       residue-specific chi torsions + forward kinematics
ensemble             K=4 softmax-occupancy slots
starts               50 per site
active occupancy     >= 0.05
Stage 2              Adam reset, 200 full-resolution steps, lr 0.1
lambda_vdw           1.0
lambda_rot           0.5
lambda_clash         5.0
VDW soft distance    3.0 A
symmetry soft cutoff 2.5 A
symmetry hard gate   2.0 A
```

Per-residue-class density schedule:

- **1-3 chi:** 500 full-resolution density-only Adam steps at `lr=1.0`.
- **4 chi baseline:** 100 steps at 4 A FWHM/lr 1.0, 100 at 2 A/lr 0.1,
  100 full-resolution/lr 0.01.
- **Latest 4-chi ablation:** 200+200+200 density steps at the same blur/lr
  sequence, followed by the unchanged 200-step physics stage at `lr=0.1`.

The 4 A and 2 A targets are Gaussian-smoothed copies of the same map, not maps
with added noise. The Gaussian kernel performs a local weighted average. FWHM
is the width of that Gaussian at half its maximum height. Blurring creates a
broader, smoother basin for long sidechains before full-resolution refinement.

Strict joint success requires:

1. Both deposited A and B conformers found with **conventional**,
   chemically symmetry-aware RMSD `<1.0 A`:
   `sqrt(mean_atoms(sum_xyz(delta**2)))`.
2. Recovered occupancies within `±0.20` of deposited occupancies.
3. Every active slot canonical under the shared residue/chi-specific rotamer
   centers and widths.
4. No direct or crystallographic-symmetry contact below `2.0 A`.
5. Every active slot has finite tmol energy no worse than its assigned
   deposited A or B control, scored in that control's frozen environment.

Recovery alone is not success. `all_active_strict_physical` is sometimes
reported independently; it is not a cascade count. For monotone tables,
condition it on recovery and occupancy. The final all-active `strict` count is
the intersection of recovery, occupancy, and all physical gates. Also report
the assigned-A/B-pair version, which selects the lowest-RMSD found A and B and
does not charge extra active slots as independent failure opportunities.

Current rule identifiers are versioned. The frozen 251/1000 diagnostic uses:

```text
geometry  2026-07-24-altloc-minstate-rotwidth-matched-tmol-v1
tmol      frozen_matched_deposited_minstate_v1
tolerance 0.0
```

The newer rotamer-control rule
`2026-07-24-altloc-minstate-rotwidth-hisunion-aromatic45-matched-tmol-v2`
has not been promoted to an official 20-site headline.

## 4. Frozen 20-site panel

Original five:

```text
3A1C B ARG447
4C16 A MET258
6H59 B ARG144
7F72 A MET103
8Q6Q B ASP81
```

Expanded fifteen:

```text
1ZV8 E ASN1     2V05 A HIS168    2VFP A TYR417
3GMI A GLU5     3K8W A SER337    3NY7 B LYS19
4MKM A THR77    5DBA A TRP325    5KWB A PHE591
5Z8H A MET730   6Y4G B CYS260    7T7A A LEU396
7UO8 A GLN53    8DJ2 A VAL893    8FBE B ILE92
```

All 20 are from the untouched 99-protein test directory. Site selection used
chemistry, occupancy, resolution, A/B separation, and representability—not
recovery performance.

## 5. Latest denoised experimental-target result

This composite uses the per-residue-schedule 20-site run, replacing ARG447,
ARG144, and LYS19 with their denoised 200+200+200 ablation results.

```text
Both conformers found:       561 / 1000
+ occupancy ±0.20:          423 / 1000
+ full physical audit:      280 / 1000 strict
```

Mutually exclusive failure cascade:

```text
not both found              439 / 1000
both found, occupancy fail  138 / 1000
physical fail after both    143 / 1000
strict success              280 / 1000
```

Slide figure:

```text
results/figures/twenty_site_recovery_cascade.png
```

Authoritative pod roots:

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_per_residue_schedule_v1
/home/dev/qfit_unet_data/density_denoiser/heldout_three_four_chi_200step_v1
```

The denoised 4-chi extension improved the three-site aggregate from 14/150 to
21/150 strict. ARG144 and LYS19 improved; ARG447 remained 0/50 because its A
basin was never found. Full details:

```text
results/heldout_three_four_chi_200step_report.md
results/heldout_three_four_chi_200step_comparison.csv
```

## 6. Latest synthetic-target optimizer control

This composite uses the synthetic 20-site per-residue run, replacing ARG447,
ARG144, and LYS19 with their synthetic 200+200+200 results.

```text
Both conformers found:        673 / 1000
+ occupancy ±0.20:           653 / 1000
physically valid independent  622 / 1000
strict joint success:         494 / 1000  (49.4%)
```

Authoritative local composite:

```text
results/figures/twenty_site_synthetic_200step_composite.csv
results/figures/twenty_site_synthetic_200step_composite.png
```

Authoritative pod roots:

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_per_residue_schedule_v1
/home/dev/qfit_unet_data/density_denoiser/heldout_three_four_chi_200step_synthetic_v1
```

Per-site synthetic composite:

| Site | Both | + occupancy | Physical independent | Strict |
|---|---:|---:|---:|---:|
| 1ZV8 ASN1 | 6 | 6 | 0 | 0 |
| 2V05 HIS168 | 2 | 0 | 49 | 0 |
| 2VFP TYR417 | 13 | 11 | 23 | 11 |
| 3A1C ARG447 | 43 | 41 | 40 | 36 |
| 3GMI GLU5 | 11 | 11 | 3 | 0 |
| 3K8W SER337 | 50 | 50 | 39 | 39 |
| 3NY7 LYS19 | 45 | 45 | 35 | 35 |
| 4C16 MET258 | 21 | 21 | 14 | 10 |
| 4MKM THR77 | 50 | 50 | 50 | 50 |
| 5DBA TRP325 | 30 | 30 | 0 | 0 |
| 5KWB PHE591 | 48 | 48 | 48 | 46 |
| 5Z8H MET730 | 18 | 18 | 43 | 17 |
| 6H59 ARG144 | 45 | 41 | 42 | 36 |
| 6Y4G CYS260 | 50 | 50 | 50 | 50 |
| 7F72 MET103 | 37 | 33 | 37 | 30 |
| 7T7A LEU396 | 41 | 35 | 49 | 34 |
| 7UO8 GLN53 | 14 | 14 | 0 | 0 |
| 8DJ2 VAL893 | 50 | 50 | 50 | 50 |
| 8FBE ILE92 | 49 | 49 | 0 | 0 |
| 8Q6Q ASP81 | 50 | 50 | 50 | 50 |
| **Total** | **673** | **653** | **622** | **494** |

The gap between the synthetic control (494 strict) and denoised experimental
target (280 strict) is evidence that both target quality and optimizer search
remain limiting. The synthetic control is not a claim of experimental
recovery.

### 2026-07-24 synthetic physical-audit revision

The `494/1000` strict value above uses the historical rule:

```text
generic 30-degree rotamer gate
all-altloc/all-water hard clash audit
tmol <= better(deposited A, deposited B) + 10
```

A revised audit corrected rotamer centers/widths, direct and symmetry altloc
partitioning, and matched-control tmol. Candidate-specific tmol environments
diverged from their deposited references in 30/293 sampled endpoints, so tmol
now freezes the deposited-A environment for every A-assigned endpoint and the
deposited-B environment for every B-assigned endpoint. Every control and
endpoint was regenerated from scratch under:

```text
geometry  2026-07-24-altloc-minstate-rotwidth-matched-tmol-v1
tmol      frozen_matched_deposited_minstate_v1
```

The authoritative zero-tolerance revised composite is:

```text
Both conformers found:        673 / 1000
+ occupancy +/-0.20:          653 / 1000
physically valid independent: 270 / 1000
strict joint success:         251 / 1000
```

The earlier copied-energy intermediate `237/1000` is invalid and must not be
cited. The rise from 237 to 251 is removal of an audit artifact, not optimizer
improvement. The historical 494 and revised 251 use materially different
physical rules and are not model-progress comparisons.

The exact `candidate - matched deposited <= 0` tmol gate is highly sensitive:
351/588 finite failures are within +1. Near-reproduction endpoints below
0.1 A RMSD have a positive-side maximum of about +0.473. Sensitivity:

```text
tmol tolerance    strict
0.0               251 / 1000
0.5               372 / 1000
1.0               409 / 1000
2.0               460 / 1000
5.0               546 / 1000
10.0              556 / 1000
```

Full report and pod roots:

```text
results/synthetic_frozen_tmol_v5_report.md
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_audit_rules_v5_frozen_tmol_aligned
/home/dev/qfit_unet_data/density_denoiser/heldout_three_four_chi_200step_synthetic_audit_rules_v5_frozen_tmol_aligned
```

### 2026-07-24 frozen-tmol gate diagnostic

No endpoints were modified and no tmol energy was rescored. Across all 1,898
finite A/B-matched active conformers, tmol margin versus matched-deposited RMSD
has Pearson `-0.1941` and Spearman `-0.4138`. tmol is therefore not a redundant
positive RMSD proxy; displaced conformers often score lower. The `+0.5`
reproduction observation applies to the 442 conformers within 0.1 A. Applying
it globally would also admit 104 positive-margin conformers outside that
regime, so no tolerance was promoted.

There are 2,496 active conformers: 1,898 matched and finite, 598 unmatched, and
zero non-finite matched conformers. Thus 588 zero-tolerance tmol failures are
`588/1,898 = 31.0%`. The “physically valid independent” counts are endpoint
ensembles/starts, not conformers.

At the exact 11 recovered+occupancy 3GMI endpoints that hard-fail symmetry, the
raw Stage-2 soft symmetry loss is nonzero in every case (`0.5015` to `0.5188`;
`2.507` to `2.594` after `lambda_clash=5`). The soft cutoff is 2.5 A and the
hard gate is 2.0 A. Those values conflict with the zero losses stored by the
original optimizer run and prompted a fresh current-environment rerun; see the
new 3GMI diagnostic below.

Positive-margin per-site 99th percentiles within the matched `<=0.1 A`
reproduction population range from `0.0333` to `0.4709`, a spread of `0.4375`.
Only eight sites have any positive margin in that population, and 3K8W has 84
near-deposited conformers but no positive margin. No global tolerance was
promoted.

At tmol tolerance 0.0, the occupancy-conditioned all-active strict count is
251/1000 and the assigned-A/B-pair count is 350/1000. At tolerance 0.5 they
are 372/1000 and 479/1000. The all-active rule sees 537 actual extra slots and
41 missing slots relative to two per start, but assigned-pair reporting leaves
both 3K8W and 8Q6Q at 0/50. Their zeroes are therefore not explained solely by
extra active conformers.

The schedule-tuned 3A1C ARG447, 6H59 ARG144, and 3NY7 LYS19 replacements are
inside the panel and contribute `31 + 23 + 14 = 68` of the 251 all-active
strict successes.

A deterministic broad deposited-control audit covered 170 A/B sites, 340
conformers, 17 residue types, and 63 held-out test proteins. Under geometry
rule `2026-07-24-altloc-minstate-rotwidth-matched-tmol-v1`, 30/340 deposited
conformers were rejected (8.8%); all 30 failed rotamer. HIS was 16/20 rejected,
PHE 4/20, TYR 4/20, LYS 3/20, and ASP/CYS/GLN 1/20 each. tmol is structurally
exempt from this control because deposited-versus-self has margin exactly
zero.

After unioning HIS chi2 `+/-80` with `+/-170` and widening PHE/TYR chi2 from
30 to 45 degrees, the identical floor under the v2 geometry rule rejects
12/340 (3.5%). HIS is 0/20 rejected; PHE and TYR remain 3/20 each. This is a
large control improvement but not a clean false-rejection floor, so v2 is not
an official headline rule.

Authoritative report and pod roots:

```text
results/frozen_tmol_gate_diagnostic_v1_report.md
/home/dev/qfit_unet_data/density_denoiser/frozen_tmol_gate_diagnostic_v1
/home/dev/qfit_unet_data/density_denoiser/deposited_altloc_false_rejection_floor_v2/audit
/home/dev/qfit_unet_data/density_denoiser/site_tmol_and_assigned_pair_diagnostic_v2
/home/dev/qfit_unet_data/density_denoiser/deposited_altloc_false_rejection_floor_v4_his_aromatic_rule_v2/audit
```

The obsolete raw v3 and v4 audit trees were intentionally deleted from the
PVC on 2026-07-24 to prevent accidental reuse. The derived v3-to-v5
artifact-attribution table remains frozen under
`frozen_tmol_gate_diagnostic_v1/artifact_delta_v3_to_v5.csv`. The two v5
aligned audit roots above remain authoritative.

### 2026-07-24 3GMI stale-environment and barrier diagnostic

The original 3GMI endpoints stored Stage-2 symmetry loss as zero, but current
evaluation at those exact coordinates gives raw loss `0.5015-0.5188`. A fresh
50-start run with identical scientific hyperparameters and no barrier changes
the result decisively:

```text
both found + occupancy          41 / 50
all-active geometry physical   50 / 50
sub-2 A symmetry failures       0 / 50
```

The exact 11 historical failures all move from `1.7795-1.7920 A` to
`2.6254-2.8514 A`. The old 3GMI 0/50 is therefore a run-time
symmetry-environment artifact. All 20 baseline endpoint files and the three
schedule-tuned replacements predate the current symmetry-environment
implementation, so all sites carry stale-provenance risk; only 3GMI has been
rerun to establish a numerical effect.

A single-factor quartic barrier was also implemented:

```text
base       max(2.5 - d, 0)^2
shoulder   scale * max((2.0 + 0.25 - d) / 0.25, 0)^4
tested scale 1.0
```

At 2.0 A this raises weighted symmetry cost from 1.25 to 6.25. The 50-start
barrier run also has zero hard symmetry failures, and all old 11 remain at
`2.625-2.889 A`. Its 40/50 recovery versus 41/50 without the barrier differs
by only about 0.35 standard errors at `n=50`, so it is evidence of neither
benefit nor harm. Its one physical failure is rotamer-only. Keep the barrier
available for a corrected site that still balances below 2.0 A; do not
promote it globally from 3GMI.

```text
/home/dev/qfit_unet_data/density_denoiser/3gmi_current_symmetry_environment_rerun_v1
/home/dev/qfit_unet_data/density_denoiser/3gmi_symmetry_barrier_quartic_v1
```

## 7. Important completed diagnostics

### Two-stage physics

Adding physics during initial basin search damaged recovery. The successful
design is density-first, then reset Adam and polish with physics. Probe 4c.2
established that soft physics can eliminate clashes and produce canonical
endpoints, but it cannot recover a basin that Stage 1 never finds.

### Hard ARG447 site

More iterations alone do not solve denoised 3A1C ARG447. Fixed/released
occupancy, sequential fitting, Dunbrack initialization, transferred ARG
initialization, and longer coarse-to-fine schedules were explored. The
consistent issue is failure to reach deposited A under the denoised target.
Promising future changes are multi-basin proposal/search, adaptive slot
creation, residual/sequential proposals with a better joint refinement, or a
set/slot-attention model whose number of active conformers is not hard-coded.

### Occupancy

Several sites recover both structures but fail occupancy. Occupancy logits can
receive weaker or poorly calibrated gradients than chi angles. Relevant ideas:

- fixed-occupancy navigation followed by released occupancy;
- occupancy-temperature or entropy schedules;
- site-specific amplitude calibration;
- explicit residual-density fitting;
- reporting sensitivity at ±0.20, ±0.10, and ±0.05.

The recomputation utility is:

```text
density_denoiser/recompute_strict_occupancy_tolerances.py
```

### Low-VDW ablation

`lambda_vdw=0.05` was tested on 11 selected failure sites. It is not globally
better: all 50/550 strict successes came from SER337. It helped that rotamer
case but did not rescue clash-prone GLU/TRP/ILE sites. Use residue/failure-class
specific changes rather than adopting 0.05 globally.

```text
docs/heldout_eleven_lambda_vdw_005_report.md
/home/dev/qfit_unet_data/density_denoiser/heldout_eleven_lambda_vdw_005_v1
```

### Canonical frame and U-Net 2.0

Canonical residue-frame extraction was implemented and validated, requiring
regenerated patches. Canonical-model training did not yield a sufficiently
clear downstream advantage to replace the original model. U-Net 2.0 added a
candidate-landscape/ranking auxiliary loss; early checkpoints showed at most
small downstream changes and did not solve the missing ARG447 basin.

Do not claim these models failed universally, but do not use them as the
production checkpoint without a controlled held-out comparison.

### SampleWorks/qFit comparison

Local scaffolding exists:

```text
density_denoiser/prepare_sampleworks_benchmark.py
density_denoiser/calibrate_sampleworks_density.py
density_denoiser/audit_sampleworks_benchmark.py
density_denoiser/run_sampleworks_3a1c_pair.sh
external/
```

Treat this as work in progress until its run root contains a terminal status
and audited output. SampleWorks may not accept omit-map density in the same way
as this optimizer, so calibrate the map representation before interpreting a
comparison. A future qFit comparison should use identical structures,
reflections/maps, sites, and strict audits.

## 8. Density-map visualization assets

The ASP81 triplet used for slides is under:

```text
artifacts/maps/8Q6Q_B_ASP81_density_triplet_v1/
```

It includes the local-frame PDB plus experimental omit `mFo-DFc`, denoised, and
synthetic ground-truth CCP4 maps. PyMOL images must use an identical saved view
and `set auto_zoom, off`; map contour values are not numerically comparable
across independently normalized maps. The chosen slide contours were roughly:

```text
experimental  0.75
denoised      11
ground truth  12
```

These unequal numeric contours are acceptable for a qualitative morphology
panel only if the slide says each map was contoured independently. Do not imply
equal-sigma quantitative comparison.

Utilities:

```text
density_denoiser/export_density_triplet.py
scripts/compose_pymol_density_triptych.py
scripts/render_density_triplet_png.py
scripts/render_density_triplet_slice_png.py
```

## 9. Local/pod topology

```text
local source    /Users/utkarsh/qfitonsteroids
pod project     qfit-unet
namespace       diffuse
pod source      /home/dev/workspace
persistent PVC  /home/dev/qfit_unet_data
main Python     /home/dev/qfit_unet_data/.venv/bin/python
tmol Python     /home/dev/qfit_unet_data/.venv-tmol/bin/python
```

Source is edited locally and synchronized. Never edit
`/home/dev/workspace` directly. Large datasets, checkpoints, and raw results
remain on the PVC.

## 10. Routine commands

From the local repository:

```bash
cd /Users/utkarsh/qfitonsteroids
actl pod list
actl pod status qfit-unet
actl pod exec qfit-unet
```

Start/restore synchronization in a dedicated terminal:

```bash
cd /Users/utkarsh/qfitonsteroids
actl pod sync qfit-unet
```

Verify a changed file reached the pod:

```bash
LOCAL=$(shasum -a 256 density_denoiser/five_site_optimizer.py | awk '{print $1}')
REMOTE=$(actl pod exec qfit-unet --no-forwarding -- bash -lc \
  "sha256sum /home/dev/workspace/density_denoiser/five_site_optimizer.py | awk '{print \\\$1}'")
printf 'local=%s\nremote=%s\n' "$LOCAL" "$REMOTE"
```

Run the test suite:

```bash
actl pod exec qfit-unet --no-forwarding -- bash -lc \
  'cd /home/dev/workspace && /home/dev/qfit_unet_data/.venv/bin/python -m pytest -q'
```

Check a detached run:

```bash
OUT=/home/dev/qfit_unet_data/density_denoiser/RUN_NAME
cat "$OUT/status.txt"
ps -p "$(cat "$OUT/controller.pid")" -o pid,etime,%cpu,%mem,stat,cmd
find "$OUT/pids" -maxdepth 1 -name '*.status' -print -exec cat {} \;
tail -n 30 "$OUT/logs/controller.log"
```

Completion requires `status.txt=complete`, terminal shard statuses, parseable
strict summaries, and no relevant errors. GPU idleness alone is not evidence.

Safely stop the pod, preserving its PVC:

```bash
actl pod down qfit-unet
```

Never use `--delete-data`.

## 11. Launch commands

Always choose a new versioned output directory. The launchers refuse to
overwrite an existing scientific run.

### Full 20-site denoised experimental benchmark

```bash
cd /home/dev/workspace
OUT=/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_per_residue_schedule_v2
mkdir -p "$OUT/logs"
nohup env OUTPUT="$OUT" \
  bash density_denoiser/run_heldout_twenty_per_residue_schedule_shards.sh \
  > "$OUT/logs/controller.log" 2>&1 < /dev/null &
echo $! > "$OUT/controller.pid"
```

### Full 20-site synthetic optimizer control

```bash
cd /home/dev/workspace
OUT=/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_per_residue_schedule_v2
mkdir -p "$OUT/logs"
nohup env OUTPUT="$OUT" \
  bash density_denoiser/run_heldout_twenty_synthetic_per_residue_schedule_shards.sh \
  > "$OUT/logs/controller.log" 2>&1 < /dev/null &
echo $! > "$OUT/controller.pid"
```

### Three 4-chi sites, 200+200+200, denoised target

```bash
cd /home/dev/workspace
OUT=/home/dev/qfit_unet_data/density_denoiser/heldout_three_four_chi_200step_v2
mkdir -p "$OUT/logs"
nohup env OUTPUT="$OUT" \
  bash density_denoiser/run_heldout_three_four_chi_200step_shards.sh \
  > "$OUT/logs/controller.log" 2>&1 < /dev/null &
echo $! > "$OUT/controller.pid"
```

### Three 4-chi sites, 200+200+200, synthetic target

```bash
cd /home/dev/workspace
OUT=/home/dev/qfit_unet_data/density_denoiser/heldout_three_four_chi_200step_synthetic_v2
mkdir -p "$OUT/logs"
nohup env OUTPUT="$OUT" \
  bash density_denoiser/run_heldout_three_four_chi_200step_synthetic_shards.sh \
  > "$OUT/logs/controller.log" 2>&1 < /dev/null &
echo $! > "$OUT/controller.pid"
```

Immediately verify any launch:

```bash
ps -p "$(cat "$OUT/controller.pid")" -o pid,etime,%cpu,%mem,stat,cmd
cat "$OUT/status.txt"
tail -n 30 "$OUT/logs/controller.log"
nvidia-smi
```

Resume tmol only after a completed geometry audit:

```bash
cd /home/dev/workspace
OUT=/home/dev/qfit_unet_data/density_denoiser/RUN_NAME
/home/dev/qfit_unet_data/.venv-tmol/bin/python five_site_tmol_audit.py \
  --input-root "$OUT/audit/PANEL" --output "$OUT/audit/PANEL" --resume
/home/dev/qfit_unet_data/.venv/bin/python \
  -m density_denoiser.summarize_endpoint_audit \
  --audit-root "$OUT/audit/PANEL"
```

## 12. Recommended next experiments

Prioritize controlled experiments that address the diagnosed failure class:

1. **Experimental-versus-synthetic paired analysis.** Join the latest two
   20-site tables by site/start and quantify which sites lose recovery,
   occupancy, or physical validity specifically when moving to denoised
   experimental targets.
2. **Adaptive multi-basin proposals for ARG447/HIS168.** The failure is not
   simply insufficient steps. Test diverse rotamer/state proposals, residual
   peaks, or adaptive slot birth, then keep the same joint refinement/audit.
3. **Occupancy-specific objective.** Test fixed-then-released occupancy,
   calibrated density amplitude, or occupancy-temperature scheduling on
   VAL893/ASN1/TYR417/GLN53 without retraining the U-Net.
4. **Residue-specific physics calibration.** The low-VDW result supports a
   special treatment for SER337, not a global lower VDW. Calibrate deposited
   A/B before any per-class weight change.
5. **Fair qFit/SampleWorks baseline.** Run the same 20 sites if the tool accepts
   the map/model representation; apply the identical conventional RMSD,
   occupancy, clash, rotamer, and tmol audit.
6. **Only then revisit U-Net training.** If paired analysis shows the denoiser
   erases or merges alternate basins, explore altloc-weighted reconstruction,
   local contrast/amplitude losses, or landscape objectives. Evaluate via
   downstream held-out recovery, not validation L2 alone.

For every new experiment: change one factor, calibrate deposited A/B first,
use a new output root, checkpoint during execution, run all geometry/tmol
audits, and report raw successes/starts.

## 13. 2026-07-24 labeled-water min-state diagnostic

The optimizer physics environment now has a separate versioned rule:

`2026-07-24-altloc-minstate-water-minstate-v2`

Labeled waters use min-over-state selection in both direct and symmetry soft
environments; partial labeled waters include an absent state. Unlabeled waters
remain invariant and occupancy weighted. No weight, cutoff, schedule, rotamer
table, geometry-audit rule, or tmol rule changed.

All 40 deposited A/B conformers were re-evaluated. 7UO8 GLN53's impossible
water terms disappear: A VDW falls 6.5544 -> 1.4002 and B symmetry falls
0.26879 -> 0. The remaining A VDW is the normal nonzero 3 A squared-hinge
floor, not a water clash. All deposited symmetry floors are now zero.

The exact two-site rerun is complete:

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_7uo8_2vfp_water_minstate_v1
```

- 7UO8 GLN53: 18/50 both found and occupancy; 16/50 pass the
  rotamer/direct/symmetry cascade; tmol gives 0/50 at tolerance 0, 15/50 at
  0.5/1/2, and 14/50 at site q99 0.1128.
- 2VFP TYR417: 1/50 both found, occupancy, geometry, and strict tmol at every
  requested fixed tolerance.

The derived splice replaces only these sites:

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_water_minstate_splice_v1
```

It has 730 both found, 716 occupancy, 708 geometry, and all-active tmol
322/533/559/574 at tolerances 0/0.5/1/2. Assigned-pair tolerance-0 remains
394. Its `site_rule_provenance.csv` records the two replacements under
water-minstate-v2 and the other 18 under the retrospective
water-invariant-v1 label. This is mixed-rule diagnostic evidence, not a
candidate frozen metric.

The 2VFP occupancy-split hypothesis is rejected on the completed-current
endpoints: median A + unmatched occupancy is 0.2326 versus deposited 0.420,
and the unmatched conformer is about 5.95 A from A.

Full interpretation and artifact paths are in
`results/water_minstate_two_site_rerun_v1_report.md`.

## 14. 2026-07-28 frozen synthetic 20-site metric v3

The single-rule composite is complete at:

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_water_minstate_v2_single_rule_v1
```

The frozen metric is:

```text
qfit-synth20-merge050-one-to-one-tmol044-v3
```

It uses an optimal one-to-one A/B assignment protected during a 0.5 A
single-linkage near-duplicate merge, sums occupancy within each merged
component, and retains the matched-tmol tolerance of `+0.44`. Its frozen
baseline cascade is 742 both found, 714 occupancy-qualified, 710 assigned-pair
rotamer/direct/symmetry, and 626 strict tmol-qualified starts.

It supersedes both earlier definitions before any subsequent model experiment:

- v1 `qfit-synth20-assignedpair-tmol044-v1` used independent greedy labels,
  undercounting recovery by 13 and summing occupancy across some
  geometrically distinct slots; cascade 729/715/711/621.
- v2 `qfit-synth20-one-to-one-assignedpair-tmol044-v2` fixed recovery but
  omitted genuine near-duplicate occupancy; cascade 742/700/696/615.

These deltas are assignment corrections, not model progress.

Do not modify this metric during subsequent model experiments. Any change
requires a new version string and a complete baseline re-audit.

The definition, source hashes, tolerance validation, extra-slot decision,
joint-library false-acceptance rates, 5Z8H diagnosis, limitations, and
authoritative artifact paths are in:

```text
results/synthetic_20site_metric_freeze_v3.md
```

The superseded definitions remain at
`results/synthetic_20site_metric_freeze_v1.md` and
`results/synthetic_20site_metric_freeze_v2.md`.

### 2026-07-28 assigned-pair separation diagnostic

Across all 742 v3 recovered starts, assigned/deposited A–B separation has
median 0.985 and IQR 0.956–1.031, so assigned pairs are not systematically
compressed panel-wide. 5KWB is clean: deposited separation 0.617 A versus
assigned median 0.597 A over 49 starts.

All eight assigned pairs below half the deposited separation occur at 2VFP.
Its deposited separation is 0.567 A, but assigned median is 0.226 A and 8/14
recoveries sit at 0.190–0.230 A. The A/B-protected merge preserves those
anchors by construction, so v3 has a concentrated known double-counting risk
at 2VFP even though the other 19 sites do not show the pattern.
All eight pass occupancy and the complete strict cascade, contributing 8/626
strict successes. Diagnostic removal would give 734 both-found and 618
strict, but this diagnostic did not change the production metric.

Production RMSD already minimizes over valid equivalent terminal-atom swaps.
For 8Q6Q ASP81, fixed-label same-state pairwise median is 1.110 A and the
OD1/OD2-swap-minimized median is 0.975 A (IQR 0.899–1.073); only 5/50 pairs
fall below 0.25 A. The prior 0.975 A value already included the swap and does
not collapse toward zero.

Full results:

```text
results/assigned_pair_separation_diagnostic_v1.md
```

### 2026-07-28 occupancy-gradient/activity diagnostic

The signed post-softmax occupancy gradient is not a useful extra-conformer
classifier: 54.76% of extras and 51.58% of matched conformers have positive
`dL_density/docc`. The distributions straddle zero at every useful aggregate
level.

Under the superseded v1 assignment, using `occupancy > 0.10` as the reported
extra-active threshold removes 195/248 extras attached to frozen-headline
successes with 0/1,242 matched conformers lost; headline extra-bearing starts
become 52/621 instead of 208/621. Across all starts it removes 376/1,030
extras with 0/1,458 matched lost, and the companion all-active count becomes
598 instead of 534. The optimizer's `>0.05` physics mask is unchanged. These
counts remain historical v1 interpretive diagnostics; v2 matching is
authoritative for future results.

The 14 occupancy-gate failures have median extra-plus-submask mass 0.204
versus 0.0475 for the 715 passes (AUC 0.873). Full results:

```text
results/occupancy_gradient_activity_threshold_diagnostic_v1.md
```

An absent-slot ablation was briefly prepared but stopped before endpoints at
the user's request. Its partial remote tree is marked `stopped_by_user`; the
optimizer was restored to frozen source hash
`367acfaba8f6d0da660fac45ace5c0c696f705bbdb05b60d2072b8724b87cbd6`.

## 15. 2026-07-28 Stage-1 fixed-occupancy prefix sweep

The first model experiment under frozen metric
`qfit-synth20-merge050-one-to-one-tmol044-v3` is complete:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_fixed_occupancy_sweep_v1
```

It swept `fixed_occupancy_steps=100/200/300` against the unchanged frozen
control. Occupancy logits remained in a separate Adam group with learning
rate zero during the prefix, so gradients and moments stayed warm. A required
zero-step grouped-Adam check reproduced 3GMI exactly at 41/50 with identical
endpoint rows.

The total cascades were:

```text
fixed 0:   742 / 714 / 710 / 710 / 710 / 626
fixed 100: 728 / 719 / 715 / 715 / 715 / 632
fixed 200: 695 / 680 / 677 / 677 / 677 / 609
fixed 300: 659 / 637 / 635 / 635 / 635 / 567
```

The primary raw-greedy minor/major missed-state counts were 142/45 control,
173/35 at 100, 152/53 at 200, and 178/52 at 300. No arm improved absolute
minor-state recovery. The 100-step strict gain of 6 accompanies 14 fewer
recovered pairs and is site redistribution rather than the target capability.

Release produced a visible first-step density-loss discontinuity in
594/643/637 starts for the three arms. New-arm slots crossing below 0.05
subsequently traveled median 0.724/0.974/1.215 A, so those trajectories do
not show positional paralysis. The frozen control did not record trajectories,
leaving the original control-specific mechanism claim untestable.

Full tables and interpretation:

```text
results/fixed_occupancy_sweep_v1_report.md
```

### Mode splitting versus mass dumping

The saved-endpoint diagnostic favors mass dumping as the panel-wide
explanation for high-occupancy extras in failed-recovery starts. Among 363
extras above 0.10, median RMSD to the missed state is 1.525 Å and only 13
(3.6%) lie within the 1 Å recovery neighbourhood. All 13 occur at 2VFP,
and every one is closer to the recovered B state than to missed A.

By comparison, 44/53 high extras in the 52 headline comparison starts lie
within 1 Å of an already recovered deposited state, showing that local mode
duplication exists but does not explain most missed-state failures.

For the 196 starts missing exactly one state, recovered occupancy plus all
active-extra occupancy is within 0.05 of deposited A+B in 169 cases, but this
is mostly the K=4 softmax accounting identity. Using only extras within 1 Å
of the missed state leaves median mass error 0.272, and only 12/196 starts
contain a high near-missed extra.

Full report and raw-artifact paths:

```text
results/mode_splitting_vs_mass_dumping_v1.md
```

## 16. 2026-07-29 unmatched-slot target-density prerequisite

The frozen 187 single-recovery starts contain 259 unmatched active slots.
Their atom positions have median native target density 65.9% of the within-site
deposited A/B reference (IQR 49.7–90.6%); 189/259 retain at least half the
reference and none is below 10%. The slots therefore occupy real tail density,
not vacuum.

Authoritative artifacts:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_water_minstate_v2_single_rule_v1/
analysis/unmatched_target_density_prerequisite_v1
```

Local report:

```text
results/unmatched_target_density_prerequisite_v1.md
```

## 17. 2026-07-29 Stage-1 merge-and-respawn R1

R1 is complete at:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_respawn_R1_v1
```

It ran 20 sites × 50 starts with `seed=41+start`, cadence 100, active-pair
merge RMSD below 0.5 A, and the unchanged frozen-v3 audit. The reused control
guard passed exactly `742/714/710/710/710/626`.

Primary raw missed-minor / missed-major counts were:

```text
control: 142 / 45
R1:      142 / 46
```

The R1 cascade is `743/716/712/712/712/628`, but the +2 strict total is
downstream site redistribution rather than primary placement progress. All
five requested tail sites have zero cascade delta. R2/R3 were not launched
because R1 did not improve the primary endpoint.

R1 fired 143 respawns in 123/1000 starts. The merged-away slot was already
within 1 A of a deposited state in 134/143 events; 132/143 replacements ended
worse than the slot replaced; only 22/140 unique respawned slots survived
above 0.10 occupancy. Among those 132 worse events, 120 peaks were near a
deposited state, eight near the midpoint, and four unrelated. The dominant
failure is therefore not overlap-peak placement, but underdetermined
atom-peak-to-whole-conformer inversion and poor survival.

Gram condition >=100 was neither meaningfully earlier nor more reliable than
the 0.5 A RMSD trigger. Endpoint duplicate precision/recall were 35.9%/20.5%
for Gram and 37.4%/20.1% for RMSD.

Authoritative comparison:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_respawn_R1_v1/
analysis/frozen_v3_respawn_comparison_v4
```

Full report:

```text
results/respawn_R1_v1_report.md
```

Do not implement respawn in production. The tested naive residual-argmax
insertion rule is rejected.

## 18. 2026-07-29 raw-residual canonical-rotamer gate

The read-only frozen-v3 diagnostic is complete at:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_water_minstate_v2_single_rule_v1/
analysis/frozen_v3_residual_rotamer_gate_v2
```

It reconstructed all four saved endpoint slots for the guarded historical
142 missed-minor / 45 missed-major starts and evaluated the exact production
Cartesian canonical-chi pool in native additive density before z-scoring.
No optimizer was run and the metric was unchanged. Coordinate reconstruction
agreed within `8.31e-6 A`; raw target reconstruction re-z-scored to the saved
target at maximum relative L2 error `3.86e-6`.

The deposited missed-conformer ceiling has median Pearson `0.340` and fitted
occupancy `0.070`. The best canonical rotamer falls to median Pearson `0.119`
and fitted occupancy `0.023`; only 49/187 are within 1 A of the missed state.
Using the descriptive conjunction within 1 A, Pearson at least 0.2, and fitted
occupancy at least 0.05, only 27/187 qualify.

The decisive tail results are heterogeneous. 2VFP has deposited ceiling
median `0.181`, best canonical median `-0.007`, and 0/44 actionable fits.
5Z8H has only 5/28 actionable fits. 1ZV8 retains a strong deposited ceiling
but its canonical pool does not cover the missed state. 7UO8 canonical
densities correlate but remain outside 1 A. 4C16 is positive in 15/24 starts,
mostly from the missed-major cohort.

A low-occupancy active slot (`0.05 < q < 0.10`) is available without a
degenerate pair in 81/187 starts. At the two sites with zero R1 merge events,
this covers 23/44 2VFP missed-minor starts and 6/26 5Z8H missed-minor starts,
but useful canonical fits occur in only 0/44 and 3/26, respectively.

The gate therefore rejects a panel-wide run based on direct insertion from
the current canonical table. It does not prove every conceivable
whole-conformer proposal impossible, but a merge-independent trigger alone
does not repair the missing insertion targets. Do not implement or launch
whole-conformer respawn from this result.

Full report:

```text
results/frozen_v3_residual_rotamer_gate_v2_report.md
```
