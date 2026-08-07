# Stage-1 merge-and-respawn R1

**Metric:** `qfit-synth20-merge050-one-to-one-tmol044-v3` (unchanged)  
**Control:** frozen endpoints reused; not rerun  
**R1:** respawn every 100 Stage-1 steps; active-pair merge RMSD < 0.5 Å  
**Seeds:** `41 + start`  
**Scope:** 20 sites × 50 starts

## Decision

R1 does not improve the primary endpoint. The raw control-comparable
missed-minor / missed-major split changes from **142 / 45** to **142 / 46**.
R2 and R3 were therefore not launched under the prespecified R1-first compute
guard.

The frozen-v3 cascade guard passed exactly:

```text
control: 742 → 714 → 710 → 710 → 710 → 626
R1:      743 → 716 → 712 → 712 → 712 → 628
```

The +2 strict total is downstream reshuffling, not primary placement progress:
2V05 gains two found and one strict start, 8Q6Q gains two strict starts without
a found-count change, and 7F72 loses one found/strict start. All five requested
tail sites have zero cascade delta.

## Prerequisite: unmatched slots are not in vacuum

Across the frozen 187-start cohort, the 259 unmatched active slots have median
target density **65.9%** of the within-site deposited A/B reference (IQR
49.7–90.6%). A total of 189/259 (73.0%) retain at least half the deposited
reference density, 256/259 (98.8%) retain at least one quarter, and none are
below one tenth.

Respawn therefore displaces locally density-supported configurations rather
than empty-space slots.

## Implementation

- Only pairs for which both slots exceed the frozen 0.05 active threshold are
  eligible to merge.
- The closest active pair under symmetry-aware conventional RMSD triggers the
  arm. Threshold crossings at 0.3, 0.5, and 0.8 Å are logged at every check.
- A density-space Gram condition number is logged independently. A diagnostic
  threshold of 100 does not create extra respawns.
- On merge, pair occupancy is summed in the keeper. After the bookkeeping
  merge loss is measured, the freed slot is placed and receives exactly its
  pre-merge occupancy back from the keeper. Total occupancy is conserved.
- The peak is `argmax(target - rendered)` on the current masked Stage-1 grid.
- Torsion inversion runs 50 Adam steps at learning rate 0.1 in parallel for
  each possible side-chain heavy atom and retains the χ solution putting one
  atom closest to the peak. Reaching within 0.5 Å is counted as success.
- Adam `exp_avg` and `exp_avg_sq` are cleared for the freed χ row and for the
  freed and keeper occupancy-logit elements. Adam's scalar step is shared and
  retained.
- A no-trigger R1 smoke reproduced frozen 5Z8H start 0 exactly at Stage 1 and
  the final endpoint: χ, occupancies, density, VDW, rotamer, and symmetry
  losses all match.

## Primary endpoint by site

Values are missed-minor / missed-major.

| Site | Control | R1 | Delta |
|---|---:|---:|---:|
| 1ZV8_E_ASN1 | 20 / 1 | 20 / 1 | 0 / 0 |
| 2V05_A_HIS168 | 4 / 6 | 3 / 6 | -1 / 0 |
| 2VFP_A_TYR417 | 44 / 0 | 44 / 0 | 0 / 0 |
| 3A1C_B_ARG447 | 1 / 2 | 1 / 2 | 0 / 0 |
| 3GMI_A_GLU5 | 9 / 0 | 9 / 0 | 0 / 0 |
| 3K8W_A_SER337 | 0 / 0 | 0 / 0 | 0 / 0 |
| 3NY7_B_LYS19 | 1 / 1 | 1 / 1 | 0 / 0 |
| 4C16_A_MET258 | 7 / 17 | 7 / 17 | 0 / 0 |
| 4MKM_A_THR77 | 0 / 0 | 0 / 0 | 0 / 0 |
| 5DBA_A_TRP325 | 5 / 8 | 5 / 8 | 0 / 0 |
| 5KWB_A_PHE591 | 1 / 0 | 1 / 0 | 0 / 0 |
| 5Z8H_A_MET730 | 26 / 2 | 26 / 2 | 0 / 0 |
| 6H59_B_ARG144 | 0 / 0 | 0 / 0 | 0 / 0 |
| 6Y4G_B_CYS260 | 0 / 0 | 0 / 0 | 0 / 0 |
| 7F72_A_MET103 | 3 / 2 | 3 / 3 | 0 / +1 |
| 7T7A_A_LEU396 | 5 / 0 | 5 / 0 | 0 / 0 |
| 7UO8_A_GLN53 | 15 / 6 | 16 / 6 | +1 / 0 |
| 8DJ2_A_VAL893 | 0 / 0 | 0 / 0 | 0 / 0 |
| 8FBE_B_ILE92 | 1 / 0 | 1 / 0 | 0 / 0 |
| 8Q6Q_B_ASP81 | 0 / 0 | 0 / 0 | 0 / 0 |
| **Total** | **142 / 45** | **142 / 46** | **0 / +1** |

Under exact frozen-v3 one-to-one assignment, the split is 129 / 45 for control
and 129 / 46 for R1; nine equal-occupancy single-state starts remain excluded
from rank classification.

## Mechanism

There were 4,150 cadence checks. R1 fired **143 events in 123 starts**; 877
starts were unchanged by construction. The 143 events involve 140 unique
start-slot pairs.

- The merged-away slot was already within 1 Å of a deposited state in
  **134/143 events**.
- **132/143 replacements ended farther from the nearest deposited state than
  the slot they replaced.**
- Only **22/140 unique respawned slots** survived above 0.10 occupancy.
- Event-wise, 78/143 ended within 1 Å of some deposited state.
- Torsion inversion failed to reach within 0.5 Å in **16/143** events. Failed
  residual distances have median 0.590 Å and range 0.504–0.888 Å.

For the 132 worse replacements, peak locations are:

| Peak region | Events |
|---|---:|
| Within 1 Å of deposited A or B | 120 |
| A–B midpoint region | 8 |
| Unrelated to A, B, and midpoint | 4 |

Thus the added overlap hypothesis is not the dominant failure: only 8/132
worse replacements are midpoint-directed. The more common failure is that an
atom-level peak near a deposited state does not specify a good whole-conformer
χ solution, or the inserted occupancy is subsequently suppressed.

Among the 13 events whose endpoint is a single-recovery start, seven peaks are
near the missed state and six are near the recovered state; none is in the
midpoint or unrelated category. Even when the raw peak points at the missed
state, R1 does not reduce the aggregate missed-state count.

Close-separation behavior:

| Deposited separation | Events | Midpoint-region peaks | Median peak-to-midpoint |
|---|---:|---:|---:|
| < 2.5 Å | 105 | 5 (4.8%) | 1.356 Å |
| ≥ 2.5 Å | 38 | 3 (7.9%) | 2.176 Å |

Neither 5Z8H nor 2VFP generated an eligible Stage-1 merge event, so R1 never
tested respawn at those two close-separation tail sites.

## Gram versus RMSD duplication detection

At the start level:

- Gram condition ≥100 detects 131 starts.
- RMSD <0.3 detects 104; <0.5 detects 123; <0.8 detects 152.
- Against the 229 starts with endpoint same-state duplication, Gram ≥100 has
  precision 35.9% and recall 20.5%; RMSD <0.5 has precision 37.4% and recall
  20.1%.
- Where Gram and 0.5 Å RMSD both detect, they fire on the same first check in
  110 starts; Gram is earlier in one and RMSD is never earlier.

Gram conditioning is therefore not meaningfully earlier or more reliable than
the 0.5 Å RMSD criterion here.

## Secondary endpoints

| Metric | Control | R1 |
|---|---:|---:|
| Same-state duplicate starts | 237 / 1000 | 229 / 1000 |
| Duplicate groups | 257 | 250 |
| Post-merge distinct conformers, median | 2 | 2 |
| Post-merge distinct conformers, mean | 2.265 | 2.267 |
| Single-recovery unmatched active slots | 259 | 254 |
| No A/B/midpoint reference within 1 Å | 213/259 (82.2%) | 211/254 (83.1%) |
| Unmatched occupancy, median | 0.143 | 0.149 |
| Frozen-v3 matched A+B deficit, median | 0.03178 | 0.03136 |

The occupancy deficit change is negligible and does not close the stated
0.048 control-scale deficit in a scientifically meaningful way.

Aggregate geometry-stage losses are essentially unchanged. For respawned
event slots specifically, 16/143 have a direct clash below 2 Å, 1/143 has a
symmetry clash below 2 Å, 20/143 are noncanonical, and 32/143 fail at least one
of those geometry checks.

All five requested tail sites—1ZV8, 2VFP, 5Z8H, 7UO8, and 4C16—have zero
frozen-v3 cascade delta.

## Interpretation

R1 rejects cadence-100, active-duplicate merge followed by raw residual-peak
atom placement as a useful recovery intervention. It modestly reduces endpoint
duplication but does not convert that bookkeeping gain into missed-conformer
recovery.

The mechanism log does **not** support closing every placement method because
peaks are usually near real deposited density, not unrelated or predominantly
at the midpoint. It instead exposes the underdetermined inversion from one
peak/one atom to a whole conformer and weak endpoint survival. What is closed
by this result is the tested naive residual-argmax insertion rule. No production
implementation is justified.

## Artifacts

Run:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_respawn_R1_v1
```

Authoritative analysis:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_respawn_R1_v1/
analysis/frozen_v3_respawn_comparison_v4
```

The incomplete/superseded `v1`, `v2`, and `v3` analysis directories are
preserved. R2 and R3 were not run.
