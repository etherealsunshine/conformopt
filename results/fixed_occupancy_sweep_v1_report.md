# Stage-1 fixed-occupancy prefix sweep

Date: 2026-07-28

Frozen metric (unchanged):
`qfit-synth20-merge050-one-to-one-tmol044-v3`

Control: `fixed_occupancy_steps=0`, 626 strict / 742 both found.

Experiment arms: `fixed_occupancy_steps=100`, `200`, and `300`. Each arm
contains the same 20 sites, 50 starts per site, with `seed = 41 + start`.
All model, schedule, physics, audit, assignment, merge, activity, and tmol
settings were held fixed.

## Outcome

Freezing occupancy did not improve minor-conformer recovery. The 100-step arm
raised strict success by 6 starts, but lost 14 recovered pairs and made the
minor-state failure imbalance worse. The 200-step arm produced a slightly
smaller minor/major ratio only because both failure counts increased and major
failures increased more. Longer freezes progressively reduced recovery.

| Fixed steps | Both found | Occupancy | Rotamer | Direct | Symmetry | Strict | Strict delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 control | 742 | 714 | 710 | 710 | 710 | 626 | — |
| 100 | 728 | 719 | 715 | 715 | 715 | 632 | +6 |
| 200 | 695 | 680 | 677 | 677 | 677 | 609 | -17 |
| 300 | 659 | 637 | 635 | 635 | 635 | 567 | -59 |

The +6 strict result at 100 steps is not evidence for the target capability:
it is accompanied by lower recovery and a worse minor-state endpoint.

## Implementation and zero-step reproduction

The requested warm-state freeze required splitting Stage-1 Adam into chi and
occupancy parameter groups. The occupancy group remains in Adam with live
gradients and moment updates while its learning rate is zero; its learning
rate is restored in place without resetting Adam.

Because this is the parameter-group case, a 50-start zero-step reproduction
was run at 3GMI GLU5 before the sweep. It reproduced the frozen run exactly:
41/50 both found and byte-identical final/stage-1 chi values, occupancies,
RMSDs, loss terms, labels, and occupancy accuracy for all 50 starts.

The implementation is in
`density_denoiser/five_site_optimizer.py` (`_stage1_adam` and
`_set_occupancy_learning_rate`). Its production source hash was:

```text
629d34a754faf8774784c3a0f5c3437f619c9c3713fe3b6d75ba19de16c06583
```

The full pod test suite passed: 51 tests.

## Primary endpoint: missed deposited occupancy rank

The requested control-comparable diagnostic uses the historical raw-greedy
classification that produced 142 minor / 45 major misses. The frozen-v3
one-to-one assignment is shown separately because it corrects 13 2VFP starts.
Equal-occupancy failures are excluded from the rank split.

| Fixed steps | Minor missed, raw | Major missed, raw | Ratio | Minor missed, v3 | Major missed, v3 |
|---:|---:|---:|---:|---:|---:|
| 0 control | 142 | 45 | 3.16 | 129 | 45 |
| 100 | 173 | 35 | 4.94 | 169 | 35 |
| 200 | 152 | 53 | 2.87 | 148 | 53 |
| 300 | 178 | 52 | 3.42 | 176 | 52 |

No arm meets the minor-recovery success criterion. At 200 steps, the apparent
ratio improvement is caused by deterioration in both numerator and
denominator, not better minor recovery.

## Full per-site cascade

Cells are `found / occupancy / rotamer / direct / symmetry / strict`.

| Site | Control | Fixed 100 | Fixed 200 | Fixed 300 |
|---|---:|---:|---:|---:|
| 1ZV8 ASN1 | 1/0/0/0/0/0 | 0/0/0/0/0/0 | 0/0/0/0/0/0 | 0/0/0/0/0/0 |
| 2V05 HIS168 | 22/22/22/22/22/21 | 30/29/29/29/29/29 | 29/28/28/28/28/28 | 27/25/25/25/25/25 |
| 2VFP TYR417 | 14/8/8/8/8/8 | 2/2/2/2/2/2 | 1/0/0/0/0/0 | 1/0/0/0/0/0 |
| 3A1C ARG447 | 47/45/45/45/45/44 | 47/46/46/46/46/46 | 44/44/44/44/44/44 | 45/44/44/44/44/43 |
| 3GMI GLU5 | 41/41/41/41/41/39 | 32/31/31/31/31/30 | 30/26/26/26/26/23 | 23/22/21/21/21/20 |
| 3K8W SER337 | 50/50/50/50/50/50 | 50/50/50/50/50/50 | 50/50/50/50/50/50 | 50/50/50/50/50/50 |
| 3NY7 LYS19 | 48/47/43/43/43/38 | 50/50/50/50/50/46 | 50/50/49/49/49/45 | 49/49/49/49/49/47 |
| 4C16 MET258 | 20/20/20/20/20/19 | 24/24/20/20/20/19 | 17/17/16/16/16/15 | 17/17/16/16/16/15 |
| 4MKM THR77 | 50/50/50/50/50/50 | 50/50/50/50/50/50 | 50/50/50/50/50/50 | 50/50/50/50/50/50 |
| 5DBA TRP325 | 37/37/37/37/37/37 | 44/44/44/44/44/44 | 39/39/39/39/39/37 | 36/36/36/36/36/36 |
| 5KWB PHE591 | 49/49/49/49/49/49 | 31/31/31/31/31/31 | 34/34/34/34/34/34 | 20/20/20/20/20/20 |
| 5Z8H MET730 | 16/16/16/16/16/1 | 20/20/20/20/20/2 | 14/14/13/13/13/3 | 9/9/9/9/9/1 |
| 6H59 ARG144 | 41/41/41/41/41/39 | 41/41/41/41/41/40 | 40/40/40/40/40/40 | 41/41/41/41/41/41 |
| 6Y4G CYS260 | 50/50/50/50/50/50 | 50/50/50/50/50/50 | 50/50/50/50/50/50 | 50/50/50/50/50/50 |
| 7F72 MET103 | 45/38/38/38/38/5 | 44/44/44/44/44/14 | 43/42/42/42/42/12 | 41/39/39/39/39/8 |
| 7T7A LEU396 | 44/41/41/41/41/34 | 47/46/46/46/46/44 | 46/45/45/45/45/42 | 42/41/41/41/41/34 |
| 7UO8 GLN53 | 18/18/18/18/18/18 | 16/16/16/16/16/16 | 11/11/11/11/11/11 | 11/10/10/10/10/10 |
| 8DJ2 VAL893 | 50/50/50/50/50/50 | 50/50/50/50/50/50 | 49/49/49/49/49/49 | 49/49/49/49/49/49 |
| 8FBE ILE92 | 49/49/49/49/49/49 | 50/50/50/50/50/48 | 48/48/48/48/48/48 | 48/48/48/48/48/46 |
| 8Q6Q ASP81 | 50/42/42/42/42/25 | 50/45/45/45/45/21 | 50/43/43/43/43/28 | 50/37/37/37/37/22 |

## Tail-site deltas

Values are `found / strict`; parentheses are deltas from control.

| Site | Control | Fixed 100 | Fixed 200 | Fixed 300 |
|---|---:|---:|---:|---:|
| 1ZV8 ASN1 | 1/0 | 0/0 (-1/+0) | 0/0 (-1/+0) | 0/0 (-1/+0) |
| 2VFP TYR417 | 14/8 | 2/2 (-12/-6) | 1/0 (-13/-8) | 1/0 (-13/-8) |
| 5Z8H MET730 | 16/1 | 20/2 (+4/+1) | 14/3 (-2/+2) | 9/1 (-7/+0) |
| 7UO8 GLN53 | 18/18 | 16/16 (-2/-2) | 11/11 (-7/-7) | 11/10 (-7/-8) |
| 4C16 MET258 | 20/19 | 24/19 (+4/+0) | 17/15 (-3/-4) | 17/15 (-3/-4) |

The strongest negative effect is the collapse of 2VFP recovery. Improvements
at 2V05, 7F72, and 7T7A explain why the 100-step strict total rises despite
lower panel-wide recovery.

## Same-state duplication and distinct conformers

| Fixed steps | Starts with duplicates | Non-primary members | Median occupancy | Members >0.20 | Mean post-merge distinct count |
|---:|---:|---:|---:|---:|---:|
| 0 control | 237 | 280 | 0.0868 | 52 | 2.265 |
| 100 | 305 | 363 | 0.0978 | 86 | 2.206 |
| 200 | 254 | 293 | 0.1003 | 78 | 2.207 |
| 300 | 244 | 284 | 0.0948 | 71 | 2.194 |

All frozen-arm non-primary duplicates have occupancy 0.25 at the release
boundary. Their median occupancy falls by the end of Stage 1, but duplication
persists at the endpoint; 100 steps makes it substantially worse. Therefore
the slots reweight after release but do not reliably separate into distinct
states.

## Unmatched extras

| Fixed steps | Extra-bearing starts >0.05 | Extra-bearing starts >0.10 | Extra conformers | Median extra occupancy |
|---:|---:|---:|---:|---:|
| 0 control | 348 | 253 | 598 | 0.170 |
| 100 | 358 | 252 | 537 | 0.174 |
| 200 | 380 | 291 | 606 | 0.209 |
| 300 | 400 | 314 | 639 | 0.202 |

The 100-step arm leaves the >0.10 extra-bearing-start rate essentially flat.
Longer freezes increase it.

## Occupancy accuracy

| Fixed steps | Median abs. A error | Median abs. B error | Median A+B deficit | Deficit q95 |
|---:|---:|---:|---:|---:|
| 0 control | 0.0235 | 0.0205 | 0.0318 | 0.1758 |
| 100 | 0.0247 | 0.0201 | 0.0335 | 0.1773 |
| 200 | 0.0256 | 0.0172 | 0.0296 | 0.1830 |
| 300 | 0.0257 | 0.0149 | 0.0333 | 0.1831 |

There is no meaningful matched-occupancy accuracy improvement.

## Assigned/deposited separation

| Fixed steps | Median separation ratio | Pairs below 0.5× |
|---:|---:|---:|
| 0 control | 0.9853 | 8 |
| 100 | 0.9850 | 4 |
| 200 | 0.9851 | 3 |
| 300 | 0.9854 | 1 |

The apparent reduction in compressed pairs is driven by loss of 2VFP
recoveries, where all eight control exceptions occurred. It is selection, not
evidence of improved pair geometry.

## Unfreeze discontinuity

| Fixed steps | Starts with loss increase | Median absolute delta | Median relative delta | Relative increase >10% |
|---:|---:|---:|---:|---:|
| 100 | 594/1000 | +0.00208 | +43.2% | 560 |
| 200 | 643/1000 | +0.01993 | +97.2% | 609 |
| 300 | 637/1000 | +0.00500 | +53.3% | 600 |

The first live occupancy step produces a visible density-loss discontinuity
in every arm. Per instruction, no learning-rate ramp was added.

## Starvation timing and subsequent travel

The frozen control did not record per-step occupancy or chi trajectories.
Consequently, the stated control-specific claim—early-starved control slots
move little afterward—cannot be tested directly. The new arms recorded all
12,000 slot trajectories.

| Fixed steps | Slots crossing 0.05 | Median crossing step | Median later RMSD | Slots crossing 0.02 | Median crossing step | Median later RMSD |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2453/4000 | 103 | 0.724 Å | 2088/4000 | 103 | 0.763 Å |
| 200 | 2545/4000 | 203 | 0.974 Å | 2249/4000 | 204 | 1.041 Å |
| 300 | 2640/4000 | 304 | 1.215 Å | 2325/4000 | 304 | 1.277 Å |

Most crossings happen within a few steps of release. Slots continue moving
substantially after crossing; the new-arm data do not show effective
positional paralysis below 0.05 or 0.02. That does not prove the historical
control behaved identically, because its trajectory was not logged.

## Interpretation

The experiment rejects a fixed 100–300-step uniform-occupancy prefix as the
next production direction under the current optimizer:

1. Absolute minor-state failures do not improve in any arm.
2. Pair recovery deteriorates monotonically as the freeze gets longer.
3. The 100-step strict gain is a redistribution across sites, not a
   minor-conformer capability gain.
4. Same-state duplication increases sharply at 100 steps.
5. Occupancy release creates a measurable density-loss discontinuity.
6. Newly released low-occupancy slots still travel appreciable distances, so
   the available mechanism evidence does not support paralysis.

This result does not change the frozen metric or baseline.

## Authoritative artifacts

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_fixed_occupancy_sweep_v1/
  run_manifest.json
  fixed_100/
  fixed_200/
  fixed_300/
  analysis_summary_v1/
    summary.json
    cascade_per_site.csv
    single_state_failures_per_site.csv
    tail_site_deltas.csv
    assigned_pair_separation_per_site.csv
  analysis_trajectories_v1/
    summary.json
    slot_starvation_and_travel.csv
    starvation_and_travel_per_site.csv
```

Zero-step grouped-Adam reproduction:

```text
/home/dev/qfit_unet_data/density_denoiser/
fixed_occupancy_grouped_zero_repro_3gmi_v1/reproduction_report.json
```
