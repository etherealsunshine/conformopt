# Signed occupancy gradient and activity-threshold diagnostic

Date: 2026-07-28  
Scope: saved endpoints from the frozen 20-site synthetic run only  
Frozen metric: `qfit-synth20-assignedpair-tmol044-v1`  
Production geometry, optimizer, tmol, weights, and thresholds: unchanged

## Result

The signed density gradient does not cleanly distinguish extra conformers from
the matched A/B pair.

For a structural conformer \(k\), the reported quantity is the ambient
post-softmax occupancy partial

```text
dL_density / docc_k = 2 <residual, rho_k>
```

holding the other occupancies fixed. Positive means that adding more of the
conformer increases density MSE and the map locally wants less of it.

| Population | n | Negative | Positive | Signed median | q25 to q75 |
|---|---:|---:|---:|---:|---:|
| Matched A/B | 1,458 | 706 (48.42%) | 752 (51.58%) | +2.22e-6 | -3.49e-4 to +6.28e-4 |
| Extra active | 1,030 | 462 (44.85%) | 564 (54.76%) | +2.29e-5 | -7.88e-4 to +2.33e-3 |

Four extra gradients are within `1e-12` of zero. No matched gradient is.
The positive-rate difference is only 3.18 percentage points, and both
distributions straddle zero broadly. Sign is therefore not a usable
density-support classifier on these endpoints.

Physics is not occupancy-weighted in the frozen objective. Away from the
hard activity-mask crossover,

```text
dL_total / docc_k = dL_density / docc_k
```

For the 961 starts with at least two active slots, the within-start range of
these occupancy partials has median `0.001685`, q75 `0.008232`, q95 `0.05019`,
and maximum `0.2599`. They are not at a smooth interior KKT stationary point.
This does not by itself identify an objective defect: the `>0.05` physics mask
makes the objective discontinuous at the crossover, and coordinates and
occupancies were optimized jointly for a finite number of Adam steps.

## Per-site signed gradients

The columns give `positive / n` and the signed median. Sites with only two
matched conformers—1ZV8 and 2VFP—and 8FBE's five extras are too small for a
site-level conclusion.

| Site | Matched positive / n | Matched median | Extra positive / n | Extra median |
|---|---:|---:|---:|---:|
| 1ZV8 ASN1 | 1 / 2 | -1.63e-3 | 89 / 183 | -7.84e-6 |
| 2V05 HIS168 | 25 / 44 | +9.58e-5 | 53 / 102 | +4.65e-4 |
| 2VFP TYR417 | 1 / 2 | +1.75e-3 | 52 / 99 | +2.57e-7 |
| 3A1C ARG447 | 52 / 94 | +7.68e-5 | 10 / 25 | -1.90e-3 |
| 3GMI GLU5 | 35 / 82 | -7.26e-5 | 15 / 24 | +6.23e-5 |
| 3K8W SER337 | 35 / 100 | -9.84e-5 | 29 / 41 | +8.13e-4 |
| 3NY7 LYS19 | 60 / 96 | +2.38e-4 | 14 / 28 | +3.28e-5 |
| 4C16 MET258 | 23 / 40 | +2.30e-5 | 57 / 95 | +5.23e-5 |
| 4MKM THR77 | 54 / 100 | +3.91e-6 | 21 / 31 | +1.47e-3 |
| 5DBA TRP325 | 37 / 74 | -7.28e-6 | 24 / 45 | +2.24e-12 |
| 5KWB PHE591 | 50 / 98 | +2.77e-7 | 6 / 10 | +2.50e-3 |
| 5Z8H MET730 | 17 / 32 | +2.55e-6 | 35 / 62 | +6.83e-6 |
| 6H59 ARG144 | 46 / 82 | +2.22e-6 | 16 / 26 | +2.29e-5 |
| 6Y4G CYS260 | 52 / 100 | +3.55e-6 | 5 / 8 | +2.16e-2 |
| 7F72 MET103 | 52 / 90 | +3.05e-5 | 25 / 36 | +1.04e-3 |
| 7T7A LEU396 | 52 / 88 | +5.61e-5 | 16 / 33 | -1.84e-5 |
| 7UO8 GLN53 | 12 / 36 | -7.62e-5 | 36 / 83 | -5.40e-5 |
| 8DJ2 VAL893 | 46 / 100 | -5.00e-5 | 34 / 48 | +1.15e-3 |
| 8FBE ILE92 | 53 / 98 | +4.43e-5 | 1 / 5 | -4.08e-2 |
| 8Q6Q ASP81 | 49 / 100 | -3.89e-7 | 26 / 46 | +5.46e-6 |

## Reported activity threshold

Adopt `occupancy > 0.10` as the reported extra-active-conformer threshold.
This is a reporting convention for over-modeling, not a production objective
change and not a change to the frozen metric.

| Report threshold | Extras removed, all starts | Matched lost, all starts | Extra-bearing starts | All-active composite at tmol +0.44 |
|---:|---:|---:|---:|---:|
| >0.05 | 0 / 1,030 | 0 / 1,458 | 540 | 534 |
| >0.075 | 269 / 1,030 | 0 / 1,458 | 401 | 581 |
| **>0.10** | **376 / 1,030** | **0 / 1,458** | **365** | **598** |
| >0.15 | 489 / 1,030 | 9 / 1,458 | 327 | 602 |

The headline-success subset is where extras cluster at the mask:

| Report threshold | Headline extras removed | Headline matched lost | Headline extra-bearing starts |
|---:|---:|---:|---:|
| >0.05 | 0 / 248 | 0 / 1,242 | 208 / 621 |
| >0.075 | 160 / 248 | 0 / 1,242 | 82 / 621 |
| **>0.10** | **195 / 248** | **0 / 1,242** | **52 / 621** |
| >0.15 | 222 / 248 | 7 / 1,242 | 26 / 621 |

At `0.10`, 78.6% of extras attached to headline successes disappear with
zero matched loss. The zero-loss result is partly definitional: the frozen
assigned-pair recovery rule already requires each selected A/B conformer to
have occupancy `>0.10`. This threshold should therefore be described as
aligned with the recovery definition, not independently optimized against
ground truth.

## Occupancy-stage losses

Among the 729 both-found starts, the 14 that fail the `±0.20` occupancy gate
have elevated mass outside the selected A/B pair:

| Population | n | Extra-active + sub-mask mass median | q25 to q75 | q95 |
|---|---:|---:|---:|---:|
| Occupancy pass | 715 | 0.0475 | 0.0187–0.0892 | 0.1983 |
| Occupancy fail | 14 | 0.2041 | 0.1092–0.2529 | 0.3636 |

The AUC for higher spare-slot mass predicting occupancy failure is `0.873`.
Seven of the fourteen failures are at 7F72, three at 7T7A, two at 3A1C, and
one each at 1ZV8 and 3NY7. Spare-slot mass is strongly associated with the
occupancy-stage losses, although it is not sufficient by itself: individual
failure masses range from `0.0241` to `0.5513`.

## Provenance

Authoritative remote artifacts:

```text
/home/dev/qfit_unet_data/density_denoiser/
  heldout_twenty_synthetic_water_minstate_v2_single_rule_v1/
  analysis/occupancy_gradient_threshold_diagnostic_v2/
```

Files:

```text
signed_grad_occ_by_site.csv
activity_threshold_sweep.csv
occupancy_gate_failures.csv
summary.json
```

The proposed absent-slot ablation was stopped at user request before any
optimization endpoints completed. Its partial, non-scientific calibration
tree was preserved and marked `stopped_by_user`. The optimizer source was
restored exactly to frozen hash:

```text
367acfaba8f6d0da660fac45ace5c0c696f705bbdb05b60d2072b8724b87cbd6
```

No new objective change is proposed from this diagnostic.
