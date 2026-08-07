# Frozen-v3 occupancy-error curves

Frozen control guard: **742 → 714 → 710 → 710 → 710 → 626**.

This report reshapes the 80 saved occupancy-candidate RSCC values. It performs no density rendering, optimizer run, endpoint read, or metric recomputation.

## SampleWorks cross-check — discrepancy precedes interpretation

| Site | Occupancy error | qfit margin | SampleWorks margin | Relative difference |
|---|---:|---:|---:|---:|
| 3A1C | 0.06 | 0.009383 | 0.021700 | 56.8% |
| 3A1C | 0.19 | 0.071183 | 0.118300 | 39.8% |
| 3A1C | 0.31 | 0.214451 | 0.436500 | 50.9% |
| 3A1C | 0.46 | 0.374564 | 0.656300 | 42.9% |
| 5Z8H | 0.03 | 0.000109 | 0.000143 | 23.9% |
| 5Z8H | 0.22 | 0.005962 | 0.007800 | 23.6% |
| 5Z8H | 0.47 | 0.027082 | 0.035600 | 23.9% |
| 5Z8H | 0.62 | 0.046485 | 0.060800 | 23.5% |

**The saved qfit margins materially disagree with the supplied SampleWorks values (>10% relative difference), so the panel-wide qfit trend should not be treated as a numerical reproduction of that two-site measurement.**

## Pooled relationship

Across all 80 site–decoy pairs, Spearman correlation between occupancy error and RSCC margin is **ρ = 0.8630**.

| Occupancy-error bin | Pairs | Median margin |
|---|---:|---:|
| [0.0, 0.1) | 16 | 0.002461 |
| [0.1, 0.2) | 20 | 0.017635 |
| [0.2, 0.3) | 10 | 0.034646 |
| [0.3, 0.4) | 15 | 0.083351 |
| [0.4, 0.5) | 12 | 0.094789 |
| [0.5, 0.6) | 6 | 0.360039 |
| [0.6, 0.7) | 1 | 0.046485 |

## Smallest tested error exceeding the site's coverage margin

These are sampled-grid thresholds, not interpolated physical limits.

| Site | Coverage margin | Smallest tested error | Decoy | Margin at threshold | Status |
|---|---:|---:|---|---:|---|
| 1ZV8 | 0.095336 | 0.420 | A0.75_B0.25 | 0.256939 | observed_on_tested_grid |
| 2V05 | 0.122678 | 0.360 | A0.25_B0.75 | 0.150317 | observed_on_tested_grid |
| 2VFP | 0.070773 | 0.480 | A0.90_B0.10 | 0.094279 | observed_on_tested_grid |
| 3A1C | 0.238070 | 0.460 | A0.90_B0.10 | 0.374564 | observed_on_tested_grid |
| 3GMI | 0.040532 | 0.270 | A0.50_B0.50 | 0.108065 | observed_on_tested_grid |
| 3K8W | 0.113698 | not reached | none_tested | not reached | not_reached_on_tested_grid |
| 3NY7 | 0.130337 | 0.450 | A0.90_B0.10 | 0.149774 | observed_on_tested_grid |
| 4C16 | 0.129403 | 0.380 | A0.75_B0.25 | 0.220956 | observed_on_tested_grid |
| 4MKM | 0.051924 | 0.490 | A0.90_B0.10 | 0.079567 | observed_on_tested_grid |
| 5DBA | 0.126072 | not reached | none_tested | not reached | not_reached_on_tested_grid |
| 5KWB | 0.044402 | 0.460 | A0.90_B0.10 | 0.050858 | observed_on_tested_grid |
| 5Z8H | 0.009108 | 0.470 | A0.75_B0.25 | 0.027082 | observed_on_tested_grid |
| 6H59 | 0.138210 | not reached | none_tested | not reached | not_reached_on_tested_grid |
| 6Y4G | 0.173665 | 0.461 | A0.90_B0.10 | 0.232090 | observed_on_tested_grid |
| 7F72 | 0.023441 | not reached | none_tested | not reached | not_reached_on_tested_grid |
| 7T7A | 0.102261 | 0.370 | A0.75_B0.25 | 0.133122 | observed_on_tested_grid |
| 7UO8 | 0.096192 | 0.410 | A0.75_B0.25 | 0.218549 | observed_on_tested_grid |
| 8DJ2 | 0.035230 | 0.410 | A0.25_B0.75 | 0.058538 | observed_on_tested_grid |
| 8FBE | 0.062220 | 0.370 | A0.75_B0.25 | 0.070554 | observed_on_tested_grid |
| 8Q6Q | 0.160329 | not reached | none_tested | not reached | not_reached_on_tested_grid |

## Sampling limitation

The tested decoy grid is coarse and site-independent, so occupancy errors are unevenly sampled across sites. For example, 5Z8H has a decoy at 0.03 error, while 6H59's nearest tested error is 0.25. Plotting margin against the actual error mitigates the bar-chart comparability artifact but does not eliminate the sampling gap.

A finer occupancy grid is worth computing if a quantitative detectability threshold is needed: the current four-point grid can only bracket thresholds coarsely and leaves several sites with no tested decoy beyond their coverage margin. It is not needed to establish the qualitative monotonic pooled relationship.

## Verification

- Frozen metric: `qfit-synth20-merge050-one-to-one-tmol044-v3`.
- Candidate rows read: 80 occupancy decoys from `per_candidate.csv`.
- Density renders: 0; optimizer runs: 0; endpoint rows read: 0.
