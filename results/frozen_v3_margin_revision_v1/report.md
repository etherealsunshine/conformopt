# Frozen-v3 deposited-candidate margin revision

Frozen control guard: **742 → 714 → 710 → 710 → 710 → 626**.

Coverage margin is the primary quantity. It depends only on deposited candidate densities: `RSCC(correct A+B) - RSCC(best single-state candidate)`.

Across all 20 sites, Spearman correlation between local fixed-label A–B separation and coverage margin is **ρ = 0.3549**.

The prior σ values remain in `per_site_margin_revision.csv` only for provenance. Their denominator is failed-endpoint scatter, so they are confounded by how consistently a site fails and are not interpreted.

| Site | Separation Å | Correct RSCC | Best coverage wrong | Coverage margin | Best occupancy wrong | Occupancy margin |
|---|---:|---:|---|---:|---|---:|
| 5Z8H | 1.230 | 1.000000 | B_alone (0.990892) | 0.009108 | A0.25_B0.75 (0.999891) | 0.000109 |
| 7F72 | 1.408 | 1.000000 | B_alone (0.976559) | 0.023441 | A0.50_B0.50 (0.999958) | 0.000042 |
| 2VFP | 1.751 | 1.000000 | B_alone (0.929227) | 0.070773 | A0.50_B0.50 (0.997050) | 0.002950 |
| 6Y4G | 1.784 | 1.000000 | B_alone (0.826335) | 0.173665 | A0.50_B0.50 (0.994799) | 0.005201 |
| 3K8W | 1.792 | 1.000000 | A_alone (0.886302) | 0.113698 | A0.50_B0.50 (0.999197) | 0.000803 |
| 5KWB | 1.880 | 1.000000 | B_alone (0.955598) | 0.044402 | A0.50_B0.50 (0.999089) | 0.000911 |
| 4MKM | 2.076 | 1.000000 | B_alone (0.948076) | 0.051924 | A0.50_B0.50 (0.997124) | 0.002876 |
| 4C16 | 2.223 | 1.000000 | B_alone (0.870597) | 0.129403 | A0.25_B0.75 (0.980190) | 0.019810 |
| 8Q6Q | 2.261 | 1.000000 | A_alone (0.839671) | 0.160329 | A0.50_B0.50 (0.993417) | 0.006583 |
| 8DJ2 | 2.312 | 1.000000 | A_alone (0.964770) | 0.035230 | A0.75_B0.25 (0.997258) | 0.002742 |
| 8FBE | 2.344 | 1.000000 | B_alone (0.937780) | 0.062220 | A0.50_B0.50 (0.992333) | 0.007667 |
| 7T7A | 2.467 | 1.000000 | B_alone (0.897739) | 0.102261 | A0.50_B0.50 (0.985088) | 0.014912 |
| 6H59 | 2.787 | 1.000000 | A_alone (0.861790) | 0.138210 | A0.75_B0.25 (0.959303) | 0.040697 |
| 5DBA | 2.811 | 1.000000 | A_alone (0.873928) | 0.126072 | A0.50_B0.50 (0.997936) | 0.002064 |
| 2V05 | 2.854 | 1.000000 | A_alone (0.877322) | 0.122678 | A0.50_B0.50 (0.984834) | 0.015166 |
| 3NY7 | 3.114 | 1.000000 | B_alone (0.869663) | 0.130337 | A0.50_B0.50 (0.997821) | 0.002179 |
| 1ZV8 | 3.116 | 1.000000 | B_alone (0.904664) | 0.095336 | A0.25_B0.75 (0.991843) | 0.008157 |
| 3GMI | 4.354 | 1.000000 | A_alone (0.959468) | 0.040532 | A0.75_B0.25 (0.999538) | 0.000462 |
| 7UO8 | 4.688 | 1.000000 | B_alone (0.903808) | 0.096192 | A0.25_B0.75 (0.990546) | 0.009454 |
| 3A1C | 5.609 | 1.000000 | B_alone (0.761930) | 0.238070 | A0.50_B0.50 (0.990617) | 0.009383 |

## Verification

- A duplicated at 0.5 + 0.5 was numerically identical to A alone at all 20 sites: zero density and RSCC discrepancy.
- Maximum raw-target reconstruction relative L2 error remained `3.858824e-06`.
- No density was re-rendered, no endpoint was read, and no σ or scatter value was recomputed.
