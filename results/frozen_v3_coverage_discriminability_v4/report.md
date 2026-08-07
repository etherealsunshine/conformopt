# Frozen-v3 20-site coverage discriminability

Frozen control guard: **742 → 714 → 710 → 710 → 710 → 626**.

RSCCs use native additive density before z-scoring on the saved production Stage-1 mask. Scatter is the population standard deviation of RSCC across frozen-v3 endpoints that failed two-state recovery. Values with fewer than 5 failed endpoints are marked unreliable.

| Site | Sep. Å | Correct | Coverage best wrong | Margin | Failed n | Scatter | σ | Reliability | Occ. margin | Endpoint margin |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|
| 5Z8H_A_MET730 | 1.230 | 1.000000 | B_alone (0.990892) | 0.009108 | 34 | 0.040469 | 0.225 | reliable | 0.000109 | 0.005916 |
| 7F72_A_MET103 | 1.408 | 1.000000 | B_alone (0.976559) | 0.023441 | 5 | 0.001666 | 14.068 | reliable | 0.000042 | 0.012105 |
| 2VFP_A_TYR417 | 1.751 | 1.000000 | B_alone (0.929227) | 0.070773 | 36 | 0.200948 | 0.352 | reliable | 0.002950 | 0.029940 |
| 6Y4G_B_CYS260 | 1.784 | 1.000000 | B_alone (0.826335) | 0.173665 | 0 | — | — | unavailable_no_scatter | 0.005201 | — |
| 3K8W_A_SER337 | 1.792 | 1.000000 | A_alone (0.886302) | 0.113698 | 0 | — | — | unavailable_no_scatter | 0.000803 | — |
| 5KWB_A_PHE591 | 1.880 | 1.000000 | B_alone (0.955598) | 0.044402 | 1 | 0.000000 | — | unavailable_no_scatter | 0.000911 | 0.025482 |
| 4MKM_A_THR77 | 2.076 | 1.000000 | B_alone (0.948076) | 0.051924 | 0 | — | — | unavailable_no_scatter | 0.002876 | — |
| 4C16_A_MET258 | 2.223 | 1.000000 | B_alone (0.870597) | 0.129403 | 30 | 0.034036 | 3.802 | reliable | 0.019810 | 0.021903 |
| 8Q6Q_B_ASP81 | 2.261 | 1.000000 | A_alone (0.839671) | 0.160329 | 0 | — | — | unavailable_no_scatter | 0.006583 | — |
| 8DJ2_A_VAL893 | 2.312 | 1.000000 | A_alone (0.964770) | 0.035230 | 0 | — | — | unavailable_no_scatter | 0.002742 | — |
| 8FBE_B_ILE92 | 2.344 | 1.000000 | B_alone (0.937780) | 0.062220 | 1 | 0.000000 | — | unavailable_no_scatter | 0.007667 | 0.063998 |
| 7T7A_A_LEU396 | 2.467 | 1.000000 | B_alone (0.897739) | 0.102261 | 6 | 0.008283 | 12.346 | reliable | 0.014912 | 0.037015 |
| 6H59_B_ARG144 | 2.787 | 1.000000 | A_alone (0.861790) | 0.138210 | 9 | 0.011247 | 12.289 | reliable | 0.040697 | 0.015095 |
| 5DBA_A_TRP325 | 2.811 | 1.000000 | A_alone (0.873928) | 0.126072 | 13 | 0.049037 | 2.571 | reliable | 0.002064 | 0.078559 |
| 2V05_A_HIS168 | 2.854 | 1.000000 | A_alone (0.877322) | 0.122678 | 28 | 0.020079 | 6.110 | reliable | 0.015166 | 0.015202 |
| 3NY7_B_LYS19 | 3.114 | 1.000000 | B_alone (0.869663) | 0.130337 | 2 | 0.003329 | 39.151 | unreliable_sparse | 0.002179 | 0.041803 |
| 1ZV8_E_ASN1 | 3.116 | 1.000000 | B_alone (0.904664) | 0.095336 | 49 | 0.002546 | 37.449 | reliable | 0.008157 | 0.024825 |
| 3GMI_A_GLU5 | 4.354 | 1.000000 | A_alone (0.959468) | 0.040532 | 9 | 0.015611 | 2.596 | reliable | 0.000462 | 0.029988 |
| 7UO8_A_GLN53 | 4.688 | 1.000000 | B_alone (0.903808) | 0.096192 | 32 | 0.023368 | 4.116 | reliable | 0.009454 | 0.029669 |
| 3A1C_B_ARG447 | 5.609 | 1.000000 | B_alone (0.761930) | 0.238070 | 3 | 0.004659 | 51.096 | unreliable_sparse | 0.009383 | 0.050456 |

## Anchor comparison

- 3A1C_B_ARG447: qfit 51.0956σ; SampleWorks reference 9.40σ; failed endpoint n=3.
- 5Z8H_A_MET730: qfit 0.2251σ; SampleWorks reference 0.29σ; failed endpoint n=34.
