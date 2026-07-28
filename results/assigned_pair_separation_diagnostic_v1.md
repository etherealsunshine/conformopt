# Assigned-Pair Separation Diagnostic

**Date:** 2026-07-28
**Population:** all 742 starts counted as recovering both states under
`qfit-synth20-merge050-one-to-one-tmol044-v3`
**Production rules changed:** none

## Result

Assigned A/B separation is not systematically compressed across the panel:

```text
assigned / deposited separation median      0.9853
IQR                                          0.9555–1.0311
assigned − deposited median                 −0.0294 A
IQR                                         −0.1048 to +0.0602 A
assigned separation < half deposited        8 / 742
```

All eight `<0.5×` cases are 2VFP TYR417. Thus the concern is not panel-wide,
but it is material at the site whose A/B states are closest.

### 5KWB PHE591

```text
deposited A–B separation                    0.6170 A
assigned median                             0.5966 A
assigned IQR                                0.5942–0.5990 A
assigned range                              0.5596–0.6029 A
assigned / deposited median                 0.9669
starts below half deposited separation      0 / 49
```

5KWB is genuine under this diagnostic. Its two assigned conformers reproduce
the deposited separation rather than double-counting one geometry.

### 2VFP TYR417

```text
deposited A–B separation                    0.5668 A
assigned median                             0.2260 A
assigned IQR                                0.1978–0.5074 A
assigned / deposited median                 0.3988
starts below half deposited separation      8 / 14
```

The fourteen assigned separations comprise eight at 0.190–0.230 A, five at
0.463–0.514 A, and one at 0.978 A. The first eight are consistent with one
geometry represented by two protected slots. The protected merge kept these
A/B anchors separate by construction, so the current recovery definition
cannot detect that duplication. This is a concentrated known limitation of
the v3 recovery count, not evidence against the other 19 sites.

All eight compressed pairs pass the occupancy and complete strict cascade.
They therefore contribute 8/626 v3 strict successes, not merely 8/742
recovery labels. Removing them diagnostically would give 734 both-found and
618 strict, but no production metric count is changed by this diagnostic.

## RMSD symmetry equivalence

All production recovery, merge, and this diagnostic use conventional
heavy-atom RMSD minimized over valid equivalent terminal-atom permutations.
The implemented swaps are:

```text
ARG NH1/NH2
ASP OD1/OD2
GLU OE1/OE2
LEU CD1/CD2
PHE CD1/CD2 together with CE1/CE2
TYR CD1/CD2 together with CE1/CE2
VAL CG1/CG2
```

## 8Q6Q ASP81 terminal-swap recomputation

The previously reported 0.975 A within-group median already included the
ASP OD1/OD2 swap. Explicit identity-versus-swap recomputation over all 50
same-state pairs gives:

| RMSD treatment | Median | IQR |
|---|---:|---:|
| Fixed OD1/OD2 labels | 1.110 A | 0.935–1.284 A |
| Minimized over OD1/OD2 swap | **0.975 A** | **0.899–1.073 A** |

The swap changes 19/50 pair values, but only 5/50 fall below 0.25 A afterward.
The distribution does not collapse toward zero. Most 8Q6Q same-state slots
are geometrically distinct endpoints sharing the same broad `<1 A` deposited
state label, not terminal-label duplicates. The v3 0.5 A merge therefore
correctly does not restore v1's blanket occupancy summation at that site.

## Artifacts

```text
remote detailed output
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_water_minstate_v2_single_rule_v1/
analysis/metric_v3_protected_merge_sweep/assigned_pair_separation_v1/

per-start assigned pair table
.../recovered_pair_separation.csv

8Q6Q identity/swap pair table
.../8q6q_same_state_swap_check.csv
```

The local per-site table is
`results/assigned_pair_separation_diagnostic_v1.csv`.
