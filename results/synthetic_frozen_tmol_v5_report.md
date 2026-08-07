# Frozen matched-environment tmol audit

**Date:** 2026-07-24

## Rule versions

```text
geometry/audit:
2026-07-24-altloc-minstate-rotwidth-matched-tmol-v1

tmol environment:
frozen_matched_deposited_minstate_v1
```

This report supersedes the invalid provisional 237/1000 calculation. It does
not modify optimizer endpoints.

## Environment-contamination diagnostic

Before freezing environments, a stratified sample compared each endpoint's
geometry-selected neighboring states with its matched deposited A/B reference.

```text
divergent environments: 30 / 293 endpoints (10.2%)
```

The divergence was concentrated:

| Site | Divergent / sampled |
|---|---:|
| 3GMI GLU5 | 19 / 20 |
| 2VFP TYR417 | 10 / 20 |
| 7UO8 GLN53 | 1 / 20 |
| Other 12 expanded-panel sites | 0 / 233 |

Candidate-specific state selection was therefore a material, site-dependent
contaminant.

## Fix

Each site now contains two immutable tmol base segments:

- deposited-A coordinates select the frozen A environment;
- deposited-B coordinates select the frozen B environment.

Every A-assigned endpoint is scored in the exact deposited-A base and compared
with deposited A scored in that same base. B uses the corresponding frozen-B
base. The environment offset therefore cancels exactly within each matched
comparison. Unmatched active endpoints remain tmol-invalid.

All controls and endpoints were regenerated from scratch. No legacy tmol rows
were copied. The four audit panels contain 2,889 newly scored rows in total,
all labeled `frozen_matched_deposited_minstate_v1`.

## Tmol versus clash-audit scope

The two gates now use the same min-over-altloc selection rule for their
overlapping local protein residues, but they do not score the same complete
environment:

- tmol uses a chemically complete contiguous same-chain protein segment,
  normally +/-12 residues;
- the direct clash audit includes all model heavy atoms, other chains, and
  occupancy-aware waters;
- the symmetry audit additionally includes crystallographic mates;
- tmol currently contains neither waters nor explicit crystallographic mates.

Thus the gates are complementary, not interchangeable. A symmetry or water
clash can correctly be absent from tmol. The earlier 8FBE absolute energy shift
was caused by selecting a different local protein altloc state, not by adding
the complete clash-audit environment to tmol.

## 8FBE control

```text
old shared-base A                577.6722
old shared-base B                358.6232
old A-B gap                      219.0490

revised frozen-A control         372.8804
revised frozen-B control         152.3279
revised matched A-B gap          220.5526
```

The relative gap remains, but the large absolute shifts confirm why frozen
matched environments are required.

## Final zero-tolerance composite

The authoritative replacement-aware 20-site synthetic composite is:

```text
Both conformers found:        673 / 1000
+ occupancy +/-0.20:          653 / 1000
physically valid independent: 270 / 1000
strict joint success:         251 / 1000
```

Comparison:

| Audit | Strict |
|---|---:|
| Original better(A,B)+10, pre-rule-fix | 494 / 1000 |
| Invalid copied-energy intermediate | 237 / 1000 |
| Frozen matched environment, delta <= 0 | 251 / 1000 |

The +14 change from 237 to 251 is removal of an audit artifact, not an
optimizer improvement. The 494 and 251 values use materially different
physical rules and must not be compared as model progress.

## Failure-margin histogram

There are 588 finite A/B-matched conformers failing the exact zero tmol gate.

| Positive margin | Failures |
|---|---:|
| (0, 1] | 351 |
| (1, 2] | 80 |
| (2, 5] | 127 |
| (5, 10] | 20 |
| >10 | 10 |

Median failure margin is 0.849. Of all finite matched failures:

- 59.7% are <=1;
- 73.3% are <=2;
- 94.9% are <=5;
- 98.3% are <=10.

For 442 endpoints within 0.1 A RMSD of their matched deposited conformer, the
positive-side 99th percentile is 0.438 and the maximum is 0.473. This supports
testing a +0.5 tolerance as a ground-truth-reproduction-scale threshold rather
than treating every positive floating-point/near-reproduction margin as a hard
failure.

Sensitivity without overwriting the zero-threshold audit:

| Matched tmol tolerance | Physical independent | Strict joint |
|---:|---:|---:|
| 0.0 | 270 | 251 |
| 0.5 | 396 | 372 |
| 1.0 | 446 | 409 |
| 2.0 | 497 | 460 |
| 5.0 | 594 | 546 |
| 10.0 | 605 | 556 |

## Stage-2 symmetry check

The current 3GMI calibration generates:

```text
invariant symmetry atoms          9
alternate symmetry groups         8
deposited-A symmetry loss          0.010527
deposited-B symmetry loss          0
deliberate-overlap probe loss      5.558888
```

The symmetry environment is populated and the soft term is capable of a
nonzero Stage-2 loss. The prior zero values on the four hard-tail endpoint
groups do not demonstrate missing mate generation; those saved endpoints
happened to lie outside the soft-overlap region under the recorded run.

Calibration root:

```text
/home/dev/qfit_unet_data/density_denoiser/symmetry_stage2_probe_3gmi_v1
```

## Authoritative pod roots

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_audit_rules_v5_frozen_tmol_aligned
/home/dev/qfit_unet_data/density_denoiser/heldout_three_four_chi_200step_synthetic_audit_rules_v5_frozen_tmol_aligned
```
