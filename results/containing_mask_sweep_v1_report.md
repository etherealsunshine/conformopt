# Containing-Mask Sweep v1

**Date:** 2026-07-29  
**Frozen metric:** `qfit-synth20-merge050-one-to-one-tmol044-v3`  
**Metric changed:** no

## Result

The containing uniform mask was aggregate-neutral and the variance-weighted
containing mask was harmful.

| Arm | Both found | Occupancy | Rotamer | Direct | Symmetry | Strict |
|---|---:|---:|---:|---:|---:|---:|
| Frozen v3 control | 742 | 714 | 710 | 710 | 710 | 626 |
| F: containing, uniform | 741 | 718 | 713 | 713 | 713 | 622 |
| G: containing, variance weighted | 690 | 669 | 665 | 665 | 665 | 591 |

The raw optimizer minor/major single-state miss split was:

| Arm | Minor missed | Major missed |
|---|---:|---:|
| Frozen v3 control | 142 | 45 |
| F | 139 | 63 |
| G | 161 | 66 |

F therefore exchanged three fewer minor misses for eighteen more major misses.
G worsened both populations. Neither arm should replace production.

The five predeclared tail sites changed in both-found count as follows:

| Site | Control | F | Delta F | G | Delta G |
|---|---:|---:|---:|---:|---:|
| 1ZV8 ASN1 | 1 | 2 | +1 | 0 | -1 |
| 2VFP TYR417 | 14 | 25 | +11 | 35 | +21 |
| 4C16 MET258 | 20 | 16 | -4 | 3 | -17 |
| 5Z8H MET730 | 16 | 14 | -2 | 8 | -8 |
| 7UO8 GLN53 | 18 | 15 | -3 | 12 | -6 |

F increased the median deposited major-collapse signal from `0.0988` to
`0.1168` while leaving aggregate recovery effectively unchanged. Both new
masks contained every enumerated reachable atom position and every deposited
A/B atom.

## Variance-weight concentration

The implemented weighting is **per voxel**, not per atom. Consequently there
is no exact “fraction of total weight on one atom”: atomic density footprints
overlap, and the variance contains inter-atom covariance. Assigning voxels to
atoms would require an additional arbitrary partitioning rule.

The exact stored concentration statistic is the fraction of total voxel
weight carried by the single highest-weighted voxel. Weights are normalized
to mean one, so this is `weight_max / voxel_count`.

| Site | Voxels | Min weight | Max weight | Highest voxel / total |
|---|---:|---:|---:|---:|
| 1ZV8 ASN1 | 421 | 0.00129 | 3.283 | 0.780% |
| 2V05 HIS168 | 859 | 0.000455 | 3.388 | 0.394% |
| 2VFP TYR417 | 736 | 1.40e-12 | 37.112 | 5.042% |
| 3A1C ARG447 | 4,982 | 0.01298 | 3.255 | 0.065% |
| 3GMI GLU5 | 930 | 7.13e-7 | 9.839 | 1.058% |
| 3K8W SER337 | 133 | 8.85e-5 | 2.954 | 2.221% |
| 3NY7 LYS19 | 2,435 | 0.000385 | 8.817 | 0.362% |
| 4C16 MET258 | 1,276 | 3.42e-5 | 8.912 | 0.698% |
| 4MKM THR77 | 145 | 0.00404 | 3.713 | 2.561% |
| 5DBA TRP325 | 1,987 | 3.30e-6 | 4.427 | 0.223% |
| 5KWB PHE591 | 615 | 1.23e-8 | 8.989 | 1.462% |
| 5Z8H MET730 | 1,219 | 5.75e-5 | 9.576 | 0.786% |
| 6H59 ARG144 | 4,981 | 0.000466 | 17.799 | 0.357% |
| 6Y4G CYS260 | 151 | 3.03e-5 | 5.321 | 3.524% |
| 7F72 MET103 | 1,240 | 0.00135 | 4.837 | 0.390% |
| 7T7A LEU396 | 457 | 0.000574 | 4.789 | 1.048% |
| 7UO8 GLN53 | 912 | 1.67e-5 | 8.623 | 0.946% |
| 8DJ2 VAL893 | 183 | 0.00108 | 4.233 | 2.313% |
| 8FBE ILE92 | 438 | 0.00438 | 4.329 | 0.988% |
| 8Q6Q ASP81 | 354 | 0.000278 | 3.160 | 0.893% |

2VFP is the clear concentration outlier: one voxel carries `5.04%` of all
site weight, versus less than `3.53%` everywhere else. This is consistent
with G's strong site-specific 2VFP response, but it does not establish an
atom-level causal attribution.

For 2VFP, a local reconstruction placed that maximum at
`(26.8196, -2.1170, 1.0317) Å`. It is `0.133 Å` from `CE2` in the reachable
canonical TYR state with physical chis approximately `(-180°, +90°)`.
Therefore it is a ring-atom location in the **enumerated reachable-state**
sense. It is not near the deposited ring positions: its closest deposited
ring atoms are A/CD1 at `4.908 Å` and B/CD2 at `5.255 Å`. Voxels within
`1 Å` of any deposited A/B CD1/CD2/CE1/CE2 atom collectively carry `20.28%`
of total site weight.

The crystal-frame mask center is the mean position of all target-sidechain
heavy-atom records, not C-alpha. For 2VFP it is
`(24.5696, -5.8670, -2.7183) Å`. An earlier conversational statement that the
production sphere was C-alpha-centered was incorrect.

## Corrected control provenance and reporting bug

The first compiled comparison used control audits from:

```text
analysis/metric_v3_merge_sweep/0p5/
```

Those are the unprotected/stale matching outputs. The authoritative frozen-v3
control is:

```text
analysis/metric_v3_protected_merge_sweep/0p5/
```

The first `summary.json` hard-coded the correct headline `742/626`, while its
nested `arms.production_control` object was computed from stale rows and read
`733/618`. The arm audits themselves were loaded correctly.

Affected outputs in the first compilation:

- `summary.json`:
  - top-level `control.found=742` and `control.strict=626` were correct;
  - every experimental-arm cascade and diagnostic was correct;
  - audit-dependent fields under `arms.production_control` were stale:
    cascade, frozen-v3 miss split, duplication, unmatched extras, and matched
    occupancy;
  - control optimizer-only fields were unaffected: raw minor/major misses and
    Stage-2 physics-loss summaries.
- `cascade_by_site.csv`:
  - every `production_control` row came from the stale audit provenance;
  - only 2VFP and 8Q6Q differ numerically between the stale and authoritative
    control cascades; the other site rows happen to agree;
  - all F and G rows were correct.
- `tail_site_deltas.csv`:
  - every delta was derived from the wrong control provenance;
  - among the five predefined tail sites, only the 2VFP row changes
    numerically because the other four have identical stale and v3 cascades;
  - corrected tail deltas are represented in the table above.
- `minor_major_misses_by_site.csv`:
  - `production_control/frozen_v3` rows were stale;
  - `production_control/raw_optimizer` and all F/G rows were correct.
- `geometry_losses_by_site.csv` was unaffected because it reads optimizer
  endpoint rows rather than audit assignments.
- `realized_signal_by_site.csv` was unaffected because it contains only F/G
  calibration and mask data.

The corrected remote output is versioned separately:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_containing_mask_sweep_v1/
analysis/frozen_v3_comparison_v2_corrected
```

The original erroneous output is preserved for provenance and must not be
used:

```text
.../analysis/frozen_v3_comparison_v1
```

The local analyzer now includes a frozen-control cascade assertion so a
control other than `742→714→710→710→710→626` is rejected instead of silently
mixed into a report. The local change could not be synchronized during this
session because the remote-sync approval was unavailable; the corrected v2
tables themselves were generated successfully using the unmodified analyzer
with the authoritative audit roots.
