# Residual-probe normalization follow-up

**Completed:** 2026-07-29  
**Diagnostic only:** yes  
**Production changed:** no  
**Metric changed:** no

## 1. Exact normalization

`five_site_optimizer.py` defines:

```python
def _normalize(values):
    return (values - values.mean()) / values.std().clamp_min(1e-6)
```

The production renderer first forms the selected radial density vector

```text
fixed_density + Σ softmax(logits)[k] × sidechain_density[k]
```

and then z-score normalizes that complete vector. Raw, denoised, and loaded
synthetic target patches are independently reduced to the same radial mask and
z-score normalized once at site setup; each blurred target is independently
normalized as well. For the production synthetic experiment, the loaded
synthetic vector is subsequently replaced by a native A/B control rendered
through the same normalized renderer.

Consequences:

- A positive scalar multiplying the *whole raw rendered vector* is erased.
- Scaling only the moving sidechain contribution is not generally erased,
  because fixed shared-atom density is present.
- The implemented occupancies are a softmax and therefore always sum to one.
  A common shift of all occupancy logits is an exact no-op. There is no
  parameter representing absolute sidechain occupancy; only slot ratios are
  identifiable.
- The matched A+B occupancy deficit is mass assigned to unmatched or sub-mask
  slots, not missing simplex mass.

## 2. Close-separation success travel

The 26 fixed-probe successes at sites with deposited local unsymmetrized A-B
separation at or below 2.5 A were replayed with identical seeds and settings,
adding diagnostic-only initial-to-final coordinate logging.

```text
median fixed-label travel       1.696 A
IQR                             1.299-2.418 A
travel <0.5 A                   0/26
travel <1.0 A                   2/26
successes at separation <=1.5  8/26
successes at separation <=2.0 18/26
```

| Site | Deposited A-B | Successes | Travel median | Travel range | Median travel/separation |
|---|---:|---:|---:|---:|---:|
| 2VFP TYR417 | 1.751 | 9 | 1.969 | 1.155-5.217 | 1.124 |
| 4C16 MET258 | 2.223 | 3 | 2.028 | 1.623-2.441 | 0.912 |
| 5KWB PHE591 | 1.880 | 1 | 0.916 | 0.916 | 0.487 |
| 5Z8H MET730 | 1.230 | 7 | 1.770 | 0.941-3.992 | 1.439 |
| 7F72 MET103 | 1.408 | 1 | 2.616 | 2.616 | 1.858 |
| 7T7A LEU396 | 2.467 | 4 | 1.358 | 1.285-1.511 | 0.551 |
| 8FBE ILE92 | 2.344 | 1 | 1.775 | 1.775 | 0.757 |

Close sites contribute 18 of the 26 successes at separation at or below
2.0 A, so the fixed 1.0 A recovery threshold and site composition contribute
to the observed separation anti-correlation. It is not, however, merely
trivial threshold crossing: no successful slot moved less than 0.5 A and only
two moved less than 1.0 A.

## 3. Site properties

Only 14 sites had frozen-v3 starts eligible for this missed-minor probe.
Altloc-group columns use the unique-group soft-VDW census, not repeated audit
records. `Active` means nonzero soft VDW against at least one state at
least one frozen endpoint; `sensitive` means materially different soft-VDW
values across states.

| Site | Probe | A-B A | Minor occ | chis | Minor residual +integral | Altloc groups total/active/sensitive | Resolution A |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1ZV8 ASN1 | 1/20 | 3.116 | .33 | 2 | 6.747 | 3/1/1 | 1.94 |
| 2V05 HIS168 | 1/4 | 2.854 | .39 | 2 | 5.018 | 10/0/0 | 1.70 |
| 2VFP TYR417 | 9/31 | 1.751 | .42 | 2 | 9.386 | 60/7/7 | 1.55 |
| 3A1C ARG447 | 0/1 | 5.609 | .44 | 4 | 7.260 | 1/1/1 | 1.85 |
| 3GMI GLU5 | 8/9 | 4.354 | .23 | 3 | 14.811 | 23/8/6 | 1.91 |
| 3NY7 LYS19 | 0/1 | 3.114 | .45 | 4 | 5.704 | 27/2/2 | 1.92 |
| 4C16 MET258 | 3/7 | 2.223 | .37 | 3 | 5.507 | 10/2/2 | 1.93 |
| 5DBA TRP325 | 1/5 | 2.811 | .45 | 2 | 22.004 | 24/8/6 | 1.96 |
| 5KWB PHE591 | 1/1 | 1.880 | .44 | 2 | 6.031 | 31/2/2 | 1.91 |
| 5Z8H MET730 | 7/26 | 1.230 | .28 | 3 | 3.237 | 20/4/3 | 1.79 |
| 7F72 MET103 | 1/3 | 1.408 | .48 | 3 | 4.527 | 22/5/5 | 1.64 |
| 7T7A LEU396 | 4/5 | 2.467 | .38 | 2 | 9.950 | 2/1/1 | 1.79 |
| 7UO8 GLN53 | 1/15 | 4.688 | .34 | 3 | 8.930 | 29/10/9 | 1.60 |
| 8FBE ILE92 | 1/1 | 2.344 | .38 | 2 | 12.974 | 19/2/2 | 1.73 |

No measured scalar explains the 3GMI/7T7A versus 1ZV8/7UO8 contrast. It does
not track separation, minor occupancy, chi count, resolution, residual
integral, or altloc-group burden. In particular, 7UO8 has a strong residual
integral and many contested groups but 1ZV8 fails with only one contested
group; 3GMI succeeds despite six state-sensitive groups. The defensible
description is a residue/site-specific residual landscape, not a discovered
general predictor.

## 4. Correct interpretation of probe v1

Probe v1 optimized against

```text
zscore(target raw density) - zscore(frozen endpoint raw density)
```

using a separately normalized single-conformer footprint. Because z-scoring
is nonlinear with respect to addition, that difference is not an additive
missing-conformer density. Fixed additions can therefore worsen the stated
residual by construction, and the free probe's median occupancy of 0.006 is
not interpretable as production evidence that the map wants the minor slot
off.

The competition hypothesis is **neither supported nor refuted**. The observed
38/129 fixed-probe recovery is retained as a lower bound for this mismatched
probe, not an estimate of a correctly formulated sequential method.

A corrected no-competition probe must keep the frozen endpoint in raw density
space, add the new slot before normalization, and optimize

```text
MSE(zscore(frozen_raw + occupancy * new_slot_raw), target_zscore)
```

with gradients disabled for the frozen endpoint but with the complete sum
inside the same renderer and normalization used by production.

## Pod artifacts

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_remaining_residual_minor_probe_v1/normalization_followup_v2/
```

The directory contains the 26-row travel table, the per-site travel summary,
the corrected property table, and the JSON summary.
