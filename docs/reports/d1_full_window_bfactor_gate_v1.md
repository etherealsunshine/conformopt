# A'' full-window B-factor gate

## Scope

This note records the deposited-geometry gate after correcting A'' slot B-factor
assignment. It is a fixed-geometry measurement only: no coordinate optimisation
was run.

## B-factor assignment

The prior window renderer passed the deposited-A B-factor vector to both slots.
That was corrected so slot 1 uses deposited A and slot 2 uses deposited B, in
the same model-atom order. A global, site-wide `delta_B` is profiled with the
affine occupancies and intercept; it is reported in A^2 as a map/model-width
diagnostic.

At 7T7A A:LEU396, direct rendering verifies that slot B is identical to an
explicit deposited-B-vector render (maximum absolute difference 0.0). It
differs from the old deposited-A-vector render by up to 0.005744 density units.

## 7T7A A:LEU396 deposited gate

Configuration: corrected seven-residue backbone mask (1,610 voxels),
neighbour subtraction outside the whole window, Torch renderer, deposited
occupancies fixed at 0.38 / 0.62 while calibrating `delta_B`.

| quantity | value |
|---|---:|
| fitted global `delta_B` | +6.5295 A^2 |
| profile interval | +6.5286 to +6.5304 A^2 |
| fixed-occupancy intercept | -0.02963 |
| fixed-occupancy RSS | 14.2343 |
| residual / calculated-density correlation | -0.1873 |
| re-solved A / B occupancies at that `delta_B` | 0.0000 / 0.9322 |
| re-solved occupancy total | 0.9322 |
| re-solved intercept / RSS | +0.03009 / 13.2078 |

### Fixed-geometry model selection

All models below profile their own global `delta_B`, occupancies, and intercept
unless occupancy is explicitly fixed. BIC uses `n = 1,610`, with three free
parameters for one conformer (`w`, intercept, `delta_B`) and four for two.

| model | occupancies | `delta_B` (A^2) | c | RSS | BIC |
|---|---:|---:|---:|---:|---:|
| deposited B only | 0.8365 | +2.1946 | +0.07845 | 12.5184 | **-7797.28** |
| deposited A only | 0.8214 | +1.9507 | +0.09091 | 14.2253 | -7591.48 |
| deposited A/B, fixed 0.38/0.62 | 0.38 / 0.62 | +6.5295 | -0.02963 | 14.2343 | -7583.08 |
| deposited A/B, free | 0.0000 / 0.8365 | +2.1946 | +0.07845 | 12.5184 | -7789.90 |

The deposited-B-only and free two-column fits have identical RSS because the
two-column QP sets A exactly to zero. BIC therefore selects deposited B only,
by 7.38 BIC units.

Five blocked spatial folds refit all free parameters on the training voxels.
Held-out RSS for B-only was 3.2627, 2.8904, 1.8352, 2.9853, and 2.0570. The
two-column model produced the identical values in every fold (paired
two-column minus B-only difference: 0 to numerical precision).

Thus this gate does not support a two-conformer density model at the deposited
geometry. No A'' geometry optimisation should be reopened on the basis of this
site.

## 5OHJ A:SER540 panel decision

5OHJ is excluded from the working panel pending the broader fixed-geometry
screen. A fresh full-structure MapScaler measurement is 0.947 (the previously
quoted 0.658/`b=0.717` belongs to 6P2N, not 5OHJ). Under the fixed-deposited
B-offset diagnostic, 5OHJ nevertheless required `delta_B = +119.75 A^2` at
1.60 A resolution. That is not a plausible deposited-versus-map B-factor
refinement difference.

This exclusion is a site-quality decision, not evidence about the optimizer.
Earlier 5OHJ occupancy and geometry comparisons should not be used as working
panel evidence.
