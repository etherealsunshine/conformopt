# 7UTC A′: analytic density positional uncertainty

**Status:** complete (`v3` result root). This is a fixed-geometry local-Gaussian analysis of the
completed `7UTC A:ARG52` A′ fit. No coordinate optimisation was performed.

## Main result

The density-only propagated central-backbone RMS positional sigmas are **0.398
Å** for slot 1 and **1.577 Å** for slot 2. They agree with the pre-registered
brute-force blocked-CV ranges (0.38 Å and 1.46 Å) to **4.6%** and **8.0%**,
respectively. This validates the **marginal covariance scale** of the local
density-Hessian machinery at the two recovered occupancies from this site.

| slot | joint QP occupancy | analytic central RMS sigma (Å) | brute-force band (Å) |
|---|---:|---:|---:|
| 1 | 0.4694 | 0.398 | 0.38 |
| 2 | 0.1569 | 1.577 | 1.46 |

The analytic ratio is **3.97**, close to the measured **3.87** and greater
than the occupancy ratio **2.99**. The low occupancy is therefore an important
but incomplete explanation of slot 2's uncertainty.

### Held-out RSS-rise conversion

For the requested RSS-rise conversion, the quadratic curvature was evaluated
separately on each of the same five held-out spatial slabs (not from the
full-mask Hessian). It profiles both QP occupancies at every slab. The slot-1
quadratic one-SD range is 0.533 Å, versus the brute-force one-SD range of
0.381 Å: it is directionally conservative by about 40%. Slot 2's quadratic
accepted set is non-contiguous over the sampled torsion path; its outer
envelope is 1.242 Å, close to the brute-force one-SD range of 1.220 Å on that
same convention. Thus the marginal sigma validation is successful, while a
single local quadratic contour should not be presented as an exact replacement
for the finite-path scan, particularly for slot 1.

## Definition and conditioning

The joint parameter vector contains 20 torsions for each slot and both interior
QP occupancies. The density Hessian is `H = JᵀJ`: qFit's CVXPY QP and the A′
density residual use unweighted masked-voxel RSS after the run's map scaling,
so `W = I` over 1,539 mask voxels. The original A′ rendering is retained:
both slots use the deposited-A B-factor array.

The unregularized Hessian is singular to numerical precision (minimum
eigenvalue `−3.62e−13`, maximum `1045.57`). A Tikhonov ridge of `1.046e−7`
(`1.0e−10 × lambda_max`) was added before inversion, giving condition number
`1.0e10`. This regularization is deliberately reported: unresolved directions
cannot be treated as data-determined.

The occupancy covariance is

```text
[[ 0.004335, -0.004296],
 [-0.004296,  0.006763]]
```

The negative occupancy correlation is approximately −0.79, confirming that
profiling both QP occupancies matters for positional uncertainty.

## Directionality

The collective central-backbone leading covariance direction is not strongly
aligned with the deposited A→B displacement (absolute cosine 0.150 for slot 1,
0.328 for slot 2). At the carbonyl scale, however, the long axis is aligned:
the O-atom alignment is 0.689 and 0.985 for slots 1 and 2; C is 0.625 and
0.903. Thus the uncertainty is anisotropic locally around the carbonyl, but it
is not a single global central-backbone A→B mode.

## Density versus A′ restraints

The reported sigmas use the **density-only** Hessian. The actual sequential A′
stage curvature has a material restraint contribution:

| slot | penalty / density trace | penalty / density Frobenius norm |
|---|---:|---:|
| 1 | 1.63× | 1.64× |
| 2 | 299.8× | 309.1× |

Consequently, a covariance taken from the full seam+Rama+omega A′ objective
would be restraint-dominated, especially for slot 2; it should not be called
positional uncertainty from density. The density-only result above is the
appropriate comparison with the blocked-CV density bands.

## Artifacts

Pod result root:
`/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_analytic_uncertainty_v3`

- `result.json`: full Hessian, conditioning, occupancy covariance, principal
  axes, and per-fraction directional calculations.
- `per_atom.csv`: per-atom Cartesian and principal positional sigmas.
