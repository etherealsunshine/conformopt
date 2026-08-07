# A' follow-up report: 7UTC leakage-corrected CV and 6P2N joint-QP rerun

## Status and essential caveat

Both jobs completed, but neither is a final scientific result. The newly added
in-loop joint QP constrained weights to be non-negative (and gave slot 2 a
temporary 0.02 floor) but **omitted qFit's total-occupancy constraint**
`sum(w) <= 1`. It is visible in
[`run_d1_aprime_sequential.py`](../../scripts/run_d1_aprime_sequential.py#L163-L174).
qFit explicitly requires the constraint in
[`solvers.py`](../../external/qfit-3.0/src/qfit/solvers.py#L80-L85) and its
CVXPY implementation includes it at
[`solvers.py`](../../external/qfit-3.0/src/qfit/solvers.py#L291-L295).

Consequently, the coordinate trajectories were optimized against a relaxed,
unphysical occupancy objective. The final reported occupancies are valid qFit
QPs, but they do not repair the coordinate optimization that preceded them.
The technical correction is to add `sum(w) <= 1` to the in-loop QP. The
decision taken after these results is narrower: rerun **6P2N
only**; do not rerun the 7UTC five-fold comparison because its beat-deposited
claim is dropped. No comparison below should be used as a claim about the
physical constrained objective.

## 1. 7UTC A:ARG52 — five-fold leakage-corrected optimization

### Design

The five pre-existing blocked spatial splits were reproduced exactly. Every
fold used 1,231 training voxels and 308 held-out voxels from the 1,539-voxel
mask. Both slot geometries were optimized on its own training voxels only.
For each held-out comparison, occupancies were refit only on that fold's
training voxels, with deposited A B factors on slot 1/A and deposited B B
factors on slot 2/B.

The paired statistic is `RSS(deposited A+B) - RSS(split-trained A' pair)` on
held-out voxels. Positive values would favour the learned A' pair.

| Fold | Split-trained RSS | Deposited A+B RSS | Paired difference | Slot 1 RMSD to A/B (A) | Slot 2 RMSD to A/B (A) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 43.227 | 37.732 | -5.495 | 0.897 / 0.947 | 0.835 / 1.130 |
| 1 | 31.053 | 32.792 | +1.739 | 0.794 / 0.681 | 0.702 / 0.799 |
| 2 | 23.863 | 27.531 | +3.668 | 0.685 / 0.772 | 1.011 / 0.759 |
| 3 | 64.517 | 52.486 | -12.031 | 0.385 / 1.259 | 0.613 / 1.294 |
| 4 | 41.721 | 42.647 | +0.926 | 0.824 / 0.782 | 0.851 / 1.010 |

**Result under the relaxed inner QP:** mean paired difference **-2.238 +/-
6.466 RSS**, with mixed signs (two folds favour deposited A+B and three favour
the split-trained pair). The prior leakage-prone result, `+5.96 +/- 1.45` with
all five positive, is therefore not reproduced by a fully split-trained fit.

The geometry is also unstable across folds: final slot-1 RMSD to A ranges from
0.385 to 0.897 A, and slot-2 RMSD to B from 0.759 to 1.294 A. This supports the
interpretation that this map/split design does not determine a stable two-slot
backbone solution.

Primary artifacts on the pod:

```text
/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_leakage_corrected_cv_v1/
  summary.json
  per_split.csv
  split_0/ ... split_4/
```

## 2. 6P2N A:GLY161 — sequential occupancy-fix rerun

### Design

The run used the requested sequential protocol: fit slot 1; freeze its
geometry; fit slot 2 while jointly re-fitting both occupancies at every inner
evaluation; apply a temporary slot-2 floor of 0.02 and release it for the final
qFit QP. It used 20 torsional parameters per slot (14 phi/psi plus 6 omega),
six augmented-Lagrangian outer updates, and 80 inner evaluations per update.

### Result

| Quantity | Value |
| --- | ---: |
| Resolution | 1.350 A |
| Mask voxels | 631 |
| Deposited A-to-B central backbone RMSD | 1.667 A |
| Deposited occupancies A/B | 0.89 / 0.11 |
| Final physical QP occupancies, slot 1/slot 2 | 0.844 / 0.156 |
| Final physical QP RSS | 88.250 |
| Slot 1 RMSD to A / B | 0.294 / 1.655 A |
| Slot 2 RMSD to A / B | 1.264 / 2.366 A |
| Verdict | PARTIAL |

Slot 1 recovered the deposited-A state to 0.294 A. The floor did keep slot 2
from a zero-weight degeneracy during its stage, but slot 2 did not approach B;
its final distance to B was 2.366 A. The final physical QP nevertheless gave
the second slot 0.156 occupancy, close in magnitude to the deposited 0.11.

The caveat is concrete here: the final stage-state weights were 0.8227 and
0.3378 (sum **1.1606**), demonstrating that the relaxed inner QP was used.
The final qFit QP subsequently enforced a total of 1.000, but its valid final
weights cannot make the optimized slot-2 geometry a constrained optimum.

Primary artifact on the pod:

```text
/home/dev/qfit_unet_data/qfit_audit/d1_aprime_6p2n_joint_qp_slot2_v4/result.json
```

## Required correction before comparison or scaling

Change the inner constraint from only `weights >= lower` to
`weights >= lower` **and** `sum(weights) <= 1`; retain the slot-2 floor only
during the slot-2 stage and release it for the final QP. This was implemented
and a fresh 6P2N-only result was launched as
`d1_aprime_6p2n_joint_qp_slot2_v5_constrained`. The five 7UTC folds are not
being repeated.

## 3. 7UTC fold-to-fold geometry scatter (existing five-fold data)

This uses the central `N/CA/C/O` coordinates saved by the five completed 7UTC
folds, in their shared deposited-A frame (so no superposition was performed).
For each slot, the reported scatter is the RMS 3D sample positional standard
deviation across the four atoms and five folds:

| Slot | Fold-to-fold positional SD | Analytic positional sigma | Scatter / analytic sigma | Mean pairwise fold RMSD |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.355 A | 0.398 A | 0.893 | 0.468 A |
| 2 | 0.337 A | 1.577 A | 0.214 | 0.451 A |

The folds are correlated subsamples, not independent realizations. Their mean
pairwise training-set overlap is **83.8%** (range 75.0–93.1%). Under the
simplifying independent-voxel/influence-function approximation, that overlap
would reduce fold scatter by `sqrt(1 - 0.838) = 0.403` relative to an
independent-resample SD. Applied mechanically to the analytic sigmas, that
would suggest 0.160 A (slot 1) and 0.635 A (slot 2).

The observed values are 0.355 A and 0.337 A. Slot 2 is substantially below its
analytic sigma, as expected for strongly overlapping folds; slot 1 is not as
reduced as the simplistic overlap calculation. Because the fits are nonlinear
and voxels are spatially correlated, this overlap calculation is only a
directional expectation. The five-fold scatter should be presented as a
correlated-subsample **lower-bound handle** on positional uncertainty, not as
an independent validation or a direct replacement for the Hessian sigma.

Artifact on the pod:

```text
/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_leakage_corrected_cv_v1/geometry_scatter.json
```

## 4. 6P2N constrained inner-QP rerun (v5; valid occupancy objective)

The rerun added `sum(w) <= 1` to every slot-2 objective and finite-difference
evaluation, while retaining the temporary 0.02 slot-2 floor. The floor was
released for the final QP. The final stage-state weights were **0.8110 / 0.1890**
(sum **1.0000**), confirming that the coordinate optimizer now saw the correct
occupancy budget throughout.

| Quantity | Constrained v5 |
| --- | ---: |
| Final QP occupancies, slot 1 / slot 2 | 0.811 / 0.189 |
| Final QP RSS | 87.336 |
| Slot 1 RMSD to A / B | 0.294 / 1.655 A |
| Slot 2 RMSD to A / B | 0.965 / 2.180 A |
| Verdict | PARTIAL |

The constrained-QP correction improved slot 2's distance to B from 2.366 A in
the relaxed v4 run to 2.180 A, but it remains far from B. Slot 1 again reached
A (0.294 A). Thus the 0.02 floor successfully prevents the zero-occupancy/
zero-gradient failure without recovering the minor deposited conformer at this
site under the constrained A' objective.

Primary artifact on the pod:

```text
/home/dev/qfit_unet_data/qfit_audit/d1_aprime_6p2n_joint_qp_slot2_v5_constrained/result.json
```
