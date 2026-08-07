# A′ sequential PoC diagnostics — 7UTC A:ARG52

## Result

**FAIL** under the pre-registered criterion: neither sequential slot reached
either deposited endpoint within 0.300 Å over central `{N, CA, C, O}`. The
deposited A→B distance is **1.3244 Å**.

This is the valid run: `v2` was a null-Jacobian implementation error and was
deleted before interpretation. `v3` used an explicit centred 0.25° finite-
difference Jacobian.

| quantity | slot 1: full-map fit | slot 2: residual fit |
|---|---:|---:|
| RMSD to deposited A (Å) | 0.6275 | 1.2194 |
| RMSD to deposited B (Å) | 0.7799 | 0.7288 |
| fraction of A→B distance covered | 41.1% | 45.0% |
| single-slot fitted occupancy | 0.5664 | 0.0617 |
| RSS for its fitting target | 129.0730 | 127.3581 |
| final joint-QP occupancy | 0.4694 | 0.1569 |

The closer unique assignment is slot 1→A (0.6275 Å) and slot 2→B (0.7288 Å);
both are above 0.300 Å. The alternative assignment is 0.7799/1.2194 Å. Thus
this is neither a label reversal nor a one-slot success.

## What was run

- One 7-residue window around `7UTC A:ARG52`, resolution 1.8470 Å, 1,539 mask
  voxels, with neighbour subtraction.
- Each slot had **20 parameters**: qFit's 14 φ/ψ parameters plus six internal
  ω rotations. Slot 1 fit the residual map alone; it was frozen, then slot 2
  fit the density residual. Final occupancies were solved jointly by QP.
- The deposited-map calibration was retained: `rho_calc` → residual-map factor
  **3.35406** (inverse 0.298146).
- The augmented-Lagrangian constraint used the explicit terminal-frame SE(3)
  residual: translation xyz followed by rotation xyz multiplied by 1.5 Å.
  `rho=0.754988`, chosen so that the measured B-like seam reference has unit
  normalized penalty. λ began at zero and was updated after each 80-evaluation
  inner solve.
- The Ramachandran barrier was active below 0.05, with a soft ω-restraint
  around deposited A (20° scale). Optimisation used trust-region
  Gauss–Newton with a centred, absolute 0.25° coordinate Jacobian; no Adam.

The implementation is in
[`run_d1_aprime_sequential.py`](/Users/utkarsh/qfitonsteroids/scripts/run_d1_aprime_sequential.py:126):
the objective and explicit SE(3) residual are evaluated at lines 126–146, the
centred Jacobian at lines 157–176, the sequential schedule at lines 235–242,
and the pre-registered verdict calculation at lines 243–247.

## Occupancy and density

Deposited occupancies are A/B = **0.24/0.33** (total 0.57). The final joint
QP returns **0.4694/0.1569** (total **0.6263**) and RSS **124.7108**, down from
the one-slot A-start RSS of **160.3092**. Therefore the scale correction is
active and the QP did assign nonzero density to both final slot geometries.

The sequential slot-2 fit alone initially had zero occupancy because it was
fit to the residual after slot 1; it grew to 0.0617. The final joint QP raised
that contribution to 0.1569. This is not the previous zero-occupancy,
zero-movement artefact.

## Terminal-frame closure diagnostics

Final seam residuals are all tiny relative to the stated diagnostic scales.

| component | slot 1 | slot 2 |
|---|---:|---:|
| tx / ty / tz (Å) | −0.001358 / −0.001378 / +0.000725 | +0.000289 / −0.000304 / −0.000008 |
| rx / ry / rz (deg) | +0.02063 / +0.01634 / −0.03572 | +0.00288 / −0.00293 / −0.01269 |
| translation / 0.02 Å | −0.0679 / −0.0689 / +0.0363 | +0.0145 / −0.0152 / −0.0004 |
| rotation / 1.5° | +0.0138 / +0.0109 / −0.0238 | +0.0019 / −0.0020 / −0.0085 |

The terminal-frame constraint did not prevent motion by being badly violated:
both final configurations are very nearly seam-closed.

## Trajectories and convergence

### Slot 1: map fit

| AL update | LM evaluations | RMSD A / B (Å) | occupancy | RSS |
|---:|---:|---:|---:|---:|
| start | — | 0.0000 / 1.3244 | 0.5231 | 160.3092 |
| 1 | 31 | 0.6191 / 0.7863 | 0.5661 | — |
| 2 | 21 | 0.6206 / 0.7863 | 0.5662 | — |
| 3 | 31 | 0.6320 / 0.7833 | 0.5666 | — |
| 4 | 22 | 0.6356 / 0.7729 | 0.5663 | — |
| 5 | 22 | 0.6273 / 0.7790 | 0.5663 | — |
| 6 | 23 | 0.6275 / 0.7799 | 0.5664 | 129.0730 |

The scalar gradient norm fell from **0.05826** to **0.000970**. The final
inner solve stopped by `xtol`, not the 80-evaluation ceiling.

### Slot 2: residual fit

| AL update | LM evaluations | RMSD A / B (Å) | occupancy | RSS |
|---:|---:|---:|---:|---:|
| start | — | 0.0000 / 1.3244 | 0.0000 | 129.0730 |
| 1 | 55 | 1.1592 / 0.7280 | 0.0612 | — |
| 2 | 42 | 1.2075 / 0.7251 | 0.0617 | — |
| 3 | 32 | 1.2109 / 0.7271 | 0.0618 | — |
| 4 | 26 | 1.2165 / 0.7308 | 0.0617 | — |
| 5 | 23 | 1.2149 / 0.7312 | 0.0617 | — |
| 6 | 27 | 1.2194 / 0.7288 | 0.0617 | 127.3581 |

Slot 2 moved decisively away from A toward B in the first update, then plateaued
near 0.73 Å from B. Its scalar gradient norm is low but did not decrease
monotonically (0.001802 at start; 0.002095 at the final state); the final
inner solve stopped by `ftol`, not by the evaluation limit. The residual fit
therefore found a stable intermediate basin rather than showing an obvious
unspent gradient caused by the limit.

## ω, Ramachandran, and internal geometry

ω deviations from deposited A (degrees) were:

- Slot 1: −0.596, −0.060, +1.368, −1.110, +1.233, +3.822.
- Slot 2: +0.027, +0.134, +0.135, −0.967, +1.410, +2.138.

Thus free ω was used, but modestly. The fifth internal Ramachandran probability
was marginally below the 0.05 floor for both slots (0.04973 and 0.04993);
all other internal values were at or above the floor. This reflects the soft
barrier—not a hard exclusion—and should be retained as a diagnostic caveat.

Internal geometry remained rigid to numerical precision:

| measure | slot 1 | slot 2 |
|---|---:|---:|
| largest bond-length change | 7.33e-15 Å | 1.02e-14 Å |
| largest bond-angle change | 4.12e-13° | 5.12e-13° |

## Interpretation bounded by this PoC

The result refutes neither the measured 0.0459 Å geometry floor nor the
adequacy of the 20-DOF parameterisation for deposited-B reachability: those
are a separate no-density result. It shows that, with the calibrated real-map
objective and this sequential two-slot schedule, both slots settle in
intermediate configurations rather than recovering the two deposited states.
The failure is not explained by an inactive scale correction, an unassigned
second slot, a broken terminal frame, or internal-geometry distortion.

Raw remote artefacts are under
`/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_sequential_v3/`:
`result.json`, `trajectory.csv`, `checkpoint.npz`, and `final_slots.npz`.

## Fixed-geometry objective evaluation

This follow-up used exactly the same calibrated residual-map target and mask,
but **did not optimise any coordinates**. Every occupancy below was re-fit by
QP at fixed geometry.

| fixed model(s) | fitted occupancy/occupancies | RSS |
|---|---:|---:|
| converged slot 1 + converged slot 2 | 0.4694 / 0.1569 | **124.7108** |
| deposited A + deposited B | 0.3000 / 0.2717 | **143.0990** |
| deposited A only | 0.5231 | 160.3092 |
| deposited B only | 0.5191 | 164.1790 |

Thus deposited A/B is **18.3882 RSS units higher** than the converged pair.
Under this particular calibrated density/QP objective, the deposited pair is
not the preferred explanation. Deposited A is also preferred to deposited B
as a one-conformer explanation (RSS lower by 3.8698).

### Fixed-coordinate path from recovered slot 2 to deposited B

Slot 1 was held at its recovered geometry. Slot 2's Cartesian window
coordinates were interpolated toward deposited B, and both occupancies were
re-fit jointly at each point.

| fraction toward B | slot-2 RMSD to B (Å) | QP RSS |
|---:|---:|---:|
| 0.0 | 0.7288 | 124.7108 |
| 0.1 | 0.6560 | 124.6713 |
| 0.2 | 0.5831 | 124.7465 |
| 0.3 | 0.5102 | 124.7776 |
| 0.4 | 0.4373 | 124.6026 |
| 0.5 | 0.3644 | 124.3620 |
| 0.6 | 0.2915 | **124.2280** |
| 0.7 | 0.2187 | 124.4551 |
| 0.8 | 0.1458 | 124.9028 |
| 0.9 | 0.0729 | 125.5102 |
| 1.0 | 0.0000 | 126.2152 |

The RSS is **non-monotonic**. It initially decreases, reaches its best value at
60% of the line (0.4828 below the recovered pair), then rises toward deposited
B. Therefore the recovered slot-2 geometry is not the minimum of this
unconstrained density/QP line probe; the literal `ftol` stop left a downhill
density direction on this path. However, this is not by itself proof that the
full augmented-Lagrangian optimiser should traverse it: linear Cartesian
interpolation is not guaranteed to preserve the φ/ψ/ω kinematic manifold,
terminal-frame closure, or the Ramachandran/ω penalties. It does rule out a
simple monotone density barrier between the recovered slot and B.

Raw objective-only artefacts are under
`/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_objective_eval_v1/`.

## Held-out-voxel cross-validation

The three fixed geometry pairs above were evaluated without coordinate
optimisation. For each split, their two occupancies were fit by QP on 1,231
training voxels (80%), then RSS was measured on the 308 held-out voxels (20%).
Five random voxel splits and five blocked spatial-slab splits were used.

| holdout scheme | geometry | held-out RSS, mean ± SD | held-out MSE, mean ± SD |
|---|---|---:|---:|
| random voxels | converged pair | 24.551 ± 0.713 | 0.07971 ± 0.00231 |
| random voxels | deposited A+B | 28.429 ± 0.702 | 0.09230 ± 0.00228 |
| random voxels | slot 1 + 60%-to-B slot 2 | 24.576 ± 0.845 | 0.07979 ± 0.00274 |
| blocked spatial slab | converged pair | 32.680 ± 9.016 | 0.10610 ± 0.02927 |
| blocked spatial slab | deposited A+B | 39.791 ± 9.342 | 0.12919 ± 0.03033 |
| blocked spatial slab | slot 1 + 60%-to-B slot 2 | **32.439 ± 8.559** | **0.10532 ± 0.02779** |

Both split types preserve the ordering: deposited A+B is worse than the
recovered/intermediate geometries. The random split is optimistically precise
because neighbouring voxels are correlated at the 1.847 Å resolution scale;
the blocked result is the appropriate comparison and has substantially larger
between-split spread. It nevertheless agrees in direction.

The map grid spacing is 0.4428 × 0.4352 × 0.4579 Å, hence a mask voxel volume
of 0.08823 Å³ and total mask volume of 135.793 Å³. With correlation volume
`(resolution / 2)^3 = 0.78762 Å³`, the requested effective-observation
estimate is **n_eff = 172.41**, not 1,539 independent observations.

Raw cross-validation artefacts are under
`/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_objective_cv_v1/`.

### Paired blocked-split check

The requested paired differences, `RSS(deposited A+B) − RSS(converged pair)`,
using the original inherited-A-B-factor rendering, were:

| blocked split | deposited − converged held-out RSS |
|---:|---:|
| 0 | +8.9888 |
| 1 | +4.4979 |
| 2 | +7.3168 |
| 3 | +7.1438 |
| 4 | +7.6069 |
| mean ± SD | **+7.1108 ± 1.6312** |

Every split has the same sign. Thus the original blocked result was not an
artefact of averaging mismatched spatial slabs.

## Deposited-altloc B-factor correction

The initial renderer used `self.a_residue.b` for every candidate model
([`run_d1_8d_sequential_poc.py`](/Users/utkarsh/qfitonsteroids/scripts/run_d1_8d_sequential_poc.py:147)),
and passed that same array into every density calculation
([`run_d1_8d_sequential_poc.py`](/Users/utkarsh/qfitonsteroids/scripts/run_d1_8d_sequential_poc.py:184)).
Therefore **deposited B had inherited deposited A's B-factors** in the first
fixed-geometry comparison.

The comparator was rerun without coordinate optimisation, with deposited A
rendered with its own B-factors and deposited B rendered with its own,
atom-name-reordered B-factors. Those B values differ substantially at several
atoms (up to +10.43 Å² for B−A), so this is a material correction.

| full-mask geometry | fitted occupancies | RSS |
|---|---:|---:|
| converged pair (A B-factor candidate model) | 0.4694 / 0.1569 | **124.7108** |
| deposited A+B, B inheriting A B-factors | 0.3000 / 0.2717 | 143.0990 |
| deposited A+B, each altloc's own B-factors | 0.2775 / 0.3022 | **141.0398** |

The correction improves the deposited comparison by 2.0592 RSS, but deposited
A/B remains 16.3290 RSS above the converged pair.

The corresponding corrected paired blocked differences are +7.8546, +4.6825,
+4.4600, +5.8434, and +6.9493: **mean +5.9580, SD 1.4546**. Again every
split has the same sign. The fixed-geometry conclusion survives the fair
B-factor comparator, albeit with a smaller gap.

Raw B-factor-corrected artefacts are under
`/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_bfactor_actual_v1/`.
