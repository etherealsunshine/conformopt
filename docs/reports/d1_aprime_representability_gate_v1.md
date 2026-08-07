# A′ PoC — 7UTC A:ARG52 representability gate

**Status:** gate complete; density optimisation not started.

**Pod result:** `/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_representability_v1`

qFit's `BackboneRotator` has 14 degrees of freedom for a seven-residue window
(2 × 7 phi/psi) and no omega variables. A′ uses qFit's 14 phi/psi rotations
plus six downstream C(i)-N(i+1) omega rotations, for 20 degrees of freedom.

The wrapper passed its forward-kinematic validation: zero parameters reproduce
deposited A with maximum coordinate error 0 Å; +1 degree on every internal
omega gives a maximum measured omega error of 0.0 degrees.

`compute_jacobian` is a differential closure Jacobian, not a seam-violation
function. A later augmented-Lagrangian A′ run would need an explicit terminal
frame SE(3) residual; this gate reports that residual but uses no seam, density,
or Ramachandran term.

## Result

Independent Levenberg-Marquardt fits from zero and deposited torsion deltas
converged to the same minimum.

| Quantity | Result |
|---|---:|
| Full seven-residue N/CA/C/O RMSD to B | **0.04589 Å** |
| Central N/CA/C/O RMSD to B | **0.05117 Å** |
| Central A→B distance | 1.3244 Å |
| Central distance covered | 96.1% |
| Function evaluations | 35 / 40 |

Free omega removes most of the reachability obstruction, but this does **not**
meet the pre-registered ~0.01 Å gate. Under that gate, representability has
not passed, so no density optimisation was launched.

The unconstrained B-like solution has a terminal-frame seam norm of **1.628 Å**
when rotation is converted using a 1.5 Å lever arm:

| Component | Value |
|---|---:|
| Translation (Å) | (+0.790, -1.288, -0.064) |
| Rotation (degrees) | (-11.84, -18.06, -7.75) |
| Rotation at 1.5 Å (Å equivalent) | (-0.310, -0.473, -0.203) |

The consistent minimum and exact omega validation make an omega-wrapper error
unlikely. The remaining 0.0459 Å is more plausibly the global effect of the
deposited A/B internal bond and angle differences; the earlier 0.0091 Å was a
local lower-bound diagnostic, not a demonstrated attainable global floor.

## Decision

Do not launch Step 1 yet. First determine whether the 0.0459 Å residual is
fully explained by deposited covalent-geometry differences, or whether the A′
forward kinematics still omit a degree of freedom.
