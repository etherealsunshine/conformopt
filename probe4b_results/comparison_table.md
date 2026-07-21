# Probe 4b comparison

## Success criterion

An experiment passes only if more than 15/50 trajectories reach deposited B at
less than 0.50 Å RMSD on at least two of the three training sites.

| Experiment | A_MET112 <0.50 Å | A_ARG129 <0.50 Å | B_MET112 <0.50 Å | Training sites passed | Verdict |
|---|---:|---:|---:|---:|---|
| A: synthetic Fobs | 0/50 | 0/50 | 50/50 | 1/3 | Fail; closest result |
| B: localized complex SF | 0/50 | 0/50 | 0/50 | 0/3 | Fail |
| C: real-space local | 0/50 | 0/50 | 0/50 | 0/3 | Fail |

Experiment D was not launched because none of A–C passed the required gate.

## Endpoint RMSD

| Experiment | A_MET112 | A_ARG129 | B_MET112 | Held-out B_ASP114 | Held-out B_ARG129 |
|---|---:|---:|---:|---:|---:|
| A: synthetic Fobs | 0.730 Å | 0.997 Å | **0.414 Å** | 0.545 Å | 1.577 Å |
| B: localized complex SF | 0.632 Å | 0.854 Å | 2.189 Å | 1.811 Å | 2.393 Å |
| C: real-space local | 0.771 Å | 2.608 Å | 2.223 Å | 1.510 Å | 1.075 Å |

All 50 starts at each site converged to effectively one endpoint. The failures
are therefore stable attractors, not high-variance inference.

## Training diagnostics

| Experiment/site | Early → late loss | Early → late chi error |
|---|---:|---:|
| A / A_MET112 | 0.531 → 0.040 | 1.814 → 1.284 rad |
| A / A_ARG129 | 0.367 → 0.104 | 1.419 → 1.690 rad |
| A / B_MET112 | 0.193 → 0.033 | 1.198 → 0.923 rad |
| B / A_MET112 | 40.183 → 39.976 | 2.215 → 2.330 rad |
| B / A_ARG129 | 36.084 → 35.755 | 1.400 → 1.344 rad |
| B / B_MET112 | 37.311 → 37.104 | 1.687 → 1.662 rad |
| C / A_MET112 | 2.311 → 1.898 | 1.787 → 1.672 rad |
| C / A_ARG129 | 1.737 → 1.373 | 1.538 → 1.510 rad |
| C / B_MET112 | 1.416 → 1.069 | 1.684 → 1.641 rad |

A substantially optimized its synthetic objective but did not consistently
approach B. B's complex experimental residual is dominated by a large fixed
term and supplies very little A-vs-B contrast. C supplies more contrast than B,
but its lower-density solutions are often non-B conformers.

## Oracle A/B and learned-endpoint losses

The losses below are computed directly, without the energy network. B is the
closest torsion-space representation of deposited B. Lower is better.

| Experiment/site | Loss at A | Loss at B | Loss at learned endpoint | Interpretation |
|---|---:|---:|---:|---|
| A / A_MET112 | 0.5626 | 0.0422 | **0.0083** | Non-B endpoint fits amplitudes better |
| A / A_ARG129 | 0.4736 | **0.0081** | 0.0764 | Learner misses better B basin |
| A / B_MET112 | 0.2129 | 0.0303 | **0.0157** | Non-B endpoint fits amplitudes better |
| B / A_MET112 | 40.3855 | 39.8785 | **39.8276** | Experimental residual favors non-B |
| B / A_ARG129 | 36.5830 | 36.0302 | **36.0005** | Experimental residual favors non-B |
| B / B_MET112 | 37.2652 | 37.0765 | **36.8073** | Experimental residual favors non-B |
| C / A_MET112 | 2.8085 | 2.2066 | **1.8400** | Local map favors non-B solution |
| C / A_ARG129 | 2.3413 | **1.1624** | 1.2276 | B is better, but learner misses it |
| C / B_MET112 | 1.2921 | 1.1458 | **0.8524** | Local map favors non-B solution |

All three objectives contain a detectable B-over-A signal. However, merely
amplifying that contrast does not make B the unique or deepest solution. The
network often follows the requested objective to a lower-loss non-B endpoint.

## Crystallographic and perturbation checks

| Model | Held-out R-factor |
|---|---:|
| Deposited | **0.323462** |
| A: synthetic Fobs | 0.323528 |
| B: localized SF | 0.323964 |
| C: real-space local | 0.330286 |

None improves the deposited held-out R-factor. A is nearly identical, which is
consistent with its synthetic amplitude optimization.

No experiment showed the desired occupancy response. A and B remained 100% B-
like under both nominal and 70/30 inputs; C remained 100% A-like. The original
perturbation implementation changes the density feature, not a retrained loss
target, so this is best read as evidence that the learned site attractor is
insensitive to the feature perturbation.

## Decision

Probe 4b A–C all fail the predefined absolute-recovery criterion. The direct
supervised control still proves that the energy/inner-loop mechanism can learn
the correct B basins. The new result is narrower: these crystallographic losses
do not uniquely select deposited B in the current single-sidechain setup.

The next diagnostic should not be Experiment D yet. First test a target whose
global minimum is known and unique in the representable torsion space, such as
a synthetic **complex** sidechain structure-factor target (amplitude + phase)
or a synthetic local density generated from the exact representable B
coordinates. This separates phase/amplitude degeneracy from EBT learnability.
