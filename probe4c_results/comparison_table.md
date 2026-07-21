# Probe 4c: Complex Targets and Physics-Regularized Losses

## Executive conclusion

Probe 4c is complete. Experiment 1 and all three default-lambda Experiment 2
runs finished 10,000 training steps, all evaluation stages, and the mandatory
physical audit of 1,000 endpoints. Experiment 3 was correctly not launched
because Experiment 1 failed its gate.

The result separates two problems that had been conflated:

1. **Physics regularization works as a validity filter.** Across the three
   regularized losses, all 450 training-site endpoints were physically valid,
   and none of the 750 regularized endpoints had a sub-2 A direct or symmetry
   clash. This is a large improvement over Probe 4b.
2. **Physics regularization does not recover B.** Every regularized trajectory
   converged to a valid, mostly A-like minimum. There were zero <0.50 A B
   recoveries and zero joint physical-plus-recovery successes.
3. **The complex target did not establish the intended bridge.** It recovered
   B_MET112 in 50/50 starts but recovered 0/50 at A_MET112 and A_ARG129, so it
   passed only one of three training sites instead of the required two.

The most important diagnostic is that the differentiable tmol term scores the
torsion-kinematic B geometry at 116.38 (A_MET112) and 139.53 (A_ARG129), while
the independent audit scores the exact deposited B structures at 48.69 and
53.74. Thus the composite loss suppresses the representable B basin at two of
three training sites. Physics removed garbage minima, but in this setup it also
made the desired torsion-space B approximation expensive.

## Decision table

| Experiment | Training sites with >15/50 below 0.50 A | Probe 4c gate | Physical result | Decision |
|---|---:|---|---|---|
| 1: synthetic complex target | 1/3 | Fail | 0/150 training endpoints valid | Do not run Experiment 3 |
| 2A: synthetic amplitudes + physics | 0/3 | Fail | 150/150 training endpoints valid | Valid wrong minima |
| 2B: localized SF + physics | 0/3 | Fail | 150/150 training endpoints valid | Valid wrong minima |
| 2C: local real-space + physics | 0/3 | Fail | 150/150 training endpoints valid | Valid wrong minima |

No experiment satisfies the joint criterion of physical validity and RMSD-to-B
below 0.50 A.

## Experimental implementation

All runs used the requested shared setup: the same five 2O1K sites, 6-layer
width-512 MLP, 8x8x8 density features, three truncated first-order training
updates, 10,000 steps, and 50 starts with 20 inference steps.

Experiment 1 compared the candidate sidechain's **complex** structure-factor
contribution with the deposited B sidechain contribution after subtracting the
fixed protein contribution. The subtraction and loss remained complex-valued.

Experiment 2 used the requested default weights:

```
lambda_tmol  = 0.01
lambda_rot   = 0.5
lambda_clash = 5.0
```

Physics was applied to the final training endpoint only, never inside learned-
energy inference. The differentiable tmol path rotates attached sidechain
hydrogens with the heavy atoms. Symmetry neighbors were generated explicitly
from the 2O1K space group and unit cell, because tmol scores only the asymmetric
unit. The rotamer prior used crystallographic chi angles and the requested
residue-specific states, including +/-90 degrees for ASP chi2.

All stages checkpointed to the named Modal volume every 100 training steps.

## A. Altloc recovery

### Experiment 1: complex target

| Site | Split | <0.50 A | <0.75 A | Mean RMSD-B | Mean RMSD-A |
|---|---|---:|---:|---:|---:|
| A_MET112 | Train | 0/50 | 0/50 | 0.856 | 1.319 |
| A_ARG129 | Train | 0/50 | 0/50 | 0.797 | 1.229 |
| B_MET112 | Train | **50/50** | **50/50** | **0.214** | 1.052 |
| B_ASP114 | Held out | 0/50 | 0/50 | 1.346 | 0.380 |
| B_ARG129 | Held out | 0/50 | 0/50 | 0.825 | 1.385 |

Experiment 1 therefore fails the Probe 4b/4c pass condition. Complex phase
information materially improves B_MET112 but does not produce a general B basin.

### Experiment 2: default physics regularization

| Loss | A_MET112 <0.50 / mean B | A_ARG129 <0.50 / mean B | B_MET112 <0.50 / mean B | Training sites passed |
|---|---:|---:|---:|---:|
| A: synthetic amplitudes + physics | 0/50 / 0.577 | 0/50 / 1.112 | 0/50 / 0.945 | 0/3 |
| B: localized SF + physics | 0/50 / 0.593 | 0/50 / 1.201 | 0/50 / 0.995 | 0/3 |
| C: local real-space + physics | 0/50 / 0.628 | 0/50 / 1.597 | 0/50 / 0.903 | 0/3 |

Each regularized loss reached 50/50 within 0.75 A at A_MET112, but none crossed
the absolute 0.50 A recovery threshold. The endpoints are closer to deposited A
than B at every site.

Compared with Probe 4b, regularization removes Experiment A's previous B_MET112
success (50/50 at 0.414 A). The price of physical validity at the default weights
is a strong bias toward A-like states.

## B. Mandatory physical audit

The independent audit reconstructed every saved endpoint and calculated:

- corrected tmol beta2016 energy after rebuilding hydrogens per conformation;
- crystallographic chi angles and residue-aware canonical deviations;
- direct asymmetric-unit and full symmetry-mate minimum distances;
- local observed 2Fo-Fc-like density z score;
- physical validity and joint success.

The criterion was exactly the requested one: no direct or symmetry contact below
2.0 A, tmol dE below 10 relative to the better deposited conformer, and all chi
angles within 30 degrees of an allowed center.

### Representative endpoint audit

All 50 endpoints in a site/experiment group converged to the same or nearly the
same attractor.

| Experiment | Site | RMSD-B | tmol dE | Chi angles (deg) | Canonical | Min distance | Density z | Valid |
|---|---|---:|---:|---|---|---:|---:|---|
| Complex | A_MET112 | 0.856 | +11.47 | -60.8, -90.2, 89.5 | No | 3.233 | 0.703 | No |
| Complex | A_ARG129 | 0.797 | +14.91 | -112.3, 125.5, 91.2, -153.8 | No | 2.874 | 0.595 | No |
| Complex | B_MET112 | **0.214** | +13.58 | -92.5, 139.4, -145.5 | No | 3.084 | 1.812 | No |
| Complex | B_ASP114 | 1.346 | +4.17 | 151.0, -117.0 | Yes | 2.877 | 0.139 | Yes, but A-like |
| Complex | B_ARG129 | 0.825 | +20.41 | -84.8, -121.2, -16.3, 170.5 | No | 2.030 | 0.268 | No |
| Reg A | A_MET112 | 0.577 | +3.55 | -73.4, -157.7, 87.8 | Yes | 3.018 | 0.892 | Yes |
| Reg A | A_ARG129 | 1.112 | -0.22 | -71.6, 155.0, 67.7, 178.8 | Yes | 2.673 | 0.378 | Yes |
| Reg A | B_MET112 | 0.945 | +1.00 | -60.9, 171.5, 86.7 | Yes | 3.084 | 2.000 | Yes |
| Reg B | A_MET112 | 0.593 | +4.16 | -78.4, -158.8, 85.2 | Yes | 3.080 | 0.875 | Yes |
| Reg B | A_ARG129 | 1.201 | -1.08 | -69.8, 164.5, 73.8, 177.3 | Yes | 2.778 | 0.771 | Yes |
| Reg B | B_MET112 | 0.995 | +0.17 | -63.4, 167.0, 80.3 | Yes | 3.084 | 2.051 | Yes |
| Reg C | A_MET112 | 0.628 | +4.49 | -81.7, -161.8, 82.7 | Yes | 3.171 | 0.816 | Yes |
| Reg C | A_ARG129 | 1.597 | +2.30 | -76.6, -175.1, 59.7, 158.3 | Yes | 2.309 | 0.494 | Yes |
| Reg C | B_MET112 | 0.903 | +2.05 | -60.5, 178.0, 93.6 | Yes | 3.084 | 1.971 | Yes |

Across held-out sites, all regularized B_ASP114 endpoints also pass physical
validation but remain 1.12-1.18 A from B and essentially A-like. Regularized
B_ARG129 endpoints are low-energy and nonclashing but miss the 30-degree rotamer
criterion by 1-7 degrees at chi2. Each regularized experiment therefore has
200/250 physically valid endpoints, but 0/250 joint successes.

### Clash result

There is **one sub-2 A clash among all 1,000 Probe 4c endpoints**: complex-target
B_ARG129 start 41 reaches 1.913 A from a symmetry mate. There are zero sub-2 A
clashes among the 750 regularized endpoints. This is a clear improvement over
the six clashing representative attractors in Probe 4b. The symmetry term did
its intended job; no regularized learned endpoint incurred a nonzero final
symmetry-clash penalty in the training-site oracle table.

### Energy result

The independent corrected tmol audit shows that regularized endpoints are close
to deposited energies: training-site dE ranges from -1.08 to +4.49. Thus the
regularized outputs are genuinely plausible conformations, not merely endpoints
that exploit the differentiable training scorer.

The complex-target endpoints are different: all three training-site attractors
have dE above 10, including the geometrically recovered B_MET112 endpoint at
+13.58. Experiment 1 therefore has coordinate recovery at one site but no joint
physical recovery.

## C. Training diagnostics

Values below compare the first and last 200 examples for each site.

| Experiment/site | Loss early -> late | Chi error early -> late |
|---|---:|---:|
| Complex / A_MET112 | 0.748 -> 0.209 | 1.783 -> 1.596 |
| Complex / A_ARG129 | 0.583 -> 0.300 | 1.347 -> 1.277 |
| Complex / B_MET112 | 0.456 -> 0.123 | 1.010 -> **0.545** |
| Reg A / A_MET112 | 1.553 -> 0.821 | 1.842 -> 1.820 |
| Reg A / A_ARG129 | 3.093 -> 0.906 | 1.249 -> 1.157 |
| Reg A / B_MET112 | 1.388 -> 0.409 | 1.196 -> 1.073 |
| Reg B / A_MET112 | 41.442 -> 40.775 | 1.834 -> 1.807 |
| Reg B / A_ARG129 | 38.875 -> 36.767 | 1.275 -> 1.212 |
| Reg B / B_MET112 | 38.517 -> 37.643 | 1.223 -> 1.127 |
| Reg C / A_MET112 | 3.475 -> 2.638 | 1.823 -> 1.792 |
| Reg C / A_ARG129 | 4.607 -> 2.393 | 1.196 -> 1.072 |
| Reg C / B_MET112 | 2.538 -> 1.536 | 1.173 -> 1.014 |

Every loss decreases, but chi error improves strongly only for complex-target
B_MET112. This reproduces the central Probe 4/4b signature: objective
optimization does not generally imply movement toward deposited B.

## D. Oracle analysis

### Complex target

| Site | A loss | B loss | Learned loss | Ordering |
|---|---:|---:|---:|---|
| A_MET112 | 1.2767 | 0.2553 | **0.1402** | Non-B learned endpoint beats representable B |
| A_ARG129 | 1.0938 | **0.0511** | 0.2322 | B is best; learner misses it |
| B_MET112 | 0.6870 | 0.3149 | **0.0536** | Non-exact learned endpoint beats representable B |

The complex target is not zero at the reported B control because the target was
generated from exact deposited B coordinates, while the torsion model can only
reach an approximate kinematic B (0.164-0.337 A irreducible coordinate residual
in the shared setup). A learned non-B torsion state can therefore match the exact
complex target better than `true_delta` at two sites. This prevents Experiment 1
from being the clean unique-target test it was intended to be.

The next complex-target control must generate its target from
`coords_from_chi(site, site["true_delta"])`, not the unrepresentable exact B
coordinates. That guarantees loss(B) = 0 inside the model's search space and
cleanly tests reciprocal-space learnability.

### Regularized composite losses

The default tmol term changes the A/B ranking at A_MET112 and A_ARG129:

| Site | Differentiable tmol(A) | Differentiable tmol(kinematic B) | Independent tmol(deposited B) |
|---|---:|---:|---:|
| A_MET112 | 48.88 | **116.38** | 48.69 |
| A_ARG129 | 48.88 | **139.53** | 53.74 |
| B_MET112 | 25.18 | 31.68 | 26.87 |

Consequently, in regularized Experiment A the composite B loss is worse than A
at A_MET112 (1.260 vs 1.078) and A_ARG129 (1.561 vs 1.030). The same issue
appears in localized SF, and the learned endpoint often scores below B after
regularization. The problem is not that lambda is too weak; at two sites the
tmol term actively reverses the crystallographic preference for B.

The independent reparse-based audit proves the final learned minima themselves
are physically reasonable. The discrepancy is specific to scoring the fixed-
geometry torsion approximation to B during training. The kinematic model keeps
the A conformer's bond lengths and angles while attempting to reach B by chi
rotations only; small irreducible geometric differences can be strongly
penalized by a full all-atom score.

## E. Trajectory analysis

- **Complex B_MET112:** all 50 trajectories approach below 0.50 A and remain
  there. This is Pattern 3 and the one clear learned B basin in Probe 4c.
- **Complex A_MET112:** one trajectory briefly enters 0.50 A and then leaves;
  the other 49 never reach it. A_ARG129 never enters the B basin.
- **All regularized training trajectories:** none ever enter 0.50 A. This is
  Pattern 2, not approach-then-leave degeneracy. The regularized learned field
  lacks a B basin at the tested weights.

The regularizers therefore did not convert Probe 4b's wrong minima into a B
basin. They converted them into stable canonical A-like basins.

## Additional crystallographic checks

| Model | Held-out R-factor |
|---|---:|
| Deposited | **0.323462** |
| Complex target | 0.324439 |
| Reg A synthetic amplitudes | 0.330626 |
| Reg B localized SF | 0.332331 |
| Reg C real-space | 0.333460 |

No learned ensemble improves the deposited structure. Complex target is closest,
as expected for the phase-informed synthetic control.

The occupancy perturbation test remains negative: complex target stays 100% B-
like under both 50/50 and 70/30 features, while all regularized models remain
100% A-like. The models learn fixed site attractors rather than an occupancy-
responsive density interpretation.

## Experiment 3 gate

Experiment 3 was not run. Experiment 1 passed only B_MET112 and failed the
required >15/50 recovery on at least two of three training sites. Bootstrapping
phases from a model that cannot reliably use perfect complex targets would not
be interpretable and would violate the explicit Probe 4c run order.

## Scientific interpretation

The Probe 4b diagnosis was partly right: physical constraints eliminate the
noncanonical/clashing solutions. But that was not sufficient because the
crystallographic objective and the physical prior still do not agree on B in
the restricted torsion representation.

The result is best summarized as:

> Probe 4c replaces physically nonsensical wrong minima with physically valid
> wrong minima. Phase information creates one true B basin, but not a robust
> solution across sites.

This does not justify abandoning EBT or reciprocal-space learning. It identifies
two concrete control failures that must be repaired before scaling:

1. the complex target must be generated from an exactly representable
   kinematic B;
2. the training physics score must be calibrated so that representable B is not
   penalized relative to A before the model is trained.

## Recommended next experiments

### 1. Probe 4c.1: exact kinematic complex target

Generate each target structure factor from the exact output of
`coords_from_chi(true_delta)`. Before training, assert numerically that complex
loss(B) < 1e-6 and that B beats A at all three sites. This is the decisive
oscillatory-landscape control that Experiment 1 was intended to provide.

### 2. Probe 4c.2: physics calibration gate

Before any lambda sweep, score A, kinematic B, and a canonical rotamer panel
with the exact differentiable training physics. Require:

- finite gradients;
- no hydrogen-coordinate artifact;
- kinematic B within 10 score units of deposited A/B after local hydrogen or
  sidechain relaxation;
- composite(B) < composite(A) for every training site.

If full tmol remains hypersensitive to fixed bond geometry, use a differentiable
nonbonded environment term plus rotamer and symmetry penalties during training,
and retain full tmol for post-hoc validation.

### 3. Small lambda sweep after calibration

The default run establishes that rotamer/symmetry physics can produce valid
endpoints. After fixing the B-energy calibration, sweep a narrower tmol range
such as 0, 0.001, 0.003, and 0.01 while retaining rotamer and symmetry terms.
Select only settings where B remains lower than A in the oracle composite.

### 4. Phase refinement remains gated

Run iterative phase refinement only if the exact-kinematic complex control
passes two of three training sites. If that corrected control still fails, add
Fourier features or a reciprocal-space-aware architecture before attempting
phase bootstrapping.

## Reproducibility artifacts

- `experiment_1_complex_target/altloc_recovery.csv`
- `experiment_1_complex_target/physical_audit.csv`
- `experiment_1_complex_target/oracle_analysis.csv`
- `experiment_1_complex_target/training_curves.png`
- `experiment_1_complex_target/trajectory_rmsd_to_B.png`
- `experiment_1_complex_target/energy_landscape.png`
- equivalent outputs under all three `experiment_2_regularized/` directories;
- `endpoint_audit/physical_audit_all.csv` (all 1,000 merged endpoints);
- `endpoint_audit/endpoint_metrics.csv` (rotamers, clashes, density, contacts);
- `endpoint_audit/tmol_energies.csv` (corrected tmol scores);
- `probe4_modal.py` (training/evaluation pipeline);
- `probe4c_audit_modal.py` and `probe4c_tmol_modal.py` (detached audits).
