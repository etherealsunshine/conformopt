# Probe 4c.1 and 4c.2: Kinematic Target Fix and Soft-Physics Calibration

## Executive conclusion

Probe 4c.1 and 4c.2 are complete. All four detached Modal pipelines finished
10,000 training steps and every evaluation stage. The independent endpoint
audit scored all 1,000 saved conformations with corrected, reparse-based tmol,
rotamer geometry, direct contacts, and crystallographic symmetry contacts.

The kinematic target fix did exactly what it was designed to do at the control
level: the representable kinematic B conformation has zero target loss and
beats A at all three training sites. Nevertheless, the learned optimizer finds
B at only B_MET112 (50/50 starts). It misses A_MET112 and A_ARG129 in every
start. Representation mismatch was therefore a real defect in Probe 4c, but it
was not the main remaining cause of failure.

The calibrated soft-physics losses also pass their construction checks: they do
not reverse the desired B-versus-A ordering before training. They eliminate the
large tmol pathology of Probe 4c and produce many physically reasonable
endpoints. However, all three crystallographic variants recover B in 0/150
training starts and 0/100 held-out starts. Soft physics makes the wrong minima
more plausible; it does not create the missing B basin.

The strongest result is the B_MET112 kinematic-complex run: 50/50 endpoints are
below 0.50 A RMSD to B, canonical, nonclashing, and within 3.30 tmol energy
units of the better deposited conformer. These are 50 genuine joint successes.
No other experiment/site combination has a joint success.

## Decision table

| Experiment | Training sites passing >15/50 at <0.50 A | Held-out <0.50 A | Physically valid endpoints | Joint valid + B recovery | Verdict |
|---|---:|---:|---:|---:|---|
| 4c.1 kinematic complex target | 1/3 | 0/100 | 50/250 | **50/250** | Clean control works at B_MET112 only |
| 4c.2A synthetic amplitudes + soft physics | 0/3 | 0/100 | 100/250 | 0/250 | Valid, mostly A-like minima |
| 4c.2B localized SF + soft physics | 0/3 | 0/100 | 50/250 | 0/250 | Validity improved; signal still insufficient |
| 4c.2C local real-space + soft physics | 0/3 | 0/100 | **200/250** | 0/250 | Best validity, no B basin |

No run passes the intended recovery gate of at least two of three training
sites. Probe 4c.2C is best for physical plausibility; Probe 4c.1 is the only run
that recovers any deposited B state.

## What changed from Probe 4c

Probe 4c's complex target used exact deposited-B coordinates that the
chi-only kinematic model could not reproduce exactly. Probe 4c.1 instead
generates the complex target from `coords_from_chi(site, true_delta)`, placing
the target exactly inside the model's search space.

The mandatory pre-training assertions all passed:

| Site | Kinematic residual to deposited B (A) | Loss at kinematic B | Loss at A |
|---|---:|---:|---:|
| A_MET112 | 0.264 | **0.000** | 1.4529 |
| A_ARG129 | 0.164 | **0.000** | 1.0092 |
| B_MET112 | 0.337 | **0.000** | 0.5094 |

Probe 4c.2 removes full tmol from the training loss and replaces it with local,
differentiable terms: direct VDW repulsion, symmetry-clash repulsion, and a
rotamer prior. Full tmol remains an independent calibration/audit reference.
The weights were `lambda_vdw=1.0`, `lambda_rot=0.5`, and
`lambda_clash=5.0`, with `lambda_tmol=0`.

All nine pre-training physics calibration cases passed: soft physics did not
penalize kinematic B enough to reverse the crystallographic preference, and the
composite B loss remained below A for all three losses and all three training
sites. For example, in synthetic-amplitude 4c.2A the composite A/B losses were
0.5892/0.1695 at A_MET112, 0.5414/0.1653 at A_ARG129, and 0.2379/0.0735 at
B_MET112. This is the ordering Probe 4c's full-tmol term had broken.

## Altloc recovery

### Probe 4c.1

| Site | Split | <0.50 A | <0.75 A | Mean RMSD-B | Mean RMSD-A |
|---|---|---:|---:|---:|---:|
| A_MET112 | Train | 0/50 | 0/50 | 1.093 | 1.383 |
| A_ARG129 | Train | 0/50 | 0/50 | 0.799 | 1.151 |
| B_MET112 | Train | **50/50** | **50/50** | **0.333** | 1.003 |
| B_ASP114 | Held out | 0/50 | 0/50 | 1.598 | 0.895 |
| B_ARG129 | Held out | 0/50 | 0/50 | 2.796 | 2.089 |

The clean zero-loss B control is not sufficient for learnability. At A_MET112
and A_ARG129, the learned endpoints reduce target loss substantially but stop at
non-B states. B_MET112 is the only site where the learned field discovers and
retains the exact target basin.

### Probe 4c.2

| Loss | A_MET112 <0.50 / mean B | A_ARG129 <0.50 / mean B | B_MET112 <0.50 / mean B | Training sites passed |
|---|---:|---:|---:|---:|
| A: synthetic amplitudes | 0/50 / 0.521 | 0/50 / 1.062 | 0/50 / 0.883 | 0/3 |
| B: localized SF | 0/50 / 0.548 | 0/50 / 0.903 | 0/50 / 0.956 | 0/3 |
| C: local real-space | 0/50 / 0.645 | 0/50 / 1.613 | 0/50 / 0.843 | 0/3 |

All three runs reach 50/50 below 0.75 A at A_MET112, but none crosses the
predefined 0.50 A threshold. The held-out sites also have zero recoveries.

## Independent physical audit

Each saved endpoint was reconstructed independently. Physical validity requires
all of the following: no direct or symmetry contact below 2.0 A, corrected tmol
energy less than 10 units above the better deposited A/B conformer, and every
chi angle within 30 degrees of the residue-aware canonical centers. Joint
success additionally requires RMSD to B below 0.50 A.

All 50 starts in a site/run group converge to the same or an effectively
identical attractor, so the representative values below also describe the group.

| Run | Site | RMSD-B | tmol dE | Chi angles (degrees) | Canonical | Minimum distance (A) | Valid |
|---|---|---:|---:|---|---|---:|---|
| 4c.1 | A_MET112 | 1.093 | +14.15 | -133.7, 64.3, 35.2 | No | 2.86 | No |
| 4c.1 | A_ARG129 | 0.799 | +19.40 | -126.5, 124.9, 93.5, -140.2 | No | 2.78 | No |
| 4c.1 | B_MET112 | **0.333** | **+3.30** | -68.6, 161.5, -167.4 | Yes | 3.08 | **Yes** |
| 4c.1 | B_ASP114 | 1.598 | +17.95 | 128.2, -175.5 | No | 2.42 | No |
| 4c.1 | B_ARG129 | 2.796 | +10.95 | -119.7, 130.4, -63.2, -172.1 | No | **1.27 symmetry** | No |
| 4c.2A | A_MET112 | 0.521 | +7.00 | -75.2, -146.1, 90.0 | No (34 degrees at chi2) | 3.13 | No |
| 4c.2A | A_ARG129 | 1.062 | +2.74 | -68.9, 149.8, 69.6, 173.9 | No (30.15 degrees) | 2.98 | No |
| 4c.2A | B_MET112 | 0.883 | +2.96 | -56.5, 172.7, 91.3 | Yes | 3.08 | Yes |
| 4c.2A | B_ASP114 | 1.103 | +0.10 | 172.2, -109.1 | Yes | 2.89 | Yes |
| 4c.2A | B_ARG129 | 1.411 | +1.36 | -70.8, -143.4, -48.0, 154.0 | No | 2.25 | No |
| 4c.2B | A_MET112 | 0.548 | +8.38 | -80.8, -147.7, 92.5 | No | 3.10 | No |
| 4c.2B | A_ARG129 | 0.903 | +3.69 | -80.7, 136.0, 83.1, 168.7 | No | 2.94 | No |
| 4c.2B | B_MET112 | 0.956 | +1.84 | -63.1, 158.2, 85.0 | Yes | 3.08 | Yes |
| 4c.2B | B_ASP114 | 1.223 | +1.47 | 163.7, -124.3 | No | 2.68 | No |
| 4c.2B | B_ARG129 | 1.314 | +2.44 | -77.7, -140.7, -44.9, 148.3 | No | **1.86 symmetry** | No |
| 4c.2C | A_MET112 | 0.645 | +7.02 | -86.6, -159.0, 81.2 | Yes | 3.15 | Yes |
| 4c.2C | A_ARG129 | 1.613 | +6.68 | -63.0, -170.0, 52.8, 150.4 | Yes | 2.50 | Yes |
| 4c.2C | B_MET112 | 0.843 | +4.28 | -56.2, -177.7, 96.9 | Yes | 3.08 | Yes |
| 4c.2C | B_ASP114 | 1.117 | +1.20 | 167.9, -96.3 | Yes | 3.06 | Yes |
| 4c.2C | B_ARG129 | 1.361 | +2.20 | -72.1, -138.5, -55.1, 131.7 | No | 2.06 | No |

The corrected tmol energies show that every 4c.2 endpoint is within 10 units of
the better deposited conformer. Their failures are therefore not caused by a
gross all-atom energy catastrophe. The strict rotamer criterion is the largest
validity discriminator, with several states just outside 30 degrees. Only one
4c.2 group has a sub-2 A clash: 4c.2B B_ARG129, where all 50 runs converge to a
1.86 A symmetry contact. In 4c.1, B_ARG129 has a severe 1.27 A symmetry clash.

Most importantly, 4c.1 B_MET112 passes every test. Its learned chi state is
near the deposited B `g-/t/t` rotamer, its closest contact is 3.08 A, and its
tmol energy is comparable to deposited A/B and far below the mean random-
rotamer energy. This is a physically credible learned alternate, not just a
coordinate-level hit.

## Optimization and trajectory diagnostics

| Run | Mean loss, first 1,000 -> last 1,000 | Mean chi error (rad), first -> last |
|---|---:|---:|
| 4c.1 | 0.437 -> 0.237 | 1.365 -> 1.049 |
| 4c.2A | 1.869 -> 0.251 | 1.415 -> 1.337 |
| 4c.2B | 39.173 -> 37.942 | 1.437 -> 1.377 |
| 4c.2C | 3.482 -> 1.713 | 1.408 -> 1.292 |

Loss falls in every run, but chi error improves only modestly except where the
objective provides a usable B basin. This repeats the Probe 4/4b pattern: the
models learn to optimize the supplied loss, but loss improvement is not a
reliable proxy for movement to deposited B.

Trajectory histories sharpen that conclusion. In 4c.1, all 50 B_MET112 starts
enter the <0.50 A basin and remain there; no other site ever enters it. In 4c.2A
and 4c.2B, respectively two and one A_MET112 trajectories briefly cross 0.50 A
but leave before the endpoint. No other 4c.2 trajectory ever crosses the
threshold. The dominant failure is a missing/stable wrong basin, not general
instability after successful recovery.

## Oracle analysis

The 4c.1 oracle is now unambiguous:

| Site | A loss | Kinematic-B loss | Learned loss |
|---|---:|---:|---:|
| A_MET112 | 1.4529 | **0.0000** | 0.1886 |
| A_ARG129 | 1.0092 | **0.0000** | 0.2135 |
| B_MET112 | 0.5094 | **0.0000** | **0.0001** |

At the two failed sites, B is the unique zero-loss oracle state but the learned
update field stops in a higher-loss basin. This rules out the old exact-B versus
kinematic-B mismatch as the complete explanation. The remaining bottleneck is
learning/optimization of the energy field: three truncated first-order updates
and the local density representation do not reliably turn the global target
ordering into a reachable B attractor.

The 4c.2 oracle also confirms that calibration fixed the previous physics
reversal. B beats A at all three training sites for all three composite losses.
Yet learned endpoints can still beat or approach B without being geometrically
B-like, especially in local real-space and amplitude-only objectives. Correct
A/B ordering is necessary but not sufficient; the loss landscape remains
non-identifiable or contains easier off-target minima.

## Crystallographic checks

| Model | Held-out R factor | Difference from deposited |
|---|---:|---:|
| Deposited | **0.323462** | -- |
| 4c.1 kinematic complex | 0.331818 | +0.008356 |
| 4c.2A synthetic amplitudes | 0.327252 | +0.003790 |
| 4c.2B localized SF | **0.325435** | **+0.001973** |
| 4c.2C local real-space | 0.331722 | +0.008261 |

No learned ensemble improves on the deposited structure. 4c.2B is closest, but
its advantage over the other learned ensembles does not correspond to B-state
recovery.

The occupancy perturbation test is also negative. Each model retains a fixed
site preference under both 50/50 and 70/30 inputs rather than moving in a
consistent occupancy-dependent manner. The system is learning attractors, not
yet a calibrated interpretation of alternate-conformer occupancy.

## Combined interpretation across Probe 4c, 4c.1, and 4c.2

1. Probe 4c exposed a real implementation/scientific mismatch: exact deposited
   B was not reachable by the chi-only generator, and full tmol strongly
   penalized the kinematic approximation at A_MET112 and A_ARG129.
2. Probe 4c.1 removes the target mismatch completely. The fact that two sites
   still fail proves that representability alone does not solve the learned
   optimization problem.
3. Probe 4c.2 removes the full-tmol reversal and passes all calibration gates.
   The resulting minima are much more physically reasonable, but remain mostly
   A-like or intermediate.
4. Therefore the current limiting factor is **signal geometry and learned
   dynamics**, not simply the absence of physics. The objectives rank B above A
   at the controls, yet do not provide a sufficiently isolated, reachable B
   basin under the current features, inference depth, and training scheme.

## Recommended next experiment

Do not spend the next run merely increasing soft-physics weights: that is likely
to make the same wrong minima more canonical. The most informative next probe
is a direct optimization control for each site using the same kinematic target
loss but optimizing chi angles themselves with many steps and multiple starts,
without the learned MLP. If direct optimization reliably finds B at A_MET112
and A_ARG129, the bottleneck is the learned three-step energy dynamics. If it
does not, the reciprocal-space target has off-target basins and needs a more
identifying local phase/density construction. In parallel, increase inference
depth only after this control, and evaluate basin entry versus retention rather
than endpoint loss alone.

## Deliverables

- `probe4c12_results/endpoint_audit/physical_audit_all.csv`: all 1,000 merged
  geometry and corrected-tmol endpoint records.
- `probe4c1_results/physical_audit.csv`: 250 Probe 4c.1 endpoint records.
- `probe4c2_results/*/physical_audit.csv`: 250 endpoint records per 4c.2 loss.
- Each result directory also contains the stage manifest, run configuration,
  10,000-step training log, recovery table, oracle table, trajectories, and
  diagnostic plots.
