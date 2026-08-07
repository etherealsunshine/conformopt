# Probe 4 combined report: reciprocal-space training and supervised chi control

## Executive conclusion

The implementation and optimization machinery work, but the experimental
reciprocal-space objective did not teach the network the correct torsion-space
landscape.

The supervised control recovered all three trained altloc-B conformers from all
50 random starts within 0.50 Å. This rules out a basic failure of the MLP,
torsion kinematics, truncated first-order differentiation, or iterative
inference. The original reciprocal-space run instead converged every start to
site-specific but mostly incorrect attractors, failed the density perturbation
test, worsened held-out R-factor, and did not improve chi error during training.

The original A:ARG129 result of 50/50 “B-like” endpoints is a false positive
under the relative classifier: its endpoint is 2.259 Å from B and only slightly
farther from A (2.344 Å). It recovered B within 0.75 Å in 0/50 trials.

## Experimental design

Both experiments used the same:

- protein: 2O1K;
- five deposited A/B altloc sites;
- 8×8×8 fixed local density features;
- six-layer, width-512 scalar energy network;
- sidechain chi representation and differentiable forward kinematics;
- three truncated first-order inner steps during training;
- 20 energy-gradient steps and 50 random starts during evaluation;
- training sites: A_MET112, A_ARG129, B_MET112;
- held-out sites: B_ASP114, B_ARG129.

Only the endpoint supervision changed:

1. **Reciprocal run:** normalized experimental |F_calc|−|F_obs| residual over
   95% of reflections; 10,000 steps.
2. **Supervised control:** circular error to deposited B chi angles; 5,000
   steps. The B coordinates were not given to the energy network as input.

## Altloc recovery

“Relative B” only means RMSD-to-B is lower than RMSD-to-A. “Absolute recovery”
requires RMSD-to-B below 0.50 Å.

| Site | Split | Reciprocal relative B | Reciprocal <0.50 Å | Reciprocal mean RMSD-B | Supervised relative B | Supervised <0.50 Å | Supervised mean RMSD-B |
|---|---|---:|---:|---:|---:|---:|---:|
| A_MET112 | Train | 50/50 | 0/50 | 0.693 Å | 50/50 | 50/50 | 0.223 Å |
| A_ARG129 | Train | 50/50 | 0/50 | 2.259 Å | 50/50 | 50/50 | 0.209 Å |
| B_MET112 | Train | 0/50 | 0/50 | 1.599 Å | 50/50 | 50/50 | 0.340 Å |
| B_ASP114 | Held out | 0/50 | 0/50 | 1.075 Å | 0/50 | 0/50 | 1.235 Å |
| B_ARG129 | Held out | 50/50 | 0/50 | 2.561 Å | 50/50 | 0/50 | 1.554 Å |

All 50 endpoints at a given site were nearly identical in both experiments.
For supervised training this is the intended single-target funnel. For the
reciprocal run it demonstrates collapse to the wrong attractor rather than
multimodal A/B recovery.

The torsion kinematics cannot reproduce every deposited B coordinate exactly
because the backbone and non-torsional geometry are fixed. The B-coordinate
kinematic residuals are 0.164–0.337 Å for the three training sites. The
supervised endpoints of 0.209–0.340 Å are therefore near the representational
limit of this implementation.

## Training behavior

### Reciprocal run

Per-site diffraction loss decreased slightly, but endpoint chi error did not:

| Training site | Early → late loss | Early → late chi error |
|---|---:|---:|
| A_ARG129 | 2.594 → 2.468 | 1.790 → 1.885 rad |
| A_MET112 | 2.801 → 2.688 | 1.441 → 1.435 rad |
| B_MET112 | 2.478 → 2.400 | 1.522 → 1.512 rad |

Energy decreased over the inner steps for essentially every trajectory, so
gradient descent followed the learned field correctly. The field simply led to
coordinates that marginally reduced the global diffraction loss rather than to
the deposited B conformer.

### Supervised control

The control reduced both circular chi loss and chi error, and its 20-step
inference reached near-kinematic-limit RMSDs from every random start on all
three training sites. This establishes that:

- network parameters receive useful gradients through the final inner update;
- first-order detaches do not prevent learning;
- periodic chi features are sufficient;
- the energy network can build a broad, navigable basin;
- 20-step test-time refinement can exploit that basin.

## Energy landscapes

For A:ARG129, the reciprocal-space landscape's low-energy basin is far from the
deposited B marker. The supervised landscape places its basin around B.

- Reciprocal landscape: `probe4_results_download/energy_landscape/landscape_plot.png`
- Supervised landscape: `probe4_supervised_results/energy_landscape/landscape_plot.png`

These plots visually agree with the endpoint statistics: optimization works,
but reciprocal supervision sculpted the wrong field.

## Crystallographic and perturbation tests

| Model | Held-out R-factor |
|---|---:|
| Deposited structure | 0.32346 |
| Supervised-control output | 0.32988 |
| Reciprocal-trained output | 0.33752 |

Neither learned output beats the deposited structure. The supervised output is
closer because it accurately restores the trained deposited B conformers, but
this is not an independent crystallographic success.

The reciprocal model produced 100% B-like endpoints under both its nominal
50/50 density and the synthetic 70/30 A/B perturbation. Thus it did not respond
to the density shift. The supervised control also remained at B, which is
expected because its objective explicitly teaches a fixed B target; that
perturbation result is not a failure of the supervised diagnostic.

## Held-out generalization

The supervised model failed absolute recovery on both held-out sites. B_ARG129
was relatively closer to B, but remained 1.554 Å away; B_ASP114 remained closer
to A and 1.235 Å from B. Therefore the model demonstrated per-site learning,
not reliable density-conditioned generalization.

With only three training sites and two held-out sites, this is a diagnostic
rather than a statistically meaningful generalization benchmark. In
particular, one held-out ASP residue type never occurs in training.

## Interpretation

### What is ruled out

The negative reciprocal result is not explained by insufficient network
capacity, broken torsion differentiation, a detached final update, or an
inability of gradient descent to cross the desired torsion displacement. Direct
endpoint supervision solves all three trained sites.

### Most likely remaining bottlenecks

1. **Global loss swamps the local conformer signal.** One half-occupancy
   sidechain contributes only a small fraction of scattering from 792 atoms.
2. **Amplitude scaling/model mismatch.** Training compares raw SFcalculator
   amplitudes to experimental amplitudes without optimizing a crystallographic
   scale inside the loss. A large fixed mismatch can dominate the small
   coordinate-dependent term.
3. **Experimental F_obs is not exactly explained by the deposited model.** Bulk
   solvent, anisotropic scaling, waters, disorder, B-factors, and model error
   remain fixed while only one sidechain moves.
4. **Density conditioning is underdetermined.** Three fixed density patches let
   the network memorize site-specific outputs and provide no incentive to learn
   an occupancy response.
5. **Too little diversity for generalization.** Three training sites cannot
   teach robust residue- and environment-dependent behavior.

## Recommended next experiment

Do not move directly to multi-protein training, but do not reject the learned
energy architecture. The next decisive bridge should be a **synthetic,
scale-matched reciprocal-space control**:

1. Generate synthetic F_obs with the same SFcalculator instance from the exact
   deposited A/B coordinates.
2. Train against those amplitudes with all scale conventions matched.
3. Evaluate with the new absolute <0.50 Å and <0.75 Å criteria.
4. If this succeeds, progressively add experimental complications: optimized
   amplitude scale, bulk-solvent/scaling terms, noise, then experimental F_obs.
5. If it fails while direct chi supervision succeeds, localize the reciprocal
   signal by subtracting the fixed-atom structure-factor contribution or using
   a residue-local difference target.

After that bridge passes, train with varied synthetic A/B occupancies and
recompute density features for each mixture. That is necessary before the
70/30 perturbation test can reasonably be expected to work.

## Decision

The correct status is **mechanism validated, crystallographic supervision not
validated**. The original Probe 4 success criterion based only on “closer to B
than A” should be replaced with an absolute RMSD threshold. Under the stricter
criterion, the reciprocal run recovered A:ARG129 in 0/50 trials, while the
supervised control recovered it in 50/50 trials.
