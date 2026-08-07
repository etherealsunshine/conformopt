# Learned Energy for Crystallographic Altloc Recovery

## Combined report for Probe 4 and Probe 4b

## Executive summary

These experiments tested whether a learned scalar energy can create a smooth,
navigable torsion-space landscape that guides random sidechain conformations to
experimentally deposited alternative conformations.

The central result is now clear:

1. **The learned-energy mechanism works.** With direct chi-angle supervision,
   the model recovered all three training altloc-B conformers from 50/50 random
   starts at less than 0.50 Å RMSD.
2. **The original experimental reciprocal-space loss does not work.** It reduced
   diffraction loss without improving chi error and created incorrect
   site-specific attractors.
3. **Localizing the crystallographic signal was not sufficient.** Synthetic
   amplitudes, localized complex structure factors, and local real-space density
   all failed the predefined recovery criterion.
4. **The losses contain a real B-over-A signal, but B is often not the unique or
   deepest optimum.** In many cases, the learned incorrect endpoint has lower
   crystallographic loss than the deposited B conformer.

The model is therefore frequently doing what the objective requests. The
remaining bottleneck is objective identifiability: the current crystallographic
losses allow non-B conformations that explain the chosen signal as well as or
better than deposited B.

The correct project status is:

> **Energy-learning mechanism validated; crystallographic target formulation
> not validated.**

This is not evidence that gradient-based learned energies are fundamentally
incapable of crystallographic fitting. It is evidence that amplitude-only and
phase-approximated single-sidechain objectives do not uniquely specify the
desired conformer.

---

## 1. Scientific question

Probe 2 established that direct gradient descent on a hand-designed density or
physics-plus-density objective has difficulty crossing rotamer barriers. For
2O1K A:ARG129, the density-only baseline reached a B-like endpoint in only 3 of
50 trajectories.

Probe 4 asked whether a neural energy could learn a smoother optimization
landscape. Instead of directly predicting chi angles, the model assigns a
scalar energy to a candidate conformation. Chi angles are refined by gradient
descent on that energy, and only the final refined structure receives endpoint
supervision.

The hypothesis was:

> If the endpoint crystallographic loss contains enough information about the
> desired altloc, training should reshape the learned energy so random starts
> flow toward that conformer despite the barriers that defeated Probe 2.

---

## 2. Shared system and implementation

### Protein and sites

All experiments used 2O1K, which contains five A/B altloc sites supported by the
existing PDB and MTZ data:

| Site | Role | Chi dimensions |
|---|---|---:|
| A_MET112 | Training | 3 |
| A_ARG129 | Training | 4 |
| B_MET112 | Training | 3 |
| B_ASP114 | Held out | 2 |
| B_ARG129 | Held out | 4 |

The two held-out sites were excluded from all model updates.

### Model

The learned energy is a six-layer, width-512 MLP conditioned on:

- a fixed 8×8×8 local density feature patch;
- candidate chi angles represented as periodic sine/cosine pairs;
- residue-type one-hot features.

It produces one scalar energy. Lower energy represents greater compatibility.

### Optimization

Training uses three inner gradient steps in chi space. State-to-state paths are
detached between inner steps to implement the first-order approximation. The
final update remains connected to model parameters; detaching the final update
would eliminate the outer learning signal.

Evaluation uses:

- 50 random chi initializations per site;
- 20 learned-energy gradient steps per initialization;
- RMSD to deposited A and B sidechain coordinates;
- energy-landscape visualization for A:ARG129.

### Corrected success metric

The original metric labeled an endpoint “B-like” whenever it was closer to B
than A. This produced false positives even when both RMSDs were very large.

All final conclusions therefore use an absolute criterion:

> **Successful recovery requires RMSD-to-B below 0.50 Å.**

Probe 4b was defined to pass only if more than 15/50 trajectories met this
threshold on at least two of the three training sites.

---

## 3. Probe 4: experimental global reciprocal-space loss

### Objective

The original Probe 4 trained against experimental structure-factor amplitudes:

```text
L = mean[((|F_calc| - |F_obs|) / max(|F_obs|, 1))²]
```

Only one altloc-B sidechain moved during each training step; the rest of the
protein remained fixed.

### Training behavior

The diffraction loss decreased slightly at every training site, but chi error
did not improve:

| Training site | Early → late diffraction loss | Early → late chi error |
|---|---:|---:|
| A_ARG129 | 2.594 → 2.468 | 1.790 → 1.885 rad |
| A_MET112 | 2.801 → 2.688 | 1.441 → 1.435 rad |
| B_MET112 | 2.478 → 2.400 | 1.522 → 1.512 rad |

Energy reliably decreased during inference. The optimizer was following the
learned field, but the field pointed to incorrect structures.

### Recovery

| Site | Relative B-like count | <0.50 Å count | Mean RMSD-to-B |
|---|---:|---:|---:|
| A_MET112 | 50/50 | 0/50 | 0.693 Å |
| A_ARG129 | 50/50 | 0/50 | 2.259 Å |
| B_MET112 | 0/50 | 0/50 | 1.599 Å |
| B_ASP114 | 0/50 | 0/50 | 1.075 Å |
| B_ARG129 | 50/50 | 0/50 | 2.561 Å |

A:ARG129 appeared to improve from Probe 2’s 3/50 to 50/50 under the relative
metric. In reality, its endpoint was 2.344 Å from A and 2.259 Å from B. It was
called B-like because of a difference of only 0.085 Å. Under the absolute
criterion it recovered B in 0/50 trials.

All 50 starts at a site converged to nearly identical coordinates, indicating
collapse to a stable but incorrect site-specific attractor.

### Crystallographic checks

| Model | Held-out R-factor |
|---|---:|
| Deposited | **0.32346** |
| Original reciprocal EBT | 0.33752 |

The output did not improve held-out diffraction agreement. It also remained
100% B-like under both nominal 50/50 and synthetic 70/30 density features,
showing no occupancy response.

---

## 4. Supervised chi-angle control

### Purpose

The supervised control replaced the crystallographic endpoint loss with direct
circular error to deposited B chi angles while preserving:

- the same network;
- the same density and residue inputs;
- random chi initialization;
- the same first-order inner loop;
- the same training/held-out split;
- the same inference procedure.

This was not proposed as a production method. It was designed to isolate the
learned-energy mechanism from the crystallographic target.

### Results

| Site | Split | Relative B-like count | <0.50 Å count | Mean RMSD-to-B |
|---|---|---:|---:|---:|
| A_MET112 | Train | 50/50 | **50/50** | 0.223 Å |
| A_ARG129 | Train | 50/50 | **50/50** | 0.209 Å |
| B_MET112 | Train | 50/50 | **50/50** | 0.340 Å |
| B_ASP114 | Held out | 0/50 | 0/50 | 1.235 Å |
| B_ARG129 | Held out | 50/50 | 0/50 | 1.554 Å |

The 0.209–0.340 Å training-site RMSDs are near the representational limit of
the fixed-backbone torsion kinematics, whose deposited-B residuals range from
0.164 to 0.337 Å.

### Meaning

This control establishes that:

- the outer loss differentiates through the learned-energy gradient;
- first-order truncation does not prevent learning;
- the MLP has enough capacity;
- the periodic chi representation is suitable;
- the model can create broad basins that capture 50 random starts;
- 20 inference steps can traverse large rotamer changes.

Its held-out failure also demonstrates that three training sites support
memorization, not reliable density-conditioned generalization.

---

## 5. Probe 4b: localized-loss experiments

Probe 4b tested whether the original failure was caused by the moving
sidechain’s tiny fraction of total scattering. Three objectives were evaluated.

### Experiment A: synthetic global amplitudes

The target amplitudes were generated from the complete deposited structure
using the same SFcalculator instance. This removes experimental scale mismatch,
bulk-solvent error, measurement noise, and disagreement between the target and
the forward model.

### Experiment B: localized complex structure factors

For each site, the fixed-atom contribution was subtracted:

```text
F_total = F_fixed + F_sidechain
F_target,residual = scaled |F_obs| exp(i φ_fixed) - F_fixed
L = |F_sidechain - F_target,residual|²
```

This compares the complex sidechain contribution rather than the global
amplitude.

### Experiment C: local real-space density residual

Experimental 2Fo−Fc-like coefficients and fixed-atom calculated coefficients
were transformed onto a differentiable 0.5 Å grid within 4 Å of each A/B
sidechain region. The candidate sidechain density was trained against the local
observed-minus-fixed residual.

This inverse-Fourier formulation keeps target and candidate densities in
consistent SFcalculator units and avoids an arbitrary Gaussian/map scale.

### Implementation issue found and corrected

SFcalculator mutates its stored occupancy tensor when an occupancy override is
provided. An initial fixed-atom calculation therefore silently left later
candidate calculations with zero occupancy for the moving sidechain, producing
zero gradients.

The final implementation preserves the deposited occupancies once and passes
them explicitly to every candidate calculation. Smoke tests then confirmed
nonzero sidechain and model gradients for A, B, and C.

---

## 6. Probe 4b recovery results

### Absolute recovery counts

| Experiment | A_MET112 | A_ARG129 | B_MET112 | Training sites passed | Verdict |
|---|---:|---:|---:|---:|---|
| A: synthetic Fobs | 0/50 | 0/50 | **50/50** | 1/3 | Fail; closest result |
| B: localized SF | 0/50 | 0/50 | 0/50 | 0/3 | Fail |
| C: real-space local | 0/50 | 0/50 | 0/50 | 0/3 | Fail |

None meets the requirement of more than 15/50 recoveries on at least two
training sites. Experiment D, which depended on an A–C success, was therefore
not launched.

### Endpoint RMSDs

| Experiment | A_MET112 | A_ARG129 | B_MET112 | Held-out B_ASP114 | Held-out B_ARG129 |
|---|---:|---:|---:|---:|---:|
| A: synthetic Fobs | 0.730 Å | 0.997 Å | **0.414 Å** | 0.545 Å | 1.577 Å |
| B: localized SF | 0.632 Å | 0.854 Å | 2.189 Å | 1.811 Å | 2.393 Å |
| C: real-space local | 0.771 Å | 2.608 Å | 2.223 Å | 1.510 Å | 1.075 Å |

Again, the 50 starts at each site converged to almost exactly the same endpoint.
These results reflect stable learned minima rather than insufficient inference
time or stochastic variability.

### Training diagnostics

Experiment A substantially reduced its synthetic objective, but chi accuracy
was inconsistent:

| Site | Early → late loss | Early → late chi error |
|---|---:|---:|
| A_MET112 | 0.531 → 0.040 | 1.814 → 1.284 rad |
| A_ARG129 | 0.367 → 0.104 | 1.419 → 1.690 rad |
| B_MET112 | 0.193 → 0.033 | 1.198 → 0.923 rad |

Experiment B’s localized loss changed very little and provided weak chi
improvement. Experiment C’s density loss decreased more clearly, but chi error
remained almost flat. As in the original run, decreasing crystallographic loss
was not sufficient evidence of conformational recovery.

### Held-out R-factors

| Model | Held-out R-factor |
|---|---:|
| Deposited | **0.323462** |
| A: synthetic Fobs | 0.323528 |
| B: localized SF | 0.323964 |
| C: real-space local | 0.330286 |

Experiment A nearly reproduces the deposited held-out R-factor, consistent with
its scale-matched synthetic objective. No learned result improves it.

---

## 7. Oracle loss analysis

To determine whether the losses lacked all B signal or were merely ambiguous,
each objective was evaluated directly at:

- deposited/representable A;
- the closest torsion-space representation of deposited B;
- the endpoint learned by the corresponding EBT.

Lower values are better.

| Objective/site | A loss | B loss | Learned endpoint loss |
|---|---:|---:|---:|
| A synthetic / A_MET112 | 0.5626 | 0.0422 | **0.0083** |
| A synthetic / A_ARG129 | 0.4736 | **0.0081** | 0.0764 |
| A synthetic / B_MET112 | 0.2129 | 0.0303 | **0.0157** |
| B localized / A_MET112 | 40.3855 | 39.8785 | **39.8276** |
| B localized / A_ARG129 | 36.5830 | 36.0302 | **36.0005** |
| B localized / B_MET112 | 37.2652 | 37.0765 | **36.8073** |
| C density / A_MET112 | 2.8085 | 2.2066 | **1.8400** |
| C density / A_ARG129 | 2.3413 | **1.1624** | 1.2276 |
| C density / B_MET112 | 1.2921 | 1.1458 | **0.8524** |

### Interpretation

Every objective scores B below A. The crystallographic signal distinguishing
the deposited alternatives is therefore present.

However:

- synthetic amplitudes score a non-B learned endpoint below B on both MET
  sites;
- localized complex residuals score non-B endpoints below B on all three sites;
- local real-space density scores non-B endpoints below B on both MET sites.

This changes the interpretation from “the model cannot see B” to:

> **The model can see that B is better than A, but the loss often says that a
> third conformation is even better than B.**

For A:ARG129 under synthetic amplitudes and local density, B remains better than
the learned endpoint. Those cases show a remaining landscape-learning or
optimization failure. Most other failures are objective-identifiability
failures: the learned endpoint genuinely optimizes the specified target better
than representable B.

---

## 8. Why the losses are ambiguous

### Amplitude degeneracy

Structure-factor amplitudes omit phase. Multiple coordinate arrangements can
produce similar or lower amplitude error, particularly when only a small
half-occupancy sidechain is allowed to move.

### Phase approximation in localized SF

Experiment B supplies experimental amplitudes with phases estimated from fixed
atoms. The resulting residual is not guaranteed to equal the deposited
sidechain’s true complex contribution. The large residual baseline and weak
A-to-B contrast confirm that fixed-atom/model disagreement remains dominant.

### Local-map model error

The local density residual includes phase bias, neighboring atoms, solvent,
B-factor error, occupancy uncertainty, and density that the single movable
sidechain cannot fully explain. Moving to a non-deposited conformation can
compensate for these fixed errors and reduce the local map loss.

### Missing physical constraints

Torsion space preserves bond geometry but does not impose a rotamer prior,
steric compatibility, or a complete physical energy. A crystallographically
convenient non-B torsion can therefore win even if it is chemically less
plausible.

### Too little conditional diversity

Three fixed training density patches allow strong per-site memorization. The
network has no training distribution of occupancy perturbations or alternative
maps that forces it to interpret density rather than memorize a site attractor.

---

## 9. Conclusions

### What worked

- Differentiable chi-to-coordinate kinematics.
- SFcalculator gradients through the final coordinates.
- First-order learned-energy training.
- Smooth, strong energy funnels from 50 random starts.
- Direct recovery of deposited B when the endpoint target is unambiguous.
- Durable detached execution, checkpointing, and per-stage artifacts.

### What did not work

- Global experimental amplitude loss.
- Global synthetic amplitude loss as a unique conformational target.
- Fixed-phase localized complex residuals.
- Experimental local real-space residuals.
- Held-out-site generalization.
- Density-occupancy perturbation response.
- Absolute B recovery under the predefined Probe 4b criterion.

### Decision

Experiment D should remain gated off. Increasing the number of simultaneously
movable sidechains may strengthen total scattering signal, but it also greatly
increases the number of compensating conformations. It does not address the
demonstrated non-uniqueness of the objective.

The current result does not justify multi-protein EBT training against these
losses. It does justify one more controlled bridge that makes the correct
crystallographic target unique.

---

## 10. Recommended next experiments

### 10.1 Synthetic complex sidechain target

Generate the exact complex structure-factor contribution of the representable
B sidechain using SFcalculator:

```text
F_sidechain,B = F_full,B - F_fixed
L = |F_sidechain,candidate - F_sidechain,B|²
```

Unlike amplitude-only supervision, this preserves phase and explicitly encodes
the sidechain’s position. It is the reciprocal-space analogue of the successful
chi control.

Interpretation:

- If this reaches B in 50/50 starts, phase/amplitude degeneracy is the key wall.
- If it fails despite B being the exact unique target, debug the learned
  landscape approximation and first-order training on oscillatory reciprocal
  objectives.

### 10.2 Synthetic local density from representable B

Generate a local density target only from the exact candidate-space B
coordinates, using the same differentiable renderer used for candidates. This
removes experimental phases, neighboring-model error, and representation
mismatch.

Then add complications one at a time:

1. exact B sidechain density;
2. A/B occupancy mixture;
3. fixed neighboring atoms;
4. B-factor and coordinate noise;
5. approximate model phases;
6. experimental map.

The first transition that breaks recovery identifies the actual wall.

### 10.3 Physical regularization after target validation

Once a clean crystallographic target succeeds, add a final-coordinate rotamer,
steric, or tmol penalty to reject nonphysical compensating minima. Physics
should not be added before the clean-target bridge passes, because it would
confound signal identifiability with regularization.

### 10.4 Occupancy-conditioned training

Train on multiple synthetic occupancy mixtures and recompute both density
features and endpoint targets for each mixture. A model trained only on one
fixed target has no reason to respond correctly to a 70/30 feature
perturbation.

---

## Final statement

The experiments have moved the project from a vague negative result to a
specific mechanistic diagnosis. The EBT can learn and optimize a correct
torsion-space funnel. The present crystallographic objectives fail because they
do not make the deposited conformer uniquely optimal. The next phase should
focus on controlled complex or synthetic-density targets that preserve spatial
identity, followed by a staged reintroduction of experimental ambiguity.
