# EBT for Crystallographic Ensemble Fitting: Complete Context Document

**Last updated:** July 2026
**Status:** Active research direction within Fraser Lab / SampleWorks
**Sprint context:** 6-week sprint, EBT work is direction ③ (generative qFit)

---

## Table of Contents
1. [The Problem](#1-the-problem)
2. [The Three Directions](#2-the-three-directions)
3. [What Are Energy-Based Transformers](#3-what-are-energy-based-transformers)
4. [How EBT Maps Onto Generative qFit](#4-how-ebt-maps-onto-generative-qfit)
5. [Training Against F_obs](#5-training-against-f_obs)
6. [How EBT Sidesteps the MIQP](#6-how-ebt-sidesteps-the-miqp)
7. [Coordinate Representation](#7-coordinate-representation-cartesian-vs-torsion)
8. [The De-Risking Ladder](#8-the-de-risking-ladder)
9. [Results So Far](#9-results-so-far)
10. [Feedback From Karson and Fraser](#10-feedback-from-karson-and-fraser)
11. [Novel Approaches To Consider](#11-novel-approaches-to-consider)
12. [Separate Future Direction: Sequence-Space Protein EBT](#12-separate-future-direction-sequence-space-protein-ebt)
13. [Key References](#13-key-references)
14. [Open Questions](#14-open-questions)
15. [Glossary](#15-glossary)

---

## 1. The Problem

### 1.1 Structure predictors memorize, they don't learn physics

SampleWorks (Chrispens, Collins, Fraser, Wankowicz et al. 2026) tested three structure predictors (Boltz-2, Protenix, RosettaFold3) on 791 alternative location ("altloc") segments from 40 PDB entries. These altlocs are alternative conformations that are:
- **Physically real** — experimentally supported, nearly iso-energetic
- **Absent from training data** — structure predictors strip altlocs during data curation, keeping only the primary conformer

Key findings:
- **1.6%** of baseline (unguided) ensembles capture both altloc states
- **Figure 4B:** ensemble occupancies track training frequency (what fraction of the training cluster is in state B) rather than experimental data (which shows equal occupancy). The predicted conformational distribution reflects memorization of the PDB, not physical energy.
- **Figure 3B:** increasing diffusion guidance strength improves density fit (RSCC) but **degrades geometry** (clashscore, bond outliers). Two competing objectives — density vs physics — fighting each other with a hand-tuned weight.

### 1.2 The altloc dataset

The SampleWorks dataset contains:
- 40 high-resolution PDB entries
- 791 altloc segments total
- 724 backbone altloc segments (backbone atoms have altloc labels)
- 66 sidechain-only altloc segments
- 1 domain-level altloc (98 residues)
- All evaluated against synthetic, noise-free density maps at 1 Å resolution with equal occupancy between altlocs A and B

### 1.3 The 2k scan (our prior work)

We ran difference-map diagnostics on ~2000 structures:
- **Sidechain misses are REAL and systematic:** 25% of structures, resolution-dependent (hallmark of real signal), dominated by ARG/GLU/GLN/LYS/aromatics — all reachable by rotamer search
- **Backbone misses are NOT systematic:** 1% paired, flat across resolution = disorder/noise, not discrete alternates
- **Functional enrichment test:** are missed altlocs at functional sites? Result: weak (1.38× pockets for all; 1.00× for paired set). Strongest signals were crystal-packing interfaces. **Not a hidden-functional-state story.**
- **2O1K worked example:** qFit fits far better than old deposition (R 0.19 vs 0.26). Even qFit misses a genuine alternate at ARG 119 (confirmed by omit map). But recovering it barely moved global R-free → **validate small altlocs by LOCAL difference density, not global R-free.**

### 1.4 Core thesis

Current structure predictors sample from a learned distribution biased by training statistics — not from a physical energy landscape. This is solvable by: (①) adding physics constraints, (②) calibrating uncertainty, and (③) learning the fitting process itself.

---

## 2. The Three Directions

### Direction ① — tmol physics prior (STATUS: mechanism DONE, results pending)

**The gap:** SampleWorks steers on density agreement only (`real_space_density.py`). No physical plausibility reward anywhere in the guidance loop. The only "clash" check is a post-hoc eval metric.

**The fix:** tmol (github.com/uw-ipd/tmol) is a GPU, differentiable, PyTorch reimplementation of the Rosetta energy (`beta_nov2016_cart`). Drops into SampleWorks' `RewardFunctionProtocol`.

**Current status:**
- `core/rewards/tmol_plausibility.py` validated end-to-end
- Forward-exact vs tmol's own `pose_stack_from_pdb` energy
- Discriminates clashes (~75× energy difference)
- Differentiable (grad norm ~460)
- Wired into FK steering + DataSpaceDPS guidance on GPU and CPU

**Next:** A/B test on 2-3 systems with time-dependent resolution scheduling (see §10.1 Karson feedback). Money result: clashscore drops, RSCC holds.

### Direction ② — GP-UQ for altlocs (STATUS: DEMOTED after Fraser feedback)

**Original idea:** Gaussian Process noise model over local density for per-site significance testing (is this altloc real?) and occupancy estimation with error bars.

**Why it was demoted:** Fraser flagged the water problem — ordered waters produce density peaks that look similar to partial-occupancy sidechain altlocs. Discriminating altloc vs water vs noise is a three-way classification problem requiring chemical/geometric knowledge, not just statistical noise modeling. Also pointed to Holton 2014 (the R-factor gap paper) showing the residual density is far more complex than "altloc vs noise."

**Current status:** Parked. One-sentence mention in presentation. Needs scoping conversations with Holton and Bonomi before any commitment.

### Direction ③ — Generative qFit / EBT (STATUS: de-risking probes in progress)

**The idea:** train a model that takes the experimental data and outputs the multiconformer ensemble directly, on GPU, in seconds. Two flavors:
- **Flavor A:** distill qFit outputs → fast qFit (same answers, ~100× faster). Ceiling = qFit.
- **Flavor B:** train against |F_obs| directly → can beat qFit. Harder. **This is where EBTs enter.**

**Why EBT architecture:** see §4.

---

## 3. What Are Energy-Based Transformers

### 3.1 Core concept

Reference: Gladstone et al. "Energy-Based Transformers are Scalable Learners and Thinkers" (arXiv:2507.02092, ICML 2025 oral).

A standard transformer takes an input and produces a prediction in one forward pass. An EBT does something fundamentally different:

1. The network takes **both** the input AND a candidate prediction
2. It outputs a **single scalar** — an energy — scoring how compatible they are
3. Lower energy = better compatibility
4. Predictions are made by **gradient descent** on the energy, starting from random noise:

```
ŷ_{i+1} = ŷ_i - α · ∇_ŷ E_θ(x, ŷ_i)
```

The network NEVER directly outputs the prediction. It only outputs a number. The prediction emerges from optimizing over that number.

### 3.2 Three key properties (the "Facets")

**Facet 1 — Dynamic compute allocation:** more gradient descent steps for harder problems. Easy predictions converge fast (energy drops quickly). Hard predictions need more steps (energy stays high). The system automatically allocates more compute to harder problems.

**Facet 2 — Uncertainty:** the energy convergence pattern indicates confidence. Fast convergence to deep minimum = confident. Slow convergence or oscillation = uncertain. No separate uncertainty model needed.

**Facet 3 — Self-verification:** generate M predictions from different random starts, pick the one with lowest energy. The model is its own judge. No external reward model or scoring function needed.

### 3.3 How training works

**The energy is NEVER directly supervised.** There is no ground truth energy. The network learns to shape a landscape that gradient descent can navigate to correct answers.

Training procedure:
1. `ŷ_0 ~ N(0, I)` — initialize candidate as random noise
2. `ŷ_{i+1} = ŷ_i - α · ∇_ŷ E_θ(x, ŷ_i)` — gradient descent on learned energy for N steps
3. `L = loss(ŷ_N, ground_truth)` — loss on FINAL prediction only (not on energy)
4. `∂L/∂θ via ∂ŷ_N/∂θ (HVP)` — backprop through ENTIRE optimization chain
5. `θ ← θ - η · ∂L/∂θ` — update weights → reshapes energy landscape

Step 4 requires second-order derivatives (Hessian-vector products) because you're differentiating through a gradient. These scale linearly with model size (~3.3× overhead per training step).

**The key insight:** the loss gradient tells the network "your energy landscape's gradient field led the optimization to the wrong place — here's how to tilt the landscape so it leads to the right place next time." Over thousands of training examples, the landscape gets sculpted so gradient descent from random noise reliably converges to correct answers.

**Early training (when energy is garbage):** the energy landscape is random, gradient descent goes nowhere useful, the final prediction is garbage, the loss is huge. But backpropagation through the chain still provides signal about HOW to reshape the landscape. Each training step improves the landscape slightly. After thousands of steps, the landscape reliably funnels noise toward correct answers.

### 3.4 Key results from the paper

- EBTs scale **35% faster** than Transformer++ on data efficiency
- EBTs improve performance by up to **29%** via additional inference-time computation
- Thinking helps **MORE on out-of-distribution data** (linear trend: more OOD = more improvement)
- Self-verification capability **scales with training data**
- Despite slightly worse pretraining perplexity, EBTs outperform Transformer++ on 3/4 downstream benchmarks (better generalization)

### 3.5 How EBT differs from ProteinEBM

ProteinEBM (Roney, Ou, Ovchinnikov 2026) is an energy-**parameterized** diffusion model. Key differences:

| | ProteinEBM | EBT |
|---|---|---|
| Training | Denoising score matching (dense supervision at every noise level) | Optimization-based (endpoint supervision only, backprop through GD chain) |
| Energy | Time-dependent E_θ(x, s, t) — different landscape per noise level | Single landscape E_θ(x, ŷ) — same landscape regardless of step count |
| Inference | Reverse diffusion + Langevin dynamics | Pure gradient descent on energy |
| Architecture | AF3/Boltz-1 diffusion modules, backbone-only, 85M params | Transformer, operates on whatever representation you choose |
| Score vs energy | Parameterizes the score s_θ = -∇_x E_θ, energy falls out as byproduct | Energy is the primary output, score is its gradient |

ProteinEBM found that IPA (equivariant architecture) was **unstable under second-order derivatives**, forcing them to use a non-equivariant architecture with data augmentation. This is directly relevant to EBT feasibility — EBT training also requires second-order derivatives.

---

## 4. How EBT Maps Onto Generative qFit

### 4.1 The mapping

| EBT concept | Generative qFit equivalent |
|---|---|
| Input x | Electron density (from experimental structure factors) |
| Candidate prediction ŷ | Ensemble: conformer coordinates + occupancies |
| Energy E_θ(x, ŷ) | Scalar scoring density-ensemble compatibility + physical plausibility |
| Gradient descent on ŷ | Iterative refinement of ensemble against density |
| Self-verification (pick lowest E) | Generate 10 ensembles from different starts, pick best |
| Thinking longer | More steps for ambiguous density |
| Energy convergence | Uncertainty signal (fast = confident, slow = uncertain) |

### 4.2 Five concrete advantages over a feed-forward approach

**1. Unified energy:** density fit + tmol physics in one scalar. No competing objectives, no guidance weight to tune at inference. Gradient descent jointly optimizes both. Breaks the SampleWorks Figure 3B tradeoff (geometry degrading with guidance strength).

**2. Self-verification:** generate multiple ensembles from different random starts, pick the lowest energy. No external scoring pipeline. The model judges its own output.

**3. Adaptive compute:** well-ordered residue → energy converges in 2 steps. Disordered region → 20 steps, energy stays high = honest uncertainty signal.

**4. OOD for altlocs:** SampleWorks altlocs are "physically in distribution, out of training set." EBT paper shows thinking helps more on OOD data. Energy landscape encodes physics, not training frequency.

**5. No complex output head:** feed-forward version needs variable conformers + valid coordinates + normalized occupancies = hard structured prediction. EBT just outputs a scalar. The ensemble lives in the optimization variable.

### 4.3 tmol integration

Three options for including physics:

**Option A (explicit additive):** `E_θ(D, ŷ) + λ · E_tmol(ŷ)`. Neural network handles density, tmol handles physics. Lambda set during training, not tuned per-system at inference.

**Option B (training regularizer):** Train with composite loss: density fit primary + learned energy must agree with tmol on training structures. Network internalizes physics. At inference, only run the network (no tmol call).

**Option C (force matching):** Regularize energy gradients to match tmol gradients: `∇_ŷ E_θ ≈ ∇_ŷ E_tmol`. The EBT's gradient descent steps approximate physical energy minimization.

Start with Option A (simplest). Explore B and C later.

### 4.4 Backbone altlocs

The EBT doesn't distinguish backbone from sidechain atoms — it optimizes all continuous coordinates. If density shows two backbone conformations for a loop, gradient descent will push backbone atoms toward both density features from different random starts. This is a significant advantage over qFit, which uses a separate, more limited backbone sampling pipeline.

The SampleWorks dataset is dominated by backbone altlocs (724/791 segments). The EBT can handle these natively. qFit architecturally cannot (its rotamer library is sidechain-only).

---

## 5. Training Against F_obs

### 5.1 Why not train against density maps

The naive approach — train against 2mFo-DFc maps — is circular. These maps are calculated using phases from the deposited model: `ρ(x) = FT[ |F_obs| · exp(i · φ_model) ]`. The phases encode the model's biases, including its choice of single conformer where there should be two. Training against a phase-biased target means the model inherits the bias.

### 5.2 The principled approach: reciprocal space

X-ray crystallography measures **structure factor amplitudes** |F_obs| — these are the raw experimental data with no model phases involved. They're deposited in the PDB alongside every structure (~150k X-ray structures have them).

From any ensemble (coordinates + occupancies + B-factors), you can compute theoretical structure factors F_calc:

```
F_calc(h,k,l) = Σ_j o_j · f_j · exp(-B_j · s²) · exp(2πi(hx_j + ky_j + lz_j))
```

where o_j is occupancy, f_j is atomic scattering factor, B_j is B-factor, (x_j, y_j, z_j) are coordinates.

**SFcalculator** (Li, Dalton, Hekstra 2025) makes this computation fully differentiable in PyTorch.

### 5.3 The training loss

```
L = Σ_{h,k,l} ( |F_calc(h,k,l)| - |F_obs(h,k,l)| )²
```

This is essentially the crystallographic R-factor as a differentiable loss. No model phases anywhere. You're comparing your ensemble's calculated amplitudes directly against measured amplitudes.

### 5.4 Full EBT training loop with F_obs

```
For each training example (a PDB entry with deposited structure factors):
  1. Load |F_obs| — pure experimental data
  2. Initialize random ensemble ŷ_0 (random coordinates, uniform occupancies)
  3. For i = 0 to N-1:
     a. Feed (density_features, ŷ_i) into transformer → scalar energy E_θ
     b. Compute gradient: ∇_ŷ E_θ
     c. Update: ŷ_{i+1} = ŷ_i - α · ∇_ŷ E_θ
     (optionally add tmol gradient: ŷ_{i+1} -= α₂ · ∇_ŷ E_tmol(ŷ_i))
  4. From final ensemble ŷ_N, compute F_calc using SFcalculator
  5. Loss = Σ(|F_calc| - |F_obs|)² (+ optionally λ · E_tmol(ŷ_N))
  6. Backprop through ENTIRE chain (SFcalculator → ŷ_N → GD steps → energy → weights)
  7. Update weights θ
```

### 5.5 Why this can beat qFit

- qFit fits against 2mFo-DFc maps (phase-biased, model-dependent). This trains against |F_obs| (phase-free, model-independent).
- qFit is limited by its rotamer library — can only propose conformers in the library. The EBT optimizes in continuous coordinate space and can find conformers between rotamer states or in unusual geometries.
- This is essentially "learned refinement, amortized across 150k structures." Refinement programs optimize one structure from scratch each time. This learns what good ensembles look like and applies that knowledge to new structures in seconds.

### 5.6 A note on what the model sees at inference

At inference, the model needs local density features as input (not |F_obs| directly, which is global). The density features could be:
- A local 2mFo-DFc map patch around the residue (yes, phase-biased, but only for context — the training loss was against |F_obs|)
- Local difference density features
- Learned features from a density encoder

The training loss is against |F_obs| (phase-free). The input features at inference can use phase-biased maps for local context because the model has been trained to produce ensembles that explain the raw data, not to reproduce the input features. This is analogous to how AlphaFold uses MSA features as input but is evaluated against structure — the input provides context, the loss provides the ground truth.

---

## 6. How EBT Sidesteps the MIQP

### 6.1 Why qFit uses MIQP

qFit's pipeline: enumerate candidate rotamers from a library (discrete set) → score each against density → solve for which combination at what occupancies best explains the density with fewest conformers. That last step — "pick best subset from discrete candidates and assign continuous occupancies" — is a Mixed Integer Quadratic Program. Integer: include this conformer yes/no. Quadratic: density fit is least-squares. NP-hard in general.

### 6.2 How EBT avoids this

The EBT never discretizes. Everything is continuous:
- Fix max conformers to K (e.g., 4 — covers >95% of cases)
- All K always exist with continuous coordinates
- Occupancies parameterized through softmax: `occ_k = exp(z_k) / Σ exp(z_j)` — always sum to 1, always positive
- Gradient descent simultaneously moves coordinates toward density AND adjusts occupancies
- An "unused" conformer's occupancy is pushed toward zero naturally (it doesn't improve density fit, so the energy penalizes it)

The MIQP's two subproblems — "which conformers?" (integer) and "what occupancies?" (continuous) — both become continuous optimization. "Which conformers" is answered by which end up with nonzero occupancy.

### 6.3 The non-convex landscape concern

The raw density fit loss as a function of coordinates is non-convex — rotamer barriers, multiple minima. That's why qFit discretizes. But the EBT doesn't optimize the raw density fit. It optimizes a LEARNED energy that has been TRAINED to be navigable. The training procedure (backprop through GD) explicitly incentivizes the energy landscape to be smooth enough that gradient descent from noise reaches correct ensembles. The model learns to smooth over rotamer barriers.

Multi-start handles residual non-convexity: run from 10+ random initializations, pick lowest energy. Different starts find different local minima.

---

## 7. Coordinate Representation: Cartesian vs Torsion

### 7.1 Cartesian coordinates (optimize xyz directly)

**Pros:**
- Simplest implementation
- SFcalculator and tmol both take xyz directly
- No coordinate conversion needed

**Cons:**
- Most of 3N-dimensional space is physically meaningless (broken bonds, steric clashes)
- Energy landscape wastes capacity encoding "don't break chemistry"
- Starting from noise: optimization must fix bond geometry AND find conformers simultaneously
- A sidechain rotamer switch is a correlated movement of many atoms along a curved path

### 7.2 Torsion angles (optimize phi/psi/chi)

**Pros:**
- Every point in torsion space is chemically valid by construction (bond lengths/angles fixed)
- Search space dramatically smaller (~2-3 torsions per residue vs ~24 xyz per residue)
- Rotamer switch = moving along 1-2 axes (chi1, chi2) rather than curved manifold in 3N space
- Energy landscape only encodes conformationally relevant information

**Cons:**
- Need differentiable torsion-to-Cartesian conversion (forward kinematics — exists, is standard)
- Periodicity: chi1 = -179° and chi1 = 181° are the same but numerically distant. Fix with (sin θ, cos θ) parameterization.
- Backbone torsions are coupled through chain closure (not an issue for sidechain-only optimization)
- HVP must flow through the coordinate conversion layer

### 7.3 Recommendation

**Start with sidechain torsions only (fix backbone).** This is the simplest, lowest-dimensional version:
- 1-4 chi angles per residue
- No chain closure issues
- Matches the sidechain altloc recovery problem (25% systematic miss rate from 2k scan)
- Matches the most tractable test cases in SampleWorks
- Small enough search space that even an MLP might learn a useful energy landscape

**Test both representations in probes.** Run Probe 0 (plumbing) and Probe 2 (multi-start altloc discovery) in both Cartesian and torsion space to measure the difference empirically.

**Later extension:** add backbone torsions for backbone altlocs. Requires handling chain closure constraints.

---

## 8. The De-Risking Ladder

### Karson's advice: get to hard problems through MVPs

The full EBT generative qFit is a massive project (NeurIPS-level if it works). Don't jump straight to it. Build a ladder of MVPs where each rung is independently useful and demonstrates the next rung is worth attempting.

### Rung 0: tmol in guidance loop (DONE)

**Question:** Does physics energy help ensemble generation?
**Answer:** Yes (mechanism validated, A/B results pending).

### Rung 1: Post-hoc energy ranking (DONE — informative negative)

**Question:** Does self-verification (rank by energy, keep best) improve ensemble quality?
**Method:** Generated 80 ensemble members from Boltz-2, scored with composite energy (tmol + density), selected top 8.
**Result on 6B8X (PTP1B):**

| Selection | RSCC | Clustering Score |
|---|---|---|
| Baseline (unranked 8) | 0.847 | 0.000764 |
| Density-heavy ranking | 0.851 | 0.000770 |
| tmol-only ranking | 0.846 | 0.000792 |

**Interpretation:** Clustering scores are essentially zero across all conditions. No altloc B samples exist in the 80-member pool. The model prior is so strong that 80 draws all land on conformation A. **Selection is not the bottleneck — generation diversity is.** This motivates active exploration (Rung 2+).

### Rung 2: Gradient descent refinement with hand-designed energy (NEXT)

**Question:** Can gradient descent on tmol + |F_calc|-|F_obs| loss navigate conformational space and discover missing altlocs?
**Method:** Take predictor output (biased to altloc A), perturb 50 times, run GD on composite energy from each perturbation. See if any trajectory discovers altloc B.

### Probing tests (detailed):

**Probe 2 — Multi-start altloc discovery (1-2 days):**

This is the real test. Everything else is a byproduct. If the plumbing is broken (gradients don't flow through SFcalculator or tmol with `create_graph=True`), you'll find out on the first iteration — clear stack trace, 10-minute fix. Don't waste time on a separate plumbing test.

Protocol:
- Take residue with known altlocs A and B from SampleWorks dataset
- Compute F_obs from full deposited model (both altlocs at correct occupancies)
- Start from conformation A coordinates
- For j = 1 to 50: perturb, run 100 steps GD on E_tmol + λ·Σ(|F_calc|-|F_obs|)², record endpoint
- Plot: RMSD-to-A vs RMSD-to-B for all 50 endpoints
- Test in both Cartesian and torsion space
- Start with simple case (leucine chi1 flip), then arginine, then backbone altloc
- **This establishes the baseline the learned energy must beat**

**Probe 4 — Learned energy on one protein (3-5 days):**
- Take one well-characterized protein with several known altlocs
- Train a small model to fit THAT protein's |F_obs| using EBT procedure
- Not generalizing across proteins — just learned refinement for one structure
- Test whether it recovers known altlocs better than hand-designed energy from Probe 2
- If it can't learn a useful landscape for ONE protein, it won't generalize across 150k

### Decision matrix

| Probe 2 (hand-designed) | Probe 4 (learned) | Interpretation | Next step |
|---|---|---|---|
| GD finds altlocs | Learned beats hand-designed | **Strong green light** | Build full transformer EBT |
| GD finds altlocs | Learned doesn't beat | Learning is the bottleneck, not the mechanism | Debug training, more data/capacity |
| GD can't find altlocs | N/A | Landscape too rugged for GD-based approach | Learned energy must smooth barriers (harder); consider alternative parameterizations |
| GD can't even refine | N/A | **Kill.** Optimization on this energy surface is broken | Abandon EBT direction entirely |
| Gradients don't flow (SFcalc or tmol) | N/A | Engineering blocker, not a science problem | Fix the autograd issue, then rerun |

---

## 9. Results So Far

### 9.1 Rung 1 results
See §8 Rung 1 table above. Informative negative: selection is not the bottleneck.

### 9.2 tmol gate validation
- Forward energy matches tmol's own calculation
- 75× energy discrimination between clashing and non-clashing structures
- Gradients are finite and nonzero (grad norm ~460)
- Wired into SampleWorks guidance loop
- A/B results on real systems pending (weeks 1-2 of sprint)

### 9.3 Prior probes from strategy doc
- CA-only ProteinEBM was **near-chance** on altloc discrimination → all-atom is required
- Rosetta ref2015 gate: **90-97%** real-vs-decoy discrimination (physics works)
- Occupancy sign test: **63-73%, R²=0** → occupancy is NOT predictable from static energy (must come from density fit, not physics)

---

## 10. Feedback From Karson and Fraser

### 10.1 Karson: time-dependent resolution scheduling

**Core insight:** match the density guidance to the diffusion noise level. Currently SampleWorks uses the same map at the same resolution with the same weight throughout the entire diffusion trajectory. This is wrong.

**Three knobs:**
1. **Map resolution (downsampling):** at high noise (early steps), blur the map to low resolution (~5-8 Å). As noise decreases, increase resolution to atomic detail. The structure and guidance operate at matching scales.
2. **Gradient weighting:** normalize the guidance gradient relative to the denoising update. Weight should change with timestep t.
3. **Reward function weight λ(t):** the most general version. Density reward weight as a function of diffusion timestep.

**Schedule for composite (density + tmol):**
- Early (high noise): low-res density (gentle), tmol OFF (structure too distorted for physics)
- Middle: medium-res density (moderate), tmol ramping up
- Late (low noise): full-res density (possibly ramping down as peaks sharpen), tmol at full weight

**Key point:** these knobs interact and must be tuned in concert. This is the money experiment for weeks 1-2.

### 10.2 Karson: GP-UQ is cool but scoping needed

"Really cool and useful for sampleworks as we move towards ensembles but there's a reason no one's done it before." Advised talking to James Holton or Bonomi to find the wall before committing.

### 10.3 Fraser: water problem kills GP-UQ (for now)

Fraser directly flagged that the GP-UQ can't distinguish altloc density from ordered water density. This isn't a nitpick — it's the thing that has stopped everyone who tried this. Discriminating altloc vs water vs noise is a three-way classification requiring chemical/geometric knowledge, not just statistical noise modeling.

### 10.4 Fraser: read Holton 2014

Pointed to Holton, Classen, Frankel & Tainer (2014) "The R-factor gap in macromolecular crystallography" (FEBS Journal). Core finding: the ~20% R-free gap is NOT primarily measurement noise — it's model inadequacy (single static structures can't explain ensemble data). The residual contains many contributions beyond altlocs (anharmonic motion, disorder, partial waters, radiation damage, etc.).

**Implication for direction ③:** training ensembles against |F_obs| directly attacks the R-factor gap Holton characterized. If the EBT produces better ensembles, R-factor drops, gap closes. This is measurable.

### 10.5 Karson: EBT needs MVPs first

Advised getting to hard problems through concrete probing tests rather than committing to the full architecture. "Show me it's worth spending time on."

### 10.6 Karson: crystallographic noise details

The noise in density maps is dominated by model-phase error which is much larger than measurement noise. Even at very high resolution with low signal-to-noise, you're still seeing real signal because you're averaging over millions of molecules. The challenge is that model error dominates, and as you move to ensembles (adding parameters), you need to properly handle how the added degrees of freedom interact with model bias.

---

## 11. Novel Approaches To Consider

### 11.1 CryoDRGN (most directly relevant)

Zhong et al. "CryoDRGN" Nature Methods 2021. Learns a continuous conformational landscape from cryo-EM images. Each image = 2D projection of one molecule in one conformation. The model maps a latent variable z to a 3D density volume. Different z = different conformations.

**Connection:** same problem (deconvolving conformational heterogeneity from averaged experimental data). Could represent ensemble not as "K discrete conformers with occupancies" but as "a continuous function from latent space to structure, where the distribution over z determines the ensemble." The density from the ensemble is then an integral over z.

### 11.2 Differentiable rendering / Gaussian splatting

Kerbl et al. "3D Gaussian Splatting" SIGGRAPH 2023. Represents a scene as 3D Gaussians with positions, sizes, opacities — optimized to match observed images.

**Connection:** your ensemble is a collection of 3D Gaussians (atoms) with positions (coordinates), sizes (B-factors), opacities (occupancies) — optimized to match observed diffraction. The math is almost identical. The gaussian splatting community has solved "variable numbers of Gaussians" and "avoiding local minima" problems.

### 11.3 Auto-decoding (DeepSDF style)

Park et al. "DeepSDF" CVPR 2019. Skip the encoder. Each structure gets a learnable latent code z. Shared decoder maps (z, position) → output. At test time, freeze decoder, optimize only z.

**Connection:** optimize over low-dimensional latent space (~8-16 dims) rather than all atom coordinates. The decoder handles mapping to valid structures. Shared decoder across training structures learns general conformational variation.

### 11.4 Test-time training

Sun et al. "Test-Time Training" ICML 2020. Model continues to learn during inference — a few gradient steps on model weights (not just prediction) to fine-tune for a specific input.

**Connection:** each crystal is different. A model trained on 150k structures might generalize well on average but suboptimally for any specific structure. TTT could briefly fine-tune the energy function for each new crystal.

### 11.5 Optimal transport / Wasserstein

Use Wasserstein distance instead of L2 for comparing F_calc to F_obs. Sensitive to structural similarity rather than per-reflection differences. Natural tools for multi-modal problems (Wasserstein barycenters).

### 11.6 Riemannian optimization

Protein conformations live on manifolds (torus for torsion angles). Riemannian gradient descent respects this geometry and handles periodicity naturally. The mass-weighted metric from MD gives physically meaningful steps.

### 11.7 Blind source separation

Density map is a linear mixture: ρ_obs = o_1·ρ_1 + o_2·ρ_2 + noise. Recovering individual ρ_i and weights o_i from ρ_obs is textbook source separation (ICA, NMF, Bayesian mixtures).

### 11.8 Frenet-Serret frames (GENIE3)

Backbone as a curve with curvature and torsion. For backbone altlocs, the "distance" between conformers might be lower in this space than in Cartesian or torsion space.

---

## 12. Separate Future Direction: Sequence-Space Protein EBT

**THIS IS A SEPARATE PROJECT, NOT PART OF THE CURRENT SPRINT.**

A detailed proposal is saved as `Sequence_Space_Protein_EBT_Proposal.pdf`. The core idea: train an EBT with masked language modeling on protein sequences (UniRef50), benchmark against ESM-2 on ProteinGym fitness prediction.

Key connections to current work:
- ProteinEBM validated energy-based approaches for protein structure (0.686 Spearman on ProteinGym stability with 85M params, beating ESM3)
- SampleWorks documented memorization in diffusion-based predictors — the same problem in structure space
- tmol regularization could ground the learned sequence energy in physics

This is a different lab's paper. Preserved for future development.

---

## 13. Key References

### Core papers
- **EBT paper:** Gladstone et al. (2025). "Energy-Based Transformers are Scalable Learners and Thinkers." arXiv:2507.02092. ICML 2025 oral.
- **SampleWorks:** Chrispens, Collins, Fraser, Wankowicz et al. (2026). "sampleworks: A Modular Platform for Experimentally Guided Biomolecular Ensemble Generation." The Stacks. DOI:10.82153/jkxj-tw08.
- **ProteinEBM:** Roney, Ou, Ovchinnikov (2026). "Protein Diffusion Models as Statistical Potentials." bioRxiv.
- **Holton 2014:** Holton, Classen, Frankel, Tainer. "The R-factor gap in macromolecular crystallography." FEBS Journal 2014.

### Tools
- **tmol:** github.com/uw-ipd/tmol — GPU differentiable Rosetta energy in PyTorch. Apache-2.0.
- **SFcalculator:** Li, Dalton, Hekstra (2025). Differentiable structure factor calculation in PyTorch.
- **qFit:** Wankowicz et al. (2024). Automated multiconformer model building. eLife.

### Protein language models
- **ESM-2:** Lin et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science.
- **ProteinGym:** Notin et al. (2023). NeurIPS.

### EBM theory
- **Du et al. (2022).** "Learning iterative reasoning through energy minimization." ICML. (Optimization-based EBM training foundation)
- **LeCun (2022).** "A path towards autonomous machine intelligence." (EBM framework overview)

### Novel approaches to consider
- **CryoDRGN:** Zhong et al. (2021). Nature Methods. (Continuous conformational landscape from cryo-EM)
- **3D Gaussian Splatting:** Kerbl et al. (2023). SIGGRAPH. (Differentiable rendering with Gaussians)
- **DeepSDF:** Park et al. (2019). CVPR. (Auto-decoding, latent optimization)
- **TTT:** Sun et al. (2020). ICML. (Test-time training)
- **GENIE3:** Frenet-Serret frame representation for protein backbone.

---

## 14. Open Questions

### Architecture
- Cartesian vs torsion: which gives better altloc discovery rates? (Probe 2 comparison)
- Does HVP through tmol's CUDA kernels work stably? (Will find out immediately in Probe 2 — if it breaks, clear stack trace)
- What model size is needed to learn a useful energy for one protein vs all proteins?
- How many GD steps during training before gradients become unstable?

### Training
- Can the model learn from |F_obs| loss alone, or does it need additional supervision?
- What's the minimum training set size? One protein? 100? 150k?
- Does the landscape actually learn to be smoother than the hand-designed energy?
- How does the time-dependent scheduling from direction ① map onto the EBT's optimization steps?

### Representation
- Fixed max conformers (K=4) with softmax occupancies — does this work or do unused conformers cause optimization issues?
- Should the model also optimize B-factors, or treat them as fixed/predicted separately?
- For backbone altlocs: how to handle chain closure in torsion space?

### Evaluation
- How to compare fairly against qFit? Same structures, same F_obs, compare R-factors?
- Is the EBT's self-verification (pick lowest energy) actually calibrated — does lowest energy = best ensemble?
- Can you measure the R-factor gap improvement from EBT ensembles vs qFit ensembles vs single-conformer models?

### Fundamental concerns
- The convexity incentive from EBT training: does it merge energy basins for multi-state proteins? (Test on kinase active/inactive)
- ProteinEBM found IPA unstable under HVP. Will the same happen with whatever architecture we use?
- Is the conformational landscape over all-atom coordinates fundamentally too rugged for optimization-based training?
- The exploitation vs exploration tension: does self-verification (pick lowest energy) suppress genuinely novel but valid conformations?

---

## 15. Glossary

**Altloc (alternative location):** Multiple conformations of a residue or segment modeled within a single PDB entry. Typically stripped from structure predictor training data.

**B-factor (temperature factor):** Parameter describing the spread of an atom's electron density. Higher B-factor = more spread = more thermal motion or disorder.

**Clustering score:** SampleWorks metric measuring whether a predicted ensemble properly separates into and covers all reference altloc conformations.

**EBM (Energy-Based Model):** Model that assigns a scalar energy to input configurations. Lower energy = higher compatibility/probability.

**EBT (Energy-Based Transformer):** Transformer implementation of an EBM, where predictions are made by gradient descent on the learned energy.

**F_obs:** Experimentally measured structure factor amplitudes. The raw data from X-ray diffraction.

**F_calc:** Theoretical structure factors computed from an atomic model. Differentiable via SFcalculator.

**FK steering / Feynman-Kac steering:** A guidance method in SampleWorks based on importance weighting.

**HVP (Hessian-vector product):** Efficient computation of second-order derivatives needed for EBT training. Scales linearly with model size.

**MIQP (Mixed Integer Quadratic Program):** Optimization problem with both integer and continuous variables and a quadratic objective. Used by qFit for conformer selection + occupancy fitting.

**Occupancy:** Fraction of molecules in a crystal adopting a particular conformation. Occupancies of altlocs sum to 1.

**Phase problem:** X-ray diffraction measures amplitudes but not phases. Phases must be estimated from the model, introducing model-dependent bias.

**qFit:** Existing tool for multiconformer model building. CPU, discrete rotamer sampling + MIQP. The current standard.

**R-factor / R-free:** Measures of agreement between calculated and observed structure factors. Lower = better. R-free uses a held-out test set of reflections.

**RSCC (Real-Space Correlation Coefficient):** Correlation between observed and calculated density maps in a local region.

**RSZD (Real-Space Z-score from Difference density):** Standard metric for significance of unexplained density at a residue.

**SFcalculator:** Differentiable PyTorch tool for computing structure factors from atomic models.

**tmol:** GPU, differentiable, PyTorch reimplementation of Rosetta energy function.

**2mFo-DFc map:** Standard electron density map used in crystallography. Calculated from observed amplitudes and model phases. Phase-biased.

**mFo-DFc map:** Difference density map showing unexplained signal. Also phase-biased.
