# A_ARG129 Coarse-to-Fine Direct Optimization

## Setup

- Target: real-space Gaussian density of the kinematic B conformer
- Starts: 50 random chi initializations
- Optimizer: Adam, learning rate 1.0, with optimizer state carried across stages
- Schedule: 100 steps at 4 A FWHM, 100 steps at 2 A FWHM, then 100 steps at full resolution
- Success threshold: sidechain RMSD to kinematic B below 0.5 A
- Baseline: 50 starts, 200 full-resolution steps, Adam learning rate 1.0

## Main result

The blurred landscape successfully funnels A_ARG129 toward B, but sharpening at an unchanged learning rate ejects the trajectories from that basin.

| Measurement | Coarse-to-fine | Full-resolution baseline |
|---|---:|---:|
| Ever reached <0.5 A | **46/50** | 6/50 |
| Final endpoint <0.5 A | **0/50** | 5/50 |
| Mean best RMSD to B | **0.085 A** | 1.584 A |
| Mean final RMSD to B | 3.430 A | **3.022 A** |

Thus coarse-to-fine improves basin discovery dramatically (92% versus 12%), but the implemented schedule is worse as an endpoint optimizer because it does not preserve the discovered solutions.

## Stage-by-stage behavior

| State | Hits <0.5 A | Mean RMSD | Median RMSD |
|---|---:|---:|---:|
| Initial | 0/50 | 1.611 A | 1.440 A |
| After 4 A stage | **43/50** | **0.161 A** | **0.032 A** |
| After 2 A stage | 16/50 | 2.139 A | 3.279 A |
| After full-resolution stage | **0/50** | 3.430 A | 3.386 A |

Across each interval, 44/50 runs entered the B basin during the 4 A stage, 45/50 entered it at some point during the 2 A stage, and only 16/50 entered it during the full-resolution stage. A total of 46/50 unique runs entered the basin at least once.

## Failure localization

The failure occurs immediately after each resolution change:

| Step | Interpretation | Hits <0.5 A | Median RMSD |
|---:|---|---:|---:|
| 100 | End of 4 A stage | 43/50 | 0.032 A |
| 101 | One 2 A update | 37/50 | 0.191 A |
| 102 | Two 2 A updates | 3/50 | 1.885 A |
| 200 | End of 2 A stage | 16/50 | 3.279 A |
| 201 | One full-resolution update | 6/50 | 3.541 A |
| 202 | Two full-resolution updates | **0/50** | 3.589 A |

The median four-chi update norm is only 0.014 rad immediately before the first transition. It jumps to 0.205 rad on the first 2 A update and 1.591 rad on the second. At the full-resolution transition it jumps from 0.011 rad to 1.079 rad. These approximately 60-degree multi-torsion jumps are large enough to leave the narrow B basin.

## Interpretation

The scientific hypothesis is supported: blurring converts the difficult A_ARG129 landscape into a navigable broad basin. The negative final result is an optimizer-schedule problem, not evidence that coarse-to-fine guidance is ineffective.

Keeping Adam at learning rate 1.0 across abrupt objective changes is the key defect. The sharper objectives have much larger and narrower gradients, while Adam also carries moments accumulated under the previous objective. The same step size that is productive at 4 A is destructive at 2 A and full resolution.

## Recommended controlled follow-up

Run the same 50 starts with optimizer state reset at each transition and a decaying learning-rate schedule:

1. 4 A FWHM: 100 steps at learning rate 1.0.
2. 2 A FWHM: reset Adam; 100 steps at learning rate 0.1.
3. Full resolution: reset Adam; 100 steps at learning rate 0.01.

The decisive metric is retention: how many of the 43 B-like endpoints after the 4 A stage remain below 0.5 A after 2 A and full-resolution refinement. A smaller supplemental sweep (0.03, 0.1, and 0.3 at 2 A; 0.003, 0.01, and 0.03 at full resolution) can identify the largest stable sharpening rates before repeating all 50 starts.

## Follow-up result: decay plus Adam resets

The recommended 50-start experiment was run with the same seeds and schedule, changing only the optimizer handling:

- 4 A: learning rate 1.0
- Reset Adam
- 2 A: learning rate 0.1
- Reset Adam
- Full resolution: learning rate 0.01

| Measurement | Decay + resets | Constant lr=1.0 | Full-resolution baseline |
|---|---:|---:|---:|
| Ever reached <0.5 A | 44/50 | **46/50** | 6/50 |
| Final endpoint <0.5 A | **43/50** | 0/50 | 5/50 |
| Mean best RMSD to B | **0.083 A** | 0.085 A | 1.584 A |
| Mean final RMSD to B | **0.108 A** | 3.430 A | 3.022 A |
| Median final RMSD to B | **0.0001 A** | 3.386 A | 3.372 A |

Stage retention was perfect among the B-like trajectories:

| Boundary | Hits <0.5 A | Retention from prior boundary |
|---|---:|---:|
| After 4 A | 43/50 | -- |
| After 2 A | 43/50 | **43/43** |
| After full resolution | 43/50 | **43/43** |

The transition update norms were controlled. At the 2 A transition, the median four-chi update was 0.200 rad rather than the 1.591 rad second-step jump seen at constant lr=1.0. At the full-resolution transition, the median update was 0.020 rad rather than 1.079 rad.

### Verdict

The A_ARG129 real-space kinematic landscape is navigable. Coarse blurring solves basin discovery, while optimizer reset and learning-rate decay solve basin retention. The result improves final recovery from 5/50 for direct full-resolution optimization to **43/50**, an absolute gain of 76 percentage points and an 8.6-fold increase in successful endpoints.
