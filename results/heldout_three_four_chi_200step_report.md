# Four-chi extended density-schedule ablation

## Result

Doubling each coarse-to-fine density stage from 100 to 200 steps improved the
three-site aggregate strict joint success from **14/150 (9.3%)** to
**21/150 (14.0%)**. Both-state recovery increased from **23/150 (15.3%)** to
**38/150 (25.3%)**. The improvement came from ARG144 and LYS19. ARG447 still
never recovered deposited A and therefore remained 0/50 strict.

## Controlled change

The comparison uses the same original crystal-frame U-Net checkpoint, denoised
target, K=4 representation, 50 starts per site, base learning rate, occupancy
parameterization, soft-physics weights, physics duration, audit thresholds, and
seed setting as the recorded 20-site per-residue-schedule run. The only intended
optimizer change was the four-chi Stage 1 duration:

| Stage | Baseline | New run |
|---|---:|---:|
| 4 A blur, lr 1.0 | 100 steps | 200 steps |
| 2 A blur, lr 0.1 | 100 steps | 200 steps |
| Full resolution, lr 0.01 | 100 steps | 200 steps |
| Soft physics, lr 0.1 | 200 steps | 200 steps |
| Total | 500 steps | 800 steps |

Soft-physics weights remained lambda_vdw=1.0, lambda_rot=0.5, and
lambda_clash=5.0. Deposited A/B physics calibration passed for all three sites
before optimization. The optimizer used conventional, chemically
symmetry-aware RMSD in the audit.

## Recovery cascade

Counts are ensembles out of 50 starts per site. `Occupancy` means both A and B
were found at conventional RMSD <1.0 A and their recovered occupancies were
within +/-0.20. `Strict` additionally requires every active slot to be
canonical, free of direct and crystallographic-symmetry clashes below 2.0 A,
and within 10 tmol units of the best deposited A/B control.

| Site | Both found, old -> new | + occupancy, old -> new | + physical audit (strict), old -> new |
|---|---:|---:|---:|
| 3A1C B ARG447 | 0/50 -> 0/50 | 0/50 -> 0/50 | 0/50 -> 0/50 |
| 6H59 B ARG144 | 18/50 -> 25/50 | 14/50 -> 19/50 | 14/50 -> 19/50 |
| 3NY7 B LYS19 | 5/50 -> 13/50 | 1/50 -> 5/50 | 0/50 -> 2/50 |
| **Total** | **23/150 -> 38/150** | **15/150 -> 24/150** | **14/150 -> 21/150** |

Absolute aggregate changes:

- Both found: +15 ensembles, +10.0 percentage points.
- Recovery plus occupancy: +9 ensembles, +6.0 percentage points.
- Strict joint success: +7 ensembles, +4.7 percentage points; a 50% relative
  increase over 14 strict successes.

## Failure decomposition

The cascade partitions all 150 starts into mutually exclusive outcomes.

| Outcome | Baseline | New | Change |
|---|---:|---:|---:|
| A/B not both found | 127 | 112 | -15 |
| Both found, occupancy failed | 8 | 14 | +6 |
| Recovery/occupancy passed, physical audit failed | 1 | 3 | +2 |
| Strict success | 14 | 21 | +7 |

The increases in later-stage failures are not evidence that the longer schedule
made those gates intrinsically worse: 15 starts moved out of the dominant
not-found bucket, and seven reached strict success. The occupancy and physical
gates now receive a larger recovered population.

## Site-level interpretation

### 3A1C B ARG447

ARG447 remains the hard failure. Deposited B alone was found more often
(14/50 -> 25/50), but deposited A was found in **0/50** starts in both runs.
Thus extra time improves access to the already reachable B basin without
creating a path to the missing A basin. This is evidence against insufficient
step count as the primary explanation for ARG447.

The endpoints were generally physically reasonable despite failed recovery:
strict-physical active conformers increased from 128/179 to 144/181. The
failure is navigation/landscape recovery, not mainly clashes or rotamer
validity.

### 6H59 B ARG144

ARG144 is the clearest positive result. A recovery increased from 21/50 to
29/50, B recovery stayed at 34/50, and joint A/B recovery increased from 18/50
to 25/50. Strict success rose from 14/50 to 19/50.

All 19 new ensembles that passed recovery and occupancy also passed the full
physical audit. The active-conformer audit improved or remained strong:
canonical 113/118 -> 126/127, clash-free 92/118 -> 105/127, tmol-valid
115/118 -> 126/127. Here, additional exploration time genuinely helps.

### 3NY7 B LYS19

LYS19 also benefited: A recovery increased 10/50 -> 14/50, B recovery
40/50 -> 48/50, both-state recovery 5/50 -> 13/50, occupancy-qualified
recovery 1/50 -> 5/50, and strict success 0/50 -> 2/50.

The three occupancy-qualified ensembles that failed strict did so at the tmol
gate. All five passed ensemble geometry; only two had every active conformer
tmol-valid. The site-level endpoints remained canonical and clash-free
(165/175 canonical and 175/175 clash-free), while only 107/175 active
conformers were tmol-valid. Residual LYS19 limitations are therefore primarily
energy plausibility and occupancy, not steric clashes.

## Conclusion

The 200+200+200 schedule is a useful improvement for reachable four-chi
landscapes: it raises recovery and strict success on ARG144 and converts LYS19
from zero to nonzero strict success. It is not a universal solution. ARG447's A
basin remains inaccessible after doubling the density-stage budget, so that
site needs a change in initialization, representation, target landscape, or
multi-basin search strategy rather than still more iterations of the same
schedule.

## Sources of truth

Baseline run on pod:
`/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_per_residue_schedule_v1`

New run on pod:
`/home/dev/qfit_unet_data/density_denoiser/heldout_three_four_chi_200step_v1`

The primary inputs to this report are each run's `strict_summary.json`,
`strict_per_site.csv`, `ensemble_strict_audit.csv`, and
`active_conformer_strict_audit.csv` files.
