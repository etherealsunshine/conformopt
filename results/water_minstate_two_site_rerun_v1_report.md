# Labeled-water min-state fix and two-site synthetic rerun

Date: 2026-07-24

## Rule and implementation

New optimizer physics-environment rule:

`2026-07-24-altloc-minstate-water-minstate-v2`

Labeled waters now use the same min-over-neighbor-state soft penalty as
alternate protein residues in both the direct and crystallographic-symmetry
environments. A partial labeled water gets an explicit absent state when the
sum of its state occupancies is below one. Unlabeled waters remain invariant
and occupancy weighted.

No physics hyperparameter changed:

- Stage 2: 200 steps, reset Adam, learning-rate scale 0.1
- VDW / rotamer / symmetry weights: 1.0 / 0.5 / 5.0
- VDW soft threshold: 3.0 A
- Symmetry soft/hard thresholds: 2.5 / 2.0 A
- Quartic barrier scale: 0.0

The focused and relevant geometry/environment tests pass: 21/21.

## Deposited A/B soft floor, all 20 sites

All 40 deposited conformers were re-evaluated under the new rule.

- Every symmetry floor is now exactly zero.
- 7UO8 GLN53 A VDW drops from 6.5544 to 1.4002. The impossible
  HOH186 contributions at 0.613/1.689 A disappear.
- 7UO8 GLN53 B symmetry drops from raw 0.26879 (weighted 1.34395) to zero.
  The 1.7517 A symmetry-HOH228 contribution disappears.
- 3GMI GLU5 A's small raw symmetry floor also drops from 0.010527 to zero
  because HOH534 is a partial labeled water.

The remaining 7UO8 A VDW value is not a residual water clash. It is the normal
nonzero floor of the 3 A squared-hinge objective, largely from target-backbone
and ordinary neighbor contacts. Other deposited sites have the same kind of
baseline floor; changing it would require a separate cutoff/functional-form
experiment and was not done here.

## July-23 versus completed-current Stage-2 physics

The July-23 run did not store a source hash or source snapshot. Its environment
behavior is nevertheless recoverable: the reconstructed evaluator matches its
saved 2VFP final VDW values to 2.5e-5.

| Component | July-23 stale run | Completed current run |
|---|---|---|
| Protein altloc handling | one fixed preferred state per residue | min penalty over every deposited state |
| Waters | excluded | included; all labeled waters treated as invariant |
| Direct and symmetry construction | same fixed selected protein list | all protein/water states, then partition |
| VDW / rotamer / symmetry weights | 1.0 / 0.5 / 5.0 | 1.0 / 0.5 / 5.0 |
| Soft VDW / symmetry cutoffs | 3.0 / 2.5 A | 3.0 / 2.5 A |
| Stage-2 schedule | 200 steps, LR scale 0.1 | 200 steps, LR scale 0.1 |
| Symmetry barrier | unavailable | available but disabled (scale 0) |

At 2VFP the direct environment grows from 151 selected protein atoms to 302
all-state atoms, including 64 waters.

For the six starts recovered in both runs, median raw VDW rescoring is:

| Coordinates | July-23 protein environment | water-invariant v1 | water-minstate v2 |
|---|---:|---:|---:|
| July-23 endpoints | 1.419 | 1.836 | 1.636 |
| completed-current endpoints | 13.541 | 14.806 | 14.624 |

Thus the immediate atom-list change raises the old good endpoints by about
0.42, not 13.4. Most of the saved 1.419 -> 14.920 shift is a nonlinear change
in Stage-2 coordinates and active-slot composition under the changed
objective. It is not caused by the symmetry term, which is zero at 2VFP.

The stale-to-current +55 both-found aggregate therefore cannot be attributed
to a single "live symmetry" factor: the run also changed direct-environment
construction and the rotamer authority. It remains a changed-objective
baseline rather than model progress.

## 2VFP occupancy-splitting hypothesis

On the completed-current endpoints, 12 of the 17 recovered starts have an
unmatched active conformer.

- deposited A occupancy: 0.420
- median predicted A occupancy: 0.174
- median unmatched occupancy: 0.0588
- median A + unmatched occupancy: 0.2326
- median closest unmatched RMSD to A: 5.953 A

The unmatched conformers are not near-duplicate A conformers and their
occupancy does not restore A to approximately 0.42. Near-duplicate merging is
therefore not the fix for this collapse.

## Two-site rerun

The rerun used the identical checkpoint, target, seeds, K=4, density schedule,
and Stage-2 hyperparameters. Only the optimizer environment rule changed.

| Site | Both found | + occupancy | + rotamer/direct/symmetry | all-active tmol 0 | all-active tmol 0.5 / 1 / 2 |
|---|---:|---:|---:|---:|---:|
| 7UO8 GLN53 | 18/50 | 18/50 | 16/16/16 | 0/50 | 15/15/15 |
| 2VFP TYR417 | 1/50 | 1/50 | 1/1/1 | 1/50 | 1/1/1 |

7UO8 recovers as predicted once the incompatible waters leave the objective.
Its strict zero at tmol tolerance 0 is a tmol-margin result, not a remaining
direct or symmetry clash. At its site-specific reproduction q99 (0.1128),
14/50 all-active and 17/50 assigned-pair starts pass.

2VFP regresses from 17/50 both-found to 1/50. The new rule removes the large
VDW endpoint state but does not restore A; it changes the force balance in the
opposite direction. This is evidence against treating the water change as a
global improvement.

## Spliced 20-site composite and rule provenance

The versioned composite replaces only 7UO8 and 2VFP. It does not overwrite
endpoint data.

| Stage | Completed-current baseline | Two-site splice | Delta |
|---|---:|---:|---:|
| both found | 728 | 730 | +2 |
| + occupancy | 698 | 716 | +18 |
| + rotamer/direct/symmetry | 691 | 708 | +17 |
| all-active tmol tolerance 0 | 321 | 322 | +1 |
| all-active tmol tolerance 0.5 | 517 | 533 | +16 |
| all-active tmol tolerance 1.0 | 543 | 559 | +16 |
| all-active tmol tolerance 2.0 | 558 | 574 | +16 |
| assigned-pair tmol tolerance 0 | 394 | 394 | 0 |

Rule provenance:

- 7UO8 and 2VFP:
  `2026-07-24-altloc-minstate-water-minstate-v2`
- the other 18 sites:
  `2026-07-24-altloc-minstate-water-invariant-v1` (retrospective label for the
  hash-verified completed-current implementation)
- geometry audit:
  `2026-07-24-altloc-minstate-rotwidth-hisunion-aromatic45-matched-tmol-v2`
- tmol:
  `frozen_matched_deposited_minstate_v1`

The splice is mixed-rule diagnostic evidence, not a candidate frozen metric.

## Remote artifacts

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_current_symmetry_v1/analysis/deposited_soft_physics_floor_water_minstate_v2/
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_current_symmetry_v1/analysis/2vfp_stage2_environment_diff_v4/
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_current_symmetry_v1/analysis/2vfp_occupancy_split_v1/
/home/dev/qfit_unet_data/density_denoiser/heldout_7uo8_2vfp_water_minstate_v1/
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_water_minstate_splice_v1/
```
