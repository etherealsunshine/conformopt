# Eleven-site low-VDW experiment report

## Executive result

The run completed optimization and the full geometry, tmol, and strict audit for 11 held-out sites (50 starts per site; 550 ensembles total). The only experimental change relative to the matching sites in `heldout_twenty_per_residue_schedule_v1` was `lambda_vdw: 1.0 -> 0.05`.

The low-VDW run produced 50 strict successes out of 550 starts, but all 50 came from one site, `3K8W_A_SER337`. The other ten sites remained at zero strict success. Therefore `lambda_vdw=0.05` is not a globally superior setting. It fixes the SER337 rotamer/physics failure, improves recovery or occupancy for some sites, but leaves or worsens physical invalidity at the clash-prone sites.

## Methods

- Data: 11 untouched held-out sites selected from the original-five and expanded-fifteen panels.
- Denoiser: original crystal-frame U-Net checkpoint `model/denoiser_best.pt`.
- Target: denoised experimental omit-map density.
- Representation: residue-specific chi torsions with differentiable forward kinematics.
- Ensemble: `K=4` softmax-occupancy slots.
- Starts: 50 independent starts per site.
- Density stage:
  - 1-3 chi residues: 500 full-resolution Adam steps at learning rate 1.0.
  - 4 chi residues (`ARG447` and `LYS19`): 100 steps at 4 A FWHM/lr 1.0, 100 steps at 2 A/lr 0.1, and 100 full-resolution steps/lr 0.01.
- Physics stage: Adam reset, 200 full-resolution steps at 0.1 times the density-stage learning rate.
- Physics weights: `lambda_vdw=0.05`, `lambda_rot=0.5`, `lambda_clash=5.0`.
- Comparison: identical recorded settings and sites from the prior per-residue-schedule production run, except that its `lambda_vdw=1.0`.
- tmol version: 0.1.40.

Strict success required all of the following:

1. Both deposited A and B states found with conventional, chemically symmetry-aware RMSD below 1.0 A.
2. Recovered A/B occupancies each within 0.20 of the deposited occupancies.
3. Every active slot (occupancy at least 0.05) canonical within 30 degrees.
4. No direct or crystallographic-symmetry contact below 2.0 A.
5. Every active slot's tmol energy no more than 10 units above the better deposited A/B control.

## Aggregate cascade

| Outcome | VDW 1.0 baseline | VDW 0.05 | Change |
|---|---:|---:|---:|
| Both conformers found | 252/550 | 240/550 | -12 |
| Recovery + occupancy | 130/550 | 137/550 | +7 |
| Failed physical audit after recovery + occupancy | 130/550 | 87/550 | -43 |
| Strict joint success | 0/550 | 50/550 | +50 |

Equivalently, the mutually exclusive low-VDW failure cascade is:

- Not both found: 310/550.
- Both found but occupancy failed: 103/550.
- Recovery and occupancy passed but physical audit failed: 87/550.
- Strict success: 50/550.

The strict rate rose from 0.0% to 9.1%, but this aggregate improvement is entirely attributable to SER337.

## Per-site low-VDW failure decomposition

| Site | Both found | Recovery + occupancy | Physical failure | Strict |
|---|---:|---:|---:|---:|
| 3A1C ARG447 | 0/50 | 0/50 | 0/50 | 0/50 |
| 1ZV8 ASN1 | 0/50 | 0/50 | 0/50 | 0/50 |
| 2V05 HIS168 | 0/50 | 0/50 | 0/50 | 0/50 |
| 2VFP TYR417 | 11/50 | 0/50 | 0/50 | 0/50 |
| 3GMI GLU5 | 21/50 | 12/50 | 12/50 | 0/50 |
| 3K8W SER337 | 50/50 | 50/50 | 0/50 | 50/50 |
| 3NY7 LYS19 | 4/50 | 3/50 | 3/50 | 0/50 |
| 5DBA TRP325 | 42/50 | 41/50 | 41/50 | 0/50 |
| 7UO8 GLN53 | 12/50 | 0/50 | 0/50 | 0/50 |
| 8DJ2 VAL893 | 50/50 | 0/50 | 0/50 | 0/50 |
| 8FBE ILE92 | 50/50 | 31/50 | 31/50 | 0/50 |
| **Total** | **240/550** | **137/550** | **87/550** | **50/550** |

`Physical failure` is conditional on already passing both recovery and occupancy, so each row follows a true cascade. It is not the count of all physically invalid starts.

## Comparison by site

| Site | Both found: 1.0 -> 0.05 | Occupancy: 1.0 -> 0.05 | Strict: 1.0 -> 0.05 | Interpretation |
|---|---:|---:|---:|---|
| 3A1C ARG447 | 0 -> 0 | 0 -> 0 | 0 -> 0 | Landscape/recovery failure; VDW weight is irrelevant because neither basin is recovered. |
| 1ZV8 ASN1 | 6 -> 0 | 0 -> 0 | 0 -> 0 | Recovery worsened; occupancy remained limiting. |
| 2V05 HIS168 | 0 -> 0 | 0 -> 0 | 0 -> 0 | Recovery failure unchanged. |
| 2VFP TYR417 | 9 -> 11 | 0 -> 0 | 0 -> 0 | Small recovery gain, no occupancy benefit. |
| 3GMI GLU5 | 38 -> 21 | 29 -> 12 | 0 -> 0 | Navigation/occupancy worsened and every occupancy-qualified endpoint still clashed. |
| 3K8W SER337 | 50 -> 50 | 50 -> 50 | 0 -> 50 | Decisive win: the 50 noncanonical failures at VDW 1.0 disappeared at VDW 0.05. |
| 3NY7 LYS19 | 5 -> 4 | 1 -> 3 | 0 -> 0 | Slight occupancy gain; all three qualifying endpoints failed tmol. |
| 5DBA TRP325 | 38 -> 42 | 34 -> 41 | 0 -> 0 | Density/occupancy improved, but all 41 qualifying endpoints were noncanonical, clashing, and tmol-invalid. |
| 7UO8 GLN53 | 12 -> 12 | 0 -> 0 | 0 -> 0 | No meaningful change; occupancy remains limiting. |
| 8DJ2 VAL893 | 50 -> 50 | 0 -> 0 | 0 -> 0 | Recovery is perfect, but occupancy remains completely wrong. |
| 8FBE ILE92 | 44 -> 50 | 16 -> 31 | 0 -> 0 | Strong recovery/occupancy gain, but all 31 qualifying endpoints still clashed and failed tmol. |

## Physical-failure causes among occupancy-qualified endpoints

Failure categories overlap: one ensemble can fail rotamer, clash, and tmol simultaneously.

| Site | Occupancy-qualified | Noncanonical | Sub-2 A clash | tmol failure |
|---|---:|---:|---:|---:|
| 3GMI GLU5 | 12 | 0 | 12 | 0 |
| 3K8W SER337 | 50 | 0 | 0 | 0 |
| 3NY7 LYS19 | 3 | 0 | 0 | 3 |
| 5DBA TRP325 | 41 | 41 | 41 | 41 |
| 8FBE ILE92 | 31 | 0 | 31 | 31 |

The other six sites had no occupancy-qualified starts, so no endpoint could reach the physical stage of the strict cascade.

Across all active conformers, regardless of recovery:

| Conformer-level audit | VDW 1.0 | VDW 0.05 |
|---|---:|---:|
| Active conformers | 1,844 | 1,882 |
| Canonical | 1,498/1,844 (81.2%) | 1,784/1,882 (94.8%) |
| Clash-free | 1,407/1,844 (76.3%) | 1,447/1,882 (76.9%) |
| tmol-valid | 1,427/1,844 (77.4%) | 1,343/1,882 (71.4%) |
| Ensembles with every active conformer physically valid | 138/550 | 185/550 |

Thus the lower VDW weight substantially improved canonicality, barely changed the overall clash-free fraction, and reduced the tmol-valid fraction.

## Scientific interpretation

The hypothesis that VDW 1.0 can overpower density/occupancy optimization is partly supported. Lowering it helped ILE92 and TRP325 reach both conformers with more accurate occupancies, and it completely rescued SER337. However, weakening VDW did not make the difficult endpoints physical. The clash-prone GLU5, TRP325, and ILE92 ensembles continued to fail the strict audit; TRP325 became especially clear evidence of a density-versus-physics tradeoff.

The result argues against using `lambda_vdw=0.05` globally. The apparent 9.1% strict rate is a single-site effect. Excluding SER337, strict success is still 0/500.

A defensible next experiment would use residue- or failure-class-specific physics:

- Retain low VDW for SER-like cases where VDW displaces an otherwise valid rotamer.
- Keep stronger VDW for clash-prone GLU/TRP/ILE cases.
- Treat VAL893 and the ASN/TYR/GLN occupancy failures as occupancy-objective problems, not VDW problems.
- Treat ARG447 and HIS168 as landscape/recovery failures that require better initialization or search rather than weight tuning.

## Provenance and audit repair

Authoritative run root:

`/home/dev/qfit_unet_data/density_denoiser/heldout_eleven_lambda_vdw_005_v1`

All 550 optimizer outputs completed before the audit. The original audit invocation incorrectly supplied the full five-site selection to the one-site `original1` shard, causing it to look for `4C16_A_MET258`. The repair created frozen one-site and ten-site audit manifests inside the run and resumed only geometry, tmol, and strict summarization. No optimizer endpoint was regenerated or overwritten. Final `status.txt` is `complete`.
