# Production-mask coverage diagnostic and corrected experiment

Date: 2026-07-29

Frozen metric:
`qfit-synth20-merge050-one-to-one-tmol044-v3` (unchanged).

## Production 4.0 A boundary

The only deposited atoms outside the production sphere are both terminal NH2
atoms at 3A1C B/ARG447:

| Deposited state | Occupancy rank | Atom | Radius | Beyond boundary |
|---|---|---|---:|---:|
| A | minor | NH2 | 5.0215 A | 1.0215 A |
| B | major | NH2 | 4.9397 A | 0.9397 A |

The Gaussian density fractions outside the mask are much larger than the atom
count suggests at this site: 33.61% for minor A and 31.45% for major B.
Nevertheless, 3A1C recovers the minor state in 49/50 starts. Across the 19
non-equal-occupancy sites, minor density outside the mask is not positively
associated with minor failure:

- Pearson with all-start minor failure rate: -0.0777.
- Pearson with major-only minor-miss rate: -0.1244.

Thus production-mask truncation is real, but it does not explain the
panel-wide minor-conformer recovery pattern.

### Per-site density fraction and minor recovery

Fractions are the positive native conformer density integrated outside the
production 4.0 A mask, divided by density over the full saved 32-cube patch.
Minor failure means the deposited occupancy-minor state was not found under
the frozen-v3 one-to-one assignment.

| Site | Minor outside | Major outside | Minor failures | Major-only minor misses |
|---|---:|---:|---:|---:|
| 1ZV8 ASN1 | 2.984% | 2.971% | 48/50 | 20/50 |
| 2V05 HIS168 | 6.430% | 6.427% | 22/50 | 4/50 |
| 2VFP TYR417 | 0.028% | 0.015% | 45/50 | 40/50 |
| 3A1C ARG447 | 33.607% | 31.450% | 1/50 | 1/50 |
| 3GMI GLU5 | 3.481% | 4.340% | 9/50 | 9/50 |
| 3K8W SER337 | 0.043% | 0.043% | 0/50 | 0/50 |
| 3NY7 LYS19 | 7.203% | 4.161% | 1/50 | 1/50 |
| 4C16 MET258 | 0.888% | 0.088% | 13/50 | 7/50 |
| 4MKM THR77 | 0.016% | 0.020% | 0/50 | 0/50 |
| 5DBA TRP325 | 6.086% | 6.148% | 5/50 | 5/50 |
| 5KWB PHE591 | 0.002% | 0.002% | 1/50 | 1/50 |
| 5Z8H MET730 | 0.133% | 0.029% | 32/50 | 26/50 |
| 6H59 ARG144 | equal occupancy | equal occupancy | excluded | excluded |
| 6Y4G CYS260 | 0.002% | 0.003% | 0/50 | 0/50 |
| 7F72 MET103 | 0.125% | 0.162% | 3/50 | 3/50 |
| 7T7A LEU396 | 0.040% | 0.038% | 6/50 | 5/50 |
| 7UO8 GLN53 | 7.970% | 8.591% | 26/50 | 15/50 |
| 8DJ2 VAL893 | 0.049% | 0.049% | 0/50 | 0/50 |
| 8FBE ILE92 | 0.209% | 0.165% | 1/50 | 1/50 |
| 8Q6Q ASP81 | 0.574% | 0.551% | 0/50 | 0/50 |

## Why the 1.0 A canonical mask missed deposited atoms

The two checks used the same moving-heavy-atom name list and the same 1.0 A
padding. There is no atom-set or tolerance mismatch.

The "reachable" enumeration was not continuous torsion space. It contained
only the Cartesian product of the production marginal chi centers. Deposited
torsions occupy continuous values between those centers, while the production
gate permits deviations as large as 45--90 degrees. Consequently, 25
deposited atoms lie more than 1.0 A from the discrete canonical-center union.

Only 2/40 deposited conformers with such comparisons fail the production
marginal rotamer gate: 2V05 HIS168 A and 3NY7 LYS19 A. The other exceptions
are gate-accepted conformers sampled between discrete centers. Therefore the
contradiction is primarily a coarse discrete enumeration problem, with a
smaller genuine off-rotamer component.

Affected deposited states include ASN1 A/B, HIS168 A, ARG447 A, GLU5 B,
LYS19 A, MET258 A, TRP325 A/B, MET730 A, ARG144 B, and GLN53 A.

## Corrected F/G preflight

The containing footprint is the union of:

1. all atom positions at every production canonical-center tuple;
2. all deposited A and B moving-sidechain atoms;
3. a 1.0 A padding around that union.

Arm G uses the same footprint, but its weights are derived only from density
variance over the canonical-center states. Deposited A/B are not used to
derive weights.

| Arm | Voxels/site | Reachable outside | Deposited outside | A+B best |
|---|---:|---:|---:|---:|
| F: containing, uniform | 133--4,982 | 0 | 0 | 20/20 |
| G: containing, variance weighted | 133--4,982 | 0 | 0 | 20/20 |

Arm F's deposited major-collapse correlation loss is median 0.1168, versus
0.0988 under the production sphere. The signal strengthened rather than
weakened because the irregular containing footprint adds reachable support
while excluding many irrelevant sphere voxels. Arm G's median weighted
z-scored-MSE gap is 0.2759; it is not directly comparable in absolute scale
to Arm F.

## Run status

The two-arm run is active and detached:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_containing_mask_sweep_v1
```

Controller PID: `62240`. Arm F runs first; Arm G begins automatically after
F's optimizer, geometry audit, tmol audit, and strict summary complete.

The frozen control is reused and is not being rerun.

## Normalization null

The normalization hypothesis is not supported. Median relative separation
was 0.0731 under z-scored MSE, 0.0605 under native MSE, and 0.0584 after a
least-squares scale fit across the 129 real major-only endpoints. Deposited
coordinates ran in the opposite direction (0.1976 z-scored versus 0.2160
native). The populations disagree in sign.

Normalization's original rationale remains undocumented. Robustness to scale
mismatch between rendered and experimental/denoised maps is an inference,
not a recorded design intent.

## Artifacts

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_water_minstate_v2_single_rule_v1/
analysis/production_mask_coverage_v3

/home/dev/qfit_unet_data/density_denoiser/
containing_mask_sweep_preflight_v1
```

Relevant pod test suite: 25 passed.
