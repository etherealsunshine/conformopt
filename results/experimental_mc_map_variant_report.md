# Experimental multi-conformer map-variant report

## Outcome

All three 2O1K production experiments completed successfully, including 50 starts for each of five sites under density-only and soft-physics conditions and the detached tmol audit. None passed the predeclared two-site recovery gate. The best map was omit mFo-DFc, but its gain was concentrated in A_MET112 and did not generalize across the five sites. Soft physics did not improve overall recovery or tmol plausibility.

## Recovery against experimental maps

Each row aggregates 250 starts (50 starts x 5 sites). `Both found` requires recovery of deposited A and B endpoints. `Ensemble success` additionally requires both predicted occupancies to be within 0.20 of their deposited values.

| Map | Condition | Both found | Ensemble success | Sites with >=15/50 both found |
|---|---:|---:|---:|---:|
| Omit 2mFo-DFc | Density only | 9/250 (3.6%) | 5/250 (2.0%) | 0/5 |
| Omit 2mFo-DFc | Soft physics | 5/250 (2.0%) | 3/250 (1.2%) | 0/5 |
| Omit mFo-DFc | Density only | 30/250 (12.0%) | 13/250 (5.2%) | 1/5 |
| Omit mFo-DFc | Soft physics | 25/250 (10.0%) | 11/250 (4.4%) | 1/5 |
| Averaged kick-omit 2mFo-DFc | Density only | 10/250 (4.0%) | 3/250 (1.2%) | 0/5 |
| Averaged kick-omit 2mFo-DFc | Soft physics | 6/250 (2.4%) | 5/250 (2.0%) | 0/5 |
| Synthetic reference | Either | 153/250 (61.2%) | 139/250 (55.6%) | 4/5 by both-found count |

The strongest experimental result was A_MET112 with omit mFo-DFc: 16/50 both-found and 9/50 full successes under both conditions. A_ARG129 reached 8/50 both-found and 2/50 successes with density only, falling to 6/50 and 0/50 with soft physics. B_MET112, B_ASP114, and B_ARG129 remained below the 15/50 recovery threshold for every map and condition. B_ARG129 was never recovered in any production run.

## Occupancy behavior

The deposited targets are 0.5 A / 0.5 B. Experimental fitting usually concentrated occupancy in one conformer or in non-deposited endpoints rather than dividing it between A and B.

- Omit mFo-DFc gave the best A_MET112 recovery, but its mean predicted B occupancy was only 0.102 (density only) and 0.096 (soft physics).
- B_MET112 was the most balanced site: mean A/B was 0.548/0.361 for omit 2mFo-DFc, 0.358/0.327 for omit mFo-DFc, and 0.401/0.530 for kick-omit under density-only fitting.
- The deposited B state was almost absent for both ARG129 sites. Mean predicted B occupancy was 0.000-0.046 across all maps and conditions.
- Soft physics reduced total both-found recoveries from 49 to 36 across the three maps and reduced full successes from 21 to 19. It therefore did not resolve occupancy collapse.

## Sterics and rotamers

Counts below are active endpoints with a direct or symmetry-mate distance below 2.0 A. Canonical counts are starts for which every active endpoint was rotamer-canonical, out of 250 starts per row.

| Map | Condition | Direct clashes | Symmetry clashes | All-active canonical starts |
|---|---:|---:|---:|---:|
| Omit 2mFo-DFc | Density only | 105/480 | 67/480 | 51/250 |
| Omit 2mFo-DFc | Soft physics | 119/446 | 58/446 | 51/250 |
| Omit mFo-DFc | Density only | 151/602 | 49/602 | 36/250 |
| Omit mFo-DFc | Soft physics | 147/585 | 51/585 | 46/250 |
| Kick-omit | Density only | 114/490 | 64/490 | 42/250 |
| Kick-omit | Soft physics | 100/427 | 60/427 | 59/250 |

Soft physics produced only a modest aggregate steric/rotamer improvement and sometimes worsened direct clashes. It did not make the recovered ensembles consistently canonical or clash-free.

## tmol audit

The audit scored 926 active conformers for omit 2mFo-DFc, 1,187 for omit mFo-DFc, and 917 for kick-omit with tmol 0.1.40. Energy deltas below are relative to the lower-energy deposited endpoint for the same site.

- Across maps, only about 25-29% of active endpoints were within 10 tmol energy units of the better deposited endpoint.
- Roughly 39-49% were more than 50 units above it. Soft physics did not reduce this fraction overall.
- MET112 and ASP114 were the physically more plausible cases: median deltas were generally 4-18 units, although clashes remained.
- ARG129 was the decisive failure. Median deltas ranged from 38 to 346 units under density-only fitting and 53 to 275 under soft physics. Depending on map and chain, 49-74% of ARG endpoints were more than 50 units above the deposited reference.

This means the extra recovery from omit mFo-DFc is not sufficient evidence of a valid crystallographic ensemble. Much of the ARG signal terminates in high-energy, clashing, or noncanonical conformations.

## Interpretation

The synthetic renderer creates a smooth, model-consistent objective in which the two deposited states are recoverable. Experimental maps contain phase/model bias, noise, cancellation, neighboring-atom contributions, and local features not represented by the current differentiable sampling loss. Changing from omit 2mFo-DFc to difference density exposes more missing-state signal at A_MET112, but it does not create a broadly navigable, physically correct five-site landscape.

The current soft-physics term is not strong or well-shaped enough to steer optimization into valid rotamer basins. It slightly changes active-conformer counts and clash rates but does not close the synthetic-to-experimental recovery gap. The main bottleneck is therefore still the experimental-density objective and its coupling to conformer/occupancy optimization, not merely the choice among these three map variants.

## Decision

The two-site validation gate fails. Do not treat the learned/optimized experimental landscape as validated yet. The most informative next experiment is a constrained or library-based optimizer on omit mFo-DFc: enumerate canonical rotamers first, locally refine chi angles and occupancies, and compare that controlled upper bound against the current unconstrained optimizer. This will separate map insufficiency from optimizer/parameterization failure while preventing the high-energy ARG endpoints seen here.
