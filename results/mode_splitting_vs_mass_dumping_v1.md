# Mode splitting versus mass dumping

Date: 2026-07-28  
Scope: diagnostic on saved frozen endpoints only  
Frozen metric and optimizer: unchanged

## Definitions

- A recovered representative is selected independently for A and B as the
  best-RMSD conformer assigned to that state with occupancy `>0.10`.
- An extra is any structural slot with occupancy `>0.05` outside the
  independently selected recovered representatives.
- A high-occupancy extra has occupancy `>0.10`.
- The missed-state recovery neighbourhood is RMSD `<1.0 Å`, matching the
  frozen recovery criterion.

Selecting A and B independently is important for incomplete pairs: the
recovered state is retained as the representative and is not mislabeled as an
extra merely because the other state was missed.

## Result

Mass dumping dominates the recovery failures. High-occupancy extras generally
sit well outside the missed deposited state's recovery neighbourhood.

| Failed-recovery extra population | n | RMSD to missed median | q25–q75 | Within 1.0 Å |
|---|---:|---:|---:|---:|
| Occupancy `0.05–0.10` | 146 | 1.639 Å | 1.424–2.956 Å | 18/146 (12.3%) |
| Occupancy `>0.10` | 363 | 1.525 Å | 1.298–2.052 Å | 13/363 (3.6%) |

Among starts missing exactly one state, there are 177 high extras across 131
starts:

```text
median RMSD to missed state       1.525 Å
within 1.0 Å of missed state      13 / 177 (7.3%)
starts with any such extra        12 / 196 (6.1%)
median RMSD to recovered state    2.693 Å
closer to recovered than missed   43 / 177 (24.3%)
```

All 13 high extras within 1 Å of a missed state occur at **2VFP TYR417**.
They are distributed over 12 starts, all missing A while recovering B, and
every one is actually closer to deposited B than to the missed A. Thus even
the small near-missed tail is not clean evidence that a slot captured the
missing A mode; it is more consistent with duplicate or intermediate B-like
solutions at this already unstable site.

For the 75 starts missing both states, there are 186 high extras. None is
within 1.0 Å of either deposited state; median distance to the nearer missed
state is 1.567 Å.

The global interpretation is therefore:

- The optimizer usually never reaches the missed conformer's recovery basin.
- Occupancy is then retained by conformers elsewhere in coordinate space.
- Slot competition may still exist locally, especially at 2VFP, but it does
  not explain the panel-wide recovery gap.

No objective change follows from this diagnostic.

## Failed-recovery census

There are 271 starts that do not recover both deposited states:

| Missed state | Starts |
|---|---:|
| A only | 146 |
| B only | 50 |
| Both | 75 |

Of these, 235 carry at least one extra above 0.05 and 206 carry at least one
extra above 0.10.

This is consistent with the earlier headline partition:

```text
non-headline starts                         379
non-headline with extra >0.05              332
non-headline with extra >0.10              313
headline with extra >0.10                   52
```

## Per-site missed-state distances

The table reports high extras only (`occupancy >0.10`). A zero in the final
column means no high extra at that site enters the missed state's `<1 Å`
recovery neighbourhood.

| Site | Failed starts: A / B / both | Starts with high extra | High extras | Median RMSD to missed | Fraction <1 Å |
|---|---:|---:|---:|---:|---:|
| 1ZV8 ASN1 | 20 / 1 / 28 | 49 | 134 | 1.583 Å | 0% |
| 2V05 HIS168 | 6 / 4 / 18 | 28 | 54 | 1.583 Å | 0% |
| 2VFP TYR417 | 44 / 0 / 5 | 19 | 26 | 1.030 Å | 50.0% |
| 3A1C ARG447 | 1 / 2 / 0 | 3 | 4 | 1.244 Å | 0% |
| 3GMI GLU5 | 0 / 9 / 0 | 4 | 4 | 1.005 Å | 0% |
| 3NY7 LYS19 | 1 / 1 / 0 | 2 | 3 | 1.139 Å | 0% |
| 4C16 MET258 | 7 / 17 / 6 | 29 | 46 | 1.462 Å | 0% |
| 5DBA TRP325 | 8 / 5 / 0 | 8 | 9 | 3.296 Å | 0% |
| 5KWB PHE591 | 1 / 0 / 0 | 0 | 0 | — | — |
| 5Z8H MET730 | 26 / 2 / 6 | 13 | 13 | 1.525 Å | 0% |
| 6H59 ARG144 | 8 / 1 / 0 | 9 | 10 | 1.163 Å | 0% |
| 7F72 MET103 | 3 / 2 / 0 | 3 | 3 | 1.516 Å | 0% |
| 7T7A LEU396 | 5 / 0 / 1 | 6 | 9 | 1.386 Å | 0% |
| 7UO8 GLN53 | 15 / 6 / 11 | 32 | 47 | 1.525 Å | 0% |
| 8FBE ILE92 | 1 / 0 / 0 | 1 | 1 | 1.954 Å | 0% |

The five sites with zero failed-recovery starts are 3K8W SER337, 4MKM
THR77, 6Y4G CYS260, 8DJ2 VAL893, and 8Q6Q ASP81.

## Headline comparison population

The 52 headline starts retaining an extra above 0.10 contain 53 high extras.
Unlike failed starts, these extras are usually close to a deposited state
that was already successfully recovered:

```text
nearest-deposited RMSD median     0.500 Å
q25–q75                          0.219–0.755 Å
within 0.5 Å                     25 / 53 (47.2%)
within 1.0 Å                     44 / 53 (83.0%)
```

| Site | Starts | High extras | Median nearest-deposited RMSD | Fraction <1 Å |
|---|---:|---:|---:|---:|
| 2V05 HIS168 | 8 | 8 | 2.925 Å | 0% |
| 3A1C ARG447 | 1 | 1 | 1.493 Å | 0% |
| 3K8W SER337 | 7 | 8 | 0.032 Å | 100% |
| 4MKM THR77 | 1 | 1 | 0.035 Å | 100% |
| 5DBA TRP325 | 9 | 9 | 0.224 Å | 100% |
| 5KWB PHE591 | 1 | 1 | 0.128 Å | 100% |
| 7F72 MET103 | 1 | 1 | 0.470 Å | 100% |
| 8DJ2 VAL893 | 10 | 10 | 0.500 Å | 100% |
| 8Q6Q ASP81 | 14 | 14 | 0.694 Å | 100% |

This comparison shows that the slot system can split or duplicate modes when
both states have already been recovered. That behaviour is real, but it is
not the dominant explanation for missing-state failures.

## One-state-miss occupancy accounting

For all 196 starts missing exactly one state, the recovered representative
plus every active extra nearly exhausts the deposited A+B occupancy:

```text
absolute mass error median        0.0158
q95                               0.0812
within 0.05                       169 / 196 (86.2%)
within 0.10                       191 / 196 (97.4%)
within 0.20                       196 / 196 (100%)
```

This equality is largely imposed by the four-slot softmax: recovered
occupancy plus active-extra occupancy equals one minus sub-mask mass. It
shows where the occupancy went, but not whether the extras captured the
missed geometry.

Restricting the sum to extras actually within 1 Å of the missed state changes
the result:

```text
absolute mass error median        0.272
q25–q75                           0.087–0.483
starts with a high near-missed extra
                                   12 / 196
```

Therefore the mass belonging to the missing conformer is generally carried
by geometrically wrong slots. This is mass dumping, not mode splitting.

## Artifacts

Authoritative remote directory:

```text
/home/dev/qfit_unet_data/density_denoiser/
  heldout_twenty_synthetic_water_minstate_v2_single_rule_v1/
  analysis/mode_splitting_vs_mass_dumping_v2/
```

Files:

```text
failed_recovery_starts.csv
failed_recovery_extra_conformers.csv
failed_recovery_rmsd_distributions.csv
headline_high_extra_conformers.csv
headline_comparison_rmsd_distributions.csv
one_missed_state_mass_accounting.csv
per_site_summary.csv
summary.json
```

The raw conformer tables contain occupancy, RMSD to A, RMSD to B, RMSD to
each missed state, RMSD to the recovered state, and RMSD to the nearest
deposited state for every requested extra.
