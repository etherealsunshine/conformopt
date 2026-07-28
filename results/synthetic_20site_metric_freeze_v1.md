# Synthetic 20-Site Metric Freeze v1

**Frozen metric version:** `qfit-synth20-assignedpair-tmol044-v1`
**Frozen baseline:** 621 / 1000 starts
**Date:** 2026-07-27

> **Superseded on 2026-07-28, ultimately by
> `qfit-synth20-merge050-one-to-one-tmol044-v3`.** The v1 audit assigned
> each conformer independently to its nearest deposited state. That greedy
> labeling undercounted both-state recovery by 13 starts because it did not
> enforce a distinct conformer for A and B, while also summing occupancies
> across geometrically distinct same-state labels. No model experiment was run
> between the versions. This document is retained as the immutable v1 record.

This document freezes the metric used for subsequent synthetic 20-site model
experiments. The definition below does not change during those experiments.
Any future change to a rule, table, weight, threshold, tolerance, conformer
selection rule, or gate order requires:

1. a new metric version string; and
2. a complete re-audit of this frozen baseline under the new version.

Numbers from different metric versions are not directly comparable.

## Frozen definition

### Rule strings

```text
optimizer environment
2026-07-24-altloc-minstate-water-minstate-v2

geometry
2026-07-24-altloc-minstate-rotwidth-hisunion-aromatic45-matched-tmol-v2

tmol
frozen_matched_deposited_minstate_v1
```

### Source hashes

| Source | SHA-256 |
|---|---|
| `density_denoiser/five_site_optimizer.py` | `367acfaba8f6d0da660fac45ace5c0c696f705bbdb05b60d2072b8724b87cbd6` |
| `density_denoiser/clash_environment.py` | `ae5940329de4ccc1d1f729f1eb0004ad607152bc347bc8e92636a0a512ab44df` |
| `density_denoiser/residue_geometry.py` | `2e6d2b57338e464928024f69d704968aa78cba0f83fc0ea382782b8add06c2b4` |
| `density_denoiser/audit_five_site_endpoints.py` | `6002c5e0763b79c93ecea07a43d316fb3600c2068e6749c42555ab23d96e0cb1` |
| `five_site_tmol_audit.py` | `fd13c8f494e16ce8909e1f3202bef5510c4da85fc06f73162f1d0b5e2d9c5c8d` |

All 20 sites carry this same rule and source-hash set.

### Conformer rule

The frozen headline uses the **assigned A/B pair**:

- among active conformers, select the lowest-RMSD conformer assigned to
  deposited A and the lowest-RMSD conformer assigned to deposited B;
- apply the physical and tmol gates to those two selected conformers;
- do not make every additional active K=4 slot an independent headline
  failure opportunity.

All-active results, the number and occupancy of extra active slots, and their
gate failures remain mandatory secondary diagnostics.

### Gate order

1. Both deposited A and B are found at conventional, chemically
   symmetry-aware RMSD `< 1.0 Å`.
2. Recovered A/B occupancies are each within `±0.20` of deposited occupancy.
3. Select the assigned A/B pair as defined above.
4. Both selected conformers pass the frozen residue/chi-specific rotamer gate.
5. Both pass the direct-clash gate: no contact below `2.0 Å`.
6. Both pass the crystallographic-symmetry-clash gate: no contact below
   `2.0 Å`.
7. Both have finite matched tmol margins
   `candidate − matched deposited ≤ +0.44`.

The frozen baseline cascade is:

```text
both A/B found                 729 / 1000
+ occupancy                   715 / 1000
+ assigned pair available     715 / 1000
+ rotamer                     711 / 1000
+ direct clash                711 / 1000
+ symmetry clash              711 / 1000
+ matched tmol <= +0.44       621 / 1000
```

The companion all-active count at the same tolerance is 534 / 1000. The
previously quoted 536 all-active and 625 assigned-pair counts are the `+0.5`
column, not the exact `+0.44` result.

## 1. Current-endpoint validation of +0.44

The current composite has 1,944 finite A/B-matched active conformers in the
requested RMSD range.

| RMSD to matched deposited | Conformers | Pass at +0.44 | Pass rate |
|---|---:|---:|---:|
| `≤0.1 Å` | 532 | 530 | 99.62% |
| `0.1–0.3 Å` | 776 | 718 | 92.53% |
| `0.3–0.6 Å` | 449 | 346 | 77.06% |
| `0.6–1.0 Å` | 187 | 121 | 64.71% |

Pass rate rises monotonically as RMSD improves. In the current `≤0.1 Å`
population:

```text
positive margins             176 / 532
positive-margin q99          +0.3576
positive-margin maximum      +0.5486
pass at +0.44                530 / 532
```

The prior calibration was q99 `+0.438` and maximum `+0.473` over 442
near-reproduction conformers. The current q99 moved downward rather than
upward, so `+0.44` remains conservative at the calibrated percentile. The two
current near-reproduction exceptions are:

| Candidate | Site/assignment | RMSD | Margin |
|---|---|---:|---:|
| `4MKM_A_THR77__037__2` | THR77 B | 0.0983 Å | +0.4815 |
| `5DBA_A_TRP325__037__0` | TRP325 A | 0.0919 Å | +0.5486 |

The maximum is therefore not bounded by `+0.44`; the frozen tolerance is a
reproduction-scale q99 tolerance, not a guarantee that every `<0.1 Å`
conformer passes.

### Tolerance robustness and RMSD-cut sensitivity

| Tmol tolerance | Assigned pair | All active |
|---:|---:|---:|
| `+0.36` | 612 / 1000 | 531 / 1000 |
| `+0.44` (frozen) | 621 / 1000 | 534 / 1000 |

Moving from the current measured `≤0.1 Å` positive-margin q99 (`+0.3576`) to
the frozen `+0.44` changes the assigned-pair headline by 9 starts and the
all-active companion by 3 starts. The baseline is therefore insensitive at
the aggregate level over this tolerance interval.

| RMSD cutoff | Finite matched `n` | Positive-margin `n` | Positive q99 | Positive max |
|---:|---:|---:|---:|---:|
| `0.05 Å` | 313 | 94 | +0.2474 | +0.2476 |
| `0.10 Å` | 532 | 176 | +0.3576 | +0.5486 |
| `0.15 Å` | 799 | 197 | +1.1429 | +1.1563 |
| `0.20 Å` | 982 | 250 | +1.4814 | +2.6621 |
| `0.30 Å` | 1,308 | 353 | +1.9269 | +2.6621 |

The data support robustness within the strict reproduction regime
(`0.05–0.10 Å`), but **do not** support the stronger claim that the
`0.10 Å` cutoff is immaterial when expanded to `0.15–0.30 Å`. Those wider
populations contain real geometry-dependent margin and move q99 sharply.
The `0.10 Å` definition is therefore retained as part of the recorded
derivation rather than treated as arbitrary.

## 2. Extra active conformers and conformer-rule decision

Every active conformer for all 50 starts at 2V05 and 8Q6Q is recorded in the
diagnostic artifact, including occupancy, nearest deposited RMSD, pair
selection, and failed gates.

### 2V05 HIS168

```text
recovery + occupancy             22 / 50
assigned-pair pass at +0.44      21 / 50
all-active pass at +0.44          6 / 50
lost only because of extras      15 / 50
```

All 15 causal extra conformers fail only `tmol_unmatched`; they do not fail
rotamer, direct clash, or symmetry clash. Their occupancies are:

```text
range       0.0870–0.1721
median      0.1191
below 0.075 0 / 15
below 0.10  7 / 15
```

Across all 102 extra slots at this site, 86 fail only because tmol has no
matched deposited reference, two fail rotamer plus unmatched tmol, and 14
pass every applicable gate.

### 8Q6Q ASP81

```text
recovery + occupancy             50 / 50
assigned-pair pass at +0.44      27 / 50
all-active pass at +0.44         12 / 50
lost only because of extras      15 / 50
```

The 17 causal extra conformers in those 15 starts fail the `+0.44` tmol-margin
gate. Their occupancies are:

```text
range       0.0560–0.2436
median      0.1525
below 0.075 1 / 17
below 0.10  3 / 17
```

The extras do **not** generally cluster just above the `0.05` active mask.
Nevertheless, all-active is not the frozen headline: at 2V05 it turns
geometry-clean extra conformers into automatic failures solely because an
unmatched conformer has no matched deposited tmol reference. That is a
reference-definition artifact, not a demonstrated physical failure.

Assigned-pair therefore matches the stated A/B recovery task and is frozen as
the headline. All-active and extra-slot occupancy remain reported so that
hallucinated or fragmented occupancy is not hidden.

### Named reported quantity: extra-active-conformer rate

Across the complete 1,000-start composite, 540 starts (54.0%) produce at
least one active conformer outside the selected A/B pair. There are 1,030
such slots:

| Population | Starts with extras / starts | Extra slots | Occupancy median (IQR) | Nearest-deposited RMSD median (IQR) | Geometry pass |
|---|---:|---:|---:|---:|---:|
| All starts | 540 / 1000 | 1,030 | 0.167 (0.073–0.454) | 1.086 Å (0.268–1.567) | 962 / 1,030 (93.40%) |
| Occupancy-qualified | 257 / 715 | 309 | 0.069 (0.057–0.112) | 0.502 Å (0.210–0.907) | 304 / 309 (98.38%) |
| Frozen-headline successes | 208 / 621 | 248 | 0.066 (0.056–0.093) | 0.463 Å (0.155–0.782) | 244 / 248 (98.39%) |

Thus the assigned-pair headline does hide a measurable over-modeling signal:
33.5% of headline-successful starts contain at least one extra active slot,
and nearly all of those slots pass the geometry gates. Their generally low
occupancy should be reported with, not substituted for, the headline. The
site-specific causal-extra medians remain 0.119 at 2V05 and 0.153 at 8Q6Q;
those are appreciably above the global headline-success extra median.

### Post-freeze reporting convention for extras

The reported extra-active-conformer threshold is `occupancy > 0.10`. This
changes only the named over-modeling diagnostic; it does not change the
frozen metric, optimizer, physics mask, audit gates, or baseline.

At this reporting threshold, 195/248 extras attached to frozen-headline
successes are excluded, headline extra-bearing starts fall from 208/621 to
52/621, and 0/1,242 matched headline conformers are lost. Across all starts,
376/1,030 extras are excluded, extra-bearing starts fall from 540/1,000 to
365/1,000, and 0/1,458 matched conformers are lost. Re-evaluating the
companion all-active diagnostic with this activity definition gives
598/1,000 instead of 534/1,000.

The zero-matched-loss result is partly definitional: assigned-pair recovery
already requires each selected A/B conformer to exceed 0.10 occupancy. The
threshold is therefore documented as aligned with the recovery definition,
not as an independently optimized biological cutoff. The full signed-gradient
and threshold sweep is in
`results/occupancy_gradient_activity_threshold_diagnostic_v1.md`.

## 3. Rotamer false acceptance

MolProbity is not installed on the pod. The equivalent independent check uses
tmol's bundled backbone-dependent Dunbrack joint-state library.

The evaluated population is every current active endpoint conformer that
passes the production rotamer gate—2,423 conformers, not a subsample. A
production pass disagrees with the independent library when no joint state at
the nearest 10-degree phi/psi bin:

- has library probability at least `0.3%`; and
- covers every chi within the production gate's own per-chi width.

Terminal 2-fold chemical symmetries are respected. This specifically measures
joint-state disagreement caused by per-chi factorization; it does not replace
MolProbity's full empirical contour implementation.

### Deposited-control baseline

The identical classifier was run on the broad deposited A/B control set
(`20` conformers per residue type, `340` total). The direct apples-to-apples
control column conditions on passing the production rotamer gate, as the
endpoint population does. The final column retains all deposited controls,
including the 12 known production-gate false rejections.

| Residue | Endpoint disagreement | Deposited production-pass disagreement | All deposited disagreement |
|---|---:|---:|---:|
| ARG | 5 / 224 (2.23%) | 2 / 20 (10.00%) | 2 / 20 (10.00%) |
| ASN | 0 / 185 (0%) | 0 / 20 (0%) | 0 / 20 (0%) |
| ASP | 0 / 146 (0%) | 1 / 19 (5.26%) | 1 / 20 (5.00%) |
| CYS | 0 / 108 (0%) | 0 / 19 (0%) | 1 / 20 (5.00%) |
| GLN | 54 / 114 (47.37%) | 1 / 19 (5.26%) | 1 / 20 (5.00%) |
| GLU | 0 / 105 (0%) | 0 / 20 (0%) | 0 / 20 (0%) |
| HIS | 0 / 144 (0%) | 0 / 20 (0%) | 0 / 20 (0%) |
| ILE | 0 / 103 (0%) | 1 / 20 (5.00%) | 1 / 20 (5.00%) |
| LEU | 3 / 120 (2.50%) | 0 / 20 (0%) | 0 / 20 (0%) |
| LYS | 3 / 120 (2.50%) | 0 / 17 (0%) | 2 / 20 (10.00%) |
| MET | 4 / 339 (1.18%) | 0 / 20 (0%) | 0 / 20 (0%) |
| PHE | 0 / 108 (0%) | 0 / 17 (0%) | 0 / 20 (0%) |
| SER | 0 / 141 (0%) | 0 / 20 (0%) | 0 / 20 (0%) |
| THR | 0 / 131 (0%) | 0 / 20 (0%) | 0 / 20 (0%) |
| TRP | 0 / 115 (0%) | 0 / 20 (0%) | 0 / 20 (0%) |
| TYR | 0 / 72 (0%) | 0 / 17 (0%) | 0 / 20 (0%) |
| VAL | 0 / 148 (0%) | 0 / 20 (0%) | 0 / 20 (0%) |
| **Total** | **69 / 2,423 (2.85%)** | **5 / 328 (1.52%)** | **8 / 340 (2.35%)** |

The endpoint total is close to the unconditioned deposited-control total and
only 1.33 percentage points above the production-pass control rate. The
binary 2.85% must therefore be described as a **lower bound on false
acceptance**, not as a calibrated false-acceptance estimate. GLN remains a
real classifier/population outlier: its endpoint conformers all come from one
site (7UO8 GLN53), so 54/114 has the same reduced ratio as 9/19 but not 114
independent site-level observations.

### Rotameric versus semi-rotameric treatment

| Library treatment | Endpoint disagreement | Deposited production-pass disagreement | All deposited disagreement |
|---|---:|---:|---:|
| Rotameric | 15 / 1,434 (1.05%) | 3 / 176 (1.70%) | 6 / 180 (3.33%) |
| Semi-rotameric | 54 / 989 (5.46%) | 2 / 152 (1.32%) | 2 / 160 (1.25%) |

The semi-rotameric set is ASN, ASP, GLN, GLU, HIS, PHE, TRP, and TYR. The
current classifier enumerates the library's tabulated terminal-chi bins as
discrete joint states above the `0.3%` probability floor; it does **not**
integrate the continuous terminal-chi density. That makes it a poor absolute
classifier for semi-rotameric residues. However, the elevation is not shared
by every semi-rotameric type—only GLN is elevated—so the current evidence
does not justify labeling the entire 5.46% grouped rate a generic
semi-rotameric artifact. It is a GLN/site-specific result confounded by that
discrete-bin treatment.

### Continuous distance to a qualifying joint state

Distances below are periodic angular distances to the nearest joint state
with library probability at least `0.3%`. `max χ` is the maximum per-chi
distance for each conformer. Values are median / q95 / maximum in degrees.
The authoritative CSV also records minimum, q25, and q75.

| Residue | Endpoint per-chi distance | Deposited per-chi distance | Endpoint max χ | Deposited max χ |
|---|---|---|---:|---:|
| ARG | χ1 7.9/15.9/106.3; χ2 6.0/28.8/106.8; χ3 6.3/23.0/92.4; χ4 6.5/16.3/85.0 | χ1 11.4/29.9/32.1; χ2 11.1/33.6/45.3; χ3 8.7/25.1/46.1; χ4 6.2/14.9/20.5 | 15.9/29.6/106.8 | 20.7/45.3/46.1 |
| ASN | χ1 4.5/11.4/20.2; χ2 3.1/12.8/14.1 | χ1 6.3/28.9/36.0; χ2 8.1/19.9/35.3 | 5.1/13.0/20.2 | 11.0/28.9/36.0 |
| ASP | χ1 8.3/12.5/24.5; χ2 7.7/29.1/34.9 | χ1 7.8/34.7/119.7; χ2 9.5/43.9/60.7 | 8.9/29.1/34.9 | 10.4/46.8/119.7 |
| CYS | χ1 7.1/10.3/25.6 | χ1 9.1/40.8/46.0 | 7.1/10.3/25.6 | 9.1/40.8/46.0 |
| GLN | χ1 17.4/114.4/114.4; χ2 10.0/18.1/35.3; χ3 17.7/45.4/47.3 | χ1 11.1/32.9/38.9; χ2 6.0/30.7/83.1; χ3 10.0/27.3/30.3 | 26.3/114.4/114.4 | 14.4/41.1/83.1 |
| GLU | χ1 18.6/25.2/31.4; χ2 3.7/12.5/12.9; χ3 6.5/7.7/25.3 | χ1 9.9/31.8/32.6; χ2 10.8/22.7/43.2; χ3 10.1/15.1/40.2 | 18.6/25.2/31.4 | 16.9/33.2/43.2 |
| HIS | χ1 12.0/31.3/38.8; χ2 7.9/13.8/55.4 | χ1 9.4/22.9/32.1; χ2 8.1/25.0/26.3 | 12.0/31.3/55.4 | 13.4/25.3/32.1 |
| ILE | χ1 13.8/13.9/34.6; χ2 21.7/21.8/24.7 | χ1 6.8/19.1/29.0; χ2 6.4/34.6/71.6 | 21.7/21.8/34.6 | 10.6/34.6/71.6 |
| LEU | χ1 1.2/11.9/73.6; χ2 6.5/17.4/20.9 | χ1 5.0/21.9/27.8; χ2 3.9/18.6/53.4 | 8.4/17.9/73.6 | 8.8/29.0/53.4 |
| LYS | χ1 4.7/16.0/30.9; χ2 16.7/36.8/101.6; χ3 8.4/14.5/29.9; χ4 12.3/17.1/23.5 | χ1 9.2/33.2/47.2; χ2 9.2/31.9/37.2; χ3 4.3/40.2/47.2; χ4 7.7/26.3/43.2 | 19.9/36.8/101.6 | 13.1/47.2/47.2 |
| MET | χ1 6.9/25.4/87.4; χ2 5.8/29.0/41.1; χ3 13.3/28.1/49.5 | χ1 6.4/29.0/29.9; χ2 8.1/15.5/20.7; χ3 10.1/30.5/48.7 | 15.0/33.6/87.4 | 15.4/30.8/48.7 |
| PHE | χ1 2.3/10.7/10.9; χ2 7.2/10.5/11.2 | χ1 7.1/16.5/20.5; χ2 5.4/13.3/16.2 | 7.2/10.8/11.2 | 9.3/16.5/20.5 |
| SER | χ1 3.5/4.8/8.6 | χ1 6.5/18.8/25.5 | 3.5/4.8/8.6 | 6.5/18.8/25.5 |
| THR | χ1 2.0/5.4/26.5 | χ1 5.3/16.9/18.5 | 2.0/5.4/26.5 | 5.3/16.9/18.5 |
| TRP | χ1 27.9/32.3/35.8; χ2 13.0/27.0/35.7 | χ1 6.8/20.9/21.3; χ2 7.5/14.3/14.6 | 28.1/32.3/35.8 | 12.2/20.9/21.3 |
| TYR | χ1 1.6/8.4/16.6; χ2 7.3/12.2/31.3 | χ1 6.1/17.4/19.1; χ2 5.4/10.4/10.7 | 7.3/15.0/31.3 | 8.2/17.4/19.1 |
| VAL | χ1 3.6/5.1/21.4 | χ1 8.4/17.0/17.6 | 3.6/5.1/21.4 | 8.4/17.0/17.6 |
| **All** | χ1 6.0/30.6/114.4; χ2 7.3/29.0/106.8; χ3 8.7/42.0/92.4; χ4 6.6/17.1/85.0 | χ1 7.5/28.6/119.7; χ2 7.5/27.0/83.1; χ3 9.1/30.7/48.7; χ4 6.4/24.9/43.2 | **9.3/34.7/114.4** | **11.0/33.1/119.7** |

The aggregate endpoint and deposited distance distributions are comparable,
which again shows why the binary disagreement is only a lower bound. The
production widths are used only for the binary coverage call; these
continuous distances expose deviations that a wide χ1 or terminal-χ window
can otherwise accept.

### Rejected classifier variants

Two exploratory methods were discarded and are not evidence for this metric:

- converting Dunbrack energies with `exp(−E)` and applying a `0.3%` cutoff;
- a fixed `3σ` angular threshold.

Both flagged deposited controls as outliers. Recording these rejected methods
prevents them from being mistaken for independent confirmations of the
reported joint-state test.

## 4. 5Z8H MET730

### Assignment breakdown

| Assignment | Conformers | RMSD median (range) | Margin median (range) | Pass +0.44 | Pearson / Spearman |
|---|---:|---:|---:|---:|---:|
| A | 20 | 0.328 (0.181–0.377) Å | +3.610 (−0.138–+7.403) | 3/20 | 0.900 / 0.851 |
| B | 43 | 0.092 (0.039–0.990) Å | +0.287 (+0.124–+7.088) | 36/43 | 0.989 / 0.825 |

Deposited occupancies are A `0.28`, B `0.72`.

A-assigned endpoints fail systematically, but the margin tracks their poorer
RMSD. This is not a constant reference offset.

### Frozen environments

| Reference | Atoms | Residues | Polymer span |
|---|---:|---:|---|
| frozen A | 176 | 24 | A:718–A:741 |
| frozen B | 176 | 24 | A:718–A:741 |

The PDB files differ only in the deposited MET730 sidechain coordinates. All
four of those target-sidechain coordinates are replaced by the candidate
before scoring. The atom count, residue set, and effective candidate
environment are therefore the same. There is no A/B environment-size or
residue-span mismatch.

### Deposited soft floors

| Deposited | VDW raw | Rotamer raw | Symmetry raw | Weighted total |
|---|---:|---:|---:|---:|
| A | 0.7482 | 0.0330 | 0 | 0.7648 |
| B | 0.7536 | 0.0102 | 0 | 0.7587 |

The VDW floor is the ordinary 3.0 Å squared hinge against the target
backbone:

- A: CB–N 2.4487 Å, CB–C 2.4942 Å, CG–CA 2.5659 Å.
- B: CB–N 2.4497 Å, CB–C 2.5032 Å, CG–CA 2.5484 Å.

There is no symmetry floor and no anomalous environment term. The evidence
supports a genuine endpoint-quality failure—especially A recovery—not a
site-specific frozen-reference bug. The frozen assigned-pair result remains
1/50 at `+0.44`.

## Known limitations retained in v1

- Partial labeled waters may be displaced without penalty in both optimizer
  and audit under the free-absent-state rule. Coupled candidate/water-state
  assignment is deferred because it is a joint objective/audit redesign with
  deposited-state-leakage risk.
- Occupancy has no smooth physics gradient. Physics uses the hard active mask
  and charges active slots equally.
- Assigned-pair headline scoring does not make extra active slots independent
  failures. Extra count, occupancy, all-active results, and failed gates must
  remain visible companion diagnostics. The named extra-active-conformer rate
  is 540/1000 starts overall and 208/621 frozen-headline successes.
- The production rotamer rule remains a per-chi factorization. The measured
  2.85% endpoint joint-library disagreement has a 2.35% all-deposited and
  1.52% production-pass deposited-control baseline. GLN's 47.37% endpoint
  outlier is recorded but not repaired in this metric version.

## Authoritative artifacts

```text
20-site composite
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_water_minstate_v2_single_rule_v1

tolerance, extra-slot, and 5Z8H diagnostics
.../analysis/freeze_diagnostics_v1

freeze interpretive addendum
.../analysis/freeze_interpretive_addendum_v2

authoritative endpoint joint-state classifier
.../analysis/dunbrack_joint_false_acceptance_v4

deposited-control and continuous-distance interpretation
.../analysis/joint_library_interpretation_v2
```

The earlier `dunbrack_false_acceptance_v1` and
`dunbrack_joint_false_acceptance_v1/v2` directories are exploratory,
superseded classifier attempts and are not metric evidence.
