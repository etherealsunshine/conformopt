# Frozen-v3 single-recovery occupancy and pooled-NNLS diagnostic

**Date:** 2026-07-29
**Scope:** saved frozen-v3 endpoints only
**Frozen metric:** `qfit-synth20-merge050-one-to-one-tmol044-v3` (unchanged)
**Optimizer/audit reruns:** none
**Production change:** none

## Population and definitions

The requested `142 major-only / 45 minor-only` split is the historical
raw-greedy `found_A/found_B` diagnostic on the frozen endpoint rows. It is not
the exact protected-merge v3 partition. For provenance, exact v3 has 129
major-only, 45 minor-only, and 9 equal-occupancy single-state starts excluded
from minor/major ranking.

For each requested start, the recovered representative is the occupancy
`>0.10` slot with the lowest conventional symmetry-aware RMSD to the recovered
state. Every other optimizer-active slot (`occupancy >0.05`) is reported as an
unmatched slot. The tables also retain the raw-greedy recovered-state occupancy
sum and the recovered representative's v3 merged-component occupancy.

The A–B midpoint is atomwise after applying the valid equivalent-atom
permutation of deposited B that minimizes deposited A–B RMSD.

## 1. Occupancy structure of single-recovery starts

### Major-only starts

| Quantity | Result |
|---|---:|
| Starts | 142 |
| Deposited major occupancy | median 0.630; IQR 0.580–0.670 |
| Recovered representative occupancy | median 0.819; IQR 0.674–0.891 |
| Raw recovered-state occupancy sum | median 0.878; IQR 0.681–0.935 |
| V3 merged occupancy of representative | median 0.819; IQR 0.674–0.902 |
| Unmatched active occupancy | median 0.160; IQR 0.059–0.294 |
| Sub-mask occupancy (`<=0.05`) | median 0.017; IQR 0.004–0.041 |
| Representative closer to deposited major than 1.0 | 73 / 142 |
| Representative within ±0.20 of deposited major | 74 / 142 |
| Representative within 0.20 of 1.0 | 75 / 142 |
| Representative occupancy `>=0.95` | 24 / 142 |
| Raw recovered-state sum `>=0.95` | 31 / 142 |

This is not a clean fork to either extreme. Major-only starts are overweighted
relative to the deposited major on average, but most do not put essentially
all mass into the recovered representative: the median balance outside it is
0.181 after including unmatched active and sub-mask slots. Only 24/142
representatives are at least 0.95 occupancy.

The occupancy result therefore agrees qualitatively with the residual probe's
larger positive minor-lobe residual (`8.663` versus `1.678` at the major lobe):
minor mass was not generally absorbed into one unit-occupancy major conformer.
However, only 74/142 representatives are within the frozen ±0.20 occupancy
tolerance of the deposited major, so “occupancy is right and only position is
wrong” is not uniformly true. The panel contains a mixture of partial
single-state collapse, approximately correct major weight with misplaced
balance, and intermediate cases.

### Minor-only starts and label-swap test

| Quantity | Result |
|---|---:|
| Starts | 45 |
| Deposited minor occupancy | median 0.370; IQR 0.370–0.440 |
| Recovered minor representative occupancy | median 0.346; IQR 0.282–0.532 |
| Unmatched active occupancy | median 0.639; IQR 0.423–0.710 |
| Sub-mask occupancy | median 0.0046; IQR approximately 0–0.019 |
| Representative within ±0.20 of deposited minor | 32 / 45 |
| Representative closer to deposited major than minor | 13 / 45 |
| Representative within ±0.20 of deposited major | 8 / 45 |
| Both closer to major and within ±0.20 of major | **6 / 45** |

Using the explicit conjunction “closer to deposited major than deposited
minor” and “within ±0.20 of deposited major,” 6/45 are major-weight label-swap
cases. They occur at 3NY7 (1), 5DBA (3), and 5Z8H (2). The broader one-criterion
counts are reported because close deposited occupancies make the classification
sensitive: 13/45 are closer to major, while 8/45 lie within ±0.20 of major.
Most minor-only starts are not label swaps under the combined definition.

The complete per-start table contains the representative, raw recovered-state
sum, v3 merged occupancy, every unmatched-slot occupancy, and sub-mask total.

## 2. Unmatched-slot geometry

There are 259 optimizer-active slots outside the recovered representative in
the 187 requested starts.

| Distance | Median | IQR | Within 1 Å |
|---|---:|---:|---:|
| To missed deposited state | 1.608 Å | 1.367–2.476 Å | 23 / 259 |
| To recovered deposited state | 2.690 Å | 1.568–3.558 Å | 36 / 259 |
| To A–B midpoint | 2.035 Å | 1.512–2.389 Å | 35 / 259 |

An exclusive closest-location read gives:

| Outcome | Slots |
|---|---:|
| No missed/recovered/midpoint distance below 1 Å | **213 / 259 (82.2%)** |
| Midpoint below 1 Å and closest | 15 / 259 (5.8%) |
| Recovered state below 1 Å and closest | 29 / 259 (11.2%) |
| Missed state below 1 Å and closest | 2 / 259 (0.8%) |

There is a real threshold-edge tail: 75/259 slots are 1.0–1.5 Å from the
missed state, and the missed state is the closest of the three references for
61 of them. But it is not the dominant distribution; the median missed-state
RMSD is 1.608 Å and 82.2% have no reference within 1 Å.

Of the 23 slots within 1 Å of the missed state, 21 are at 2VFP. That site has
overlapping A/B recovery neighborhoods and the known protected-anchor
double-counting caveat, so these are not general evidence that the threshold
alone causes single recovery. The remaining two are at 7UO8.

For high unmatched slots (`occupancy >0.10`), missed-state RMSD is median
1.525 Å and 13/167 are within 1 Å, reproducing the earlier mass-dumping
diagnostic. The position of the balance is therefore mostly unrelated to the
missed state; mode averaging exists but is small.

For comparison, among raw both-found starts, extras outside the selected A/B
representatives have median nearest-deposited RMSD 0.504 Å and 243/325
(74.8%) lie within 1 Å. Restricting to occupancy `>0.10` gives 73/95 (76.8%).
This is directionally consistent with the earlier 83% comparison, but not
numerically identical because that value used the frozen-headline/high-extra
cohort rather than all raw both-found starts.

## 3. Fixed-position pooled-conformer NNLS

For each site, all optimizer-active saved conformers from 50 starts were
clustered using the frozen protected single-linkage threshold of 0.5 Å.
Cluster representatives were held fixed. Unit-occupancy Gaussian densities
used the exact production atom model. The pre-z-score deposited A/B mixture
was reconstructed and checked against the saved optimizer synthetic target:
median relative error was `2.82e-7`, with median correlation effectively 1.

The requested system was then solved:

```text
min ||Gq - b||²  subject to q >= 0
G = DᵀD
b = Dᵀ target
```

Nonzero means `q > 1e-6`. A/B occupancy below uses nearest-state geometric
assignment under the 1 Å recovery threshold; the raw tables also include
one-to-one values.

| Site | Minor-cluster contributing starts | 0.5 Å clusters | Nonzero | Fitted A / deposited A | Fitted B / deposited B | A+B deficit | cond(G) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1ZV8 ASN1 | 2 | 9 | 4 | 0.262 / 0.330 | 0.659 / 0.670 | 0.079 | 6,722 |
| 2V05 HIS168 | 28 | 8 | 4 | 0.645 / 0.610 | 0.055 / 0.390 | 0.300 | 753 |
| 2VFP TYR417 | 45 | 4 | 4 | 0.363 / 0.420 | 0.543 / 0.580 | 0.093 | 14.6 |
| 3A1C ARG447 | 49 | 18 | 6 | 0.351 / 0.440 | 0.435 / 0.560 | 0.214 | 3,507 |
| 3GMI GLU5 | 40 | 7 | 6 | 0.762 / 0.770 | 0.214 / 0.230 | 0.024 | 15.9 |
| 3K8W SER337 | 50 | 2 | 2 | 0.534 / 0.535 | 0.462 / 0.465 | 0.003 | 3.57 |
| 3NY7 LYS19 | 49 | 23 | 9 | 0.406 / 0.450 | 0.516 / 0.550 | 0.078 | 2,042 |
| 4C16 MET258 | 37 | 17 | 11 | 0.346 / 0.370 | 0.533 / 0.630 | 0.121 | 369 |
| 4MKM THR77 | 50 | 2 | 2 | 0.412 / 0.410 | 0.584 / 0.590 | 0.003 | 6.66 |
| 5DBA TRP325 | 45 | 10 | 4 | 0.546 / 0.550 | 0.424 / 0.450 | 0.030 | 152 |
| 5KWB PHE591 | 50 | 2 | 2 | 0.388 / 0.440 | 0.605 / 0.560 | 0.007 | 9.47 |
| 5Z8H MET730 | 18 | 8 | 6 | 0.297 / 0.280 | 0.673 / 0.720 | 0.030 | 146 |
| 6H59 ARG144 | N/A (equal occupancy) | 12 | 8 | 0.388 / 0.500 | 0.469 / 0.500 | 0.143 | 339 |
| 6Y4G CYS260 | 50 | 2 | 2 | 0.438 / 0.439 | 0.559 / 0.561 | 0.002 | 1.56 |
| 7F72 MET103 | 47 | 6 | 5 | 0.439 / 0.480 | 0.539 / 0.520 | 0.022 | 91.7 |
| 7T7A LEU396 | 44 | 8 | 5 | 0.378 / 0.380 | 0.619 / 0.620 | 0.003 | 270 |
| 7UO8 GLN53 | 24 | 13 | 9 | 0.170 / 0.340 | 0.587 / 0.660 | 0.243 | 1,067 |
| 8DJ2 VAL893 | 50 | 3 | 3 | 0.616 / 0.660 | 0.298 / 0.340 | 0.086 | 12.6 |
| 8FBE ILE92 | 49 | 3 | 2 | 0.338 / 0.380 | 0.632 / 0.620 | 0.029 | 8.97 |
| 8Q6Q ASP81 | 50 | 5 | 4 | 0.557 / 0.570 | 0.445 / 0.430 | -0.002 | 76.5 |

The pooled absolute A+B deficit has median **0.030**, versus the supplied
per-start median 0.048; 11/20 sites fall below 0.048. Pooling therefore closes
the median deficit partially, but not uniformly. The mean absolute deficit is
0.076 and the maximum is 0.300. Large residual deficits remain at 2V05,
7UO8, 3A1C, 6H59, and 4C16.

The solution is usually not a clean two-conformer refit: 15/20 sites assign
nonzero weight to more than two clusters; the median is four and the maximum
is eleven. Gram conditioning is also poor at several sites: median
`cond(G)=119`, with a maximum of 6,722. Thus individual fitted weights can be
unstable even when aggregate density fit improves.

The pool-limitation framing is confirmed. 1ZV8 has only two starts containing
an active `>0.10` minor-neighborhood conformer (one becomes a full both-found
start), and its refit remains deficient and extremely ill-conditioned. 5Z8H
has 18 such contributing starts (16 both-found plus two minor-only) and its
aggregate deficit falls to 0.030. Pool richness is necessary but not
sufficient: 2V05 has 28 minor contributors yet retains a 0.300 deficit.

This is precision improvement over already-discovered positions, not a
discovery mechanism. It also does not overcome the independent density-margin
floor of roughly 0.19 occupancy at wide separation and 0.62 at close
separation. No production refit is justified from this diagnostic.

## Conclusion

The major-only population is not predominantly a unit-occupancy collapse.
There is usually meaningful mass outside the recovered representative, in
qualitative agreement with the positive minor residual. But that mass is
usually geometrically misplaced: mode averaging and threshold-edge misses are
secondary, while unrelated mass dumping dominates.

Pooling demonstrates that the cross-start conformer library can reduce the
median occupancy deficit, especially at sites with reliable A/B modes. Its
multi-cluster solutions, high condition numbers, and failure at sparse or
ambiguous sites argue against production implementation. The next discovery
problem remains positional proposal/search, not merely an occupancy-only
refit.

## Artifacts

Authoritative pod directory:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_water_minstate_v2_single_rule_v1/
analysis/frozen_v3_occupancy_pooling_diagnostic_v3
```

Files:

```text
single_recovery_per_start.csv
single_recovery_unmatched_active_slots.csv
successful_start_unmatched_active_slots.csv
pooled_nnls_per_site.csv
pooled_nnls_clusters.csv
pooled_cluster_members.csv
midpoint_equivalent_atom_permutations.json
summary.json
```

The earlier `...diagnostic_v1` and `...diagnostic_v2` directories are
superseded development diagnostics with narrower unmatched-slot or mismatched
target definitions. They were preserved rather than overwritten. Only
`...diagnostic_v3` is authoritative.
