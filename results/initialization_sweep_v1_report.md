# Synthetic 20-site initialization sweep v1

**Completed:** 2026-07-29  
**Metric:** `qfit-synth20-merge050-one-to-one-tmol044-v3` (unchanged)  
**Starts:** 50 per site, 1,000 per arm, `seed = 41 + start`  
**Optimizer source hash:** `c0958a13aa83c243b4a40472f2c7559600a4193ab22e1ef4cdd07ec38437548d`

The frozen control was reused. Three initialization-only arms were run:

- `canonical_free`: four stratified canonical starts with 12-degree jitter,
  independent of deposited coordinates.
- `canonical_a_anchor`: one deposited-A anchor plus three stratified canonical
  starts.
- `deposited_a_cloud_120`: the control's deposited-A-centered Gaussian cloud,
  widened from approximately 60 to 120 degrees.

## Main result

No experimental arm improved the primary minor-conformer recovery endpoint.
The wider cloud was the closest aggregate result: it produced 739 rather than
742 recovered starts and 632 rather than 626 strict starts, while increasing
minor misses from 142 to 149 and major misses from 45 to 54. The six-start
strict increase is therefore not evidence for the stated mechanism.

| Initialization | Found | + occupancy | + rotamer | + direct | + symmetry | Strict | Minor missed | Major missed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen control | 742 | 714 | 710 | 710 | 710 | 626 | 142 | 45 |
| Canonical, A-free | 720 | 698 | 696 | 696 | 696 | 608 | 169 | 57 |
| Canonical + A anchor | 719 | 704 | 703 | 703 | 703 | 615 | 183 | 42 |
| Deposited-A cloud, 120° | 739 | 723 | 717 | 717 | 717 | 632 | 149 | 54 |

Minor/major misses use the raw-greedy, control-comparable classification
requested for this experiment. The cascade uses the frozen v3 merge-then-
one-to-one metric.

## Separation-stratified primary endpoint

| Initialization | Separation >2.5 Å minor/major missed | Separation ≤2.5 Å minor/major missed |
|---|---:|---:|
| Frozen control | 55 / 24 | 87 / 21 |
| Canonical, A-free | 65 / 29 | 104 / 28 |
| Canonical + A anchor | 90 / 10 | 93 / 32 |
| Deposited-A cloud, 120° | 68 / 28 | 81 / 26 |

Initialization changes did not preferentially improve well-separated sites.
This falsifies the prediction that broader or canonical initial coverage would
mainly help above approximately 2.5 Å.

## Mechanism diagnostics

Among slots beginning within 1 Å of the minor conformer, the fraction ending
farther from it was 62.4% for control, 59.2% for A-free canonical, 73.5% for
canonical+A, and 53.9% for the 120-degree cloud. Median net chi travel was
124–137 degrees in every arm.

The anchored arm supplies the clearest negative result: explicitly placing a
slot near deposited A increased the number of near-minor initial slots from
696 to 1,162, but 854 of those moved away and minor misses rose from 142 to
183. Initialization coverage alone is therefore not the limiting mechanism.

## Other diagnostics

| Initialization | Starts with same-state duplicates | Starts with unmatched extra >0.10 | Median A+B occupancy deficit | Assigned pairs <0.5× deposited separation |
|---|---:|---:|---:|---:|
| Frozen control | 237 | 253 | 0.0318 | 8 |
| Canonical, A-free | 247 | 261 | 0.0332 | 4 |
| Canonical + A anchor | 250 | 277 | 0.0353 | 4 |
| Deposited-A cloud, 120° | 248 | 237 | 0.0294 | 5 |

The wider cloud modestly reduced unmatched extras and occupancy deficit but did
not reduce same-state duplication or minor misses.

## Per-site found / strict

| Site | Control | Canonical, A-free | Canonical + A | 120° cloud |
|---|---:|---:|---:|---:|
| 1ZV8 ASN1 | 1 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2V05 HIS168 | 22 / 21 | 19 / 16 | 30 / 27 | 21 / 19 |
| 2VFP TYR417 | 14 / 8 | 13 / 3 | 6 / 4 | 8 / 3 |
| 3A1C ARG447 | 47 / 44 | 48 / 47 | 48 / 47 | 47 / 40 |
| 3GMI GLU5 | 41 / 39 | 39 / 34 | 37 / 31 | 40 / 35 |
| 3K8W SER337 | 50 / 50 | 50 / 50 | 50 / 50 | 50 / 50 |
| 3NY7 LYS19 | 48 / 38 | 50 / 41 | 49 / 40 | 49 / 38 |
| 4C16 MET258 | 20 / 19 | 20 / 19 | 14 / 14 | 28 / 22 |
| 4MKM THR77 | 50 / 50 | 50 / 50 | 50 / 50 | 50 / 50 |
| 5DBA TRP325 | 37 / 37 | 40 / 40 | 33 / 33 | 40 / 40 |
| 5KWB PHE591 | 49 / 49 | 42 / 42 | 47 / 46 | 48 / 48 |
| 5Z8H MET730 | 16 / 1 | 10 / 0 | 9 / 0 | 16 / 2 |
| 6H59 ARG144 | 41 / 39 | 37 / 37 | 39 / 37 | 39 / 38 |
| 6Y4G CYS260 | 50 / 50 | 50 / 50 | 50 / 50 | 50 / 50 |
| 7F72 MET103 | 45 / 5 | 45 / 10 | 43 / 5 | 42 / 8 |
| 7T7A LEU396 | 44 / 34 | 39 / 30 | 47 / 38 | 47 / 39 |
| 7UO8 GLN53 | 18 / 18 | 18 / 18 | 17 / 16 | 14 / 13 |
| 8DJ2 VAL893 | 50 / 50 | 50 / 50 | 50 / 50 | 50 / 50 |
| 8FBE ILE92 | 49 / 49 | 50 / 49 | 50 / 50 | 50 / 49 |
| 8Q6Q ASP81 | 50 / 25 | 50 / 22 | 50 / 27 | 50 / 38 |

The wider cloud's net strict gain is heterogeneous: 8Q6Q gains 13 strict and
4C16 gains 3, while 2VFP and 7UO8 each lose 5 and 3A1C loses 4.

## Provenance

Authoritative pod run:

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_initialization_sweep_v1
```

Compiled analysis:

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_initialization_sweep_v1/analysis_summary_v1
```

The frozen metric, geometry rules, optimizer environment, tmol rule, tolerance,
merge threshold, one-to-one assignment, schedules, weights, checkpoint, and
seeds were not changed.
