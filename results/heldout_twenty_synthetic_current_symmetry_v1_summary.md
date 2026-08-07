# Current-symmetry 20-site synthetic baseline

Remote result root:

`/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_current_symmetry_v1`

Controller status: `complete`

Rules:

- Geometry: `2026-07-24-altloc-minstate-rotwidth-hisunion-aromatic45-matched-tmol-v2`
- tmol environment: `frozen_matched_deposited_minstate_v1`
- tmol tolerance is a post-hoc sweep; no tolerance is promoted.

The stale baseline used an inactive optimizer symmetry environment. Its 673
both-found count is shown only to expose site-level changes, not as a
model-progress comparator.

## Aggregate cascade

| Stage | Starts passing |
|---|---:|
| Both A/B found | 728 / 1000 |
| + occupancy | 698 / 1000 |
| + all-active rotamer | 691 / 1000 |
| + all-active direct clash | 691 / 1000 |
| + all-active symmetry clash | 691 / 1000 |
| + all-active tmol, tolerance 0.0 | 321 / 1000 |
| + all-active tmol, tolerance 0.5 | 517 / 1000 |
| + all-active tmol, tolerance 1.0 | 543 / 1000 |
| + all-active tmol, tolerance 2.0 | 558 / 1000 |
| + all-active tmol, per-site reproduction q99 | 428 / 1000 |

The corresponding assigned-A/B-pair counts are 394, 620, 662, 675, and 507
at tolerances 0.0, 0.5, 1.0, 2.0, and per-site q99.

## Per-site cascade

`Δ` is current both-found minus stale both-found. `R/D/S` is the sequential
all-active rotamer/direct/symmetry cascade after occupancy. `AA0` and `AP0`
are all-active and assigned-pair strict counts at tmol tolerance 0.0.

| Site | Stale found | Found | Occupancy | Δ | R/D/S | AA0 | AP0 | AA0.5 | AA1 | AA2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1ZV8 ASN1 | 6 | 1 | 0 | -5 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| 2V05 HIS168 | 2 | 22 | 22 | +20 | 22/22/22 | 6 | 21 | 6 | 6 | 6 |
| 2VFP TYR417 | 13 | 17 | 1 | +4 | 0/0/0 | 0 | 1 | 0 | 0 | 0 |
| 3A1C ARG447 | 43 | 47 | 45 | +4 | 45/45/45 | 37 | 43 | 38 | 39 | 39 |
| 3GMI GLU5 | 11 | 41 | 41 | +30 | 41/41/41 | 0 | 0 | 35 | 36 | 36 |
| 3K8W SER337 | 50 | 50 | 50 | 0 | 50/50/50 | 48 | 50 | 50 | 50 | 50 |
| 3NY7 LYS19 | 45 | 48 | 47 | +3 | 43/43/43 | 24 | 36 | 24 | 24 | 24 |
| 4C16 MET258 | 21 | 20 | 20 | -1 | 20/20/20 | 12 | 19 | 12 | 12 | 12 |
| 4MKM THR77 | 50 | 50 | 50 | 0 | 50/50/50 | 0 | 0 | 47 | 48 | 49 |
| 5DBA TRP325 | 30 | 37 | 37 | +7 | 35/35/35 | 27 | 36 | 33 | 34 | 34 |
| 5KWB PHE591 | 48 | 49 | 49 | +1 | 49/49/49 | 0 | 0 | 45 | 46 | 47 |
| 5Z8H MET730 | 18 | 19 | 19 | +1 | 19/19/19 | 0 | 0 | 14 | 14 | 14 |
| 6H59 ARG144 | 45 | 39 | 39 | -6 | 39/39/39 | 37 | 39 | 37 | 37 | 37 |
| 6Y4G CYS260 | 50 | 50 | 50 | 0 | 50/50/50 | 44 | 47 | 46 | 46 | 47 |
| 7F72 MET103 | 37 | 45 | 38 | +8 | 38/38/38 | 0 | 1 | 4 | 23 | 33 |
| 7T7A LEU396 | 41 | 44 | 41 | +3 | 41/41/41 | 32 | 34 | 33 | 33 | 33 |
| 7UO8 GLN53 | 14 | 0 | 0 | -14 | 0/0/0 | 0 | 0 | 0 | 0 | 0 |
| 8DJ2 VAL893 | 50 | 50 | 50 | 0 | 50/50/50 | 39 | 50 | 39 | 39 | 39 |
| 8FBE ILE92 | 49 | 49 | 49 | 0 | 49/49/49 | 9 | 11 | 42 | 44 | 46 |
| 8Q6Q ASP81 | 50 | 50 | 50 | 0 | 50/50/50 | 6 | 6 | 12 | 12 | 12 |
| **Total** | **673** | **728** | **698** | **+55** | **691/691/691** | **321** | **394** | **517** | **543** | **558** |

## Immediate observations

- The live symmetry environment changed recovery unevenly. 3GMI improved from
  11 to 41 both-found starts, while 7UO8 fell from 14 to 0.
- No starts are lost at the direct- or symmetry-clash stages after the rotamer
  stage. The corrected optimizer environment removed the previous hard-audit
  mismatch in this run.
- tmol remains the dominant and tolerance-sensitive filter: the all-active
  headline spans 321 to 558 across the requested fixed tolerance sweep.
- Fresh 3K8W endpoints no longer show the stale B-reference failure:
  assigned-pair strict recovery is 50/50 at tolerance 0.
- Fresh 8Q6Q remains B-limited: only 6/87 B-assigned conformers pass tmol at
  tolerance 0, yielding 6/50 assigned pairs.

