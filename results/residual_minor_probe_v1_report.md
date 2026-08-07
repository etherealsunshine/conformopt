# Residual minor-conformer probe v1

**Completed:** 2026-07-29  
**Diagnostic only:** yes  
**Production changed:** no  
**Metric changed:** no  
**Frozen metric:** `qfit-synth20-merge050-one-to-one-tmol044-v3`

## Question

For every frozen-v3 start that recovered only the occupancy-major deposited
state, the saved endpoint ensemble was frozen and rendered. Its density was
subtracted from the synthetic target, and one fresh slot was optimized against
that residual using the production density schedule and Stage-2 physics.

The probe used the production initialization: a deposited-A-centered
`N(0,1)`-radian chi offset with `seed = 41 + start`. Two amplitude treatments
were tested from the same initial chi:

- `fixed_minor`: occupancy fixed to the deposited minor occupancy.
- `free_sigmoid`: an independent occupancy in `[0,1]`, initialized at `0.25`.

The first five-site tail pass contained 99 starts. Because it was informative,
the diagnostic was extended to all 129 frozen-v3 missed-minor starts.

## Primary result

| Probe occupancy | Recovered minor | Rate | Median final RMSD | Median final occupancy |
|---|---:|---:|---:|---:|
| Fixed deposited-minor | 38 / 129 | 29.5% | 1.589 Å | 0.340 |
| Free sigmoid | 16 / 129 | 12.4% | 2.693 Å | 0.006 |

These are measurements of the v1 normalized-residual probe, not estimates of
a correctly formulated sequential method. The renderer-normalization
follow-up below showed that the constructed residual target is not additive.

## Per-site result

| Site | Local unsym. A–B separation | Eligible | Fixed recovered | Free recovered |
|---|---:|---:|---:|---:|
| 1ZV8 ASN1 | 3.116 Å | 20 | 1 | 0 |
| 2V05 HIS168 | 2.854 Å | 4 | 1 | 0 |
| 2VFP TYR417 | 1.751 Å | 31 | 9 | 2 |
| 3A1C ARG447 | 5.609 Å | 1 | 0 | 0 |
| 3GMI GLU5 | 4.354 Å | 9 | 8 | 3 |
| 3NY7 LYS19 | 3.114 Å | 1 | 0 | 0 |
| 4C16 MET258 | 2.223 Å | 7 | 3 | 1 |
| 5DBA TRP325 | 2.811 Å | 5 | 1 | 1 |
| 5KWB PHE591 | 1.880 Å | 1 | 1 | 0 |
| 5Z8H MET730 | 1.230 Å | 26 | 7 | 7 |
| 7F72 MET103 | 1.408 Å | 3 | 1 | 0 |
| 7T7A LEU396 | 2.467 Å | 5 | 4 | 1 |
| 7UO8 GLN53 | 4.688 Å | 15 | 1 | 0 |
| 8FBE ILE92 | 2.344 Å | 1 | 1 | 1 |
| **Total** | — | **129** | **38** | **16** |

The result is strongly site-dependent. Sequential residual fitting looks
potentially useful at 3GMI (8/9 fixed) and 7T7A (4/5), but not as a general
panel-wide mechanism. It failed at the widest-separated 3A1C start and
recovered only 1/15 at 7UO8 and 1/20 at 1ZV8.

## Separation stratification

| Local unsym. separation | Eligible | Fixed recovered | Free recovered |
|---|---:|---:|---:|
| `>2.5 Å` | 55 | 12 (21.8%) | 4 (7.3%) |
| `≤2.5 Å` | 74 | 26 (35.1%) | 12 (16.2%) |

Recovery did not improve with deposited-state separation in this mismatched
probe. A diagnostic replay of all 26 close-group successes found median
initial-to-final slot travel of 1.696 Å (IQR 1.299–2.418): no slot moved less
than 0.5 Å and only two moved less than 1.0 Å. Thus close-site composition and
the fixed recovery threshold contribute to the anti-correlation, but the
successes are not merely trivial sub-Å threshold crossings.

## Was a minor-lobe signal present?

Yes. Before the probe, across all starts:

```text
median positive residual integral at minor lobe   8.663
median positive residual integral at major lobe   1.678
median minor-lobe residual RMS                    0.745
```

For the fixed-occupancy probe, the median minor-lobe positive integral was
`9.072` in recovered starts and `8.303` in failed starts. The constructed
normalized residual contained a positive signal, but it was not an additive
missing-conformer target, so this does not establish “present but
unreachable” under the production objective.

## Important renderer finding

The production renderer is not strictly additive in the normalized loss
space. It forms the occupancy-weighted density and then z-score normalizes the
combined vector:

```text
normalize(sum_k occupancy_k * density_k)
```

The mean and standard deviation depend on the entire ensemble. Consequently,
`target_normalized - endpoint_normalized` is not exactly the normalized
density of the missing conformer. This is why the diagnostic explicitly used
the same normalized space as production, and it helps explain two observations:

- the free probe's median occupancy collapsed to `0.006`;
- adding a fixed-occupancy conformer increased median minor-lobe residual RMS
  from `0.745` to `0.951`.

Thus this probe is mathematically mismatched to the renderer. A corrected
probe must retain the frozen endpoint in raw density space, add the new slot
to that raw density, and apply the production z-score to the combined render
before comparing it to the normalized target. No such change was made here.

## Interpretation

The shared-residual competition hypothesis is **neither supported nor
refuted**. The 38/129 fixed-probe result is a lower bound from a mismatched
target, not an estimate of a correctly formulated no-competition probe.
Sequential fitting should not be judged until the probe is constructed inside
the same raw-sum-then-z-score convention as production.

The full normalization and travel follow-up is recorded in
`results/residual_probe_normalization_followup_v1.md`.

## Definitions

- Missed minor: frozen-v3 merge-then-one-to-one assignment found only the
  higher-occupancy deposited state.
- Residual space: production z-score-normalized radial density vector,
  synthetic target minus frozen endpoint render.
- Lobe mask: selected-grid voxels within 1.0 Å of any deposited sidechain atom;
  A/B-overlapping voxels retained and overlap reported.
- Recovery: conventional chemically symmetry-aware RMSD `<1.0 Å`.
- Physics: unchanged Stage-2 VDW/rotamer/symmetry terms and weights.

## Validation and provenance

Focused plus relevant internal tests: `28 passed`. A real one-start 4C16 smoke
completed both occupancy modes before launch. All 14 site shards completed and
the logs contained no errors.

Tail result root:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_five_tail_residual_minor_probe_v1
```

All-site extension and combined 129-start table:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_remaining_residual_minor_probe_v1
```

Authoritative raw combined table:

```text
.../heldout_remaining_residual_minor_probe_v1/
residual_minor_probe_all_20site.csv
```

Each raw row records initial chi and RMSDs, residual magnitude and integrated
positive/signed density at the minor and major lobes, final occupancy, final
minor/major RMSD, losses, and physics terms.
