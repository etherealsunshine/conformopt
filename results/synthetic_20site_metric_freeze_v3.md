# Synthetic 20-Site Metric Freeze v3

**Frozen metric version:** `qfit-synth20-merge050-one-to-one-tmol044-v3`
**Frozen baseline:** 626 / 1000 starts
**Date:** 2026-07-28

This is the authoritative frozen metric for subsequent synthetic 20-site
experiments. It keeps optimal one-to-one A/B recovery assignment and restores
physically appropriate occupancy summation only for geometrically
near-duplicate slots.

No model experiment was run between v1, v2, and v3. Their differences are
audit assignment corrections, not optimizer progress.

This definition does not change during subsequent model experiments. Any
future change requires a new metric version and a complete re-audit of this
same frozen baseline.

## Frozen rule

### Unchanged scientific rules

```text
optimizer environment
2026-07-24-altloc-minstate-water-minstate-v2

geometry
2026-07-24-altloc-minstate-rotwidth-hisunion-aromatic45-matched-tmol-v2

tmol
frozen_matched_deposited_minstate_v1

matched tmol tolerance
+0.44

reported activity/recovery occupancy
>0.10
```

The optimizer's Stage-2 physics mask remains `occupancy >0.05`.

### Merge and assignment

1. Compute a preliminary optimal one-to-one assignment over slots with
   occupancy `>0.10`, maximizing valid A/B matches and then minimizing total
   conventional symmetry-aware RMSD under the strict `<1.0 Å` cutoff.
2. Treat those distinct A and B slots as protected anchors: they may never
   enter the same merge component.
3. Over all optimizer-active slots (`occupancy >0.05`), form single-linkage
   components using pairwise conventional symmetry-aware RMSD `<0.50 Å`.
4. Sum slot occupancy within each component.
5. Use the protected anchor geometry for an anchored component; use the
   occupancy-weighted medoid for an unanchored component.
6. Run the final optimal one-to-one A/B assignment over the merged components.
7. Score the distinct assigned A/B representatives through rotamer, direct
   clash, symmetry clash, and matched tmol.

The preliminary protection is necessary. A literal unconstrained
merge-before-assignment merged genuine nearby A/B modes at 2VFP: its
both-found count fell from 14 to 7/5/0 at thresholds 0.3/0.5/0.8 Å. That
violated the explicit requirement to retain the correct one-to-one recovery
fix. The protected formulation separates the two questions: distinct-state
coverage is preserved, while same-mode satellite occupancy can be merged.

### Gate order

1. Distinct A and B merged components found.
2. Summed assigned-component occupancies each within `±0.20` of deposited
   occupancy.
3. Both representative geometries pass the frozen rotamer gate.
4. Both pass the `2.0 Å` direct-clash gate.
5. Both pass the `2.0 Å` symmetry-clash gate.
6. Both matched tmol margins are finite and `≤+0.44`.

Extra components remain mandatory secondary over-modeling diagnostics.

## Source hashes

| Source | SHA-256 |
|---|---|
| `density_denoiser/five_site_optimizer.py` | `367acfaba8f6d0da660fac45ace5c0c696f705bbdb05b60d2072b8724b87cbd6` |
| `density_denoiser/clash_environment.py` | `ae5940329de4ccc1d1f729f1eb0004ad607152bc347bc8e92636a0a512ab44df` |
| `density_denoiser/residue_geometry.py` | `2e6d2b57338e464928024f69d704968aa78cba0f83fc0ea382782b8add06c2b4` |
| `density_denoiser/audit_five_site_endpoints.py` | `b5187d1e71a4003a796b3ee47cc4af29899fb5912e9df589908ec1c10fa1fed7` |
| `density_denoiser/summarize_endpoint_audit.py` | `1dce5e6133d8f0dcd349be0955ba306345797279e17128a7a5db0fba444d9d3c` |
| `scripts/five_site_tmol_audit.py` | `932618da4a859b7c6df49465c9c195e1310e804afecd300cf59d13d2146f136a` |
| `scripts/analyze_merge_assignment_sweep.py` | `c536f5d4b7124112dadd5263cf5871174b11894467c4d8c6d5b921d2f5e83631` |

All 20 sites carry this same definition.

## Threshold sensitivity

The protected A/B assignment keeps recovery fixed at 742 for every tested
merge threshold.

| Version | Found | Occupancy | Rotamer | Direct | Symmetry | Tmol ≤0.44 |
|---|---:|---:|---:|---:|---:|---:|
| v1 greedy | 729 | 715 | 711 | 711 | 711 | 621 |
| v2 one-to-one, no merge | 742 | 700 | 696 | 696 | 696 | 615 |
| Protected merge 0.3 Å | 742 | 712 | 708 | 708 | 708 | 625 |
| **Protected merge 0.5 Å — v3** | **742** | **714** | **710** | **710** | **710** | **626** |
| Protected merge 0.8 Å | 742 | 714 | 710 | 710 | 710 | 626 |

The 0.5 Å threshold is the smallest tested value on the 0.5–0.8 Å cascade
plateau. It recovers one additional strict start over 0.3 Å without relying on
the more permissive 0.8 Å threshold.

## Full per-site cascade

Each cell is `v1 / v2 / v3`, with 50 starts per site.

| Site | Found | Occupancy | Rotamer | Direct | Symmetry | Tmol |
|---|---:|---:|---:|---:|---:|---:|
| 1ZV8 | 1/1/1 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| 2V05 | 22/22/22 | 22/21/22 | 22/21/22 | 22/21/22 | 22/21/22 | 21/20/21 |
| 2VFP | 1/14/14 | 1/7/8 | 1/7/8 | 1/7/8 | 1/7/8 | 1/7/8 |
| 3A1C | 47/47/47 | 45/45/45 | 45/45/45 | 45/45/45 | 45/45/45 | 44/44/44 |
| 3GMI | 41/41/41 | 41/41/41 | 41/41/41 | 41/41/41 | 41/41/41 | 39/39/39 |
| 3K8W | 50/50/50 | 50/46/50 | 50/46/50 | 50/46/50 | 50/46/50 | 50/46/50 |
| 3NY7 | 48/48/48 | 47/47/47 | 43/43/43 | 43/43/43 | 43/43/43 | 38/38/38 |
| 4C16 | 20/20/20 | 20/20/20 | 20/20/20 | 20/20/20 | 20/20/20 | 19/19/19 |
| 4MKM | 50/50/50 | 50/50/50 | 50/50/50 | 50/50/50 | 50/50/50 | 50/50/50 |
| 5DBA | 37/37/37 | 37/34/37 | 37/34/37 | 37/34/37 | 37/34/37 | 37/34/37 |
| 5KWB | 49/49/49 | 49/48/49 | 49/48/49 | 49/48/49 | 49/48/49 | 49/48/49 |
| 5Z8H | 16/16/16 | 16/16/16 | 16/16/16 | 16/16/16 | 16/16/16 | 1/1/1 |
| 6H59 | 41/41/41 | 41/41/41 | 41/41/41 | 41/41/41 | 41/41/41 | 39/39/39 |
| 6Y4G | 50/50/50 | 50/50/50 | 50/50/50 | 50/50/50 | 50/50/50 | 50/50/50 |
| 7F72 | 45/45/45 | 38/36/38 | 38/36/38 | 38/36/38 | 38/36/38 | 5/4/5 |
| 7T7A | 44/44/44 | 41/41/41 | 41/41/41 | 41/41/41 | 41/41/41 | 34/34/34 |
| 7UO8 | 18/18/18 | 18/18/18 | 18/18/18 | 18/18/18 | 18/18/18 | 18/18/18 |
| 8DJ2 | 50/50/50 | 50/50/50 | 50/50/50 | 50/50/50 | 50/50/50 | 50/50/50 |
| 8FBE | 49/49/49 | 49/49/49 | 49/49/49 | 49/49/49 | 49/49/49 | 49/49/49 |
| 8Q6Q | 50/50/50 | 50/40/42 | 50/40/42 | 50/40/42 | 50/40/42 | 27/25/25 |
| **Total** | **729/742/742** | **715/700/714** | **711/696/710** | **711/696/710** | **711/696/710** | **621/615/626** |

V3 preserves the +13 recovery correction, including 2VFP 1→14, and restores
14 of the 15 occupancy-stage starts lost by the no-merge v2 aggregate. It does
not blindly reproduce v1 occupancy: geometrically separated slots previously
sharing a greedy state label are not merged.

## Deposited-pair symmetry caveat

A deposited-coordinate-only diagnostic found that 2VFP TYR417 is a
symmetry-degenerate pair. Its fixed-label local A/B RMSD is `1.751495 Å`, but
the valid `CD1↔CD2` plus `CE1↔CE2` ring permutation reduces the residual to
the tabulated `0.566776 Å`. Thus the deposited records are chemically the same
ring orientation up to equivalent labels plus about `0.57 Å` of residual
motion, rather than two cleanly resolved, widely separated states.

This explains the site's recovery history: `1/50` under greedy assignment and
`14/50` under one-to-one assignment, with 8 of the 14 assigned pairs compressed
to `0.19–0.23 Å`. A single produced geometry can legitimately be within the
`<1.0 Å` recovery neighborhoods of both deposited records. This is a property
of the site, not solely a matching artifact. The frozen v3 metric and its
`626/1000` baseline remain unchanged, but 2VFP's recovery number is not
directly comparable to an ordinary genuinely distinct two-state site.

5KWB PHE591 is the only other near-degenerate ring-flip pair: fixed-label
local RMSD `1.879860 Å`, falling to `0.616975 Å` after `CD1↔CD2` and
`CE1↔CE2`. Its assigned pairs did not show 2VFP's anomalous compression
(`0.597 Å` median versus `0.617 Å` deposited across 49 recoveries), but its
structural recovery count carries the same overlapping-neighborhood caveat.
8DJ2 VAL893 and 7T7A LEU396 are also label-sensitive, but retain
`1.093311 Å` and `1.747144 Å` residual local displacement after their methyl
swaps.

The full four-definition, 20-site audit is
`results/deposited_panel_separation_diagnostic_v1.md`. The new low-separation
anchor under fixed-label local RMSD is 5Z8H MET730 at `1.229525 Å`.

## Hidden greedy same-state duplication

Under v1 greedy labels:

```text
starts with >=2 slots assigned to the same state     237 / 1000
same-state duplicate groups                         257
non-primary members                                 280
non-primary occupancy median                        0.0868
non-primary occupancy IQR                           0.0607–0.1735
non-primary occupancy >0.20                         52
within-group pairwise RMSD median                    0.151 Å
within-group pairwise RMSD IQR                       0.0285–0.524 Å
```

The non-primary median is above the `0.066` unmatched-extra median, and 52—not
merely 21—non-primary members carry more than 0.20 occupancy.

8Q6Q is qualitatively important: 38/50 starts have same-state multiplicity,
with 14 non-primary members above 0.20 occupancy, but its median within-group
RMSD is 0.975 Å. Those slots share a greedy deposited-state label but are not
near-duplicate conformers under the frozen 0.5 Å merge rule. This is why v3
recovers only 42/50 occupancy-qualified starts there rather than restoring
v1's 50/50.

The complete per-site duplication table is
`results/synthetic_20site_hidden_greedy_duplicates_v1.csv`.

## Provenance

No optimizer endpoint was modified or regenerated. Geometry was reconstructed
from the frozen endpoint chi/occupancy rows. The protected merge leaves every
selected A/B representative and environment assignment identical to v2 at all
three thresholds; this was verified candidate-by-candidate. Therefore the
assignment-specific v2 tmol energies were reused byte-for-byte.

```text
frozen endpoint composite
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_water_minstate_v2_single_rule_v1

protected merge sweep and authoritative v3 audit
.../analysis/metric_v3_protected_merge_sweep/

machine-readable comparison
.../analysis/metric_v3_protected_merge_sweep/comparison/
```

V1 and v2 remain preserved as superseded historical definitions with their
respective reasons recorded.
