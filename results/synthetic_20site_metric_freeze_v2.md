# Synthetic 20-Site Metric Freeze v2

**Frozen metric version:** `qfit-synth20-one-to-one-assignedpair-tmol044-v2`
**Frozen baseline:** 615 / 1000 starts
**Date:** 2026-07-28

> **Superseded before model experiments by
> `qfit-synth20-merge050-one-to-one-tmol044-v3`.** V2 correctly fixed recovery
> assignment but used only the selected slot's occupancy, omitting the
> occupancy of genuine near-duplicate slots. It is retained as the immutable
> no-merge ablation.

This version supersedes `qfit-synth20-assignedpair-tmol044-v1` before any
subsequent model experiment was run. The single change is conformer matching:
independent greedy nearest-state labels are replaced by an optimal one-to-one
A/B assignment. The frozen optimizer endpoints are unchanged and were
re-audited; the delta is an assignment correction, not additional optimizer
success or model progress.

This definition does not change during subsequent model experiments. Any
future change requires a new metric version and a complete re-audit of this
same frozen baseline.

## Frozen definition

### Rule strings

```text
optimizer environment
2026-07-24-altloc-minstate-water-minstate-v2

geometry
2026-07-24-altloc-minstate-rotwidth-hisunion-aromatic45-matched-tmol-v2

tmol
frozen_matched_deposited_minstate_v1

conformer matching
optimal_one_to_one_AB_rmsd_v1
```

No geometry rule, optimizer environment, rotamer table, weight, cutoff, tmol
rule, or tmol tolerance changed from v1.

### Source hashes

| Source | SHA-256 |
|---|---|
| `density_denoiser/five_site_optimizer.py` | `367acfaba8f6d0da660fac45ace5c0c696f705bbdb05b60d2072b8724b87cbd6` |
| `density_denoiser/clash_environment.py` | `ae5940329de4ccc1d1f729f1eb0004ad607152bc347bc8e92636a0a512ab44df` |
| `density_denoiser/residue_geometry.py` | `2e6d2b57338e464928024f69d704968aa78cba0f83fc0ea382782b8add06c2b4` |
| `density_denoiser/audit_five_site_endpoints.py` | `2cf5d941928ae1dc983c959e260285896a22065c187729d071ca68f434ee03f7` |
| `density_denoiser/summarize_endpoint_audit.py` | `da72b93d9ec654439250159079c328ec4ddae820c6844b23ba9ffe6bdbff9522` |
| `scripts/five_site_tmol_audit.py` | `932618da4a859b7c6df49465c9c195e1310e804afecd300cf59d13d2146f136a` |
| `five_site_tmol_audit.py` compatibility entry point | `fd13c8f494e16ce8909e1f3202bef5510c4da85fc06f73162f1d0b5e2d9c5c8d` |
| `scripts/analyze_one_to_one_metric_v2.py` | `bfedb131c171424f99b7cdc149210cf7fe6b2421521aed7cca02ce5780ec806c` |

All 20 sites use this same rule and source-hash set.

### Optimal A/B assignment

For every start:

1. Candidate slots must have occupancy `> 0.10` to be eligible for reported
   A/B recovery.
2. A candidate-to-state edge is valid only when its conventional,
   chemically symmetry-aware RMSD is strictly `< 1.0 Å`.
3. Choose distinct candidate slots for deposited A and B by maximizing the
   number of valid matches, then minimizing total A+B RMSD.
4. Each deposited state can be claimed at most once and each candidate can
   supply at most one state.
5. Active candidates not selected for A or B remain unmatched extras.

Because there are exactly two deposited states and K=4 slots, the
implementation exhaustively evaluates the 2×K assignment. This is exactly the
two-state Hungarian objective without adding SciPy as a production runtime
dependency.

The optimizer's Stage-2 physics mask remains `occupancy > 0.05`. The frozen
reported activity/recovery threshold remains `occupancy > 0.10`.

### Headline conformer rule

The headline scores the distinct A/B pair selected above. Additional active
slots are not independent headline failure opportunities. Their count,
occupancy, RMSD, and gate outcomes remain mandatory over-modeling diagnostics.

### Gate order

1. Distinct deposited A and B states found under the optimal assignment.
2. The selected A and B occupancies are each within `±0.20` of deposited
   occupancy.
3. Both selected conformers pass the frozen rotamer gate.
4. Both pass the direct-clash gate: no contact below `2.0 Å`.
5. Both pass the crystallographic-symmetry-clash gate: no contact below
   `2.0 Å`.
6. Both have finite matched tmol margins
   `candidate − matched deposited ≤ +0.44`.

The `+0.44` tolerance and its near-reproduction derivation are unchanged from
v1. The v1 evidence remains the tolerance provenance record.

## v1 versus v2 total cascade

| Metric stage | v1 greedy | v2 one-to-one | Δ |
|---|---:|---:|---:|
| Both A/B found | 729 | 742 | +13 |
| + occupancy | 715 | 700 | −15 |
| + rotamer | 711 | 696 | −15 |
| + direct clash | 711 | 696 | −15 |
| + symmetry clash | 711 | 696 | −15 |
| + matched tmol ≤ +0.44 | 621 | **615** | **−6** |

The recovery increase and later decrease are both consequences of correcting
assignment:

- Greedy matching missed 13 starts where distinct slots jointly covered A and
  B.
- Greedy matching also summed the occupancies of multiple independently
  A-labeled or B-labeled duplicate slots. Under one-to-one matching only the
  selected slot supplies that state's occupancy, so 15 previously
  occupancy-qualified starts no longer pass the occupancy gate.

Therefore neither `+13` recovered nor `−6` strict is a modeling change.

## Full per-site cascade

Each cell is `v1 → v2`, with 50 starts per site.

| Site | Both found | + occupancy | + rotamer | + direct | + symmetry | + tmol 0.44 |
|---|---:|---:|---:|---:|---:|---:|
| 1ZV8 ASN1 | 1→1 | 0→0 | 0→0 | 0→0 | 0→0 | 0→0 |
| 2V05 HIS168 | 22→22 | 22→21 | 22→21 | 22→21 | 22→21 | 21→20 |
| 2VFP TYR417 | **1→14** | **1→7** | 1→7 | 1→7 | 1→7 | **1→7** |
| 3A1C ARG447 | 47→47 | 45→45 | 45→45 | 45→45 | 45→45 | 44→44 |
| 3GMI GLU5 | 41→41 | 41→41 | 41→41 | 41→41 | 41→41 | 39→39 |
| 3K8W SER337 | 50→50 | 50→46 | 50→46 | 50→46 | 50→46 | 50→46 |
| 3NY7 LYS19 | 48→48 | 47→47 | 43→43 | 43→43 | 43→43 | 38→38 |
| 4C16 MET258 | 20→20 | 20→20 | 20→20 | 20→20 | 20→20 | 19→19 |
| 4MKM THR77 | 50→50 | 50→50 | 50→50 | 50→50 | 50→50 | 50→50 |
| 5DBA TRP325 | 37→37 | 37→34 | 37→34 | 37→34 | 37→34 | 37→34 |
| 5KWB PHE591 | 49→49 | 49→48 | 49→48 | 49→48 | 49→48 | 49→48 |
| 5Z8H MET730 | 16→16 | 16→16 | 16→16 | 16→16 | 16→16 | 1→1 |
| 6H59 ARG144 | 41→41 | 41→41 | 41→41 | 41→41 | 41→41 | 39→39 |
| 6Y4G CYS260 | 50→50 | 50→50 | 50→50 | 50→50 | 50→50 | 50→50 |
| 7F72 MET103 | 45→45 | 38→36 | 38→36 | 38→36 | 38→36 | 5→4 |
| 7T7A LEU396 | 44→44 | 41→41 | 41→41 | 41→41 | 41→41 | 34→34 |
| 7UO8 GLN53 | 18→18 | 18→18 | 18→18 | 18→18 | 18→18 | 18→18 |
| 8DJ2 VAL893 | 50→50 | 50→50 | 50→50 | 50→50 | 50→50 | 50→50 |
| 8FBE ILE92 | 49→49 | 49→49 | 49→49 | 49→49 | 49→49 | 49→49 |
| 8Q6Q ASP81 | 50→50 | 50→40 | 50→40 | 50→40 | 50→40 | 27→25 |
| **Total** | **729→742** | **715→700** | **711→696** | **711→696** | **711→696** | **621→615** |

2VFP improves from 1/50 to 14/50 both-found, exceeding the previously
estimated lower bound of 13/50. Seven of those starts pass the complete frozen
v2 cascade.

## Provenance and authoritative artifacts

No optimizer endpoints were modified or regenerated. Geometry was
reconstructed from the frozen chi/occupancy endpoint rows, assignment-specific
direct and symmetry environments were rebuilt, and tmol was rescored in the
newly assigned frozen A/B environments.

```text
frozen endpoint composite
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_water_minstate_v2_single_rule_v1

v2 re-audit and comparison
.../analysis/metric_v2_one_to_one/

per-site machine-readable cascade
.../analysis/metric_v2_one_to_one/composite/per_site_v1_v2_cascade.csv

total machine-readable cascade
.../analysis/metric_v2_one_to_one/composite/total_v1_v2_cascade.csv
```

The v1 document is retained as a superseded historical definition. It was
replaced because independent greedy matching undercounted distinct A/B
recovery and allowed duplicate-slot occupancy aggregation.
