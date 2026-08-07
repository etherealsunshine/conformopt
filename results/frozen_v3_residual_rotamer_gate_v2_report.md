# Frozen-v3 raw-residual canonical-rotamer gate

## Scope and guards

This is a saved-endpoint diagnostic only. It ran no optimizer, changed no
endpoint, and did not modify the frozen metric
`qfit-synth20-merge050-one-to-one-tmol044-v3`.

The historical raw-greedy cohort guard passed exactly:

```text
missed minor / recovered major: 142
missed major / recovered minor:  45
total single-recovery starts:    187
```

For every start, all four saved endpoint torsion rows and occupancies were
reconstructed. The maximum RMSD between a reconstructed active slot and its
saved frozen-v3 coordinates was `8.31e-6 Å`. The raw deposited A/B mixture,
when z-scored only for validation, reproduced the saved optimizer target with
maximum relative L2 error `3.86e-6`, minimum Pearson correlation
`0.999999999993`, and maximum absolute error `3.27e-5`.

The actual fits use native additive density before z-scoring on the saved
Stage-1 radial mask. Shared fixed-atom density cancels between target and
rendered endpoint. For each residue, the analyzer enumerated the exact
Cartesian pool of physically unique marginal canonical chi centers used by
production (6–81 conformers depending on residue type). A candidate's fitted
occupancy is the non-negative scalar least-squares solution in raw density
space. “Best” means the candidate with the largest SSE reduction; Pearson
correlation is reported separately.

## Decisive result: deposited missed-conformer ceiling

The deposited missed conformer is not uniformly absent from the residual, but
the signal is modest and highly site-dependent:

| cohort | starts | ceiling Pearson median (IQR) | ceiling fitted q median | ceiling r < 0.2 |
|---|---:|---:|---:|---:|
| All single-recovery | 187 | 0.340 (0.198–0.489) | 0.070 | 48/187 |
| Missed minor | 142 | 0.349 (0.221–0.494) | 0.070 | 34/142 |
| Missed major | 45 | 0.316 (0.142–0.415) | 0.075 | 14/45 |
| Five tail sites | 138 | 0.321 (0.198–0.461) | 0.069 | 35/138 |

Only 36/187 ceilings reach `r >= 0.5`; 139/187 reach `r >= 0.2`. Thus the
raw residual often retains a recognizable missed-state component, but it is
not a clean missing lobe with the deposited occupancy. Even the exact missed
conformer receives median fitted occupancy only 0.070.

The important exception is 2VFP: across its 44 missed-minor starts, the exact
deposited minor conformer has median ceiling correlation 0.181 and fitted
occupancy 0.049. This is weak evidence for an insertable missing lobe at the
largest failure site.

## Canonical-rotamer result

Across all 187 starts, the best canonical rotamer has median correlation
0.119 and fitted occupancy 0.023. Only 49/187 best fits are within 1.0 Å of
the missed conformer. Using a descriptive “actionable” conjunction—within
1.0 Å, `r >= 0.2`, and fitted `q >= 0.05`—only 27/187 starts qualify:

| cohort | best r median | best fitted q median | within 1 Å | actionable |
|---|---:|---:|---:|---:|
| All single-recovery | 0.119 | 0.023 | 49/187 | 27/187 |
| Missed minor | 0.058 | 0.008 | 31/142 | 12/142 |
| Missed major | 0.278 | 0.067 | 18/45 | 15/45 |
| Five tail sites | 0.058 | 0.007 | 38/138 | 20/138 |

This is not a panel-wide positive gate for canonical whole-conformer
insertion. The result is stronger for the smaller missed-major cohort, but the
problem under study is dominated by missed-minor failures.

## Tail sites

| site | missed minor / major | deposited ceiling median r | best canonical median r / q | best within 1 Å | actionable | starts with active 0.05 < q < 0.10 |
|---|---:|---:|---:|---:|---:|---:|
| 1ZV8 ASN1 | 20 / 1 | 0.498 | 0.000 / 0.0001 | 0/21 | 0/21 | 9/21 |
| 2VFP TYR417 | 44 / 0 | 0.181 | -0.007 / 0.000 | 5/44 | 0/44 | 23/44 |
| 5Z8H MET730 | 26 / 2 | 0.281 | 0.085 / 0.009 | 17/28 | 5/28 | 8/28 |
| 7UO8 GLN53 | 15 / 6 | 0.476 | 0.457 / 0.085 | 0/21 | 0/21 | 5/21 |
| 4C16 MET258 | 7 / 17 | 0.338 | 0.274 / 0.066 | 16/24 | 15/24 | 13/24 |

The sites fail for different reasons:

- At 1ZV8 the deposited ceiling is clear, but no canonical state is within
  1 Å of the missed conformer and the best canonical correlation is
  essentially zero. The residual contains signal that the production
  canonical table cannot represent.
- At 2VFP both the ceiling and canonical fits are weak. This is the cleanest
  evidence for a smeared partial-overlap residual rather than an insertable
  missing lobe.
- At 5Z8H canonical geometry is often nearby, but the fitted signal is small:
  only 5/28 starts meet the descriptive actionable conjunction.
- At 7UO8 a canonical density can correlate well, but the closest selected
  states remain outside the 1 Å recovery threshold (best-fit RMSD is at least
  1.357 Å). Exact canonical insertion would explain density without scoring
  as recovery.
- 4C16 is the one clear positive site: 15/24 starts meet the conjunction.
  Its cohort is predominantly missed-major (17/24), so it does not solve the
  panel's missed-minor tail.

The two sites with zero R1 merge events contain 70/142 missed-minor failures.
For those exact missed-minor starts, 2VFP has a low-occupancy active slot in
23/44 cases but zero actionable canonical fits; 5Z8H has one in 6/26 cases
and only 3/26 actionable fits. A merge-independent trigger fixes mechanical
coverage, but does not fix the lack of a useful canonical insertion target.

## Merge-independent low-occupancy coverage, all sites

“Freeable” here means an already-active endpoint slot with
`0.05 < occupancy < 0.10`; no degenerate partner is required. Overall,
81/187 single-recovery starts contain such a slot (92 slots total).

| site | single-recovery starts | starts freeable | low-q slots |
|---|---:|---:|---:|
| 1ZV8_E_ASN1 | 21 | 9 | 10 |
| 2V05_A_HIS168 | 10 | 6 | 6 |
| 2VFP_A_TYR417 | 44 | 23 | 28 |
| 3A1C_B_ARG447 | 3 | 3 | 3 |
| 3GMI_A_GLU5 | 9 | 4 | 5 |
| 3K8W_A_SER337 | 0 | 0 | 0 |
| 3NY7_B_LYS19 | 2 | 1 | 1 |
| 4C16_A_MET258 | 24 | 13 | 15 |
| 4MKM_A_THR77 | 0 | 0 | 0 |
| 5DBA_A_TRP325 | 13 | 3 | 3 |
| 5KWB_A_PHE591 | 1 | 0 | 0 |
| 5Z8H_A_MET730 | 28 | 8 | 9 |
| 6H59_B_ARG144 | 0 | 0 | 0 |
| 6Y4G_B_CYS260 | 0 | 0 | 0 |
| 7F72_A_MET103 | 5 | 4 | 5 |
| 7T7A_A_LEU396 | 5 | 2 | 2 |
| 7UO8_A_GLN53 | 21 | 5 | 5 |
| 8DJ2_A_VAL893 | 0 | 0 | 0 |
| 8FBE_B_ILE92 | 1 | 0 | 0 |
| 8Q6Q_B_ASP81 | 0 | 0 | 0 |

## Interpretation and gate decision

Atom-level respawn failed because one peak did not determine a valid
conformer. This read shows that replacing it with the exact production
canonical table is not a general solution:

- the true missed conformer gives only a moderate median residual ceiling and
  a small fitted occupancy;
- the best canonical state is near the missed conformer in only 26.2% of
  starts and meets the actionable conjunction in 14.4%;
- at 2VFP, which alone contributes 44/142 missed-minor failures, even the
  deposited ceiling is weak and the canonical gate is null.

There are isolated positive pockets, especially 4C16 and a subset of 5Z8H,
so the result does not prove that every conceivable whole-conformer placement
rule is impossible. It does reject a panel-wide experiment based on direct
insertion from the current canonical table. A merge-independent trigger would
increase event coverage, but the two untested R1 tail sites supply useful
canonical targets in only 3/70 missed-minor starts under the stated
conjunction. No further optimizer run is justified by this gate alone.

## Artifacts

Local:

```text
results/frozen_v3_residual_rotamer_gate_v2/
  per_rotamer.csv   # 3,171 rows: every canonical candidate for every start
  per_start.csv     # 187 rows
  per_site.csv      # all 20 sites, including explicit zeros
  summary.json
  report.md
  progress.json
```

Authoritative pod copy:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_water_minstate_v2_single_rule_v1/
analysis/frozen_v3_residual_rotamer_gate_v2
```

Analyzer and focused tests:

```text
scripts/diagnose_frozen_v3_residual_rotamers.py
test_residual_rotamer_diagnostic.py
```

Focused result: `9 passed`.
