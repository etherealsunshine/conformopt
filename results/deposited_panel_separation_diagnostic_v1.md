# Deposited 20-Site A/B Separation Diagnostic

**Date:** 2026-07-28  
**Scope:** deposited panel coordinates only; no optimizer endpoints  
**Frozen metric:** unchanged (`qfit-synth20-merge050-one-to-one-tmol044-v3`)

## Definitions

All RMSDs are conventional side-chain-heavy-atom RMSDs,
`sqrt(mean_atoms(sum_xyz(delta^2)))`, over the production atom set from `CB`
outward.

1. **Local, fixed labels:** superpose deposited B onto A using the target
   residue's `N/CA/C/O`, then compare corresponding atom names.
2. **Local, symmetry-corrected:** the same local superposition, minimized over
   the production equivalent-terminal-label permutation.
3. **Global, fixed labels:** superpose B onto A using all protein `N/CA/C/O`
   atoms, changing only the target residue from A to B and holding every other
   backbone identical, then compare corresponding atom names.
4. **Global, symmetry-corrected:** the same global superposition, minimized
   over the production equivalent-terminal-label permutation.

The currently tabulated panel separation is a direct comparison in the shared
deposited crystal frame with the production symmetry correction and no Kabsch
fit. It is therefore the global/shared-frame symmetry-corrected convention
(definition 4). The explicit global fit reproduces every tabulated value to
within `0.0001 Å`. Definitions listed in the final column are all values that
numerically reproduce the tabulated value within `0.001 Å`; coincidences occur
when the residue backbone is shared and/or the identity permutation wins.

## All deposited A/B separations

The rows are ranked by definition 1, un-symmetrized local separation.

| Rank | Site | Local fixed (1) | Local sym (2) | Global fixed (3) | Global sym (4) | Tabulated | Reproduced by |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 5Z8H MET730 | 1.229525 | 1.229525 | 1.229525 | 1.229525 | 1.229525 | 1/2/3/4 |
| 2 | 7F72 MET103 | 1.408112 | 1.408112 | 1.408112 | 1.408112 | 1.408112 | 1/2/3/4 |
| 3 | 2VFP TYR417 | 1.751495 | 0.566776 | 1.751495 | 0.566776 | 0.566776 | 2/4 |
| 4 | 6Y4G CYS260 | 1.783938 | 1.783938 | 1.758573 | 1.758573 | 1.758586 | 3/4 |
| 5 | 3K8W SER337 | 1.792399 | 1.792399 | 1.815448 | 1.815448 | 1.815544 | 3/4 |
| 6 | 5KWB PHE591 | 1.879860 | 0.616975 | 1.879860 | 0.616975 | 0.616975 | 2/4 |
| 7 | 4MKM THR77 | 2.075603 | 2.075603 | 2.075603 | 2.075603 | 2.075603 | 1/2/3/4 |
| 8 | 4C16 MET258 | 2.222916 | 2.222916 | 2.222916 | 2.222916 | 2.222916 | 1/2/3/4 |
| 9 | 8Q6Q ASP81 | 2.260891 | 2.260891 | 2.304154 | 2.304154 | 2.304178 | 3/4 |
| 10 | 8DJ2 VAL893 | 2.312297 | 1.093311 | 2.312297 | 1.093311 | 1.093311 | 2/4 |
| 11 | 8FBE ILE92 | 2.344242 | 2.344242 | 2.344242 | 2.344242 | 2.344242 | 1/2/3/4 |
| 12 | 7T7A LEU396 | 2.466739 | 1.747144 | 2.418914 | 1.673139 | 1.673138 | 4 |
| 13 | 6H59 ARG144 | 2.786688 | 2.786688 | 2.786688 | 2.786688 | 2.786688 | 1/2/3/4 |
| 14 | 5DBA TRP325 | 2.811262 | 2.811262 | 2.811262 | 2.811262 | 2.811262 | 1/2/3/4 |
| 15 | 2V05 HIS168 | 2.854161 | 2.854161 | 2.854161 | 2.854161 | 2.854161 | 1/2/3/4 |
| 16 | 3NY7 LYS19 | 3.113500 | 3.113500 | 3.113500 | 3.113500 | 3.113500 | 1/2/3/4 |
| 17 | 1ZV8 ASN1 | 3.115990 | 3.115990 | 3.115990 | 3.115990 | 3.115990 | 1/2/3/4 |
| 18 | 3GMI GLU5 | 4.353754 | 4.178096 | 4.353754 | 4.178096 | 4.178096 | 2/4 |
| 19 | 7UO8 GLN53 | 4.687842 | 4.687842 | 4.687842 | 4.687842 | 4.687842 | 1/2/3/4 |
| 20 | 3A1C ARG447 | 5.609232 | 5.600694 | 5.810602 | 5.796254 | 5.796252 | 4 |

## Symmetry-degenerate candidates

The requested flag is `definition 1 - definition 2 > 0.3 Å`.

| Site | Equivalent-atom swap | Local fixed | Residual after swap | Correction |
|---|---|---:|---:|---:|
| 2VFP TYR417 | `CD1↔CD2` and `CE1↔CE2` | 1.751495 | 0.566776 | 1.184719 |
| 5KWB PHE591 | `CD1↔CD2` and `CE1↔CE2` | 1.879860 | 0.616975 | 1.262885 |
| 8DJ2 VAL893 | `CG1↔CG2` | 2.312297 | 1.093311 | 1.218985 |
| 7T7A LEU396 | `CD1↔CD2` | 2.466739 | 1.747144 | 0.719595 |

2VFP and 5KWB are the two ring-flip cases. Both become sub-0.7 Å after
relabeling and are chemically near-degenerate deposited pairs. 8DJ2 and 7T7A
are also strongly label-sensitive, but their residual separations are
`1.093311 Å` and `1.747144 Å`; they retain substantially more genuine
geometric displacement after the methyl-label ambiguity is removed.

3GMI GLU5 and 3A1C ARG447 choose their valid terminal swaps, but the corrections
are only `0.175658 Å` and `0.008538 Å`, respectively, so neither crosses the
requested flag threshold. For 3A1C, the tabulated global symmetry-corrected
value `5.796252 Å` corresponds to `5.810602 Å` with fixed NH1/NH2 labels:
the ARG swap is negligible relative to the displacement. Its local
backbone-aligned fixed-label value is `5.609232 Å`.

## Low-separation anchor

The smallest deposited pair by un-symmetrized local separation is
**5Z8H MET730, 1.229525 Å**. It is the appropriate low-separation anchor for
the reward-margin experiment in place of 2VFP.

8DJ2 VAL893 does **not** survive as the low-separation anchor: its tabulated
symmetry-corrected value is `1.093311 Å`, but its fixed-label local separation
is `2.312297 Å`.

## Frozen-v3 interpretation

2VFP is a symmetry-degenerate deposited pair. Its `1/50` recovery under greedy
matching and `14/50` under one-to-one matching must be interpreted with the
fact that 8 of those 14 assigned A/B pairs are compressed to `0.19–0.23 Å`
against a nominal symmetry-corrected deposited separation of `0.566776 Å`.
Because deposited A and B are the same tyrosine ring geometry up to the
equivalent ring labels plus about `0.57 Å` residual motion, a single produced
geometry can legitimately fall inside both recovery neighborhoods. This is a
site property, not merely a matching artifact, and 2VFP's recovery count is
not directly comparable to an ordinary genuinely distinct two-state site.

5KWB is the only other panel site with the same ring-flip/near-degenerate
deposited-pair property (`0.616975 Å` residual after the PHE ring swap).
Unlike 2VFP, its recovered assigned pairs were not anomalously compressed:
the prior diagnostic found a `0.597 Å` assigned median versus `0.617 Å`
deposited over 49 starts. Its structural recovery count carries the same
overlapping-neighborhood caveat, although the observed endpoint pathology is
specific to 2VFP. 8DJ2 and 7T7A are swap-sensitive but retain
greater-than-1 Å residual displacement and are not the same near-identical
ring-flip case.

## Provenance

Authoritative machine-readable artifacts:

```text
/home/dev/qfit_unet_data/density_denoiser/
deposited_panel_separation_diagnostic_v4/
  run_config.json
  summary.json
  deposited_panel_separations.csv
```

The coordinate source is the deposited A/B arrays in the frozen-v3
`tmol_inputs.json` files plus the corresponding held-out test PDB backbones.
No endpoint coordinate, optimizer output, frozen metric rule, or frozen
baseline count was changed.
