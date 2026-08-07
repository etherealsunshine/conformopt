# Density-mask variance and deposited A/B torsion diagnostic

**Completed:** 2026-07-29  
**Diagnostic only:** yes  
**Optimization runs:** none  
**Production changed:** no  
**Metric changed:** no

## Objective and mask definition

`_normalize` subtracts the masked-vector mean and divides by
`torch.std()`. Because PyTorch uses the sample standard deviation here, the
exact squared norm is `N-1`, not `N`; this changes only the constant factor.
For two normalized vectors, Stage-1 MSE is exactly monotone with negative
Pearson correlation.

The production radial mask is a 4.0 A sphere sampled at 0.5 A, containing
2,176 voxels at every site.

Two diagnostic focused masks were used:

- A/B-difference mask: `|rho_A-rho_B| > 0.10 * max(|rho_A-rho_B|)`.
- Atom-union mask: voxels within 1.0 A of any deposited A/B target-sidechain
  atom.

No objective or mask was changed.

## 1. Variance decomposition

The proposed fixed-density dilution mechanism is absent in this panel.
`fixed_density` contains only shared/unlabeled atoms of the target residue's
sidechain; it does not contain the neighboring protein. All 20 selected
residues have zero such atoms after the moving A/B atom set is defined.
Therefore:

```text
fixed atom count                 0 at 20/20 sites
Var(fixed)                       0 at 20/20 sites
2 Cov(fixed, sidechain)          0 at 20/20 sites
sidechain-attributable fraction  100% under every tested mask
```

| Site | Full Var(sidechain) | Difference voxels | Difference Var(sidechain) | Full-mask correlation loss from major collapse | Difference-mask loss | Atom-union loss |
|---|---:|---:|---:|---:|---:|---:|
| 1ZV8 ASN1 | 0.01992 | 774 | 0.02658 | 0.0951 | 0.1249 | 0.1104 |
| 2V05 HIS168 | 0.01895 | 1,067 | 0.01967 | 0.1250 | 0.1779 | 0.2284 |
| 2VFP TYR417 | 0.56195 | 75 | 2.73213 | 0.0777 | 0.2705 | 0.1122 |
| 3A1C ARG447 | 0.01271 | 1,020 | 0.01111 | 0.2375 | 0.3294 | 0.3783 |
| 3GMI GLU5 | 0.09000 | 351 | 0.22693 | 0.0405 | 0.0417 | 0.0409 |
| 3K8W SER337 | 0.01625 | 282 | 0.04068 | 0.1175 | 0.2524 | 0.4172 |
| 3NY7 LYS19 | 0.03126 | 665 | 0.03334 | 0.1325 | 0.2713 | 0.2924 |
| 4C16 MET258 | 0.06574 | 333 | 0.12818 | 0.1300 | 0.1931 | 0.1885 |
| 4MKM THR77 | 0.03788 | 226 | 0.10857 | 0.0502 | 0.1254 | 0.1689 |
| 5DBA TRP325 | 0.05839 | 989 | 0.04906 | 0.1258 | 0.2360 | 0.2666 |
| 5KWB PHE591 | 0.24647 | 171 | 0.79691 | 0.0444 | 0.1351 | 0.1011 |
| 5Z8H MET730 | 0.12402 | 236 | 0.43758 | 0.0113 | 0.0256 | 0.0255 |
| 6H59 ARG144 | 0.09567 | 674 | 0.08879 | 0.1413 | 0.3318 | 0.2402 |
| 6Y4G CYS260 | 0.06299 | 157 | 0.20419 | 0.1733 | 0.3342 | 0.3657 |
| 7F72 MET103 | 0.06877 | 366 | 0.14244 | 0.0224 | 0.0586 | 0.0869 |
| 7T7A LEU396 | 0.03675 | 387 | 0.06002 | 0.1009 | 0.1878 | 0.2002 |
| 7UO8 GLN53 | 0.07862 | 344 | 0.13906 | 0.0967 | 0.1461 | 0.1383 |
| 8DJ2 VAL893 | 0.02451 | 403 | 0.05698 | 0.0340 | 0.0659 | 0.0995 |
| 8FBE ILE92 | 0.03229 | 430 | 0.05385 | 0.0610 | 0.1307 | 0.1517 |
| 8Q6Q ASP81 | 0.02995 | 577 | 0.04037 | 0.1662 | 0.2743 | 0.3216 |

The fixed component is not consuming gradient budget. The broad mask still
dilutes the A/B distinction through shared near-zero background:

```text
correct A/B versus deposited-major-only correlation loss
mask                         median    IQR
full 4 A sphere             0.0988   0.0488-0.1306
A/B-difference voxels       0.1829   0.1253-0.2707
1 A atom union              0.1787   0.1081-0.2730
```

Across all 129 saved major-only endpoint ensembles used by the residual probe,
the full-mask correlation with the correct target is even higher:

```text
median correlation           0.9635
median loss versus correct   0.0365
IQR of loss                  0.0268-0.0589
range of loss                0.0081-0.1319
```

Thus correlation compression is real, but it comes from the broad,
mostly shared background and from flexible multi-slot approximations—not an
unchangeable fixed-density component.

## 2. Deposited A/B torsion differences

Differences are `abs(wrap(chi_B-chi_A))` in degrees using the optimizer's
production dihedral definitions. Indices are one-based.

| Site | Per-chi absolute A/B differences | Largest |
|---|---|---:|
| 1ZV8 ASN1 | 170.2, 173.0 | chi2 |
| 2V05 HIS168 | 60.2, 167.5 | chi2 |
| 2VFP TYR417 | 9.4, 172.1 | chi2 |
| 3A1C ARG447 | 130.4, 33.3, 56.2, 166.0 | chi4 |
| 3GMI GLU5 | 129.0, 10.9, 32.3 | chi1 |
| 3K8W SER337 | 144.5 | chi1 |
| 3NY7 LYS19 | 19.2, 52.7, 105.8, 106.4 | chi4 |
| 4C16 MET258 | 112.0, 142.2, 117.6 | chi2 |
| 4MKM THR77 | 134.5 | chi1 |
| 5DBA TRP325 | 0.8, 104.3 | chi2 |
| 5KWB PHE591 | 12.5, 167.8 | chi2 |
| 5Z8H MET730 | 2.8, 13.7, 81.3 | chi3 |
| 6H59 ARG144 | 29.3, 2.2, 18.4, 173.0 | chi4 |
| 6Y4G CYS260 | 100.3 | chi1 |
| 7F72 MET103 | 29.9, 6.7, 163.5 | chi3 |
| 7T7A LEU396 | 99.3, 114.5 | chi2 |
| 7UO8 GLN53 | 118.3, 142.3, 56.2 | chi2 |
| 8DJ2 VAL893 | 172.0 | chi1 |
| 8FBE ILE92 | 135.4, 12.2 | chi1 |
| 8Q6Q ASP81 | 88.8, 15.9 | chi1 |

### Cross-tab against the 14-site residual probe

| Location of largest A/B torsion change | Recovered | Eligible | Rate |
|---|---:|---:|---:|
| chi1 | 9 | 10 | 90.0% |
| chi2 | 21 | 88 | 23.9% |
| chi3 | 8 | 29 | 27.6% |
| chi4 | 0 | 2 | 0% |
| terminal chi | 25 | 97 | 25.8% |
| nonterminal chi | 13 | 32 | 40.6% |

These are start-weighted and highly site-confounded; the chi1 row is almost
entirely 3GMI.

The proposed late-chi explanation is rejected:

- 3GMI succeeds 8/9 even though its largest change is chi1 (`129.0` degrees);
  chi2 and chi3 change only `10.9` and `32.3` degrees.
- 7T7A succeeds 4/5 with large changes in both chi1 and terminal chi2
  (`99.3`, `114.5` degrees).
- 1ZV8 fails 19/20 with both chi1 and terminal chi2 near 180 degrees.
- 7UO8 fails 14/15 with large chi1 and chi2 changes (`118.3`, `142.3`) and a
  smaller terminal chi3 change (`56.2`).

The contrast is more consistent with the complete multi-torsion path than
with “terminal versus early,” but even that is not sufficient: 7T7A traverses
two large torsions successfully. No single per-chi location rule explains the
four-site contrast.

## Artifacts

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_water_minstate_v2_single_rule_v1/
analysis/density_mask_chi_diagnostic_v2/
```

Files include the full per-site variance table, all 20 deposited torsion
vectors, all 129 endpoint correlations, per-site endpoint summaries, and the
machine-readable summary.
