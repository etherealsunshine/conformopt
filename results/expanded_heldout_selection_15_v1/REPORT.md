# Prospective 15-protein held-out panel

## Frozen selection rule

The panel contains 15 distinct PDB IDs from the supplied untouched test split. The original development proteins (3A1C, 4C16, 6H59, 7F72, and 8Q6Q) were excluded. A site was eligible only when:

- deposited A and B occupancies were each at least 0.10;
- conventional heavy-atom A/B RMSD was at least 0.50 Å;
- conventional kinematic-to-deposited-B reconstruction RMSD was at most 0.10 Å;
- deposited A and B had complete corresponding sidechain heavy atoms;
- the site came from a unique protein;
- no denoiser loss, density-quality score, optimizer result, or recovery metric was used.

The structural audit found 633 eligible sites from 67 untouched test proteins. The frozen panel deliberately maximizes chemical breadth: 14 residue types not present in the original five-site panel, plus one MET bridge control.

## Panel

| Site | Residue | χ count | Occupancy A/B | A–B RMSD (Å) | Kinematic error (Å) |
|---|---:|---:|---:|---:|---:|
| 1ZV8 E ASN1 | ASN | 2 | 0.33/0.67 | 3.116 | 0.011 |
| 6Y4G B CYS260 | CYS | 1 | 0.47/0.60 | 1.759 | 0.096 |
| 7UO8 A GLN53 | GLN | 3 | 0.34/0.66 | 4.688 | 0.091 |
| 3GMI A GLU5 | GLU | 3 | 0.77/0.23 | 4.178 | 0.078 |
| 2V05 A HIS168 | HIS | 2 | 0.61/0.39 | 2.854 | 0.050 |
| 8FBE B ILE92 | ILE | 2 | 0.38/0.62 | 2.344 | 0.039 |
| 7T7A A LEU396 | LEU | 2 | 0.38/0.62 | 1.673 | 0.082 |
| 3NY7 B LYS19 | LYS | 4 | 0.45/0.55 | 3.114 | 0.047 |
| 5KWB A PHE591 | PHE | 2 | 0.44/0.56 | 0.617 | 0.057 |
| 3K8W A SER337 | SER | 1 | 0.46/0.40 | 1.816 | 0.044 |
| 4MKM A THR77 | THR | 1 | 0.41/0.59 | 2.076 | 0.048 |
| 5DBA A TRP325 | TRP | 2 | 0.55/0.45 | 2.811 | 0.043 |
| 2VFP A TYR417 | TYR | 2 | 0.42/0.58 | 0.567 | 0.082 |
| 8DJ2 A VAL893 | VAL | 1 | 0.66/0.34 | 1.093 | 0.029 |
| 5Z8H A MET730 | MET | 3 | 0.28/0.72 | 1.230 | 0.085 |

χ-complexity coverage is four 1-χ sites, seven 2-χ sites, three 3-χ sites, and one 4-χ site.

## Production-support validation

Production support is implemented and validated for all 15 panel residue types. The shared definitions include residue-specific chi topology, terminal-group rotamer centers, and symmetry-aware RMSD for equivalent atom labels. Remote validation passed at all 15 sites: identity reconstruction was effectively exact, deposited-B reconstruction stayed below 0.1 Å, and every chi had a finite nonzero gradient. The unchanged soft-physics calibration passed 15/15 sites, and deposited A+B produced lower denoised-density loss than A-only or B-only at every site.

Backward compatibility was checked against the original five-site calibration. All A-only, B-only, A+B, soft-physics, and B-minus-A values were numerically identical (maximum absolute difference 0.0), and all five original physics gates still passed.

The authoritative symmetry-aware remote selection is saved at `/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_15_v2_symmetry/selection.json`; its full structural audit is at `/home/dev/qfit_unet_data/density_denoiser/expanded_heldout_selection_audit_v2_symmetry/candidate_audit.csv`.
