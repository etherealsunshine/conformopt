# 3A1C ARG447 initialization tests

## Outcome

Both 50-start initialization tests completed. Neither recovered deposited A and B in the same start, so strict joint success remained 0/50.

| Initialization | Found A | Found B | Both with valid occupancies | Strict joint success | Entire K=4 endpoint physical |
|---|---:|---:|---:|---:|---:|
| Random baseline | — | — | — | 0/50 | — |
| Dunbrack top-10 ARG rotamers | 1/50 | 5/50 | 0/50 | 0/50 | 38/50 |
| Transfer from A_ARG129/B_ARG129 | 0/50 | 8/50 | 0/50 | 0/50 | 39/50 |

“Found” requires conventional heavy-atom RMSD below 1.0 Å and predicted occupancy above 0.10. Strict joint success additionally requires both A and B, each occupancy within ±0.20 of deposited occupancy (A=0.44, B=0.56), no sub-2 Å direct or symmetry clash, and canonical chi angles.

## Most informative endpoints

- Dunbrack's only A hit came from rotamer rank 3: RMSD 0.905 Å at occupancy 0.155. It did not find B, missed A's deposited occupancy by 0.285, and the A-assigned conformer was noncanonical. It is not a physical joint recovery.
- Dunbrack's best B hit was RMSD 0.329 Å at occupancy 0.466 (rotamer rank 3). Other B hits came from ranks 1, 5, 9, and 10.
- Transfer initialization's best B hit was RMSD 0.315 Å at occupancy 0.357, seeded from B_ARG129. A_ARG129-derived seeds produced 5 B hits; B_ARG129-derived seeds produced 3 B hits.
- Transfer initialization's closest approach to A was 1.324 Å, outside the conventional 1.0 Å threshold.
- No endpoint had a sub-2 Å direct or symmetry clash. The 12 Dunbrack and 11 transfer endpoints failing the whole-K=4 physical flag failed because at least one slot was noncanonical.

## Interpretation

Initialization affects single-basin reachability but does not solve the two-basin assignment problem. The target calibration favors the deposited mixture: A+B density loss was 0.598, versus 1.325 for A alone and 0.877 for B alone. Nevertheless, all successful geometrical hits were single-state recoveries. The optimizer usually assigned one substantial slot to B and spent the remaining occupancy on unrelated conformers instead of specializing another slot onto A.

Therefore the 3A1C failure is not explained solely by random chi initialization. Dunbrack and cross-site transfer make B easier to reach, but a mechanism that explicitly creates distinct slot responsibilities or fits residual density is still needed.

## Reproducibility

- Site: `3A1C_B_ARG447`
- Denoiser checkpoint: `model_unet2/epoch_002.pt`
- K=4, 50 starts per test, seed 41
- Density phase: 4 Å (100 steps, lr 1.0) → 2 Å (100 steps, lr 0.1) → full resolution (100 steps, lr 0.01), resetting Adam between stages
- Physics polish: 200 steps, lr 0.1, Adam reset
- Physics weights: VDW 1.0, rotamer 0.5, symmetry 5.0
- Dunbrack query: target phi/psi = -76.899°/-29.883°, nearest 10° bin = -80°/-30°; top 10 backbone-dependent ARG states, 5 perturbations per state
- Transfer source: five lowest-loss A_ARG129 endpoints and five lowest-loss B_ARG129 endpoints; source physical chi angles were converted into ARG447's local kinematic parameterization before adding perturbations
- Regression verification: 12 tests passed on the remote environment before launch

The machine-readable headline is in `initialization_comparison.json`; every endpoint and its complete physical audit is in `dunbrack_starts.csv` and `transfer_starts.csv`.
