# 2VFP TYR417 and 3A1C ARG447 site handoff

Pulled from `qfit-unet` on 2026-07-28. No density maps are included.

- Exact selections: `2VFP A TYR417` and `3A1C B ARG447`; deposited conformers are altlocs `A` and `B` at both sites.
- `site_metadata.json/csv`: resolution, occupancy convention, 32^3 crystal-frame box geometry, renderer B-factor rule, and conventional symmetry-aware A/B separation.
- `atom_parameters.csv`: deposited per-atom coordinates, occupancies, and B_iso values.
- `deposited_sites/`: exact residue records extracted from held-out source PDBs.
- `decoys/`: three actual failed synthetic-target endpoints per site from frozen metric `qfit-synth20-merge050-one-to-one-tmol044-v3`. Slot altlocs `1`-`4` identify optimizer slots and are not deposited A/B labels.
- `decoy_manifest.csv`: failure reasons, metrics, occupancies, and candidate IDs.
- `provenance/`: exact run configs, all 50 endpoint rows, frozen-v3 audit rows, implementation files, and SHA-256 source inventory.

Occupancy note: selection records contain raw deposited median side-chain occupancies. The optimizer uses `A/(A+B)` and `B/(A+B)`. Both sites have A+B=1.00, so raw and normalized values are numerically identical.

Separation note: 0.566776 A and 5.796252 A are conventional heavy-atom RMSDs, `sqrt(mean_atoms(sum_xyz(delta^2)))`, minimized over chemically equivalent terminal labels. The historical 3A1C selection JSON contains 3.354751 A from the older coordinate-wise RMSD; it is not the current conventional number.

Box note: first/last values are voxel-center coordinates. Their span is 15.5 A; the 32 cells at 0.5 A spacing cover a 16.0 A cell extent.
