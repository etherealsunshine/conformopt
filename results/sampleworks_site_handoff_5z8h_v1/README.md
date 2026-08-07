# 5Z8H MET730 site handoff

Pulled from `qfit-unet` on 2026-07-28. No density maps are included.

- Exact selection: `5Z8H A MET730`; deposited conformers are altlocs `A` and `B`.
- `site_metadata.json/csv`: resolution, occupancy convention, 32^3 crystal-frame box geometry, renderer B-factor rule, and conventional A/B separation.
- `atom_parameters.csv`: deposited per-atom coordinates, occupancies, and B_iso values.
- `deposited_site/`: exact residue records extracted from the held-out source PDB.
- `decoys/`: three actual failed synthetic-target endpoints from frozen metric `qfit-synth20-merge050-one-to-one-tmol044-v3`. Slot altlocs `1`-`4` identify optimizer slots and are not deposited A/B labels.
- `decoy_manifest.csv`: failure reasons, gate results, occupancies, candidate IDs, RMSDs, and tmol details.
- `provenance/`: exact run config, all 50 endpoint rows, frozen-v3 audit rows, implementation files, and SHA-256 source inventory.

Occupancy note: the selection record contains raw deposited median side-chain occupancies. The optimizer uses `A/(A+B)` and `B/(A+B)`. Here A+B=1.00, so raw and normalized values are numerically identical.

Separation note: the reported value is conventional heavy-atom RMSD, `sqrt(mean_atoms(sum_xyz(delta^2)))`, minimized over chemically equivalent terminal labels where applicable.

Box note: first/last values are voxel-center coordinates. Their span is 15.5 A; the 32 cells at 0.5 A spacing cover a 16.0 A cell extent.
