# 6P2N A:GLY161 (1.35 Å) A′ high-resolution basin test

**Status:** complete. This is the high-resolution flip-site follow-up to 7UTC.
The site was selected from frozen deposited metadata: it is the highest-
resolution flip with a compatible strict A/B seven-residue window after 3M71
(1.20 Å) failed that window gate.

## Result

The sequential A′ fit is **PARTIAL**, not a recovered A/B pair.

- Deposited central A→B backbone separation: **1.667 Å**.
- Deposited occupancies: **0.89 / 0.11**.
- Slot 1 finishes near A: **0.294 Å** from A and 1.655 Å from B.
- Slot 1 consumes its sequential occupancy capacity (**1.000**), leaving slot
  2 at zero occupancy and its deposited-A starting geometry.
- The final joint QP reassigns occupancy to **0.929 / 0.071**, but slot 2 is
  still A-like; it is not a recovered B conformer.

This is a procedural sequential-fit failure, rather than evidence that a
second density-supported geometry was found and rejected by the physical
terms.

## Map and representability

| item | value |
|---|---:|
| resolution | 1.350 Å |
| masked voxels | 631 |
| residual-map → model scale | 0.4850 |
| model → residual-map scale | 2.0619 |
| B endpoint full-window representability RMSD | 0.119 Å |

The B endpoint is adequately represented by the 20-parameter A′ chart. Thus
the failed sequential recovery is not explained by a closure/parameterization
failure at this site.

## Fixed-geometry slot-2 basin scan

After the sequential fit, slot 1 was held fixed and slot 2 was interpolated in
torsion space from its A-like converged geometry toward the fitted deposited-B
endpoint. Both QP occupancies were refitted on each training split and scored
on the five blocked held-out slabs.

| objective | best fraction | paired one-SD / non-significant fractions | interpretation |
|---|---:|---|---|
| density only | 1.5 | 1.4, 1.5 | prefers a point **past B** |
| density + frozen AL seam + Rama + omega | 0.0 | −0.1, 0.0 | remains at converged A |

Density-only held-out RSS reaches **13.976** at fraction 1.5, compared with
20.027 at the A-like converged point. But fraction 1.5 is 1.363 Å from
deposited B, has a 3.91 Å terminal-frame translation seam, 25.7° seam
rotation, and extremely poor Ramachandran support. It is therefore not a
physical recovered flip.

The full A′ objective has its minimum at fraction 0.0 (mean 1.051); its
accepted band is only −0.1 to 0.0. At this high-resolution site, the physical
terms reject density's off-manifold minimum, but they do not create a B basin.

## Reading the resolution comparison

This does **not** establish that higher resolution resolves the deposited flip.
It shows that at 1.35 Å the raw density objective is highly discriminating in a
way that points away from deposited B, while the physically regularized A′
objective prefers the A-like geometry. The next modelling question is why the
residual-map objective favors the off-manifold density feature and why the
sequential stage allows slot 1 to exhaust the capacity before slot 2 is fit.

## Artifacts

- Fit: `/home/dev/qfit_unet_data/qfit_audit/d1_aprime_6p2n_sequential_v3`
- Basin scan: `/home/dev/qfit_unet_data/qfit_audit/d1_aprime_6p2n_torsion_basin_v2`
