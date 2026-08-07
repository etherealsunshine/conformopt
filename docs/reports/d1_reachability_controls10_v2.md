# D1 reachability decomposition — ten controls

**Status:** complete, corrected audit wrapper

**Authoritative pod root:**
`/home/dev/qfit_unet_data/qfit_audit/d1_reachability_controls10_v2`

**Scope:** ten strict seven-residue non-flip controls. Three frozen-list rows
were replaced or excluded because the deposited-B window was truncated (two)
or the source PDB was absent from the pod train tree (one).

## Method

- Tier (a) calls qFit's actual `_sample_backbone`: the deposited input plus
  18 inverse-kinematic displacement solutions.
- The closure Jacobian comes from qFit's `compute_jacobian`; finite-difference
  atom Jacobians are measured through qFit's `BackboneRotator` at deposited A.
- Every finite-difference and projected tier restores the mutable qFit window
  to deposited A before the next measurement.
- The independent wrapper reproduced qFit's sampled central coordinates with
  maximum absolute error 0 Å at all ten sites.

## Results

All ten controls had closure rank 6 and an 8-dimensional closure-null space.
The central-Cβ image had dimension 3, forfeiting five directions. Adding all
seven carbonyl O atoms restored all eight closure-null directions.

| Metric | Median | Range |
|---|---:|---:|
| Actual qFit candidate central-backbone RMSD to B | 0.173 Å | 0.079–0.324 Å |
| Cβ-image projected RMSD | 0.340 Å | 0.133–0.981 Å |
| Cβ + all-O projected RMSD | 0.343 Å | 0.115–0.852 Å |
| Full closure-null projected RMSD | 0.343 Å | 0.115–0.852 Å |
| `||J_O(forfeited)|| / ||J_O(all null)||` | 0.565 | 0.363–0.726 |

The Cβ + O and full-null tiers coincide because their local image dimensions
are both eight. The projected endpoint gap is modest in this control set;
that does not negate the five-dimensional Cβ forfeiture or the carbonyl
sensitivity in those directions.

Two controls exceed the 30° phi/psi local-linearisation limit and must be
read through subspace angles rather than projected endpoint RMSD. Two controls
also have a deposited omega change above 10°, a separate component that qFit's
phi/psi-only sampler cannot reach.

## Sampler-discarding measurement

For 4HVN A:ALA28, the largest discarded window motion is 0.792 Å at the
immediate next residue (centre+1), not at a window edge. Retaining only the
central coordinates leaves one sampled outgoing peptide C–N distance at
1.668 Å: +0.339 Å, or 24.2 conventional restraint sigmas from 1.329 Å.

## Reproduction

- `scripts/run_d1_reachability.py`
- `scripts/trace_d1_sampler_discard.py`
- `external/qfit-3.0/src/qfit/backbone.py`
- `external/qfit-3.0/src/qfit/samplers.py`
- `external/qfit-3.0/src/qfit/qfit.py`
