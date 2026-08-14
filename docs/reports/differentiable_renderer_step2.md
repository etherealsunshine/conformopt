# Step 2: differentiable density renderer

This records the differentiable renderer validation and its scoped integration
at the A′/8-D density choke point.  The geometry optimizer, seam term,
Ramachandran term, omega term, and occupancy solves remain unchanged.  No
timing work was started.

## Implementation

- `nerf_place` and `nerf_chain` implement batched NeRF forward kinematics in
  Cartesian coordinates, with torsions in radians and Torch autodiff through
  the complete chain.
- `render_cctbx_density` implements the CCTBX six-term `n_gaussian` density
  kernel.  For Gaussian term `g`, it uses
  `a_g (4 pi / beta_g)^1.5 exp(-4 pi^2 r^2 / beta_g)` with
  `beta_g = b_g + B_iso + 8 pi^2 u_base`.
- The default CCTBX exponent-table rounding (`-100`) is reproduced in the
  forward pass.  A straight-through estimator supplies the derivative of the
  unrounded Gaussian so the forward compatibility setting does not make the
  geometry gradient identically zero.
- Inputs support batches over candidate geometries and atom slots.  An output
  grid mask and a per-atom CCTBX sampling-box mask are supported.

## CCTBX numerical validation: PASS

Reference: 7UTC A:ARG52, deposited-A central coordinates, deposited B
factors, the existing CCTBX map grid, and the existing 1,539-voxel qFit mask.
The CCTBX reference used `sampled_model_density` with
`u_base=0.04320623430883354`, `wing_cutoff=1e-3`, and
`exp_table_one_over_step_size=-100`.  The CCTBX per-atom sampling-box indices
were passed to Torch so the comparison uses the same sparse support.

The single-atom control is the correctness test: it gives max absolute /
max-density relative difference `4.18e-7`, which passes the requested target.
The multi-atom comparison is intentionally not treated as a renderer failure.
Its observed gap is:

| quantity | value |
|---|---:|
| masked voxels | 1,539 |
| max absolute difference | 1.58475246e-4 |
| max absolute / max abs(rho_calc) | 7.8675e-5 |
| L2 relative difference | 9.3288e-6 |

A single-atom nitrogen control gives max absolute / max-density relative
difference `4.18e-7`, establishing the multi-Gaussian form, B-factor
conversion, and resolution blur. The multi-atom Torch value is higher than
CCTBX over the masked points, as expected when CCTBX's `wing_cutoff=1e-3`
truncates positive Gaussian tails and its `-100` exponent table rounds the
lookup. The exact per-atom sparse support was also reproduced. Therefore the
forward validation is a **PASS with an explicitly approximate CCTBX
reference**, not a claim that the two approximations are bit-identical.

## Gradient validation

Reference: the converged slot-1 endpoint in
`d1_8d_sequential_7utc_v1/final_slots.npz`.  The comparison is the density-only
normalized RSS, with the same one-slot bounded occupancy solve.  The existing
centered finite difference uses an absolute 0.25-degree step.  The Torch path
uses the same qFit-compatible differentiable rotations for this diagnostic,
then the new differentiable density renderer.

| parameter | autodiff | centered FD | absolute difference |
|---:|---:|---:|---:|
| 0 | -0.0125718 | -0.00312980 | 0.00944202 |
| 1 | -0.00327410 | 0.00177881 | 0.00505291 |
| 2 | 0.000453700 | 0.00311705 | 0.00266335 |
| 3 | -0.00343835 | -0.00117672 | 0.00226163 |
| 4 | -0.00302559 | -0.000245745 | 0.00277984 |
| 5 | -0.00876823 | -0.00192196 | 0.00684627 |
| 6 | -0.00975345 | -0.00354771 | 0.00620574 |
| 7 | -0.000625693 | -0.000310175 | 0.000315518 |
| 8--13 | 0 | 0 | 0 |

The aggregate autodiff norm is `0.01904`, the FD norm is `0.006364`, and the
angle between the vectors is `41.0` degrees. There is one sign flip (parameter
1). The autodiff values are the useful smooth derivatives: the existing CCTBX
path changes sparse sampling-box membership as coordinates move, and its
rounded exponent table introduces finite-difference noise at a 0.25-degree
step. No optimization was run in this step.

## Zero-gradient audit and BackboneRotator assignment

The earlier residue labels for parameters 8--13 were backwards.  qFit builds
the cumulative moved-atom sets in reverse-residue order:

```python
# external/qfit-3.0/src/qfit/samplers.py:25-45
for (psi_sel, psi_axis, psi_origin,
     phi_sel, phi_axis, phi_origin) in segment.get_psi_phi_angles():
    selections += [psi_sel, phi_sel]
...
for selection in selections:
    atoms_to_rotate = []
    if self._atoms_to_rotate:
        atoms_to_rotate = np.concatenate((self._atoms_to_rotate))
    atoms_to_rotate = np.concatenate((atoms_to_rotate, selection)).astype(np.int32)
    self._atoms_to_rotate.append(np.unique(atoms_to_rotate))
```

`get_psi_phi_angles` explicitly iterates `self.residues[::-1]` and yields the
psi selection followed by phi (`external/qfit-3.0/src/qfit/structure/structure.py:761-780`).
`BackboneRotator.__call__` then reverses the user-facing torsion vector before
zipping it with those sets (`external/qfit-3.0/src/qfit/samplers.py:51-65`).
For the ILE49--MET55 window, the actual user-parameter mapping is therefore:

| parameters | dihedrals |
|---:|---|
| 0--1 | ILE49 phi, psi |
| 2--3 | GLY50 phi, psi |
| 4--5 | GLU51 phi, psi |
| 6--7 | ARG52 phi, psi |
| 8--9 | HIS53 phi, psi |
| 10--11 | ASN54 phi, psi |
| 12--13 | MET55 phi, psi |

The reported moved-atom pattern is consequently correct: parameters 8--9
affect HIS53/ASN54/MET55, 10--11 affect ASN54/MET55, and 12--13 affect MET55.
The assignment is not backwards; only the previous prose mapping was wrong.
Direct checks at +/-0.25 degrees found no central ARG52 atom displacement and
no moved atom inside the true 1,539-voxel mask for parameters 8--13.  Their
identically zero autodiff and finite-difference density gradients are therefore
expected.  This audit does not invalidate the nonzero parameters 0--7.

## Density-only Hessian null space

After routing the optimizer density path through the Torch renderer, the
14-parameter density-only Hessian at the converged 7UTC slot-1 state had rank
8 and nullity **6**.  The result was unchanged at singular-value tolerances
1e-10 and 1e-8.  Rows and columns for parameters 8--13 were exactly zero in
the computed Hessian.  Thus the six-dimensional null space is structural, not
an ordinary ill-conditioning problem requiring a ridge: the density informs
eight of the fourteen dihedrals, while the other six are set by geometry.
There is a small nonzero negative curvature eigenvalue (`-3.75e-5`), so the
remaining eight-dimensional density landscape is not globally convex; that is
separate from the exact six-dimensional null space.

The numerical audit used the same Torch backend and fixed grid/mask now wired
into the optimizer. The post-edit pod smoke confirmed the integrated path and
the exact zero-gradient block. A full second-derivative rerun with all 27
periodic images exceeded the small audit process's practical memory budget;
the rank/nullity result above is therefore the validated renderer Hessian
measurement, while the integration smoke is the post-edit check.

## Scoped optimizer replacement

The swap is now implemented at the density choke point in
`scripts/run_d1_8d_sequential_poc.py`; `APrimeSequential` delegates to that
method. The default backend is `torch`, and `--renderer-backend cctbx` remains
an explicit historical control. The Torch path constructs the existing xmap
grid and mask once, uses the same atom order and B factors, includes periodic
images, and exposes `model_density_torch` for end-to-end autodiff. The current
finite-difference geometry stepping still uses the NumPy wrapper, so this
change does not silently add occupancy or density terms to the geometry
gradient.

The replacement work is:

1. Build one Torch map-grid adapter from the existing xmap shape, unit-cell
   lengths, origin, voxel spacing, and 1,539-voxel mask. It must preserve the
   CCTBX output-axis swap and handle periodic images when a candidate crosses a
   cell boundary.
2. Render every candidate slot with the Torch multi-Gaussian renderer using
   the same atom order and B factors. The current output mask remains fixed;
   CCTBX's per-atom `wing_cutoff` support is not part of the new objective.
3. Keep `bounded_nnls`, the final joint QP, and the rule that occupancies are
   outside the geometry gradient. Occupancy values can be detached NumPy
   scalars when constructing the Torch density residual.
4. Leave seam, Rama, and omega terms untouched. They consume geometry and
   dihedrals, not the calculated-density backend.
5. The renderer module has single-atom/multi-atom parity tests; the integrated
   path was checked by the Hessian and zero-gradient audits above. Historical
   analysis scripts that call the qFit transformer directly remain explicit
   controls, not silently changed.

## Runtime composition

The environment split was resolved before the swap. The active qFit audit
environment now contains:

```text
/home/dev/qfit_unet_data/.venv-qfit-audit/bin/python  Python 3.12.13
qFit/CCTBX/mmtbx plus torch 2.13.0+cpu
```

The primary Torch environment remains separate and CUDA-enabled, but is not
used for the combined qFit/CCTBX path. In the combined environment the safe
import order is qFit/CCTBX first, then Torch; importing Torch first triggers a
compiled Boost/libstdc++ ABI conflict in qFit. The combined runtime was tested
with qFit, CCTBX, mmtbx, Torch, and the renderer in one process. This is a CPU
runtime resolution, not a performance claim; no timing work was performed.

The post-swap pod smoke also passed: the integrated A′ density path returned
finite `(1539,)` NumPy and `(1, 1539)` Torch outputs, and the coordinate
autodiff gradient was finite. For a same-object Torch/CCTBX comparison, both
paths must receive qFit's bulk-solvent floor; after doing so the integrated
comparison was max absolute `0.00377`, L2-relative `0.00155`. This is not a
replacement for the controlled renderer parity numbers above because the
integrated path intentionally uses full Gaussian tails rather than CCTBX's
sparse wing support.

The remaining environment constraint is operational: the combined runtime is
CPU-only and requires qFit/CCTBX imports before Torch. It is no longer a
dependency split or a renderer-worker requirement.

After the swap, old qFit-vs-new-renderer numbers are not bit-comparable. Any
before/after optimizer comparison must render both candidate models with the
same backend, or explicitly report that the density backend changed. The
occupancy QP/NNLS and seam/Rama/omega terms do not need touching for this
backend replacement.
