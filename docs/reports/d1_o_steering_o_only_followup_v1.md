# D1 O-steering: O-only refit and ANISOU-qualified interpretation

**Completed:** 2026-08-04  
**Analysis only:** no new sampling  
**Existing data:** `/home/dev/qfit_unet_data/qfit_audit/d1_o_steering_controls10_v1`  
**Refit output:** `/home/dev/qfit_unet_data/qfit_audit/d1_o_steering_controls10_v1/o_only_refit_v1`

## Plain conclusion

The intended steering-atom experiment ran on **n = 2**, not ten sites:
**5JBX C:GLU132** and **6I3B B:ALA209** are the only controls where both CB
and O have ANISOU records. The other **8/10** controls have neither atom
anisotropic, so qFit uses its geometric fallback in both arms. Because that
fallback's first direction is constructed from the chosen target atom, those
eight comparisons are **fallback-frame swaps**, not a clean CB-versus-O
steering-atom test.

Do not interpret the earlier ten-site four-atom slope reduction as a
steering-atom effect.

## Requested O-only regressions, all ten rows

The response is minimum central O RMSD; the predictor is deposited central-O
A→B displacement.

| Arm | Intercept (A) | Slope | Slope SE | r |
|---|---:|---:|---:|---:|
| CB baseline | -0.0134 | 0.6495 | 0.0626 | 0.9648 |
| O target | +0.0016 | 0.6579 | 0.0727 | 0.9545 |

On the atom the mechanism concerns, the ten-row O-target slope is actually
**0.0083 higher**, not lower. This aggregate comparison remains
methodologically confounded by the eight fallback-frame swaps, so it is not
evidence against O steering itself; it is evidence that the existing ten-site
panel does not test the claimed mechanism.

## Direct ANISOU subset (n = 2)

| Site | Deposited O A→B (A) | CB baseline min O RMSD (A) | O-target min O RMSD (A) | O−CB (A) |
|---|---:|---:|---:|---:|
| 5JBX C:GLU132 | 0.1965 | 0.0514 | 0.1236 | +0.0722 |
| 6I3B B:ALA209 | 0.3256 | 0.2054 | 0.2521 | +0.0467 |

O-target steering gives a higher minimum O residual at **both** direct-ADP
sites. A two-point fitted line is algebraically defined but has zero residual
degrees of freedom, hence no standard error or generalizable inference:

| Arm | Two-point slope | Slope SE |
|---|---:|---|
| CB baseline | 1.1933 | undefined (n=2) |
| O target | 0.9956 | undefined (n=2) |

The apparent two-point slope reduction is therefore not a result. The only
defensible direct statement is that these two sites did not improve on the
O-only endpoint under O steering.

## Scope boundary

No further sites were run. A proper experiment needs a new, resolution-
filtered panel selected for deposited anisotropic ADPs on both CB and O;
given the D3 rate of roughly 24% in the relevant resolution band, that is
separate panel construction work rather than a quick extension of this test.
