# D1 controls: carbonyl-O steering test

**Completed:** 2026-08-04  
**Pod output:** `/home/dev/qfit_unet_data/qfit_audit/d1_o_steering_controls10_v1`  
**Scope:** the exact ten complete controls from
`d1_reachability_controls10_v2`; tier (a) only.

## Intervention

During this run only, the one effective source-line change in qFit's actual
`QFitRotamericResidue._sample_backbone` was:

```python
atom_name = "O"  # instead of "CB"
```

The existing amplitudes, number of ellipsoid/fallback directions, inverse
kinematics, and 19-candidate protocol were otherwise unchanged.  The output
records the SHA-256 of the temporary imported qFit source.  After completion,
the local and pod sources were restored and checksum-verified at the original
CB-steering SHA-256 `5a501d...542aca6`.

## Result

**Superseded interpretation:** the ten-site aggregate is not a clean
steering-atom experiment because only two sites have ANISOU for both CB and
O. See [`d1_o_steering_o_only_followup_v1.md`](d1_o_steering_o_only_followup_v1.md)
for the O-only, ANISOU-qualified analysis. In particular, the O-only ten-row
slope is 0.6579 versus the CB baseline 0.6495, and both direct-ADP sites have
worse O residual under O steering.

The fitted central-backbone residual relation changes from the reproduced
CB baseline

```text
residual = 0.0388 + 0.4257 × deposited max backbone deviation
           slope SE = 0.0720, r = 0.9021
```

to all-O steering:

```text
residual = 0.0368 + 0.3761 × deposited max backbone deviation
           slope SE = 0.0618, r = 0.9068
```

The slope drops by **0.0496** (11.7% relative to the CB value), in the
predicted direction.  The O-steering median central `{N,CA,C,O}` residual is
**0.160 A**, versus the CB baseline median **0.173 A**.  With only ten
controls, the separate slope standard errors overlap, so this is directional
evidence for the channel mechanism rather than a decisive causal estimate.

| Site | A→B central RMSD | O-steered min central RMSD | Min O RMSD | Central / O fraction covered | CB baseline central RMSD | O−CB central (A) | Worse? | CB/O ANISOU |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 4HVN A:ALA28 | 0.244 | 0.177 | 0.132 | 27.3% / 30.8% | 0.230 | -0.052 | no | no / no |
| 3FXX A:LYS662 | 0.140 | 0.140 | 0.172 | 0.0% / 26.1% | 0.117 | +0.023 | **yes** | no / no |
| 7UTC D:LYS318 | 0.441 | 0.366 | 0.565 | 16.9% / 23.4% | 0.324 | +0.043 | **yes** | no / no |
| 3MKB A:GLU71 | 0.334 | 0.187 | 0.322 | 44.1% / 47.6% | 0.268 | -0.082 | no | no / no |
| 1NNW A:VAL149 | 0.292 | 0.194 | 0.237 | 33.6% / 43.9% | 0.273 | -0.079 | no | no / no |
| 4F0R A:ASP106 | 0.104 | 0.064 | 0.079 | 38.0% / 21.8% | 0.079 | -0.014 | no | no / no |
| 5JBX C:GLU132 | 0.129 | 0.111 | 0.124 | 13.9% / 37.1% | 0.105 | +0.006 | **yes** | yes / yes |
| 6I3B B:ALA209 | 0.324 | 0.190 | 0.252 | 41.5% / 22.6% | 0.266 | -0.076 | no | yes / yes |
| 8H0N A:TYR355 | 0.202 | 0.142 | 0.122 | 29.9% / 44.8% | 0.099 | +0.043 | **yes** | no / no |
| 4OIE A:PRO258 | 0.129 | 0.097 | 0.085 | 24.8% / 22.3% | 0.099 | -0.003 | no | no / no |

`fraction covered = 1 − min residual / deposited A→B distance`, calculated
with conventional four-atom RMSD and, separately, O-only distance.  It can be
zero (3FXX) when the deposited-A input itself is the best candidate.

## ANISOU comparability

Both O and CB have anisotropic ADPs at **2/10** sites (5JBX and 6I3B); neither
has one at the remaining **8/10**.  There are **0 CB-only** and **0 O-only**
sites.  Thus the comparison is not confounded by O lacking an ADP ellipsoid
where CB had one.  At the eight no-ANISOU sites, both arms use qFit's
isotropic geometric fallback; its first direction changes with the target
atom by design (CB−CA versus O−CA).

## Cost visibility

O-steering is worse in central-backbone residual for **4/10** controls:
3FXX A:LYS662 (+0.023 A), 7UTC D:LYS318 (+0.043 A), 5JBX C:GLU132
(+0.006 A), and 8H0N A:TYR355 (+0.043 A).  It is therefore not a uniform
improvement, even though the aggregate slope moves in the predicted direction.
