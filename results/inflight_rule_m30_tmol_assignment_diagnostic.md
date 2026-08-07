# In-flight rule, PHE/TYR m-30, and assignment-tmol diagnostic

**Date:** 2026-07-24

The running 20-site controller was not modified, restarted, or synchronized
during this diagnostic.

## A. In-flight rule and rotamer-loss scope

The in-flight run uses:

```text
2026-07-24-altloc-minstate-rotwidth-hisunion-aromatic45-matched-tmol-v2
```

The `residue_geometry.py` SHA256 stored at launch is
`2e6d2b57338e464928024f69d704968aa78cba0f83fc0ea382782b8add06c2b4`.
It exactly matches the remote source currently frozen on the pod.

The optimizer computes chi angles and the rotamer penalty only for each moving
conformer of the target sidechain. Neighboring sidechains contribute
coordinates to the direct and symmetry environments, but their rotamers are
not scored. Consequently, a later PHE/TYR table change affects only the target
sites 5KWB PHE591 and 2VFP TYR417 in this panel. HIS168 already ran under the
v2 HIS union and does not need rerunning for that change.

## B. Richardson m-30 check

| Deposited conformer | chi1 | chi1 bin | chi2 | m-30 supported? |
|---|---:|---|---:|---|
| 4ZXG PHE35 A | -57.351 | m | -39.439 | yes |
| 4ZXG PHE35 B | -56.493 | m | -43.053 | yes |
| 5W2O PHE239 B | -80.453 | m | -37.633 | yes |
| 5W2O TYR337 A | -176.999 | t | 32.585 | **no** |
| 8FBE TYR37 A | -61.429 | m | 7.395 | yes |
| 8FBE TYR37 B | -83.780 | m | 25.335 | yes |

Five of six conformers support the m-30 diagnosis. 5W2O TYR337 A is a t/33
tuple, not Richardson m-30, and may be strained or belong to a state absent
from the stated tuple list.

## C. Read-only proposed marginal check

No table was changed. Applying mod-180 centers `{30, 90, 150}` with a
25-degree width to the saved floor angles gives:

- all six residual conformers pass;
- maximum deviation among them is 22.605 degrees at 8FBE TYR37 A;
- PHE becomes 1/20 rejected and TYR becomes 1/20 rejected;
- the remaining PHE/TYR failures are 7RG7 PHE221 A at 25.929 degrees and
  5LRE TYR472 A at 25.815 degrees.

The proposed marginal gate cannot be described as net stricter. The current
single physical center at 90 degrees with width 45 covers 90/180 degrees of
the mod-180 circle. Three centers with width 25 cover 150/180 degrees. It is
locally tighter around each state but globally more permissive. In this sample
it would reduce the total floor from 12/340 to 8/340 while exchanging six
failures for two different boundary failures.

Because one tuple fails the m-30 prerequisite and the proposed marginal gate
is not net stricter, no v3 source rule or official rerun was created.

The sample contains only ten pairs per residue type, and the six residual
PHE/TYR conformers come from four sites. The m-30 structural omission is
evidence independent of the small sample; the observed 15% rejection rate is
not a precise population estimate. Roughly, a two-failing-pairs-of-ten
binomial interval spans about 3-56%.

The accepted floor remains:

- three heterogeneous LYS conformers, including the sole direct/symmetry
  clash pair;
- CYS247 A, 5.469 degrees beyond its chi1 width;
- ASP87 A, only 0.164 degrees beyond its chi1 width, reported as a graded
  deviation rather than treated as a scientifically meaningful discontinuity.

## D. Assignment-specific stale-endpoint tmol

| Site | Assignment | Pass | Median margin | 5th-95th percentile | Pearson margin/RMSD | Spearman |
|---|---|---:|---:|---:|---:|---:|
| 3K8W SER337 | A | 86/86 | -0.184 | -0.202 to -0.171 | -0.724 | -0.920 |
| 3K8W SER337 | B | 0/70 | 2.271 | 2.054 to 3.216 | 0.995 | 0.996 |
| 8Q6Q ASP81 | A | 58/60 | -0.117 | -0.698 to -0.014 | -0.802 | -0.520 |
| 8Q6Q ASP81 | B | 0/75 | 0.973 | 0.861 to 1.095 | -0.028 | 0.552 |

Selected-pair outcomes make the correlation explicit:

```text
3K8W: A passes 50/50; B passes 0/50; both pass 0/50
8Q6Q: A passes 48/50; B passes 0/50; both pass 0/50
```

Both sites have a systematic B-assignment reference problem. This is not
independent threshold noise and cannot be repaired by choosing a global
tolerance from pooled conformers.

Machine-readable evidence:

```text
/home/dev/qfit_unet_data/density_denoiser/
  stale_v5_3k8w_8q6q_assignment_tmol_v1/
  deposited_altloc_false_rejection_floor_v4_his_aromatic_rule_v2/
    analysis/rotamer_per_chi_v1/
```
