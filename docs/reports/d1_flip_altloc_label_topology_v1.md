# D1 flip panel: deposited altloc-label topology

**Completed:** 2026-08-04  
**Panel:** 19 flip sites with complete qFit +/-3 windows  
**Pod output:** `/home/dev/qfit_unet_data/qfit_audit/d1_flip_altloc_topology_v1`

## Conclusion

The concern is real: only **9/19** sites have an uninterrupted chain of
residues with explicit complete backbone A and B records from the flip centre
to **both** +/-3 anchors.  The other 10 have at least one intervening
nonflexible/shared or partial-label residue, so their raw A-versus-B anchor
coordinate comparison cannot be treated as evidence that the anchor belongs
to the flip's same labelled conformational group.

This does not remove the entire signal.  All nine label-connected +/-3
windows still have maximum anchor disagreement above 0.1 A (range
0.636–1.402 A).  In particular, for **7UTC A:ARG52** the - direction is
label-gapped, but the + direction is explicitly A/B-contiguous through +3
and its +3 anchor difference is 1.573 A.  Thus that directional part of the
7UTC result is not a cross-gap A/B-label artifact.

## Definition

For each residue and each N/CA/C/O atom, this audit reads the original PDB
records rather than qFit's extracted A/B structures.  A residue is
`explicit_complete_A_B` only when all four backbone atoms have explicit A
and B records.  A blank/shared residue is a **gap**: qFit puts it into both
extracted models, but it does not carry an A/B state label across the gap.

The table reports the number of consecutive explicit-A/B residues from the
centre in each direction.  A direction is label-connected to +/-k exactly
when its run length is at least k.

| Site | Explicit A/B run: - / + residues | Label-connected to -3 / +3? | k=3 anchor differences: - / + (A) |
|---|---:|---|---:|
| 5IKU A:TYR970 | 10 / 10 | yes / yes | 0.481 / 0.811 |
| 5FG8 A:LYS268 | 2 / 7 | no / yes | 0.000 / 1.111 |
| 7UTC A:GLY190 | 4 / 10 | yes / yes | 0.821 / 1.225 |
| 7UTC A:ARG52 | 1 / 5 | no / yes | 0.000 / 1.573 |
| 6P2N A:GLY161 | 1 / 0 | no / no | 0.000 / 0.247 |
| 1ZXT C:TYR14 | 6 / 4 | yes / yes | 0.805 / 0.127 |
| 5YNF A:GLN49 | 10 / 3 | yes / yes | 0.885 / 0.519 |
| 7SC4 B:PRO2317 | 10 / 4 | yes / yes | 0.734 / 1.402 |
| 4E4Y A:ALA75 | 10 / 4 | yes / yes | 0.132 / 1.004 |
| 7ZTL A:ILE257 | 0 / 2 | no / no | 0.000 / 0.000 |
| 8R7O C:THR1681 | 1 / 8 | no / yes | 0.586 / 0.667 |
| 5YNF A:ASP114 | 5 / 1 | yes / no | 0.887 / 0.000 |
| 5J1A A:PRO195 | 10 / 10 | yes / yes | 0.077 / 0.636 |
| 1RWR A:ASN294 | 2 / 1 | no / no | 0.000 / 0.000 |
| 4YZG A:SER181 | 10 / 10 | yes / yes | 0.749 / 0.473 |
| 5OHJ A:SER540 | 0 / 6 | no / yes | 0.527 / 0.607 |
| 4HFS A:TYR200 | 2 / 8 | no / yes | 0.000 / 1.118 |
| 8AJK B:ASN231 | 10 / 1 | yes / no | 0.608 / 0.253 |
| 8AJK A:VAL240 | 10 / 10 | yes / yes | 0.589 / 0.652 |

## The requested A/B-swap check

For the current anchor statistic,

```text
D = max_atoms |A_anchor - B_anchor|
```

swapping the anchor labels produces `max_atoms |B_anchor - A_anchor|`, which
is identically the same number.  Consequently, **0 of the available
site-direction-k comparisons** show a reduced value after swapping; this is
an algebraic invariance, not evidence that deposited labels are consistent.

The 6NI9/6NI6 result must therefore have used a different label-sensitive
comparison (for example, pairwise model/ensemble correspondence after a
relabel), rather than the within-anchor A-versus-B coordinate separation used
by the closure diagnostic.  The direct way to make the present test
label-sensitive is the topology criterion above: do not compare a labelled
anchor to a labelled centre across a shared/nonflexible gap.

## Consequence for the prior wider-window report

The raw wider-window anchor table should be partitioned as follows:

- **Label-connected directions:** valid deposited-model evidence of a
  different A/B backbone state at that anchor, though still not proof of a
  physical transition rather than refinement/model correlation.
- **Label-gapped directions:** uninformative about flip-to-anchor coupling;
  their A/B disagreement may be an independently assigned local group.

Accordingly, the earlier 16/19 "never reaches 0.1 A" count must not be read
as 16 globally coupled motions.  The robust statement is narrower: the nine
fully label-connected +/-3 windows all retain >0.1 A disagreement, and
7UTC A:ARG52 has a label-connected +3 discrepancy of 1.573 A.
