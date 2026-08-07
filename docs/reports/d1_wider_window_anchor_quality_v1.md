# D1 wider-window anchor and deposited-model quality check

**Completed:** 2026-08-04  
**Panel:** the 19 flip sites with a complete centre +/-3 qFit window  
**Authoritative pod output:** `/home/dev/qfit_unet_data/qfit_audit/d1_wider_anchors19_v3_chain_scoped`

## Result

Widening the closure window does **not** generally make the deposited A-to-B
motion closure-compatible *under the raw deposited A/B labels*.  At centre
/-3, 17/19 sites exceed the 0.1 A anchor-agreement criterion.  Across the
requested windows, only three sites ever fall below 0.1 A: 1RWR A:ASN294 and
7ZTL A:ILE257 already do so at +/-3, and 6P2N A:GLY161 first does so at +/-5.
The other 16/19 never do within the available range.

**Important qualification added after this run:** only 9/19 sites have an
uninterrupted explicit A/B backbone-label run to both +/-3 anchors.  The raw
metric is uninformative about centre-to-anchor coupling across the remaining
label gaps.  See
[`d1_flip_altloc_label_topology_v1.md`](d1_flip_altloc_label_topology_v1.md)
for the corrected, label-qualified conclusion: all nine connected windows
still exceed 0.1 A, but the 16/19 count must not be read as 16 globally
coupled motions.

The medians at each k are calculated only among sites having both requested
anchors.  They are therefore not a longitudinal trajectory of one fixed
population; loss of chain-end sites makes wider-k comparisons less favorable
to simplistic interpretation.

| Anchor offset k | Sites available | Median max A/B anchor difference (A) | Maximum (A) | Below 0.1 A |
|---:|---:|---:|---:|---:|
| 3 | 19 | 0.805 | 1.573 | 2 |
| 4 | 17 | 0.638 | 1.304 | 0 |
| 5 | 16 | 0.569 | 1.668 | 1 |
| 6 | 13 | 0.730 | 1.641 | 0 |
| 8 | 10 | 0.799 | 2.130 | 0 |
| 10 | 8 | 0.711 | 1.600 | 0 |

For the requested exemplar, **7UTC A:ARG52**, the anchor maximum is 1.573,
1.304, 1.158, and 1.641 A at k = 3, 4, 5, and 6 respectively; k = 8 and 10
are unavailable at the chain ends.  Thus wider anchors do not rescue this
site even before considering its nonzero omega changes.

## Deposited-model quality check

The table gives the two +/-3 anchor residues.  B is the mean of N/CA/C/O
B-factors; q is the median of their deposited occupancies.  Each entry is
`A:B/q | B:B/q`.  `best` is the least disagreement across the requested,
available k values.  The authoritative CSV contains these same B/q values
for every available k, not only k = 3.

| Site | k=3 max (A) | Best k:max (A) | -3 residue, A:B/q \| B:B/q | +3 residue, A:B/q \| B:B/q |
|---|---:|---:|---|---|
| 1RWR A:ASN294 | 0.000 | 3:0.000 | 291, 34.5/1.00 \| 34.5/1.00 | 297, 115.7/1.00 \| 115.7/1.00 |
| 1ZXT C:TYR14 | 0.805 | 4:0.585 | 11, 15.6/0.59 \| 15.6/0.39 | 17, 22.7/0.59 \| 22.8/0.39 |
| 4E4Y A:ALA75 | 1.004 | 5:0.215 | 72, 19.8/0.43 \| 19.8/0.32 | 78, 24.1/0.43 \| 25.1/0.32 |
| 4HFS A:TYR200 | 1.118 | 4:0.878 | 197, 23.2/1.00 \| 23.2/1.00 | 203, 14.9/0.50 \| 19.8/0.50 |
| 4YZG A:SER181 | 0.749 | 6:0.425 | 178, 12.8/0.39 \| 12.0/0.39 | 184, 13.0/0.39 \| 12.5/0.39 |
| 5FG8 A:LYS268 | 1.111 | 6:0.962 | 265, 45.9/1.00 \| 45.9/1.00 | 271, 45.3/0.39 \| 45.2/0.40 |
| 5IKU A:TYR970 | 0.811 | 10:0.199 | 967, 42.6/0.21 \| 42.1/0.20 | 973, 26.0/0.21 \| 26.7/0.20 |
| 5J1A A:PRO195 | 0.636 | 10:0.147 | 192, 13.7/0.42 \| 13.7/0.33 | 198, 25.1/0.42 \| 25.2/0.33 |
| 5OHJ A:SER540 | 0.607 | 3:0.607 | 537, 28.5/0.20 \| 29.2/0.15 | 543, 21.8/0.30 \| 21.8/0.31 |
| 5YNF A:ASP114 | 0.887 | 5:0.305 | 111, 34.7/0.85 \| 35.5/0.11 | 117, 38.6/1.00 \| 38.6/1.00 |
| 5YNF A:GLN49 | 0.885 | 3:0.885 | 46, 28.3/0.62 \| 26.3/0.08 | 52, 22.1/0.62 \| 22.8/0.08 |
| 6P2N A:GLY161 | 0.247 | 5:0.000 | 158, 12.3/1.00 \| 12.3/1.00 | 164, 10.4/0.51 \| 12.3/0.49 |
| 7SC4 B:PRO2317 | 1.402 | 10:0.673 | 2314, 29.7/0.44 \| 29.5/0.56 | 2320, 26.9/0.30 \| 26.9/0.15 |
| 7UTC A:ARG52 | 1.573 | 5:1.158 | 49, 65.1/1.00 \| 65.1/1.00 | 55, 20.2/0.24 \| 20.7/0.33 |
| 7UTC A:GLY190 | 1.225 | 4:1.117 | 187, 16.3/0.24 \| 15.8/0.20 | 193, 9.1/0.24 \| 9.0/0.20 |
| 7ZTL A:ILE257 | 0.000 | 3:0.000 | 254, 29.7/1.00 \| 29.7/1.00 | 260, 30.2/1.00 \| 30.2/1.00 |
| 8AJK A:VAL240 | 0.652 | 6:0.525 | 237, 29.3/0.00 \| 28.2/0.14 | 243, 29.9/0.18 \| 31.5/0.48 |
| 8AJK B:ASN231 | 0.608 | 10:0.344 | 228, 26.4/0.53 \| 26.2/0.24 | 234, 26.1/0.64 \| 25.9/0.17 |
| 8R7O C:THR1681 | 0.667 | 5:0.425 | 1678, 18.9/0.30 \| 19.2/0.74 | 1684, 25.6/0.45 \| 24.9/0.54 |

No source PDB reported an estimated coordinate error in a `REMARK 3`
coordinate-error field.  Consequently, a deposited coordinate-error estimate
is **unavailable for all 19 sites**; B/q are the usable local quality proxies.

At k = 3, disagreement has Pearson r = -0.175 with mean anchor B and
r = -0.404 with mean anchor occupancy (n = 19).  The latter is compatible
with a contribution from weakly occupied alternate positions, but it is not a
general explanation: the equivalent correlations vary in sign across k
(B: -0.175, +0.322, +0.133, -0.296, -0.246, -0.384; occupancy: -0.404,
+0.095, -0.098, +0.399, +0.514, +0.272 for k = 3, 4, 5, 6, 8, 10).  Several
substantial disagreements also occur at low-to-moderate B values, e.g. 7UTC
A:GLY190 is 1.225 A at k = 3 with anchor B means 16.3/15.8 and 9.1/9.0 A.

Thus the deposited A/B anchor differences are **not explained by B-factor or
occupancy quality alone**.  They demonstrate that, in these deposited models,
the flip-associated A/B coordinate differences commonly extend beyond three
residues.  This is evidence against a broadly applicable wider-window,
fixed-anchor Design A.  It does not by itself prove that every difference is a
physical coupled backbone motion: model/refinement correlations remain a
plausible contributor where occupancies are low.

## Measurement correction

An initial wider-window run (`d1_wider_anchors19_v2`) was invalidated before
interpretation.  Its B-window lookup searched all structure segments by
residue ID and, for identically numbered residues in a different chain,
could select the wrong chain (observed at 8AJK B:ASN231).  The v3 code scopes
the lookup to the requested chain, matching the existing `strict_window`
logic.  Direct raw-PDB validation for that site gives the corrected +/-3
maximum of 0.608 A (not the invalid 76.694 A).  The invalid v2 tree is retained
only for provenance.
