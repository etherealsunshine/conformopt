# Current-rule residual rotamer floor attribution

**Date:** 2026-07-24

This is an attribution-only diagnostic. No centers or widths were changed.

```text
geometry rule
2026-07-24-altloc-minstate-rotwidth-hisunion-aromatic45-matched-tmol-v2

population
170 deposited A/B pairs
340 deposited conformers
12 rotamer-rejected conformers
```

## Result

The 12 rejected conformers generate 13 failed-chi rows because one deposited
LYS conformer fails both chi1 and chi2.

| Site/control | Residue | Failed chi | Angle | Nearest state | Deviation | Width | Excess |
|---|---|---:|---:|---|---:|---:|---:|
| 1RZH L CYS247 A | CYS | chi1 | 110.469 | g+ | 50.469 | 45 | 5.469 |
| 2XQH A LYS345 A | LYS | chi3 | -124.004 | t | 55.996 | 45 | 10.996 |
| 2XQH A LYS345 B | LYS | chi1 | -131.932 | t | 48.068 | 45 | 3.068 |
| 2XQH A LYS345 B | LYS | chi2 | -110.650 | g- | 50.650 | 45 | 5.650 |
| 4C16 B ASP87 A | ASP | chi1 | -134.836 | t | 45.164 | 45 | 0.164 |
| 4ZXG A PHE35 A | PHE | chi2 | -39.439 | -90 | 50.561 | 45 | 5.561 |
| 4ZXG A PHE35 B | PHE | chi2 | -43.053 | -90 | 46.947 | 45 | 1.947 |
| 5W2O A PHE239 B | PHE | chi2 | -37.633 | -90 | 52.367 | 45 | 7.367 |
| 5W2O A TYR337 A | TYR | chi2 | 32.585 | +90 | 57.415 | 45 | 12.415 |
| 6Y4G A LYS480 A | LYS | chi2 | -126.683 | t | 53.317 | 45 | 8.317 |
| 7SSM A GLN252 A | GLN | chi2 | 107.709 | g+ | 47.709 | 45 | 2.709 |
| 8FBE A TYR37 A | TYR | chi2 | 7.395 | +90 | 82.605 | 45 | 37.605 |
| 8FBE A TYR37 B | TYR | chi2 | 25.335 | +90 | 64.666 | 45 | 19.666 |

## Attribution summary

By rejected conformer:

```text
chi1 implicated   3 / 12
chi2 implicated   9 / 12
chi3 implicated   1 / 12
```

The sum exceeds 12 because LYS345 B fails chi1 and chi2.

- All three PHE and all three TYR residuals are chi2 failures. The 45-degree
  widening bought one conformer per residue type but did not address these
  six deposited states.
- CYS and ASP are chi1 failures, as expected. ASP exceeds the threshold by
  only 0.164 degrees; CYS exceeds it by 5.469 degrees and may represent real
  strain.
- GLN fails chi2 by 2.709 degrees. Its effectively full-circle terminal chi3
  passes, so this is not a terminal-window failure.
- The three LYS conformers are heterogeneous: one chi3 failure, one combined
  chi1/chi2 failure, and one chi2 failure.

The residual floor is therefore concentrated in chi2, principally the six
PHE/TYR conformers—not generally in chi1. This finding does not authorize
another width or center change.

Machine-readable evidence:

```text
/home/dev/qfit_unet_data/density_denoiser/
  deposited_altloc_false_rejection_floor_v4_his_aromatic_rule_v2/
    analysis/rotamer_per_chi_v1/
      rotamer_rejection_per_chi.csv
      rotamer_rejection_summary.csv
      summary.json
```
