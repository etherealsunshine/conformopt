# D1 controls: crankshaft-visibility stratification

**Status:** complete, 33/33 frozen non-flip controls

**Pod result root:**
`/home/dev/qfit_unet_data/qfit_audit/d1_tier_a_controls33_crankshaft_v2`

## Method

Each control used qFit's actual `_sample_backbone` tier-(a) candidate
generato]oo]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]















































































































]]]]\\\\




































shouldr.  For deposited A to B, this analysis measured the displacement of
Cβ (or O for GLY), the carbonyl O displacement, and the maximum displacement
over `{N, CA, C, O}`.  Cβ-visibility is `Cβ displacement / O displacement`
(`O/O = 1` for GLY).

The fitted model was:

```text
tier-(a) residual / max backbone displacement
    ~ intercept + max backbone displacement + Cβ-visibility
```

## Result

The Cβ-visibility coefficient is **+0.0459 ± 0.0284** per unit visibility
(`t = 1.61`, two-sided `p = 0.117`, `n = 33`, 30 residual degrees of freedom).
The sign is opposite the pre-registered prediction, but the interval includes
zero.  Thus this 33-control analysis does **not** establish a crankshaft
channel effect.

The fitted intercept is 0.4488 and the deposited-deviation coefficient is
0.0782; residual standard deviation is 0.1252.

## Raw points

The fully labelled, raw 33-point scatter is preserved on the pod:

```text
d1_tier_a_controls33_crankshaft_v2/crankshaft_visibility_scatter.png
```

The point-level CSV, including all displacement measures and qFit residuals,
is `per_site.csv` in the same root.

## Reproduction

`scripts/run_d1_tier_a_controls_crankshaft.py`
