# A′ final occupancy selection: decoupled MIQP

## Scope

This is Step 1 only. The continuous QP remains the occupancy solve inside the
geometry optimizer. The MIQP below is applied once, after geometry is fixed;
occupancies are not part of a geometry derivative and no timing experiment was
run.

Before the change, qFit's coupled constraint was confirmed in the vendored
source. The formulation is documented as `z_i t_min ≤ ω_i ≤ z_i`, with
`z_i ∈ {0, 1}`, at
[`external/qfit-3.0/src/qfit/solvers.py:126-133`](/Users/utkarsh/qfitonsteroids/external/qfit-3.0/src/qfit/solvers.py:126).
The implementation adds `w - z <= 0` and `w >= threshold * z` at
[`external/qfit-3.0/src/qfit/solvers.py:218-224`](/Users/utkarsh/qfitonsteroids/external/qfit-3.0/src/qfit/solvers.py:218).

A′ now uses the explicitly decoupled variant:

```text
sum(z_i) <= K
t_min * z_i <= w_i <= z_i
sum(w_i) <= 1
```

The defaults are `K=4` and `t_min=0.02`. The implementation is labeled
“A-prime decoupled MIQP” and is configurable through `--selection-k` and
`--selection-t-min`.

## K rule

`K` is an a priori cardinality cap, not a value inferred from the recovery
result or a floor coupled to `t_min`. The requested default `K=4` is therefore
used directly. It was not selected by BIC. The 6P2N A′ run has two candidate
slots, so its effective cap is `min(4, 2)=2`; both candidates can survive.

For transparency, the same fixed endpoint geometries were rescored with
`t_min=0.02` and caps 1 through 4. qFit-style BIC is lower-is-better and uses
the number of selected conformers, not the requested cap:

| Requested K | Effective K | Slot occupancies | RSS | Selected conformers | BIC (`k`) |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.000000 / 0.000000 | 95.398641 | 1 | -1130.217266 (`9.6`) |
| 2 | 2 | 0.811523 / 0.188477 | 87.336479 | 2 | -1124.037968 (`19.2`) |
| 3 | 2 | 0.811523 / 0.188477 | 87.336479 | 2 | -1124.037968 (`19.2`) |
| 4 | 2 | 0.811523 / 0.188477 | 87.336479 | 2 | -1124.037968 (`19.2`) |

Thus a BIC-minimizing cap choice on this one site would be K=1, because the
complexity penalty outweighs the RSS improvement. K=4 was a configured
capacity for the decoupled selection experiment, not a claim that BIC selected
four conformers. The implementation now records this cap sweep under
`bic_by_cardinality_cap` while leaving the production selector at the requested
independent cap.

qFit's native BIC convention is reused as a reported criterion: it counts
selected conformers (`w >= 0.002`), not the cardinality cap. With fixed
B-factors, the A′ final model uses three geometry parameters per atom, four
central GLY atoms, and the qFit complexity factor `0.8`, giving
`k = 3 × 4 × 2 × 0.8 = 19.2` for two selected slots. BIC is reported for the
fixed-cap result; it is not used to re-couple K and the occupancy floor.

## 6P2N A:GLY161 validation

The existing constrained-A′ v5 endpoint geometries were held fixed. The audit
was written to the new pod result root:

```text
/home/dev/qfit_unet_data/qfit_audit/d1_aprime_6p2n_miqp_selection_v4/
```

Deposited occupancies are `0.89 / 0.11` (A/B). Slot numbering below is the
existing A′ slot order, not a claim that slot 1 is deposited A.

| Selection | Slot occupancies | Surviving slots | RSS | BIC (`k`) |
|---|---:|---|---:|---:|
| Continuous QP | 0.811041 / 0.188959 | 0, 1 | 87.336427 | — |
| Old 0.09 cull | 0.811041 / 0.188959 | 0, 1 | 87.336427 | — |
| Decoupled MIQP, `K=4`, `t_min=0.02` | 0.811523 / 0.188477 | 0, 1 | 87.336479 | -1124.037968 (`19.2`) |

The 11% state survives the new decoupled selection and also happens to clear
the old 0.09 cull for this endpoint. The new MIQP does not change geometry.

### qFit native coupled `t_dmin`

| `t_dmin` | Slot occupancies | Surviving slots | RSS | BIC (`k`) |
|---:|---:|---|---:|---:|
| 1.00 | 1.000000 / 0.000000 | 0 | 95.398641 | -1130.217266 (`9.6`) |
| 0.50 | 1.000000 / 0.000000 | 0 | 95.398641 | -1130.217266 (`9.6`) |
| 0.33 | 0.670000 / 0.330000 | 0, 1 | 91.828081 | -1092.393374 (`19.2`) |
| 0.25 | 0.750000 / 0.250000 | 0, 1 | 88.177737 | -1117.989027 (`19.2`) |
| 0.20 | 0.800000 / 0.200000 | 0, 1 | 87.363950 | -1123.839530 (`19.2`) |

For this fixed v5 geometry, qFit's coupled solver selected one state at two of
the five thresholds, not three; the two-state solutions at lower thresholds
are pinned to their threshold rather than representing the 11% minor state.
The measured result therefore differs from the anticipated “three of five”
count, but it still demonstrates the constraint difference: the independent
`0.02` floor permits the 18.85% final-QP minor slot, while qFit's coupled
thresholds impose 33%, 25%, or 20% when they retain it.
