# 7UTC A:ARG52 — residual-map calibration and 8-D landscape

**Status:** complete. This report covers the requested single-site scale correction and rerun, the existing 33-site flip-panel O-only re-expression, and the closure-null landscape test. No additional sites or mirror descent were run.

## 1. Residual-map scale calibration

The mismatch was in the audit wrapper's score target, not a different atomic form-factor implementation. `MapScaler` fits and applies an affine scale to the **full** experimental map ([`scaler.py`](../../external/qfit-3.0/src/qfit/xtal/scaler.py#L46), lines 71–89). qFit then subtracts calculated neighbours ([`qfit.py`](../../external/qfit-3.0/src/qfit/qfit.py#L269), lines 269–307), and `_convert` consumes that post-subtraction map directly as the target (lines 313–321). There is no residual-map re-fit.

At 7UTC, on the fixed A+B mask (1,539 voxels), the known deposited ensemble gave:

\[
\rho_{\rm residual}=0.2981461032\rho_{\rm calc}.
\]

Thus calculated density was **3.3540602721×** the residual-map amplitude. The PoC now has an opt-in `deposited_ab` control mode that multiplies the post-subtraction target by that reciprocal factor ([`run_d1_8d_sequential_poc.py`](../../scripts/run_d1_8d_sequential_poc.py#L151), lines 151–175). It is deliberately not prospective: it uses deposited A/B occupancies only to calibrate this diagnostic.

This correction is therefore a demonstrated **residual-map calibration bug in the PoC/audit path**, but not yet a panel-wide production-qFit change.

### Re-run result: 60 + 60 steps

| Quantity | Before calibration | After calibration |
|---|---:|---:|
| Deposited A/B QP weights | 0.0895 / 0.0810 | **0.3000 / 0.2717** |
| Deposited A/B weight total | 0.1704 | **0.5717** |
| Deposited occupancy total | — | 0.5700 |
| Final Slot 1 / Slot 2 QP | 0.1361 / 0.0493 | **0.4563 / 0.1654** |
| Final fitted total | 0.1854 | **0.6217** |
| Slot 1 RMSD to A / B (Å) | 0.5026 / 0.8815 | 0.5026 / 0.8815 |
| Slot 2 RMSD to A / B (Å) | 1.0499 / 0.9627 | 1.0499 / 0.9627 |

The deposited-model calibration now recovers the total occupancy accurately. The two fitted sequential slots still over-total by 0.0517, and their geometries are unchanged: scale fixed the occupancy issue, not the 8-D placement behaviour.

Authoritative pod result: `/home/dev/qfit_unet_data/qfit_audit/d1_8d_sequential_7utc_v2_residual_scaled_rerun1`.

## 2. Flip-panel metric: carbonyl-O coverage

The completed 33-site sampler panel is now reported as `1 − best O-only residual / deposited A→B O distance`, with no 1 Å threshold.

- Median coverage, all 33: **5.2%** (range 0–24.5%).
- Median coverage, 23 full 19-candidate sites: **7.8%**.

The full, auditable per-site A→B-distance table is in [d1_tier_a_flips_o_coverage_v1.md](d1_tier_a_flips_o_coverage_v1.md).

## 3. Closure-null landscape

The 14-torsion A→B least-squares direction was projected through qFit's local 8-D `null(compute_jacobian)` at deposited A ([`run_d1_8d_single_slot_landscape.py`](../../scripts/run_d1_8d_single_slot_landscape.py#L67), lines 67–79). Its A→B component is effectively zero: the predicted linear best amplitude is **3.02 × 10¹³°**. A scan from −22.5° to +90° moved the central backbone only 0.139 Å from A, so it is not a meaningful path to B.

The claimed Slot-1 intermediate is also **not a genuine local minimum**. An all-eight-null-axis perturbation test at the calibrated target found decreasing directions even at 0.5° ([`run_d1_8d_intermediate_basin.py`](../../scripts/run_d1_8d_intermediate_basin.py#L65), lines 65–93): best ΔRSS = **−0.0969** (axis 4, −0.5°); axis 0 at +2° reached ΔRSS = **−0.1502**. Therefore the 60-step Slot-1 endpoint is a line-search stopping point, not evidence for an intermediate basin that makes sequential 8-D placement impossible.

Landscape roots:

- `/home/dev/qfit_unet_data/qfit_audit/d1_8d_single_slot_landscape_7utc_v1`
- `/home/dev/qfit_unet_data/qfit_audit/d1_8d_intermediate_basin_7utc_v1`
