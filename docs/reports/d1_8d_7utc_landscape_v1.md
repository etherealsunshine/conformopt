# 7UTC A:ARG52 single-slot closure-null landscape

This is a one-site diagnostic, using the qFit `compute_jacobian` closure-null
space and the residual-map amplitude calibration measured from deposited A/B.
It does not run mirror descent or extend to another site.

## A→B tangent

At deposited A, the least-squares torsion direction towards deposited B loses
its signal when projected into `null(compute_jacobian)`: its linear best A→B
amplitude is **3.0 × 10¹³ degrees**.  In other words, this local closure-null
chart contains no usable tangent direction towards the flip.  The nominal
−22.5° to +90° scan moved the central backbone only 0.139 Å from A at most;
its shallow RSS changes are not an A→B profile.

## Claimed Slot-1 intermediate

The saved Slot-1 endpoint (0.503 Å from A; 0.881 Å from B) is **not a
certified local minimum**.  At the calibrated objective, 8-D local-axis scans
found several decreasing directions even at ±0.5°.

| Check | Result |
|---|---:|
| Baseline RSS | 132.7734 |
| Baseline occupancy | 0.5609 |
| Smallest axis step | 0.5° |
| Best small-step ΔRSS | −0.0969 |
| All 16 ±0.5° axes increase RSS? | No |
| Best sampled descent | axis 4, −0.5° |

Larger sampled moves continued to lower RSS (for example axis 0, +2.0°:
ΔRSS = −0.1502).  Thus the 60-step endpoint is a line-search stopping point,
not evidence of a genuine intermediate basin.  The result does **not** support
the conclusion that 8-D sequential placement is blocked by an intermediate
local minimum; the A→B direction is locally closure-inaccessible, and the
reported intermediate still has descent directions.

The residual-map fit was:

\[
\rho_{\rm residual}=0.298146\,\rho_{\rm calc},\qquad
\rho_{\rm calc}=3.354060\,\rho_{\rm residual}.
\]

Source roots:

- `/home/dev/qfit_unet_data/qfit_audit/d1_8d_single_slot_landscape_7utc_v1`
- `/home/dev/qfit_unet_data/qfit_audit/d1_8d_intermediate_basin_7utc_v1`
