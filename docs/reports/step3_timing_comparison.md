# Step 3: one-protein A′ versus qFit timing

## Accounting correction

The first Step 3 report was not comparable. A′ was constructed with
`residual_scale_mode="deposited_ab"`, multiplying its target by `3.35814`,
while qFit used the unscaled target. The current benchmark sets both methods to
`residual_scale_mode="none"` and evaluates quality on that same target.

For both methods the reported quality is the pure masked density residual

```text
RSS = sum((target - sum_i(w_i * rho_calc_i))**2)
```

over the same 1,539 voxels and the same Torch multi-Gaussian renderer. The
weights are solved by the same occupancy treatment, `0 <= w_i` and
`sum(w_i) <= 1`. Seam, Ramachandran, and omega terms are not included in either
reported RSS.

## Site and hardware

Both runs used `7UTC A:ARG52` on the same qfit-unet pod: Intel Xeon Platinum
8470, 208 logical CPUs, Python 3.12.13, Torch 2.13.0+cpu, and Torch
intra/inter-op threads set to 1/1. The fixed map/grid/mask contains 1,539
voxels.

## Quality-matched timing

| method | wall clock | density vectors | renderer batches | optimizer accounting | pure masked RSS | RMSE |
|---|---:|---:|---:|---|---:|---:|
| A′, autodiff, `inner_nfev=8` | 629.64 s | 226 | 225 | 12 AL updates; 96 LM residual evaluations; 52 autodiff Jacobians; 20 columns/Jacobian | 11.0826 | 0.08486 |
| qFit, native search | 483.92 s | 1,009 candidates | 257 | 1,009 candidates; 10 candidate-pool conversions; batch size 4 | 12.8736 | 0.09146 |

A′ is therefore 1.30× slower in this quality-matched run, but its pure
density RSS is 13.9% lower. The old 10× A′ gap was not a penalty-term
accounting artifact: the old benchmark’s RSS was also computed from a pure
density QP, but against the incorrectly rescaled A′ target and with a much
worse finite-difference optimizer path.

The A′ inner LM solves reached their declared `nfev=8` cap (`status=0`) rather
than reporting SciPy convergence. The result is consequently a preliminary
quality-matched timing point, not a claim that the LM stopping criterion was
fully satisfied. Its final QP occupancies were `0.1386/0.0479`; the run did
not pass the stricter deposited A/B conformer-assignment verdict.

## qFit candidate distribution

For each of qFit’s 1,009 scored candidates, the harness also computed a
one-candidate bounded-NNLS RSS on the same target and mask:

| statistic | RSS |
|---|---:|
| minimum | 12.6550 |
| median | 17.4347 |
| mean | 16.8563 |
| maximum | 19.6958 |

The qFit final `12.8736` is the selected final output after its native pool
QP/MIQP flow, not the median single-candidate score.

## Autodiff and batching

The old A′ path rendered 40 separate `+/-0.25°` density columns per Jacobian.
The new path builds the 20-parameter Jacobian with one vectorized Torch
forward-mode autodiff graph per Jacobian and no density finite differences.
The renderer accepts batched candidate/slot coordinates; A′’s sequential
objective has one active moving slot at a time, while its final two-slot model
render is batched. qFit renders candidate pools in batches of four.

The comparison is preliminary because the algorithms still have different
search spaces, selection logic, and stopping criteria. It is now a valid
renderer/target/mask/occupancy/quality comparison; it is not yet a claim of
equal scientific search effort.
