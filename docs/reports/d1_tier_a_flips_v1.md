# D1 tier-(a) sampler on all flip sites

**Status:** complete, 33/33 frozen flip-filter rows

**Authoritative pod root:**
`/home/dev/qfit_unet_data/qfit_audit/d1_tier_a_flips_v1`

## Question and method

This is tier (a) only. For each deposited-A flip site, qFit's actual
`_sample_backbone` method was run and the minimum central `{N, CA, C, O}` RMSD
and carbonyl-O-only RMSD to deposited B were measured across its returned
candidates. No Jacobian or projection tier was run.

The comparison references the ten-control result:

```text
control median residual = 0.173 Å
control fit: residual = 0.039 + 0.426 × deposited deviation
```

## Headline

- qFit generated the expected 19 candidates at **23/33** sites.
- qFit's own neighbour guard returned only the deposited input at **10/33**
  sites; these are valid sampler outcomes, not audit failures.
- **1/33** sites meets the 2009 qFit 1.0 Å central-backbone criterion. It is a
  guard-rejected one-candidate site; **0/23** full 19-candidate sites meets the
  criterion.
- Across all 33 sites, median central-backbone residual is **1.462 Å**
  (range 0.992–2.437 Å); median carbonyl-O residual is **2.649 Å**.
- Residual is strongly associated with deposited deviation: Pearson
  **r = 0.888**, Spearman **rho = 0.874**. This is not a flat failure floor.
- Flips sit above the control relation: median observed residual minus its
  control-line prediction is **+0.257 Å** (full 19-candidate subset:
  **+0.282 Å**).

The control relation is a descriptive reference, not an established flip
calibration: the controls span 0.123–0.738 Å deviation, while flips span
1.823–3.848 Å. Using the line for flips is therefore a substantial
extrapolation.

## Per-site results

`Δline` is observed tier-(a) residual minus the control-line prediction.
`guard` means qFit did not generate 18 IK samples because its own required
three-neighbour guard rejected the site.

| Site | qFit result | n | Deviation | Backbone RMSD | O RMSD | <1 Å | Δline |
|---|---|---:|---:|---:|---:|---|---:|
| 4LZK C:THR133 | guard | 1 | 1.823 | 1.006 | 1.823 | no | +.191 |
| 3P09 A:THR43 | guard | 1 | 1.968 | .992 | 1.968 | yes | +.115 |
| 4B1T A:GLY184 | guard | 1 | 2.123 | 1.201 | 2.123 | no | +.257 |
| 6ANA L:GLN27 | guard | 1 | 2.169 | 1.192 | 2.169 | no | +.229 |
| 5IKU A:TYR970 | 19 candidates | 19 | 2.172 | 1.199 | 1.937 | no | +.235 |
| 5FG8 A:LYS268 | 19 candidates | 19 | 2.189 | 1.290 | 1.935 | no | +.318 |
| 2IBN A:TRP285 | guard | 1 | 2.274 | 1.206 | 2.274 | no | +.199 |
| 7SC4 B:GLY991 | guard | 1 | 2.321 | 1.221 | 2.321 | no | +.193 |
| 7UTC A:GLY190 | 19 candidates | 19 | 2.412 | 1.140 | 2.151 | no | +.074 |
| 7UTC A:ARG52 | 19 candidates | 19 | 2.480 | 1.092 | 1.981 | no | −.003 |
| 1O6Z D:GLY30 | guard | 1 | 2.564 | 1.743 | 2.099 | no | +.612 |
| 6P2N A:GLY161 | 19 candidates | 19 | 2.615 | 1.595 | 2.325 | no | +.442 |
| 1ZXT C:TYR14 | 19 candidates | 19 | 2.640 | 1.258 | 2.307 | no | +.094 |
| 7Y9N A:LEU966 | guard | 1 | 2.719 | 1.462 | 2.719 | no | +.264 |
| 5YNF A:GLN49 | 19 candidates | 19 | 2.757 | 1.457 | 2.687 | no | +.243 |
| 7SC4 B:PRO2317 | 19 candidates | 19 | 2.774 | 1.374 | 2.389 | no | +.153 |
| 4E4Y A:ALA75 | 19 candidates | 19 | 2.791 | 1.166 | 2.107 | no | −.062 |
| 7ZTL A:ILE257 | 19 candidates | 19 | 2.880 | 1.348 | 2.552 | no | +.081 |
| 3NUG C:ALA46 | 19 candidates | 19 | 2.895 | 1.820 | 2.775 | no | +.547 |
| 8R7O C:THR1681 | 19 candidates | 19 | 2.910 | 1.415 | 2.742 | no | +.136 |
| 5YNF A:ASP114 | 19 candidates | 19 | 2.921 | 1.491 | 2.692 | no | +.208 |
| 5J1A A:PRO195 | 19 candidates | 19 | 2.946 | 1.591 | 2.649 | no | +.297 |
| 1RWR A:ASN294 | 19 candidates | 19 | 2.989 | 1.594 | 2.753 | no | +.282 |
| 4YZG A:SER181 | 19 candidates | 19 | 3.138 | 1.710 | 3.034 | no | +.334 |
| 5OHJ A:SER540 | 19 candidates | 19 | 3.382 | 1.831 | 3.261 | no | +.351 |
| 7R58 A:GLY89 | guard | 1 | 3.402 | 2.437 | 3.402 | no | +.948 |
| 4HFS A:TYR200 | 19 candidates | 19 | 3.431 | 1.975 | 3.212 | no | +.474 |
| 3M71 A:GLY161 | 19 candidates | 19 | 3.579 | 2.386 | 3.340 | no | +.822 |
| 8AJK B:ASN231 | 19 candidates | 19 | 3.601 | 1.784 | 3.322 | no | +.212 |
| 6HKG D:PRO142 | 19 candidates | 19 | 3.679 | 1.992 | 3.527 | no | +.385 |
| 6C0J B:VAL189 | 19 candidates | 19 | 3.790 | 2.117 | 3.657 | no | +.464 |
| 8AJK A:VAL240 | 19 candidates | 19 | 3.800 | 2.130 | 3.604 | no | +.472 |
| 4BTB A:GLU232 | guard | 1 | 3.848 | 2.405 | 3.848 | no | +.727 |

## Interpretation boundary

This establishes that qFit's actual central-residue backbone sampler does not
model these deposited peptide-flip alternates by the 1.0 Å criterion in the
23 sites where it produces its full candidate set. It does not identify which
downstream density-selection or refinement step would change that result, and
it should not be conflated with the discarded-neighbour geometry finding.

## Reproduction

- `scripts/run_d1_tier_a_flips.py`
- `external/qfit-3.0/src/qfit/qfit.py` (`_sample_backbone`)
- `external/qfit-3.0/src/qfit/samplers.py` (`BackboneRotator`)
