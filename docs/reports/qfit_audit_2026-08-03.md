# qFit audit — 2026-08-03

Runtime: ACTL `qfit-unet` in namespace `diffuse`; no analysis was run locally.
The local source branch is `qfitaudit`.

## Gate and environment

- qFit source: `ExcitedStates/qfit-3.0`, commit `33fd3e96bda99b896015e6569f01001eed7f53bf`, version `2025.4.dev2+g33fd3e96b`.
- qFit uses CVXPY after commits `7a88ebe` and `b38fd8d`; CPLEX is not required.
- Isolated pod environment: qFit, CVXPY 1.7.5, PySCIPOpt 6.2.1, CCTBX/mmtbx, and Gemmi 0.7.5. QP and MIQP wrapper smoke tests passed.
- `mmtbx.f_model` imported successfully. RCSB connectivity passed; PDBe timed out from the pod and was not used.

## Part 0.2 site gate

The scan covered 2,084 X-ray-supported structures with 0 parse errors. The broad metadata gate produced 232,890 A/B backbone rows; applying the requested 2015 carbonyl-flip filter produced 34 bounded sites. The 34-site panel is the downstream panel; no recovery/performance data was used to select it.

Panel metadata: `/home/dev/qfit_unet_data/qfit_audit/backbone_altloc_site_list_v4/flip_filter_sites.csv`

## D3

Deterministic random sample: 200 of 2,040 qualifying entries, seed `20260803`, resolution 1.0–2.0 Å.

| Resolution bin (Å) | n | none | waters/ions/ligands | protein main-chain+CB |
|---|---:|---:|---:|---:|
| 1.0–1.2 | 6 | 0 | 6 | 0 |
| 1.2–1.4 | 14 | 0 | 14 | 0 |
| 1.4–1.6 | 32 | 5 | 27 | 0 |
| 1.6–1.8 | 52 | 52 | 0 | 0 |
| 1.8–2.0 | 96 | 96 | 0 | 0 |

At-or-better counts were 26/200 at 1.45 Å, 42/200 at 1.50 Å, and 48/200 at 1.55 Å. The sample contained no protein ANISOU records under this source-file classification.

## D6 Tier 1

Completed 408 cases: 34 flip-filter sites plus 34 deterministic nonflip controls, each at 1.0, 1.2, 1.4, 1.6, 1.8, and 2.0 Å with 10% Gaussian noise. qFit’s CCTBX transformer, conformer mask, CVXPY QP, and SCIP MIQP wrappers were used.

DIRECT recovered both states above 0.09 in 33/34 flip sites at every resolution and 34/34 nonflip controls at every resolution. The literal PIPELINE `selected threshold + 0.09` cull retained both states in flip sites at 3/34 (1.0 Å), 1/34 (1.2 Å), and 0/34 thereafter; it retained 0/34 controls at every resolution. Per-case B-factor/scale-before-and-after values, BIC margins, occupancy errors, and sensitivity fields are in the CSV below.

For the 34 flip-filter sites × 6 resolutions (204 cases), the BIC loop selected `t_dmin=1.0` in 109 cases (the threshold itself permits only one conformer). Of the remaining cases, 95 had both states before the extra cull; 91 of those lost a state at `+0.09`. Thus 200/204 fail the final two-state cull and 4/204 survive. Thirty of the 34 unique sites select `t_dmin=1.0` at least once; 8 do so at all six resolutions.

Results: `/home/dev/qfit_unet_data/qfit_audit/d6_tier1_full_v2/`

Plots: `margin_vs_deviation.png`, `margin_by_resolution.png`, and `margin_flip_vs_nonflip.png` in that directory.

## D6 Tier 2

The conditional audit completed for all 34 flip-filter sites. It used deposited `FWT,PHWT` where available and a labeled `2Fo-Fc` proxy from `FP,SIGFP` plus `FC,PHIC` otherwise. Coverage was 17/34 sites: 13 deposited-FWT/PHWT maps and 4 derived proxies; 17 sites lacked a usable MTZ/map-coefficient route. The phase test is one-sided because both FWT/PHWT and FC/PHIC phases are refinement/model-derived. Direct both-state recovery occurred in 4/17 usable maps; the pipeline cull retained 0/17.

Results: `/home/dev/qfit_unet_data/qfit_audit/d6_tier2_realmap_v4/`

## Reproducible local scripts

- `scripts/build_backbone_altloc_site_list.py`
- `scripts/run_d3_anisou_audit.py`
- `scripts/run_d6_tier1_synthetic.py`
- `scripts/run_d6_tier2_realmap.py`

The active prospective recovery run was not opened, changed, or used for site selection.
