# qFit on Steroids

Research code for learning and optimizing crystallographic side-chain
ensembles from experimental electron density. The active pipeline combines
omit-map patches, a residual 3D U-Net, differentiable torsion optimization,
and downstream geometry/physics audits.

## Current status

The current branch is `testing`. Step 1 of the A′ occupancy work is complete:

- Geometry optimization still uses the inner continuous QP; occupancies remain
  outside geometry gradients.
- Final fixed-geometry selection now uses the decoupled MIQP variant with
  `K=4` and `t_min=0.02` by default.
- The 6P2N A:GLY161 test retains the 11% state under decoupled selection.
- qFit’s coupled threshold behavior and fixed-site BIC cap diagnostics are
  recorded in [`results/d1_aprime_miqp_selection_v1.md`](results/d1_aprime_miqp_selection_v1.md).
- The broader deposited-PDB occupancy-pileup test is being rebuilt with
  paper-defined qFit sets; no comparison result is claimed yet.

The optimizer, derivatives, and timing work were not changed for this step.

## Pipeline

```text
PDB + structure factors
        │
        ▼
experimental omit patch ──► residual 3D U-Net ──► denoised patch
                                                        │
                                                        ▼
                                      torsion/occupancy optimization
                                                        │
                                                        ▼
                              geometry, occupancy, clash, rotamer, tmol audits
```

The frozen held-out configuration uses 50 starts per site, a 500-step density
stage, an Adam reset, and a 200-step soft-physics stage. The strict audit
requires both deposited conformers, acceptable occupancies, no direct or
symmetry clash below 2 Å, allowed rotamers, and acceptable tmol energy.

## Repository map

| Path | Purpose |
|---|---|
| `density_denoiser/` | Active data, U-Net, optimizer, validation, and audit package. |
| `scripts/` | Reproducible analysis, audit, launch, and visualization scripts. |
| `tests/` | Project test suite. |
| `docs/` | Report index, figures, and long-form research context. |
| `experiments/` | Historical Probe and control experiments. |
| `results/` | Curated reports and compact evidence; full run trees remain on the pod. |
| `data/` | Small checked-in 2O1K reference inputs. |
| `AGENTS.md` | Operating rules, source catalog, pod layout, and run procedures. |
| `CURRENT_HANDOFF.md` | Latest results, interpretations, and active research backlog. |

Large runtime trees such as `artifacts/` and `external/` are local-only and
ignored by Git.

## Local source and pod runtime

Source is edited locally. Tests, preprocessing, optimization, and tmol run on
the Astera `qfit-unet` pod:

```text
local: /Users/utkarsh/qfitonsteroids
pod:   /home/dev/workspace
data:  /home/dev/qfit_unet_data
```

From the repository:

```bash
actl pod status qfit-unet
actl pod sync qfit-unet
actl pod exec qfit-unet
```

Run the test suite remotely:

```bash
actl pod exec qfit-unet --no-forwarding -- bash -lc \
  'cd /home/dev/workspace && /home/dev/qfit_unet_data/.venv/bin/python -m pytest -q'
```

The primary pod environment is `/home/dev/qfit_unet_data/.venv`; the separate
tmol environment is `/home/dev/qfit_unet_data/.venv-tmol`.

## Further reading

- [`AGENTS.md`](AGENTS.md) — mandatory operating rules and exact commands.
- [`CURRENT_HANDOFF.md`](CURRENT_HANDOFF.md) — latest scientific state and run roots.
- [`density_denoiser/README.md`](density_denoiser/README.md) — density-pair conventions.
- [`docs/reports/README.md`](docs/reports/README.md) — report index.
- [`results/README.md`](results/README.md) — curated result-artifact guide.

This is research software. Every reported number is tied to its run
configuration, data version, and audit definition.
