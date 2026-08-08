# qFit on Steroids

Research software for fitting multi-conformer protein models to crystallographic
electron density. The project combines local density preprocessing, a residual
3D U-Net, differentiable backbone kinematics, occupancy optimization, and
geometry/physics audits.

This is an experimental research repository, not yet a packaged end-user
application.

## What is different here?

The project compares two related approaches:

| | qFit | A′ (A-prime) |
|---|---|---|
| Geometry search | Samples candidate conformers | Optimizes torsions continuously |
| Density backend | CCTBX/qFit renderer | Differentiable Torch renderer |
| Occupancies during geometry search | qFit QP/MIQP flow | Continuous QP kept outside the geometry gradient |
| Final selection | Coupled `t_dmin` threshold | Decoupled MIQP: `sum(z) <= K`, `t_min z_i <= w_i <= z_i` |
| Meaning of the threshold | Limits conformers and floors each occupancy | Cardinality `K` and occupancy floor `t_min` are independent |

The upstream density model is optional. The downstream optimizer can be tested
against synthetic, experimental, or denoised targets.

## Quick start: renderer tests

The differentiable renderer is the smallest self-contained entry point:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy pytest
python -m pytest -q tests/test_differentiable_renderer.py
```

This runs the Torch NeRF-style kinematics and batched density-renderer tests.

## Full pipeline

The full crystallographic pipeline needs CCTBX/qFit/mmtbx, structure-factor
tools, SciPy, CVXPY, PyTorch, and access to the relevant PDB/MTZ data. Those
dependencies are not currently pinned in a public lockfile, and the full
pipeline is therefore not a one-command installation yet.

The main package entry points are:

```bash
python -m density_denoiser.data_pipeline --help
python -m density_denoiser.train --help
python -m density_denoiser.evaluate --help
python -m density_denoiser.five_site_optimizer --help
```

Once the full dependencies and source PDB/reflection data are available, the
minimal data-to-model workflow is:

```bash
DATA_ROOT=/path/to/qfit_data

python -m density_denoiser.data_pipeline acquire \
  --data-root "$DATA_ROOT" --split both --workers 8
python -m density_denoiser.data_pipeline prepare \
  --data-root "$DATA_ROOT" --split both \
  --map-type omit_mfo_dfc --workers 8
python -m density_denoiser.train \
  --data-root "$DATA_ROOT" --epochs 100 --batch-size 8 --resume
python -m density_denoiser.evaluate \
  --data-root "$DATA_ROOT"
```

Optimizer runs additionally require a trained checkpoint, a selected site set,
and the corresponding prepared cache; inspect the optimizer help before
launching a run. Long runs should write to a new output directory and retain
their configuration, checkpoints, and audit tables together.

The data pipeline acquires structure factors, prepares omit-map patches, and
writes resumable caches. It preserves the protein-level train/test split and
does not modify source PDB files. See
[`density_denoiser/README.md`](density_denoiser/README.md) for the detailed
data and training workflow.

The A′/qFit audit scripts are under [`scripts/`](scripts/). The timing harness
is:

```bash
PYTHONPATH=scripts python scripts/benchmark_step3_timing.py --help
```

It requires a prepared site cache and the combined qFit/CCTBX environment.

## Repository layout

| Path | Contents |
|---|---|
| [`density_denoiser/`](density_denoiser/) | Data preparation, U-Net, renderer, optimizer, and audits |
| [`scripts/`](scripts/) | A′/qFit experiments and diagnostics |
| [`tests/`](tests/) | Focused deterministic tests |
| [`data/`](data/) | Small checked-in 2O1K reference inputs |
| [`results/`](results/) | Curated reports and compact tables |
| [`docs/`](docs/) | Report index and research documentation |
| [`experiments/`](experiments/) | Historical probes and controls |

Large datasets, checkpoints, and run trees are intentionally kept outside the
source repository. `artifacts/` and `external/` are local-only directories and
are ignored by Git.

## Public-release notes

Before treating this repository as a general public package, add a license,
pin the full CCTBX/qFit environment, document how the required reflection data
are obtained, and provide a small end-to-end fixture that does not depend on
private storage or cluster tooling.

For internal research operations and the complete run protocol, see
[`AGENTS.md`](AGENTS.md) and [`CURRENT_HANDOFF.md`](CURRENT_HANDOFF.md).
