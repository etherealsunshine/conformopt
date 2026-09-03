# ConformOpt

User-friendly optimization pipeline for qFit multi-conformer protein models. Supply
one target site or a batch of sites, let the pipeline optimize the qFit
starting model, and collect the A′ and Phenix results in a structured output
directory.

The workflow is designed for site-level optimization. A protein with several
target sites is represented by several rows in the input manifest.

## What the pipeline does

For each manifest row, the pipeline:

1. loads the deposited structure, qFit model, and reflection data;
2. reads the qFit A/B conformers and occupancies;
3. runs the A′ backbone and side-chain optimization;
4. writes a Phenix-ready A′ model;
5. runs Phenix refinement when Phenix is available; and
6. records intermediate coordinates, configuration, provenance, and metrics.

```text
deposited PDB + qFit PDB + MTZ + site manifest
                         │
                         ▼
                    optimization
                         │
                         ▼
        model + Phenix refinement + diagnostics
```

## Installation

Clone the repository into the environment where you want to run the pipeline:

```bash
git clone https://github.com/etherealsunshine/conformopt.git conformopt
cd conformopt
```

Run the pipeline in a Python environment containing the scientific packages
used by the runner (`qfit`/CCTBX, NumPy, SciPy, and PyTorch). Phenix provides
the final refinement step.

For clash-weighted optimization, set `PHENIX_ROOT` to the root of the Phenix
installation so the runner can load its monomer-library connectivity data.

Because qFit/CCTBX and Phenix are external crystallographic software stacks,
their installation is environment-specific. Confirm the environment before
starting a production run:

```bash
python3 scripts/check_runtime.py --strict
python3 scripts/run_conformopt.py --help
```

`check_runtime.py` reports whether qFit, CCTBX, NumPy, SciPy, PyTorch, and
Gemmi import successfully, whether CUDA is visible to PyTorch, and whether
`phenix.refine` is on `PATH`. The runner writes the same report to
`environment.json` in each output directory.

If qFit is provided from a source checkout, expose its `src` directory through
`PYTHONPATH` before running the commands. `Gemmi` is included in the report
because it supplies the monomer-library connectivity used by clash-weighted
optimization. Clash-weighted runs also require `PHENIX_ROOT` to identify the
Phenix monomer library.

## Quick start

Prepare a panel directory containing the input files and manifest described
below. Run one target site with:

```bash
python3 scripts/run_conformopt.py \
  --panel /path/to/panel \
  --output /path/to/results \
  --site 1ABC_A_MET112
```

Run every row in the manifest as one batch:

```bash
python3 scripts/run_conformopt.py \
  --panel /path/to/panel \
  --output /path/to/results
```

Run a selected batch by repeating `--site`:

```bash
python3 scripts/run_conformopt.py \
  --panel /path/to/panel \
  --output /path/to/results \
  --site 1ABC_A_MET112 \
  --site 2XYZ_B_ARG58
```

Use a new output directory for each invocation so every run remains a separate
record.

## Preparing the input panel

Use this directory layout:

```text
panel/
├── selected_sites.csv
└── inputs/
    ├── source/
    │   ├── 1ABC.pdb
    │   └── 2XYZ.pdb
    ├── qfit/
    │   ├── 1ABC_qFit.pdb
    │   └── 2XYZ_qFit.pdb
    └── map_mtz/
        ├── 1abc.mtz
        └── 2xyz.mtz
```

The runner finds files using the PDB identifier:

```text
inputs/source/{pdb_id}.pdb
inputs/qfit/{pdb_id}_qFit.pdb
inputs/map_mtz/{pdb_id}.mtz
```

The PDB identifier matching is case-insensitive. Use the complete deposited
structure to define the local environment. The qFit PDB should contain the A/B
conformers for the target.

### `selected_sites.csv`

The manifest contains one row per target site:

| Column | Description |
| --- | --- |
| `pdb_id` | Identifier used to locate the source PDB, qFit PDB, and MTZ. |
| `chain` | Chain containing the target residue. |
| `resname` | Three-letter residue name, for example `MET` or `ARG`. |
| `residue_number` | Target residue number in the deposited model. |
| `qfit_occupancies` | JSON object containing the qFit `A` and `B` occupancies. |

Example:

```csv
pdb_id,chain,resname,residue_number,qfit_occupancies
1ABC,A,MET,112,"{""A"":0.70,""B"":0.30}"
2XYZ,B,ARG,58,"{""A"":0.55,""B"":0.45}"
```

Use ordinary doubled ASCII quotes in the actual CSV file. The example above
is shown with CSV escaping; the equivalent values are:

```text
{"A": 0.70, "B": 0.30}
{"A": 0.55, "B": 0.45}
```

The site label used by `--site` is:

```text
{pdb_id}_{chain}_{resname}{residue_number}
```

For example, the first row becomes `1ABC_A_MET112`.

## Configuration

The default command is suitable for a smoke test. Production runs can adjust
the optimization and objective settings:

```text
--inner-nfev N             inner optimizer evaluations per update
--outer-updates N          number of outer optimization updates
--chi-nfev N               side-chain chi optimization evaluations
--clash-weight X           weight of the clash objective
--rotamer-weight X         weight of the rotamer objective
--rotamer-calibration PATH provenance for a positive rotamer weight
--density-mode raw|zscore  density residual convention
--fitting-mask-radius X    radius of the observed-map fitting mask in Å
--map-protocol NAME        native_deposited or rebuilt_fmodel
--device auto|cpu|cuda     Torch device for differentiable density calculations
--free-occupancy-ratio     optimize the A/B ratio with total occupancy fixed
--normalize-clash-by-pair-count
                            normalize clash residuals by monitored pair count
--preflight-only           validate inputs and stop before optimization
```

Always inspect the version of the interface in the checkout being used:

```bash
python3 scripts/run_conformopt.py --help
```

## Results

Each site gets an independent result directory:

```text
results/
├── progress.json
├── status.txt
└── 1ABC_A_MET112/
    ├── status.json
    ├── environment.json
    ├── run_config.json
    ├── qfit_input.npz
    ├── qfit_input_objective.json
    ├── conformopt_backbone_1/
    ├── conformopt_backbone_only_2/
    ├── conformopt_sidechain_chi/
    ├── conformopt_backbone_2/
    ├── conformopt_sidechain_chi_2/
    ├── phenix_input.pdb
    ├── phenix/
    │   ├── phenix.log
    │   ├── status.json
    │   └── refined_001.pdb
    └── result.json
```

The main files are:

- `phenix_input.pdb` — the A′ endpoint sent to refinement;
- `phenix/refined_001.pdb` — the Phenix-refined model, when available;
- `result.json` — site status, parameters, provenance, and summary metrics;
- `run_config.json` — input paths and optimization settings;
- `progress.json` — batch-level completion state and per-site summaries.

Intermediate stage directories contain the checkpointed objective values and
coordinates needed to inspect or diagnose a run.

## Batch organization

Add N rows to process N sites and omit `--site`. To process a selected subset,
provide the desired site labels explicitly.

The same source and qFit files can be reused by multiple rows when a protein
has multiple target residues. Each row remains independently checkpointed and
reported.

## Reproducibility

Keep these items together for every run:

1. the exact `selected_sites.csv`;
2. the deposited PDB, qFit PDB, and MTZ files;
3. the command line used;
4. the complete output directory.

Preserve the input files with the output directory. `run_config.json` and
`result.json` record the input provenance used by each site.

## Known limitations

- The current interface optimizes target sites within protein structures.
- qFit inputs must provide the two conformers and occupancy metadata expected
  by the manifest.
- Phenix produces the refined endpoint after A′ optimization.
- A new output directory is required for each invocation.

## Source entry point

The reusable command-line entry point is:

```text
scripts/run_conformopt.py
```
