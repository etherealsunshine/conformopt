# Probe 4: learned energy for 2O1K

`probe4_modal.py` trains and evaluates a torsion-space learned energy on the five
A/B altloc residues in 2O1K. `B_ASP114` and `B_ARG129` are excluded from
training for the nearby-residue generalization test. Five percent of valid
reflections are deterministically held out from the reciprocal-space loss.

## Detached run

```bash
UV_CACHE_DIR=/private/tmp/uv-modal uvx modal run --detach probe4_modal.py
```

The default is 10,000 training steps, three first-order inner steps, 50 starts
per altloc evaluation, and 20 inference steps. A short isolated smoke run is:

```bash
UV_CACHE_DIR=/private/tmp/uv-modal uvx modal run --detach probe4_modal.py \
  --run-name probe4_smoke --steps 2 --checkpoint-every 1 \
  --eval-starts 2 --eval-steps 2 --hidden 32 --layers 2
```

## Persistence and resume

Outputs live in the named Modal Volume `qfit-probe4-results`, under
`probe4_2O1K/` (or `--run-name`). Training checkpoints and CSV rows are
committed every `--checkpoint-every` steps. Each evaluation stage commits its
own files before the next stage starts and records status in
`stage_manifest.json`.

The local entrypoint submits the pipeline with Modal's fire-and-forget call, so
it does not retain a blocking RPC that can be cancelled when the client exits.
Re-running the default command resumes `model_checkpoint.pt` and skips stages
already marked complete. `--force` starts the selected run name from scratch.
Use a new `--run-name` when changing model dimensions or hyperparameters.

Download results after or during a run with:

```bash
UV_CACHE_DIR=/private/tmp/uv-modal uvx modal volume get \
  qfit-probe4-results probe4_2O1K ./probe4_results
```

Probe 2's density-only A:ARG129 baseline is 3/50 B-like endpoints. Probe 4's
direct comparison is written to `altloc_test/summary.json`.
