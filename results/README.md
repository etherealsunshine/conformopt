# Curated local results

This directory holds compact, reviewable evidence: Markdown reports, selected
CSV summaries, configurations, and figures needed to interpret a completed
experiment.

It is **not** the home for raw datasets, model checkpoints, controller logs,
or complete optimization shards. Those remain on the pod PVC under
`/home/dev/qfit_unet_data`, where each report should name its authoritative
run root.

Before adding an artifact here:

1. Keep the corresponding raw run tree intact on the pod.
2. Include a run/config identifier and frozen metric where applicable.
3. Prefer a small summary table and figure over copied intermediate files.
4. Do not overwrite a prior scientific record; add a new versioned report.

For a route through reports, see [`../docs/reports/README.md`](../docs/reports/README.md).
