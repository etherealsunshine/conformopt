# Documentation map

This directory is the human navigation layer for the research repository.
It complements—not replaces—the operational instructions in
[`../AGENTS.md`](../AGENTS.md) and the live status in
[`../CURRENT_HANDOFF.md`](../CURRENT_HANDOFF.md).

## Start here

1. [`../README.md`](../README.md) — project purpose and production pipeline.
2. [`../AGENTS.md`](../AGENTS.md) — non-negotiable local-source/pod-runtime
   rules and complete operational catalog.
3. [`../CURRENT_HANDOFF.md`](../CURRENT_HANDOFF.md) — active work, pod roots,
   current interpretation, and monitoring commands.

## Reading routes

| If you need… | Read… |
|---|---|
| The current qFit audit evidence | [`reports/README.md`](reports/README.md) |
| The D1 reachability control result | [`reports/d1_reachability_controls10_v2.md`](reports/d1_reachability_controls10_v2.md) |
| The original research framing | [`research/EBT_RESEARCH_CONTEXT.md`](research/EBT_RESEARCH_CONTEXT.md) |
| Historical Probe conclusions | [`reports/PROBE4_FULL_EXPERIMENT_REPORT.md`](reports/PROBE4_FULL_EXPERIMENT_REPORT.md) and [`reports/PROBE4C1_4C2_REPORT.md`](reports/PROBE4C1_4C2_REPORT.md) |
| A reproducible script | [`../scripts/README.md`](../scripts/README.md) |
| Curated local outputs versus full pod artifacts | [`../results/README.md`](../results/README.md) |

## Conventions

- Scientific reports are concise, versioned interpretations.
- Raw run trees, model checkpoints, caches, and live logs remain on the pod
  PVC under `/home/dev/qfit_unet_data`.
- A report must name its authoritative pod result root and the scripts that
  generated it. It must not silently replace an earlier scientific record.
