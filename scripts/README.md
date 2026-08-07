# Script map

Scripts remain in a stable flat namespace because many analyses import them as
`scripts.<module>` and historical reports record their exact paths. This index
provides navigation without breaking those reproducibility links.

## qFit audit

- `build_backbone_altloc_site_list.py` — deterministic deposited altloc panel.
- `run_d3_anisou_audit.py` — ADP/ANISOU availability inventory.
- `run_d6_tier1_synthetic.py`, `run_d6_tier2_realmap.py`,
  `run_d6_realspace_cc.py` — D6 recovery and real-map controls.
- `audit_d6_followups.py`, `forensic_d6_bic_check.py` — D6 forensic checks.
- `run_d1_reachability.py`, `trace_d1_sampler_discard.py` — D1 reachability
  decomposition and exact sampler-discarding measurement.

## Analysis and diagnostics

Files beginning with `analyze_`, `diagnose_`, `summarize_`, `score_`, or
`interpret_` consume frozen run trees and produce compact tables or reports.
Read each module docstring and its paired test before reusing it for a new
scientific claim.

## Figures and presentation

- `plot_*.py`, `render_*.py`, and `compose_*.py` build publication or
  inspection figures.
- `inject_interactive_landscape_data.py` updates one explicitly configured
  local visualization; do not run it against an unspecified output file.

## Operational utilities

- `five_site_tmol_audit.py` — pod-side TMol endpoint scoring.
- `run_minstate_*.py` / `.sh` — restartable minor-state controls.

All scripts that consume the dataset, qFit, CCTBX, CUDA, or TMol run on the
ACTL pod. See [`../AGENTS.md`](../AGENTS.md) for the source-local/runtime-pod
contract and [`../CURRENT_HANDOFF.md`](../CURRENT_HANDOFF.md) for live roots.
