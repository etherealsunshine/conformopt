# Synthetic strict-gate and objective diagnostic

**Date:** 2026-07-24

## Scope

This diagnostic addresses two questions using the authoritative synthetic
20-site run:

1. Do the deposited A/B controls themselves satisfy the physical gates that
   eliminate all occupancy-qualified starts at five sites?
2. For four low-recovery sites, does the recorded optimizer objective prefer
   failed starts over starts that recover both deposited conformers?

Source root:

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_per_residue_schedule_v1
```

The five physical-gate sites and four low-recovery sites all belong to the
expanded-15 portion of this root, so no 200+200+200 composite substitution is
needed for this analysis.

## Result 1: the deposited ensemble fails strict physical criteria at all five sites

The 110 starts that recover both conformers with acceptable occupancy but then
fall to zero strict success do not share one universal symmetry failure. They
are eliminated by deterministic, site-specific gates.

Failure categories below overlap: one start can fail more than one gate.

| Site | Occupancy-qualified | Strict | Noncanonical | Direct clash | Symmetry clash | tmol failure |
|---|---:|---:|---:|---:|---:|---:|
| 8FBE ILE92 | 49 | 0 | 1 | 49 | 0 | 49 |
| 5DBA TRP325 | 30 | 0 | 30 | 30 | 0 | 9 |
| 7UO8 GLN53 | 14 | 0 | 0 | 14 | 14 | 14 |
| 3GMI GLU5 | 11 | 0 | 0 | 0 | 11 | 0 |
| 1ZV8 ASN1 | 6 | 0 | 6 | 0 | 0 | 0 |
| **Total** | **110** | **0** | — | — | — | — |

The existing deposited-control geometry and tmol calibration tables show that
an exact deposited A/B ensemble cannot pass the current strict definition at
any of these five sites:

| Site | Deposited-control incompatibility |
|---|---|
| 8FBE ILE92 | Deposited A has a 1.53 Å direct contact to LYS95 CE. Its tmol energy is 219.05 above deposited B, far beyond the +10 gate. |
| 5DBA TRP325 | Deposited A and B are both classified noncanonical by the current generic rotamer centers. |
| 7UO8 GLN53 | Deposited A is noncanonical, has a 0.61 Å contact to HOH186, and is 61.86 tmol units above B. Deposited B has a 1.75 Å symmetry contact to HOH228. |
| 3GMI GLU5 | Deposited B is classified noncanonical. |
| 1ZV8 ASN1 | Deposited A and B are both classified noncanonical. |

Thus the zero-strict pattern is real but is not evidence of one shared
space-group bug. It exposes several control incompatibilities:

- generic rotamer centers reject deposited conformers;
- the direct-clash audit can count deposited water contacts;
- the symmetry audit can reject deposited water contacts;
- the tmol gate is defined relative to the better deposited A/B control, so
  the worse deposited conformer can fail the gate by construction.

This does not automatically mean every physical gate should be removed. The
strict metric was designed to demand physically improved endpoints, not merely
copy the deposited model. It does mean that `strict_joint_success` cannot be
interpreted as pure synthetic recovery when the exact target ensemble is
ineligible for success.

## Result 2: failed hard-tail starts do not generally have lower objective values

Starts were divided by whether both deposited conformers were recovered below
the conventional 1.0 Å RMSD threshold. `final_loss` is the recorded Stage-2
joint objective:

```text
final density + 1.0 * VDW + 0.5 * rotamer + 5.0 * symmetry
```

| Site | Recovered / failed | Median Stage-1 density, recovered | Median Stage-1 density, failed | Median final joint, recovered | Median final joint, failed | P(failed joint < recovered joint) |
|---|---:|---:|---:|---:|---:|---:|
| 2V05 HIS168 | 2 / 48 | 0.000094 | 0.001525 | 2.215 | 2.584 | 0.385 |
| 1ZV8 ASN1 | 6 / 44 | 0.000092 | 0.001095 | 2.220 | 2.313 | 0.155 |
| 3GMI GLU5 | 11 / 39 | 0.028037 | 0.081483 | 1.698 | 2.606 | 0.086 |
| 2VFP TYR417 | 13 / 37 | 0.109596 | 0.206106 | 1.506 | 13.154 | 0.160 |

Here `P(failed joint < recovered joint)` is the fraction of all failed/recovered
start pairs in which the failed start has the lower recorded final objective.
A value below 0.5 means recovered starts are generally preferred.

At all four sites, recovered starts have lower median Stage-1 density loss and
lower median final joint loss. Therefore the data do not support a blanket
conclusion that the optimizer is reliably converging to a lower-energy wrong
answer. Navigation and convergence remain plausible limitations, so longer or
coarse-to-fine density optimization is still applicable.

There are important exceptions:

- At HIS168, the recovered sample contains only two starts, so estimates are
  noisy and objective overlap is substantial.
- At ASN1, failed starts have lower final density loss in 65.2% of
  failed/recovered pairs, although their full joint objective is generally
  higher. This is a density-versus-physics tradeoff.
- At HIS168, GLU5, and TYR417, at least one failed start has a lower final joint
  objective than the best recovered start. The objective is therefore not a
  perfect ranker even though its median behavior favors recovery.
- The recorded Stage-2 symmetry loss is zero throughout these four groups.
  Hard symmetry-clash audit failures can therefore be invisible to the soft
  optimization term.

## Consequences

1. Report synthetic structural recovery and occupancy separately from strict
   physical success.
2. Add a control-relative audit showing whether an endpoint is at least as
   physical as its matched deposited conformer.
3. Revisit water handling, generic rotamer classification, and the
   better-of-A/B tmol threshold before treating the 159 physical-stage losses
   as optimizer failures.
4. A coarse-to-fine or longer-step ablation remains justified for the
   low-recovery tail because recovered starts have better median Stage-1
   density objectives at every tested site.
5. Preserve the original strict metric as a demanding physical-quality metric
   if desired, but label it clearly as stricter than reproduction of the
   deposited synthetic target.

## Source tables

```text
audit/expanded15/deposited_control_geometry_audit.csv
audit/expanded15/tmol_calibration.json
audit/expanded15/active_conformer_strict_audit.csv
audit/expanded15/ensemble_strict_audit.csv
shards/expanded15/SITE/synthetic/SITE_starts.csv
```
