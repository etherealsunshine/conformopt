# Frozen matched-environment tmol gate diagnostic

**Date:** 2026-07-24

The frozen-tmol analysis in Sections 1-5 did not change optimizer endpoints or
rescore tmol. A later 3GMI section records two new 50-start optimizer
diagnostics under versioned output roots. No tolerance was promoted.

The frozen 20-site composite uses three replacement runs (3A1C ARG447, 6H59
ARG144, and 3NY7 LYS19) in place of their baseline rows.

```texta
geometry rule  2026-07-24-altloc-minstate-rotwidth-matched-tmol-v1
tmol rule      frozen_matched_deposited_minstate_v1
```

The historical `494`, retracted `237`, and frozen zero-tolerance `251` remain
non-comparable scientific headlines. The `237 -> 251` table below is only an
attribution of a known audit artifact.

## 1. Does tmol add information beyond RMSD?

Population: every finite active conformer assigned to deposited A or B under
the conventional `<1.0 A` assignment rule.

```text
finite matched conformers  1,898
Pearson r                  -0.1941
Spearman rho               -0.4138
```

The relationship is not the positive one expected from a redundant RMSD
proxy. More displaced matched conformers often have *lower* tmol energy than
their deposited reference. tmol therefore carries a different signal, but the
negative relationship is also a warning: the present gate readily accepts
geometrically displaced, lower-energy minima.

| Matched RMSD bin (A) | n | Margin mean | Margin median | 5th-95th percentile | Margin > 0 | 0 < margin <= 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| `<=0.1` | 442 | -0.162 | -0.117 | -1.384 to 0.394 | 142 | 142 |
| `0.1-0.3` | 710 | -0.156 | -0.359 | -2.069 to 2.238 | 253 | 70 |
| `0.3-0.6` | 468 | -16.131 | -1.560 | -135.119 to 3.140 | 123 | 28 |
| `0.6-1.0` | 278 | -5.807 | -2.187 | -42.745 to 6.518 | 70 | 6 |

The calibration-population mismatch is real. The proposed `+0.5` was inferred
from the 442 conformers within 0.1 A of deposited, where all 142 positive
margins are within `+0.5`. Applying it to the full matched population also
admits 104 positive-margin conformers outside that reproduction regime.
Therefore:

- `+0.5` is defensible as a deposited-reproduction correction only near the
  deposited geometry.
- Applying `+0.5` to every matched endpoint is gate relaxation, not merely
  numerical-noise correction.
- These results do not support promoting a global tolerance yet.

The scatter and source rows are on the pod:

```text
/home/dev/qfit_unet_data/density_denoiser/frozen_tmol_gate_diagnostic_v1/
  tmol_margin_vs_matched_rmsd.png
  matched_conformers.csv
  margin_by_rmsd_bin.csv
```

### Per-site reproduction-scale percentiles

For each site separately, the table below takes the 99th percentile of only
the positive matched-tmol margins among conformers within 0.1 A of their
matched deposited state. A missing percentile means that the site had no
positive margin in that population.

| Site | Matched within 0.1 A | Positive margins | Positive-margin q99 |
|---|---:|---:|---:|
| 2VFP TYR417 | 20 | 4 | 0.0333 |
| 3GMI GLU5 | 49 | 49 | 0.1054 |
| 4MKM THR77 | 62 | 59 | 0.3132 |
| 5KWB PHE591 | 2 | 2 | 0.1501 |
| 6Y4G CYS260 | 107 | 4 | 0.2471 |
| 7UO8 GLN53 | 21 | 21 | 0.4709 |
| 8FBE ILE92 | 1 | 1 | 0.4237 |
| 8Q6Q ASP81 | 47 | 2 | 0.0610 |

The positive per-site q99 values span `0.0333` to `0.4709`, a spread of
`0.4375`. 3K8W has 84 conformers within 0.1 A but zero positive margins, so it
has no positive-side percentile. The heterogeneity is consistent with the
observed tolerance sensitivity at 4MKM and 5KWB and the complete lack of
movement at 3K8W. No global tolerance is promoted from this diagnostic.

Machine-readable tables:

```text
/home/dev/qfit_unet_data/density_denoiser/site_tmol_and_assigned_pair_diagnostic_v2/
  per_site_positive_margin_q99.csv
  per_site_assigned_pair_strict.csv
  summary.json
```

## 2. Stage-2 symmetry loss at the exact 3GMI failures

The relevant population is the 11 3GMI starts that recovered A and B, passed
occupancy, and then failed the hard symmetry gate. Each has two active
conformers and exactly one hard-failing conformer.

```text
soft Stage-2 cutoff  2.5 A
hard audit gate      2.0 A
lambda_clash         5.0
```

| Start | Hard minimum (A) | Raw Stage-2 symmetry loss | lambda=5 weighted |
|---:|---:|---:|---:|
| 6 | 1.7893 | 0.505514 | 2.527570 |
| 9 | 1.7882 | 0.506910 | 2.534551 |
| 10 | 1.7890 | 0.505763 | 2.528815 |
| 19 | 1.7898 | 0.504826 | 2.524131 |
| 26 | 1.7893 | 0.505382 | 2.526908 |
| 32 | 1.7901 | 0.504031 | 2.520157 |
| 40 | 1.7795 | 0.518809 | 2.594043 |
| 41 | 1.7885 | 0.506511 | 2.532554 |
| 44 | 1.7897 | 0.504857 | 2.524287 |
| 45 | 1.7911 | 0.502694 | 2.513469 |
| 49 | 1.7920 | 0.501472 | 2.507358 |

None is zero. The soft term has a penalty and gradient-supporting overlap at
the coordinates rejected by the hard gate. The earlier capability probe was
insufficient evidence by itself, but the exact-coordinate check rules out the
specific zero-loss cutoff/functional-form mismatch for these 11 endpoints.

## 3. Missing counts and denominator

| Site | Active | Matched | Unmatched | Non-finite matched | Finite matched |
|---|---:|---:|---:|---:|---:|
| 1ZV8 ASN1 | 106 | 34 | 72 | 0 | 34 |
| 2V05 HIS168 | 145 | 30 | 115 | 0 | 30 |
| 2VFP TYR417 | 108 | 71 | 37 | 0 | 71 |
| 3A1C ARG447 replacement | 128 | 99 | 29 | 0 | 99 |
| 3GMI GLU5 | 129 | 66 | 63 | 0 | 66 |
| 3K8W SER337 | 156 | 156 | 0 | 0 | 156 |
| 3NY7 LYS19 replacement | 144 | 103 | 41 | 0 | 103 |
| 4C16 MET258 | 141 | 64 | 77 | 0 | 64 |
| 4MKM THR77 | 137 | 137 | 0 | 0 | 137 |
| 5DBA TRP325 | 116 | 98 | 18 | 0 | 98 |
| 5KWB PHE591 | 104 | 104 | 0 | 0 | 104 |
| 5Z8H MET730 | 96 | 73 | 23 | 0 | 73 |
| 6H59 ARG144 replacement | 127 | 104 | 23 | 0 | 104 |
| 6Y4G CYS260 | 113 | 113 | 0 | 0 | 113 |
| 7F72 MET103 | 110 | 104 | 6 | 0 | 104 |
| 7T7A LEU396 | 125 | 99 | 26 | 0 | 99 |
| 7UO8 GLN53 | 110 | 55 | 55 | 0 | 55 |
| 8DJ2 VAL893 | 152 | 143 | 9 | 0 | 143 |
| 8FBE ILE92 | 114 | 110 | 4 | 0 | 110 |
| 8Q6Q ASP81 | 135 | 135 | 0 | 0 | 135 |
| **Total** | **2,496** | **1,898** | **598** | **0** | **1,898** |

The 588 positive-margin failures are therefore `588 / 1,898 = 31.0%` of
finite matched conformers. Unmatched active conformers are auto-invalid for
tmol, but are not part of that denominator. No active conformer has a
non-finite raw tmol energy, and no matched conformer has a non-finite matched
reference or margin.

“Physically valid independent = 270” counts endpoint ensembles/optimization
starts for which every active conformer passes the full physical audit. It
does **not** count individual conformers.

## 4. Full 20-site cascade

The complete machine-readable table is:

```text
/home/dev/qfit_unet_data/density_denoiser/frozen_tmol_gate_diagnostic_v1/
  per_site_cascade.csv
```

The conformer cascade uses:

```text
active -> finite A/B match -> rotamer -> direct clash -> symmetry clash -> tmol
```

| Site | Active | Finite match | + rotamer | + direct | + symmetry | + tmol, tol 0.0 | + tmol, tol 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1ZV8 ASN1 | 106 | 34 | 34 | 34 | 34 | 24 | 34 |
| 2V05 HIS168 | 145 | 30 | 0 | 0 | 0 | 0 | 0 |
| 2VFP TYR417 | 108 | 71 | 71 | 71 | 71 | 51 | 55 |
| 3A1C ARG447 replacement | 128 | 99 | 99 | 99 | 99 | 99 | 99 |
| 3GMI GLU5 | 129 | 66 | 66 | 66 | 54 | 2 | 52 |
| 3K8W SER337 | 156 | 156 | 156 | 156 | 156 | 86 | 86 |
| 3NY7 LYS19 replacement | 144 | 103 | 98 | 98 | 98 | 88 | 90 |
| 4C16 MET258 | 141 | 64 | 64 | 64 | 64 | 64 | 64 |
| 4MKM THR77 | 137 | 137 | 137 | 137 | 137 | 74 | 133 |
| 5DBA TRP325 | 116 | 98 | 98 | 98 | 98 | 86 | 87 |
| 5KWB PHE591 | 104 | 104 | 104 | 104 | 104 | 50 | 101 |
| 5Z8H MET730 | 96 | 73 | 68 | 68 | 68 | 8 | 9 |
| 6H59 ARG144 replacement | 127 | 104 | 104 | 104 | 104 | 93 | 93 |
| 6Y4G CYS260 | 113 | 113 | 113 | 113 | 113 | 103 | 107 |
| 7F72 MET103 | 110 | 104 | 104 | 104 | 104 | 54 | 57 |
| 7T7A LEU396 | 125 | 99 | 99 | 99 | 99 | 81 | 85 |
| 7UO8 GLN53 | 110 | 55 | 55 | 55 | 55 | 26 | 52 |
| 8DJ2 VAL893 | 152 | 143 | 143 | 143 | 143 | 143 | 143 |
| 8FBE ILE92 | 114 | 110 | 110 | 110 | 110 | 105 | 108 |
| 8Q6Q ASP81 | 135 | 135 | 135 | 135 | 135 | 58 | 61 |
| **Total** | **2,496** | **1,898** | **1,858** | **1,858** | **1,846** | **1,295** | **1,516** |

Endpoint/start outcomes under both rule strings are reported as a monotone
cascade. “All-active” is conditioned on recovery and occupancy; the
unconditioned all-active diagnostic is stated separately below. The assigned
pair chooses the lowest-RMSD active A-assigned and B-assigned conformer above
the found-occupancy threshold and does not charge extra active slots.

| Site | Both | + occupancy | + all-active, tol 0.0 | + assigned pair, tol 0.0 | + all-active, tol 0.5 | + assigned pair, tol 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| 1ZV8 ASN1 | 6 | 6 | 0 | 0 | 5 | 6 |
| 2V05 HIS168 | 2 | 0 | 0 | 0 | 0 | 0 |
| 2VFP TYR417 | 13 | 11 | 6 | 7 | 10 | 11 |
| 3A1C ARG447 replacement | 43 | 41 | 31 | 41 | 31 | 41 |
| 3GMI GLU5 | 11 | 11 | 0 | 0 | 0 | 0 |
| 3K8W SER337 | 50 | 50 | 0 | 0 | 0 | 0 |
| 3NY7 LYS19 replacement | 45 | 45 | 14 | 36 | 14 | 38 |
| 4C16 MET258 | 21 | 21 | 7 | 21 | 7 | 21 |
| 4MKM THR77 | 50 | 50 | 0 | 0 | 46 | 50 |
| 5DBA TRP325 | 30 | 30 | 20 | 29 | 20 | 29 |
| 5KWB PHE591 | 48 | 48 | 0 | 0 | 45 | 48 |
| 5Z8H MET730 | 18 | 18 | 0 | 0 | 0 | 0 |
| 6H59 ARG144 replacement | 45 | 41 | 23 | 39 | 23 | 39 |
| 6Y4G CYS260 | 50 | 50 | 43 | 47 | 44 | 50 |
| 7F72 MET103 | 37 | 33 | 0 | 0 | 1 | 2 |
| 7T7A LEU396 | 41 | 35 | 25 | 31 | 28 | 31 |
| 7UO8 GLN53 | 14 | 14 | 0 | 0 | 13 | 14 |
| 8DJ2 VAL893 | 50 | 50 | 41 | 50 | 41 | 50 |
| 8FBE ILE92 | 49 | 49 | 41 | 49 | 44 | 49 |
| 8Q6Q ASP81 | 50 | 50 | 0 | 0 | 0 | 0 |
| **Total** | **673** | **653** | **251** | **350** | **372** | **479** |

The unconditioned all-active diagnostic is `270/1000` at tolerance `0.0` and
`396/1000` at tolerance `0.5`; those values are not cascade stages. There are
2,496 active conformers across 1,000 starts: a net 496 above two per start,
made up of 537 actual extra slots and 41 missing slots. The assigned-pair rule
raises the zero-tolerance result by 99, but it does not rescue 3K8W or 8Q6Q:
both remain `0/50`. Their zeroes therefore cannot be attributed solely to the
all-active penalty on extra conformers.

The schedule-tuned 3A1C ARG447, 6H59 ARG144, and 3NY7 LYS19 replacements are
inside this panel. They contribute `31 + 23 + 14 = 68` of the 251
zero-tolerance all-active strict successes.

Every strict count in this table uses geometry rule
`2026-07-24-altloc-minstate-rotwidth-matched-tmol-v1`, tmol rule
`frozen_matched_deposited_minstate_v1`, and the tolerance printed in its
column.

### Attribution of the retracted 237 intermediate

At tolerance `0.0`, under the rule strings above for v5, the per-site changes
from the retracted v3 copied-energy intermediate are:

```text
2VFP TYR417  10 -> 6   (-4)
5DBA TRP325   0 -> 20  (+20)
8FBE ILE92   43 -> 41  (-2)
all others              (0)
net                    (+14)
```

Thus the net `+14` masks movement in both directions. Of the two sites with
high sampled environment divergence, 2VFP regressed by four strict starts and
3GMI remained at zero. This is artifact attribution, not model progress.

### Original five wipeout sites

Under both rule strings:

| Site | Strict tol 0.0 | Strict tol 0.5 |
|---|---:|---:|
| 8FBE ILE92 | 41 / 50 | 44 / 50 |
| 5DBA TRP325 | 20 / 50 | 20 / 50 |
| 7UO8 GLN53 | 0 / 50 | 13 / 50 |
| 3GMI GLU5 | 0 / 50 | 0 / 50 |
| 1ZV8 ASN1 | 0 / 50 | 5 / 50 |

## 5. Deposited-control false-rejection floor

A deterministic stratified sample was drawn from supported A/B altloc pairs in
the untouched 99-protein test manifest:

```text
170 deposited A/B sites
340 deposited conformers
63 proteins
10 sites per supported residue type
at most one site per protein within each residue type
minimum A and B occupancy 0.10
```

Selection used only deposited chemistry/completeness/occupancy and a SHA256
site-key priority. It did not use optimizer, denoiser, or audit outcomes. All
170 sites completed with zero processing errors.

| Residue | Pairs | Conformers | Rotamer rejected | Direct rejected | Symmetry rejected | Any rejected | Conformer rejection | Pair rejection |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARG | 10 | 20 | 0 | 0 | 0 | 0 | 0% | 0% |
| ASN | 10 | 20 | 0 | 0 | 0 | 0 | 0% | 0% |
| ASP | 10 | 20 | 1 | 0 | 0 | 1 | 5% | 10% |
| CYS | 10 | 20 | 1 | 0 | 0 | 1 | 5% | 10% |
| GLN | 10 | 20 | 1 | 0 | 0 | 1 | 5% | 10% |
| GLU | 10 | 20 | 0 | 0 | 0 | 0 | 0% | 0% |
| HIS | 10 | 20 | 16 | 0 | 0 | 16 | 80% | 80% |
| ILE | 10 | 20 | 0 | 0 | 0 | 0 | 0% | 0% |
| LEU | 10 | 20 | 0 | 0 | 0 | 0 | 0% | 0% |
| LYS | 10 | 20 | 3 | 1 | 1 | 3 | 15% | 20% |
| MET | 10 | 20 | 0 | 0 | 0 | 0 | 0% | 0% |
| PHE | 10 | 20 | 4 | 0 | 0 | 4 | 20% | 20% |
| SER | 10 | 20 | 0 | 0 | 0 | 0 | 0% | 0% |
| THR | 10 | 20 | 0 | 0 | 0 | 0 | 0% | 0% |
| TRP | 10 | 20 | 0 | 0 | 0 | 0 | 0% | 0% |
| TYR | 10 | 20 | 4 | 0 | 0 | 4 | 20% | 20% |
| VAL | 10 | 20 | 0 | 0 | 0 | 0 | 0% | 0% |
| **Total** | **170** | **340** | **30** | **1** | **1** | **30** | **8.8%** | **10.0%** |

The current geometry rule passes `310 / 340` deposited conformers and both
conformers at `153 / 170` deposited pairs. Every rejected conformer fails the
rotamer gate. The lone direct-clash and lone symmetry-clash rejection are the
two controls of one LYS pair, and both controls already fail rotamer.

HIS is the dominant problem: `16 / 20` deposited HIS conformers are rejected.
This independently confirms that the current HIS chi2 convention/state mapping
is not ready to serve as a hard gate. PHE and TYR also show a material 20%
deposited-control rejection floor under their current chi2 centers and width.
These are audit findings only; the parked HIS/state-union and terminal/width
changes were not implemented in the v1 floor above.

tmol is structurally exempt from this false-rejection test. A deposited
conformer is its own frozen matched reference, so its margin is exactly zero
and it passes by construction.

Authoritative pod root:

```text
/home/dev/qfit_unet_data/density_denoiser/
  deposited_altloc_false_rejection_floor_v2/audit/
    run_config.json
    selected_sites.csv
    deposited_control_geometry_audit.csv
    false_rejection_by_residue.csv
    summary.json
```

### Revised HIS/aromatic rule

Two targeted rotamer changes were then made:

- HIS chi2 is the union of the approximately `+/-80` and `+/-170` states.
- PHE/TYR chi2 retains its existing centers but uses a 45-degree width instead
  of 30 degrees.

Under geometry rule
`2026-07-24-altloc-minstate-rotwidth-hisunion-aromatic45-matched-tmol-v2`,
the identical deterministic 340-conformer floor changed as follows:

| Residue | v1 rejected | v2 rejected | v2 rate |
|---|---:|---:|---:|
| HIS | 16 / 20 | 0 / 20 | 0% |
| PHE | 4 / 20 | 3 / 20 | 15% |
| TYR | 4 / 20 | 3 / 20 | 15% |
| **All 17 types** | **30 / 340** | **12 / 340** | **3.5%** |

The HIS union removes the observed HIS control failure completely. Widening
PHE/TYR improves each by one conformer but leaves a 15% deposited-control
rejection rate, so those hard gates still need calibration. The other v2
rejections are ASP 1/20, CYS 1/20, GLN 1/20, and LYS 3/20. tmol remains
structurally exempt for the same deposited-versus-self reason.

```text
/home/dev/qfit_unet_data/density_denoiser/
  deposited_altloc_false_rejection_floor_v4_his_aromatic_rule_v2/audit/
```

## 6. 3GMI endpoint provenance and current-environment rerun

The original 3GMI starts file stored Stage-2 symmetry loss as zero, while
fresh evaluation of those same coordinates under the current environment gave
raw loss `0.5015-0.5188`. The original endpoints are therefore not valid
evidence about optimization under the current symmetry environment.

A fresh 50-start GLU5 run used the same selection, checkpoint, target, seed,
K=4, 500-step density schedule, 200-step physics schedule, learning rates, and
physics weights. No barrier was enabled for this control. Its current geometry
audit reports:

```text
both found                         41 / 50
recovery + occupancy              41 / 50
all-active geometry physical      50 / 50
sub-2 A symmetry failures          0 / 50
```

The exact 11 start indices that failed historically all moved away from the
old symmetry minimum:

| Start | Historical minimum (A) | Fresh current-environment minimum (A) |
|---:|---:|---:|
| 6 | 1.7893 | 2.6669 |
| 9 | 1.7882 | 2.7607 |
| 10 | 1.7890 | 2.6716 |
| 19 | 1.7898 | 2.6980 |
| 26 | 1.7893 | 2.6980 |
| 32 | 1.7901 | 2.8514 |
| 40 | 1.7795 | 2.8025 |
| 41 | 1.7885 | 2.7584 |
| 44 | 1.7897 | 2.6254 |
| 45 | 1.7911 | 2.6641 |
| 49 | 1.7920 | 2.6874 |

This is coherent movement of the full systematic cluster, not stochastic
movement of one or two starts. The old `3GMI 0/50` is a run-time
symmetry-environment artifact.

Every starts file in the 20-site baseline completed on July 23, and all three
schedule-tuned replacement files completed by `2026-07-24 05:06 UTC`. They
all predate the current symmetry-environment implementation. Therefore all 20
sites have stale-provenance risk with respect to the soft symmetry objective;
the 3GMI rerun proves a numerical effect at 3GMI, not automatically at every
other site.

```text
/home/dev/qfit_unet_data/density_denoiser/
  3gmi_current_symmetry_environment_rerun_v1/
```

### Quartic hard-gate shoulder

The historical squared hinge gives raw loss `(2.5-d)^2`; at the 2.0 A hard
gate that is only `0.25`, or `1.25` after `lambda_clash=5`. The tested
single-factor replacement adds

```text
1.0 * max((2.0 + 0.25 - d) / 0.25, 0)^4
```

to the raw symmetry loss. At 2.0 A, the total raw loss is therefore `1.25`
and the weighted loss is `6.25`. Setting the barrier scale to zero exactly
recovers the historical squared hinge.

The 50-start barrier run passed deposited calibration and produced:

```text
both found                         40 / 50
recovery + occupancy              40 / 50
all-active geometry physical      49 / 50
sub-2 A symmetry failures          0 / 50
```

The one physical failure is a noncanonical low-occupancy unmatched conformer
at start 43; it is not a direct or symmetry clash. The exact historical 11
starts all remain clear of the hard gate at `2.625-2.889 A`. Seven have
identical audited minimum distances to the no-barrier current control, while
the other four differ by only `-0.036` to `+0.037 A`. Thus they move together
relative to the stale run. The observed 41/50 versus 40/50 recovery difference
is about `0.35` standard errors at `n=50`: evidence of neither benefit nor
harm.

Do not promote the barrier globally from this result. The primary fix is the
correct current symmetry environment; the barrier is a guardrail ablation
whose benefit would need a site where the corrected objective still balances
below 2.0 A.

```text
/home/dev/qfit_unet_data/density_denoiser/
  3gmi_symmetry_barrier_quartic_v1/
```

## Bottom line

1. Keep the frozen matched environment; that fix is valid.
2. Do not promote a global `+0.5` tolerance from the near-deposited population.
   It would admit 104 additional positive-margin conformers outside the
   reproduction regime.
3. tmol is not merely a positive RMSD proxy, but its tendency to reward some
   displaced geometries means the gate needs a geometry-stratified definition
   before it can be trusted as a headline scalar.
4. The saved 3GMI endpoints have nonzero loss when evaluated in the current
   environment, and a fresh identical-hyperparameter run clears every symmetry
   clash. The old 0/50 is a stale run-time environment artifact.
5. The two rotamer edits lower the broad deposited-conformer rejection floor
   from 8.8% to 3.5%. HIS is fixed in this sample, but PHE/TYR remain at 15%;
   do not silently promote the v2 rule to an official headline.
6. The quartic symmetry shoulder also clears the old 11, but the corrected
   no-barrier environment already does so. The 41/50 versus 40/50 difference
   is statistically uninformative; keep the barrier available for a corrected
   site that still balances below 2.0 A rather than making it a global default.
