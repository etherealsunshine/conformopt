# Synthetic 20-site audit-rule revision

**Date:** 2026-07-24  
**Rule version:** `2026-07-24-altloc-minstate-rotwidth-matched-tmol-v1`

## Changes

### Shared rotamer guardrail

The optimizer and audit now share residue/chi-specific marginal centers and
allowed widths:

- ARG chi4 adds the trans 180-degree state.
- TRP chi2 uses 0 and +/-105 degrees.
- HIS chi2 uses +/-170 degrees, retained as an explicit convention-calibration
  warning because the current stored HIS controls evaluate near +/-86 degrees.
- PHE/TYR chi2 retain the twofold-symmetry +/-90-degree convention.
- ASN/ASP chi2 and GLN/GLU chi3 use 90-degree widths.
- chi1 uses 45 degrees, internal chis 45 degrees, non-PHE/TYR terminal chis
  60 degrees, and PHE/TYR chi2 30 degrees.
- The optimizer's independent-cos penalty is width-scaled so a broad audit
  torsion is not simultaneously pulled strongly toward a narrow center.

This remains a broad marginal guardrail, not a backbone-conditioned joint
rotamer probability model.

### Altloc-aware direct and symmetry clashes

- Protein neighbors are partitioned per residue.
- Both optimizer soft physics and audit hard geometry select the least-clashing
  state independently for each neighboring alternate residue.
- This avoids assuming that A/B letters are globally coupled between residues.
- The identical partitioning applies to direct and crystallographic-symmetry
  environments.
- Labeled waters are compatible only with the candidate's assigned A/B state
  during audit.
- Unlabeled partial-occupancy waters remain present with an occupancy-scaled
  hard cutoff; they are not silently deleted.
- The assignment-neutral optimizer retains occupancy-weighted water penalties.

### Matched-control tmol

An A-assigned endpoint is compared with deposited A and a B-assigned endpoint
with deposited B. The default gate is:

```text
candidate tmol energy - matched deposited energy <= 0
```

Unmatched active conformers are tmol-invalid. The previous comparison with the
better of deposited A/B plus 10 units is no longer used.

## Deposited-control acceptance test

Under the revised clash rule, deposited A and B are direct- and
symmetry-clash-free at all five originally identified wipeout sites.

- The 7UO8 GLN53 0.61 A direct contact to altloc-B HOH186 disappears.
- The 7UO8 GLN53 1.75 A symmetry contact to altloc-A HOH228 disappears.
- The 8FBE ILE92 1.53 A contact to a mutually exclusive LYS95 state
  disappears; the revised deposited-A minimum is 2.42 A.

The five sites also pass the revised marginal rotamer gate. Across all 15
expanded-panel sites, remaining deposited rotamer exceptions are HIS168 A/B
and LYS19 A. They are retained as calibration findings rather than erased by
further indiscriminate widening.

## Provisional synthetic composite — retracted pending tmol regeneration

The optimizer endpoints are unchanged. Only their geometry/tmol audit changed.
The three 200+200+200 replacement sites remain ARG447, ARG144, and LYS19.

```text
Both conformers found:        673 / 1000
+ occupancy +/-0.20:          653 / 1000
physically valid independent: 253 / 1000
strict joint success:         237 / 1000
```

**This 237/1000 value is retracted and must not be cited.** It reused endpoint
tmol energies generated with the old shared, maximum-occupancy neighboring
altloc environment. The revised geometry audit uses candidate-compatible
min-over-altloc environments, so all deposited controls and endpoints require
tmol regeneration under the same rule before a strict composite exists.

The provisional calculation remains recorded only to make the invalidated
intermediate state reproducible.

## 8FBE tmol environment diagnostic

The old shared-base controls were:

```text
deposited A  577.6722
deposited B  358.6232
A - B gap    219.0490
```

With independently selected min-over-altloc environments:

```text
deposited A in A-specific environment  372.8804
deposited B in B-specific environment  152.3279
A - B matched-environment gap          220.5526
```

The 219-unit gap does not disappear; it increases by about 1.50 units. But both
absolute control energies shift downward by about 205 units. That proves the
old endpoint energies are not reusable for the revised audit even though this
particular A/B difference is stable.

Diagnostic roots:

```text
/home/dev/qfit_unet_data/density_denoiser/tmol_environment_diagnostic_8fbe_ile92_a_v1
/home/dev/qfit_unet_data/density_denoiser/tmol_environment_diagnostic_8fbe_ile92_b_v1
```

## Authoritative pod roots

```text
/home/dev/qfit_unet_data/density_denoiser/heldout_twenty_synthetic_audit_rules_v3
/home/dev/qfit_unet_data/density_denoiser/heldout_three_four_chi_200step_synthetic_audit_rules_v3
```

The geometry audits are valid. Their copied legacy tmol tables are retained
only as invalidated intermediate artifacts and must not be used for final
strict counts.

## Verification

Focused and complete active-package tests passed on the qfit-unet pod:

```text
28 passed
```

An unrestricted repository-wide pytest collection remains blocked by the
vendored `external/sampleworks` suite because its `atomworks` dependency is not
installed in the primary qFit environment.
