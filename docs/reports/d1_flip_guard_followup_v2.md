# D1 flip sampler: guard breakdown and deposited-A-only comparison

**Status:** complete; analysis-only follow-up to the frozen 33-site tier-(a)
flip run.  No sampler candidates were regenerated.

**Authoritative pod analysis root:**
`/home/dev/qfit_unet_data/qfit_audit/d1_tier_a_flips_v1/guard_followup_v2`

## 1. What blocks qFit's three-neighbour guard?

Ten of 33 flip-filter sites returned the deposited input only.  None is an
ordinary chain terminus and none is blocked solely because a neighbour lacks a
backbone atom.  The breakdown is **7 residue/numbering gaps, 2 peptide-chain
breaks, and 1 insertion-code discontinuity**.  Thus the guard chiefly excludes
gap/disorder-adjacent contexts, rather than unavoidable termini.

| Site | Available lower / upper neighbours | Boundary cause | Boundary context |
|---|---:|---|---|
| 4LZK C:THR133 | 15 / 0 | residue/numbering gap | upper next residue 137 |
| 3P09 A:THR43 | 23 / 0 | residue/numbering gap | upper next residue 48 |
| 4B1T A:GLY184 | 163 / 0 | peptide-chain break | upper next residue 185 |
| 6ANA L:GLN27 | 25 / 0 | peptide-chain break | upper next residue 28 |
| 2IBN A:TRP285 | 4 / 0 | residue/numbering gap | upper next residue 708 |
| 7SC4 B:GLY991 | 8 / 0 | residue/numbering gap | upper next residue 993 |
| 1O6Z D:GLY30 | 0 / 24 | insertion-code discontinuity | lower preceding residue 29A |
| 7Y9N A:LEU966 | 48 / 0 | residue/numbering gap | upper next residue 1167 |
| 7R58 A:GLY89 | 86 / 0 | residue/numbering gap | upper next residue 91 |
| 4BTB A:GLU232 | 2 / 4 | residue/numbering gap | lower segment boundary following residue 223 |

qFit builds segments by retaining consecutive residues only while
`is_next_residue` holds ([`structure.py:667`](../../external/qfit-3.0/src/qfit/structure/structure.py#L667)); for amino acids that test is the C--N
distance being below 1.5 Å ([`residue.py:107`](../../external/qfit-3.0/src/qfit/structure/residue.py#L107)).
The labels above classify the boundary that actually truncates qFit's segment,
not merely the target's immediate neighbour.

## 2. Finding #8: qFit accepts a truncated window

**2/33 flip sites** pass qFit's own guard despite having only six complete
sequential residues, hence receive 19 candidates from a truncated window:

| Site | qFit segment length / central index | Missing side |
|---|---:|---|
| 6HKG D:PRO142 | 14 / 11 | upper third neighbour |
| 6C0J B:VAL189 | 188 / 185 | upper third neighbour |

This is the previously logged `>` versus `>=` defect: qFit tests
`index + nn > len(self.segment)` and then slices through `index + nn + 1`
([`qfit.py:927`](../../external/qfit-3.0/src/qfit/qfit.py#L927)).  Equality
therefore admits a six-residue, rather than seven-residue, sampling window.

## 3. Defensible deposited-A-only comparison

The raw medians are **0.551** residual/deviation for the 10 guard-blocked
(deposited-A-only) sites and **0.540** for the 23 19-candidate sites.  That
comparison is not deviation matched.  In the final CSV the corresponding
median deposited deviations are **2.297 Å** and **2.910 Å**, respectively
(not 2.44 Å and 2.89 Å).

Nearest-neighbour matching is too loose to be the headline: all ten blocked
sites can be paired, but the median absolute deviation mismatch is **0.255 Å**
(maximum 0.349 Å), and only **3** pairs are within 0.10 Å.  I therefore use a
single descriptive OLS adjustment across all 33 sites:

```text
residual / deposited deviation ~ 1 + deposited deviation + guard-blocked
```

At the same deposited deviation, the guard-blocked indicator is **+0.070 ±
0.024** residual/deviation (two-sided p = **0.0059**, 30 residual degrees of
freedom).  In plain terms, the deposited-A-only arm retains about **7.0
percentage points more** of the deposited displacement than the sampled arm
after this linear adjustment.

The group trend lines, useful for a slide, are:

```text
guard-blocked: residual/deviation = 0.393 + 0.0736 × deviation (Å)
19 candidates: residual/deviation = 0.431 + 0.0337 × deviation (Å)
```

The figure is `ratio_vs_deviation.png` in the authoritative pod analysis root.
This is an observational comparison: segment disruption determines guard
membership and could itself affect local heterogeneity.  It supports a
conditional association with sampling, not a randomized causal estimate.

## Reproduction

- `scripts/analyze_d1_flip_guard_followup.py`
- `scripts/run_d1_tier_a_flips.py`
- `d1_tier_a_flips_v1/per_site.csv` under the pod root above
