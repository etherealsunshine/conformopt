# D1 flip panel: carbonyl-O A→B coverage

Tier-(a) was **not rerun** for this report.  It re-expresses the completed
33-site qFit `_sample_backbone` panel using carbonyl O only:

\[
\text{coverage}=1-\frac{\min_{\rm candidates}\|O_{\rm candidate}-O_B\|}
{\|O_A-O_B\|}.
\]

This avoids applying the side-chain-calibrated 1 Å criterion to a peptide
flip.  `guard` rows had only the deposited-A input, hence 0% coverage.

- All 33 sites: median **5.2%** O displacement covered (range 0–24.5%).
- Full 19-candidate sampler subset (23 sites): median **7.8%** covered.

| Site | qFit | O A→B (Å) | Best O residual (Å) | Coverage |
|---|---|---:|---:|---:|
| 4LZK C:THR133 | guard | 1.823 | 1.823 | 0.0% |
| 3P09 A:THR43 | guard | 1.968 | 1.968 | 0.0% |
| 4B1T A:GLY184 | guard | 2.123 | 2.123 | 0.0% |
| 6ANA L:GLN27 | guard | 2.169 | 2.169 | 0.0% |
| 5IKU A:TYR970 | 19 | 2.172 | 1.937 | 10.8% |
| 5FG8 A:LYS268 | 19 | 2.189 | 1.935 | 11.6% |
| 2IBN A:TRP285 | guard | 2.274 | 2.274 | 0.0% |
| 7SC4 B:GLY991 | guard | 2.321 | 2.321 | 0.0% |
| 7UTC A:GLY190 | 19 | 2.412 | 2.151 | 10.8% |
| 7UTC A:ARG52 | 19 | 2.480 | 1.981 | 20.1% |
| 1O6Z D:GLY30 | guard | 2.099 | 2.099 | 0.0% |
| 6P2N A:GLY161 | 19 | 2.615 | 2.325 | 11.1% |
| 1ZXT C:TYR14 | 19 | 2.640 | 2.307 | 12.6% |
| 7Y9N A:LEU966 | guard | 2.719 | 2.719 | 0.0% |
| 5YNF A:GLN49 | 19 | 2.757 | 2.687 | 2.5% |
| 7SC4 B:PRO2317 | 19 | 2.774 | 2.389 | 13.9% |
| 4E4Y A:ALA75 | 19 | 2.791 | 2.107 | 24.5% |
| 7ZTL A:ILE257 | 19 | 2.880 | 2.552 | 11.4% |
| 3NUG C:ALA46 | 19 | 2.895 | 2.775 | 4.2% |
| 8R7O C:THR1681 | 19 | 2.910 | 2.742 | 5.8% |
| 5YNF A:ASP114 | 19 | 2.921 | 2.692 | 7.8% |
| 5J1A A:PRO195 | 19 | 2.946 | 2.649 | 10.1% |
| 1RWR A:ASN294 | 19 | 2.989 | 2.753 | 7.9% |
| 4YZG A:SER181 | 19 | 3.138 | 3.034 | 3.3% |
| 5OHJ A:SER540 | 19 | 3.382 | 3.261 | 3.6% |
| 7R58 A:GLY89 | guard | 3.402 | 3.402 | 0.0% |
| 4HFS A:TYR200 | 19 | 3.431 | 3.212 | 6.4% |
| 3M71 A:GLY161 | 19 | 3.579 | 3.340 | 6.7% |
| 8AJK B:ASN231 | 19 | 3.601 | 3.322 | 7.8% |
| 6HKG D:PRO142 | 19 | 3.679 | 3.527 | 4.1% |
| 6C0J B:VAL189 | 19 | 3.790 | 3.657 | 3.5% |
| 8AJK A:VAL240 | 19 | 3.800 | 3.604 | 5.2% |
| 4BTB A:GLU232 | guard | 3.848 | 3.848 | 0.0% |

Source result root: `/home/dev/qfit_unet_data/qfit_audit/d1_tier_a_flips_o_coverage_v1`.
