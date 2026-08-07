# Unmatched-slot target-density prerequisite

**Frozen metric:** `qfit-synth20-merge050-one-to-one-tmol044-v3`  
**Control rerun:** no  
**Cohort:** historical raw-greedy 142 major-only + 45 minor-only starts

The 259 unmatched active slots occupy substantial target density rather than
vacuum. Target density was evaluated from the exact native Gaussian A/B mixture
used to construct the saved normalized synthetic optimizer target, at every
saved slot atom position and at deposited A/B atom positions. Ratios are
computed within site to remove site-to-site B-factor and atom-type scale.

| Quantity | Result |
|---|---:|
| Unmatched-slot mean target density | median 0.754; mean 1.321 |
| Deposited-A atom-position density | median 1.011; mean 2.097 |
| Deposited-B atom-position density | median 1.430; mean 2.346 |
| Slot / mean(deposited A, B) | median 0.659; mean 0.693 |
| Slot / mean(deposited A, B), IQR | 0.497–0.906 |
| At least 0.25 of deposited reference | 256 / 259 (98.8%) |
| At least 0.50 of deposited reference | 189 / 259 (73.0%) |
| Below 0.10 of deposited reference | 0 / 259 |

Major-only unmatched slots have a median normalized ratio of 0.653; minor-only
slots have a median ratio of 0.894. Thirty-six of the 187 starts have no
unmatched slot above the frozen 0.05 active threshold. Across the other 151
starts, the occupancy-weighted per-start ratio has median 0.694.

The prerequisite therefore rejects the “vacuum slot” interpretation. Respawn
must displace locally density-supported tail configurations, making the merge
threshold scientifically consequential.

Authoritative pod artifacts:

```text
/home/dev/qfit_unet_data/density_denoiser/
heldout_twenty_synthetic_water_minstate_v2_single_rule_v1/
analysis/unmatched_target_density_prerequisite_v1/
```

Files:

```text
unmatched_slot_target_density.csv
single_recovery_start_density.csv
summary.json
```
