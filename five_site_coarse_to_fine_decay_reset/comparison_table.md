# Direct Optimization Control

LR sweep selected `1.0` by lowest mean final complex-SF loss.

| Variant | Site | <0.50 A | <0.75 A | Mean RMSD-B | Mean final loss |
|---|---|---:|---:|---:|---:|
| coarse_to_fine_4A_2A_full_decay | A_MET112 | 14/50 | 32/50 | 0.5292 | 0.411888 |
| coarse_to_fine_4A_2A_full_decay | A_ARG129 | 46/50 | 46/50 | 0.0620 | 0.0919849 |
| coarse_to_fine_4A_2A_full_decay | B_MET112 | 50/50 | 50/50 | 0.0001 | 2.76044e-08 |
| coarse_to_fine_4A_2A_full_decay | B_ASP114 | 23/50 | 23/50 | 0.4804 | 0.000204198 |
| coarse_to_fine_4A_2A_full_decay | B_ARG129 | 25/50 | 25/50 | 0.5026 | 0.99484 |

## MLP comparison

| Site | Direct vanilla | Direct thorough | MLP 20 inference steps | MLP training inner steps |
|---|---:|---:|---:|---:|
