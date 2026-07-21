# Direct Optimization Control

LR sweep selected `1.0` by lowest mean final complex-SF loss.

| Variant | Site | <0.50 A | <0.75 A | Mean RMSD-B | Mean final loss |
|---|---|---:|---:|---:|---:|
| coarse_to_fine_4A_2A_full_decay | B_MET112 | 4/50 | 4/50 | 2.2481 | -2.16138 |

## MLP comparison

| Site | Direct vanilla | Direct thorough | MLP 20 inference steps | MLP training inner steps |
|---|---:|---:|---:|---:|
