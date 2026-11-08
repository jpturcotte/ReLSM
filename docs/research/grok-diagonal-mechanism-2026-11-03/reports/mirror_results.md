# Mirror test (H5) results

lse margins of train pairs at the final checkpoint. mirror = off-diagonal train pair whose (b, a) is also in train;
lone = off-diagonal train pair whose (b, a) is in test; diag = diagonal train pairs. ± is the standard error.
D_mirror = M_mirror − M_diag, D_lone = M_lone − M_diag, R = D_lone / D_mirror.

| run | diag | mirror | lone | D_mirror | D_lone | R | mirror − lone | acc lone | held-out, mirror in train | held-out, mirror held out |
|---|---|---|---|---|---|---|---|---|---|---|
| control_s0 | 10.65 ± 0.46 (n 44) | 14.79 ± 0.07 (n 2302) | 14.64 ± 0.06 (n 2358) | 4.14 | 3.98 | 0.96 | 0.16 | 1.000 | 14.61 ± 0.06 (n 2358) | 13.99 ± 0.07 (n 2294) |
| control_s1 | 10.27 ± 0.42 (n 44) | 15.20 ± 0.06 (n 2292) | 14.64 ± 0.07 (n 2368) | 4.93 | 4.37 | 0.89 | 0.56 | 1.000 | 14.61 ± 0.07 (n 2368) | 13.89 ± 0.07 (n 2284) |
| control_s2 | 9.78 ± 0.51 (n 49) | 14.73 ± 0.07 (n 2334) | 14.27 ± 0.07 (n 2321) | 4.95 | 4.48 | 0.91 | 0.47 | 1.000 | 14.24 ± 0.07 (n 2321) | 13.56 ± 0.07 (n 2336) |
| sepops_s0 | 16.22 ± 0.43 (n 44) | 15.99 ± 0.06 (n 2302) | 15.94 ± 0.06 (n 2358) | -0.22 | -0.27 | 1.21 | 0.05 | 1.000 | 15.69 ± 0.06 (n 2358) | 15.77 ± 0.06 (n 2294) |
| sepops_s1 | 16.39 ± 0.41 (n 44) | 17.06 ± 0.06 (n 2292) | 16.90 ± 0.06 (n 2368) | 0.67 | 0.51 | 0.76 | 0.16 | 1.000 | 16.75 ± 0.06 (n 2368) | 16.75 ± 0.06 (n 2284) |
| sepops_s2 | 15.99 ± 0.44 (n 49) | 16.43 ± 0.06 (n 2334) | 16.23 ± 0.06 (n 2321) | 0.44 | 0.24 | 0.55 | 0.20 | 1.000 | 16.17 ± 0.06 (n 2321) | 16.11 ± 0.06 (n 2336) |
| add2b_shared_s0 | 15.34 ± 0.44 (n 44) | 15.51 ± 0.06 (n 2302) | 15.48 ± 0.06 (n 2358) | 0.17 | 0.14 | 0.79 | 0.04 | 1.000 | 14.80 ± 0.07 (n 2358) | 15.12 ± 0.07 (n 2294) |
| add2b_shared_s1 | 15.13 ± 0.61 (n 44) | 16.10 ± 0.07 (n 2292) | 16.09 ± 0.06 (n 2368) | 0.97 | 0.95 | 0.99 | 0.01 | 1.000 | 15.30 ± 0.07 (n 2368) | 15.35 ± 0.07 (n 2284) |
| add2b_shared_s2 | 15.50 ± 0.57 (n 49) | 16.52 ± 0.07 (n 2334) | 16.47 ± 0.07 (n 2321) | 1.02 | 0.97 | 0.95 | 0.05 | 1.000 | 15.75 ± 0.07 (n 2321) | 15.72 ± 0.07 (n 2336) |
| add2b_sepops_s0 | 17.01 ± 0.41 (n 44) | 17.18 ± 0.06 (n 2302) | 17.22 ± 0.06 (n 2358) | 0.17 | 0.21 | 1.22 | -0.04 | 1.000 | 16.86 ± 0.06 (n 2358) | 16.98 ± 0.06 (n 2294) |
| add2b_sepops_s1 | 16.97 ± 0.38 (n 44) | 16.66 ± 0.06 (n 2292) | 16.67 ± 0.05 (n 2368) | -0.30 | -0.30 | 0.98 | -0.01 | 1.000 | 16.52 ± 0.06 (n 2368) | 16.48 ± 0.06 (n 2284) |
| add2b_sepops_s2 | 16.32 ± 0.32 (n 49) | 16.23 ± 0.06 (n 2334) | 16.12 ± 0.06 (n 2321) | -0.09 | -0.20 | 2.28 | 0.12 | 1.000 | 16.02 ± 0.06 (n 2321) | 15.88 ± 0.06 (n 2336) |
| e1_add_s0_sanity | 10.95 ± 0.46 (n 54) | 14.65 ± 0.05 (n 4650) | — ± — (n 0) | 3.70 | — | — | — | — | — ± — (n 0) | 13.44 ± 0.05 (n 4662) |

## Verdict against the registration

diag_control R per seed: 0.96, 0.89, 0.91 → **REPRESENTATION (R ≥ 0.85 in all three seeds)**

Controls with |mirror − lone| ≥ 0.5 (would flag a confound; registered controls are add2b shared and diag_sepops): none

Sanity (E1 add s0, paired split): lone pairs n = 0 (expected 0)
