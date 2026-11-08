# Restore tests (H6) results

lse margins, final weights. Base and each test show gap_train; brackets give the share of the gap closed.

| run | base gap | L1 mass up on diag | norm up on diag | both up on diag | L1 mass down on off | norm down on off | Δ diag_train (mass up / norm up) | Δ off_train (mass down / norm down) | inv RMS diag / off |
|---|---|---|---|---|---|---|---|---|---|
| e1_add_s0 | 3.70 | 4.37 (-18%) | 2.94 (21%) | 2.84 (23%) | 3.53 (5%) | 2.82 (24%) | -0.67 / 0.76 | -0.17 / -0.88 | 4.9947 / 5.3096 |
| e1_add_s1 | 4.98 | 4.85 (3%) | 3.47 (30%) | 3.11 (38%) | 4.83 (3%) | 3.14 (37%) | 0.13 / 1.51 | -0.15 / -1.84 | 4.8851 / 5.5479 |
| e1_add_s2 | 6.17 | 6.25 (-1%) | 4.93 (20%) | 4.59 (26%) | 6.06 (2%) | 4.45 (28%) | -0.07 / 1.24 | -0.12 / -1.72 | 4.8212 / 5.3902 |
| e1_mul_s0 | 4.56 | 5.15 (-13%) | 3.63 (20%) | 3.63 (20%) | 4.38 (4%) | 3.45 (24%) | -0.59 / 0.93 | -0.18 / -1.11 | 4.5648 / 4.9083 |
| e1_mul_s1 | 6.09 | 6.28 (-3%) | 5.07 (17%) | 4.85 (20%) | 5.98 (2%) | 4.67 (23%) | -0.19 / 1.03 | -0.11 / -1.42 | 4.8074 / 5.2851 |
| e1_mul_s2 | 4.64 | 5.09 (-10%) | 3.25 (30%) | 3.17 (32%) | 4.47 (4%) | 2.97 (36%) | -0.45 / 1.39 | -0.17 / -1.67 | 4.9113 / 5.5117 |
| diag_control_s0 | 4.06 | 5.37 (-32%) | 3.39 (16%) | 3.98 (2%) | 4.13 (-2%) | 3.31 (19%) | -1.31 / 0.67 | 0.07 / -0.75 | 4.9734 / 5.2359 |
| diag_control_s1 | 4.64 | 5.11 (-10%) | 3.93 (15%) | 3.82 (18%) | 4.37 (6%) | 3.74 (19%) | -0.47 / 0.72 | -0.28 / -0.90 | 4.9361 / 5.2449 |
| diag_control_s2 | 4.72 | 4.67 (1%) | 3.56 (24%) | 3.25 (31%) | 4.59 (3%) | 3.29 (30%) | 0.05 / 1.15 | -0.12 / -1.42 | 4.5787 / 5.0575 |
| diag_sepops_s0 | -0.25 | -0.37 | 0.84 | -0.02 | -0.31 | 1.47 | 0.12 / -1.09 | -0.06 / 1.72 | 3.4060 / 3.1329 |
| diag_sepops_s1 | 0.59 | 0.61 (-4%) | 0.27 (53%) | 0.79 (-35%) | 0.64 (-8%) | 0.51 (13%) | -0.02 / 0.31 | 0.05 / -0.07 | 3.0296 / 3.0666 |
| diag_sepops_s2 | 0.34 | 0.32 | 0.43 | 0.34 | 0.35 | 0.76 | 0.03 / -0.09 | 0.01 / 0.42 | 3.4467 / 3.4028 |

## Verdicts against the registration

- H6a (layer-1 operand mass): closes -18%, 3%, -1%, -13%, -3%, -10%, -32%, -10%, 1% → **NOT CAUSAL**
- H6b (final-norm scale): closes 21%, 30%, 20%, 20%, 17%, 30%, 16%, 15%, 24% → **MIXED**
- both together: closes 23%, 38%, 26%, 20%, 20%, 32%, 2%, 18%, 31%

Sanity: separate-token runs, largest |Δ gap| over all tests = 1.72 (registered bound 0.3)
Sanity: base gaps vs the v2 probe are checked by hand in the summary.

Memory for the whole job (12 checkpoints): peak RAM 1.17 GB, peak commit 2.67 GB, GPU in use 1.23 GB
