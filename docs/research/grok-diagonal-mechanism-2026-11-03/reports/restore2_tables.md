# Restore tests v2: tables (numbers only)

Final weights. Margins are lse, train groups. `Δ` = change in the mean margin of the group the change was applied to.
Runs: e1_add/e1_mul (paired split, shared tokens), diag_control (add, ordered split, shared tokens),
diag_sepops (add, ordered split, separate operand tokens; its base gap is about 0).

## T1. Checks

| run | max \|logit − model(x)\| | no-op mass | no-op norm | max row-sum error | peak RAM GB | seconds |
|---|---|---|---|---|---|---|
| e1_add_s0 | 3.1e-05 | 0.0e+00 | 0.0e+00 | 2.1e-05 | 1.20 | 2.1 |
| e1_add_s1 | 2.3e-05 | 0.0e+00 | 0.0e+00 | 9.5e-06 | 1.21 | 1.9 |
| e1_add_s2 | 2.7e-05 | 0.0e+00 | 0.0e+00 | 2.6e-05 | 1.21 | 2.0 |
| e1_mul_s0 | 2.7e-05 | 0.0e+00 | 0.0e+00 | 1.3e-05 | 1.21 | 1.9 |
| e1_mul_s1 | 3.1e-05 | 0.0e+00 | 0.0e+00 | 2.2e-05 | 1.21 | 1.9 |
| e1_mul_s2 | 2.3e-05 | 0.0e+00 | 0.0e+00 | 3.8e-05 | 1.21 | 2.0 |
| diag_control_s0 | 3.1e-05 | 0.0e+00 | 0.0e+00 | 2.6e-05 | 1.21 | 1.9 |
| diag_control_s1 | 2.7e-05 | 0.0e+00 | 0.0e+00 | 4.9e-05 | 1.21 | 1.9 |
| diag_control_s2 | 3.1e-05 | 0.0e+00 | 0.0e+00 | 5.9e-05 | 1.21 | 2.0 |
| diag_sepops_s0 | 1.9e-05 | 0.0e+00 | 0.0e+00 | 8.3e-05 | 1.21 | 1.9 |
| diag_sepops_s1 | 1.9e-05 | 0.0e+00 | 0.0e+00 | 2.1e-04 | 1.21 | 1.9 |
| diag_sepops_s2 | 2.3e-05 | 0.0e+00 | 0.0e+00 | 7.3e-06 | 1.21 | 1.9 |

## T2. Layer-1 operand-score offset, all heads at once

δ is added to the attention scores of both operand positions at query '=' in layer 1, on diagonal train examples
(target diag) or on off-diagonal train examples (target off). norm free: the final RMSNorm recomputes; norm fixed:
each example's final inverse RMS is held at its own baseline value. Δ mass = mean change of operand mass at δ = +1.

| run | base diag_train | base off_train | target | norm | Δ at δ=−1 | δ=−0.5 | δ=+0.5 | δ=+1 | Δ mass at +1 |
|---|---|---|---|---|---|---|---|---|---|
| e1_add_s0 | 10.95 | 14.65 | diag | free | 0.12 | 0.13 | -0.22 | -0.48 | 0.186 |
| e1_add_s0 | 10.95 | 14.65 | diag | fixed | -0.61 | -0.24 | 0.12 | 0.15 | 0.186 |
| e1_add_s0 | 10.95 | 14.65 | off | free | 0.15 | 0.11 | -0.13 | -0.26 | 0.098 |
| e1_add_s0 | 10.95 | 14.65 | off | fixed | -0.52 | -0.19 | 0.10 | 0.14 | 0.098 |
| e1_add_s1 | 9.62 | 14.60 | diag | free | -0.11 | -0.02 | -0.02 | -0.05 | 0.134 |
| e1_add_s1 | 9.62 | 14.60 | diag | fixed | -0.77 | -0.33 | 0.23 | 0.39 | 0.134 |
| e1_add_s1 | 9.62 | 14.60 | off | free | -0.15 | -0.03 | -0.02 | -0.06 | 0.105 |
| e1_add_s1 | 9.62 | 14.60 | off | fixed | -0.88 | -0.36 | 0.24 | 0.39 | 0.105 |
| e1_add_s2 | 9.22 | 15.39 | diag | free | -0.02 | 0.03 | -0.07 | -0.15 | 0.136 |
| e1_add_s2 | 9.22 | 15.39 | diag | fixed | -0.82 | -0.35 | 0.23 | 0.38 | 0.136 |
| e1_add_s2 | 9.22 | 15.39 | off | free | 0.01 | 0.04 | -0.07 | -0.13 | 0.089 |
| e1_add_s2 | 9.22 | 15.39 | off | fixed | -0.88 | -0.36 | 0.23 | 0.37 | 0.089 |
| e1_mul_s0 | 11.10 | 15.65 | diag | free | 0.46 | 0.27 | -0.31 | -0.62 | 0.179 |
| e1_mul_s0 | 11.10 | 15.65 | diag | fixed | -0.34 | -0.12 | 0.03 | -0.01 | 0.179 |
| e1_mul_s0 | 11.10 | 15.65 | off | free | 0.12 | 0.11 | -0.14 | -0.27 | 0.108 |
| e1_mul_s0 | 11.10 | 15.65 | off | fixed | -0.54 | -0.20 | 0.09 | 0.13 | 0.108 |
| e1_mul_s1 | 9.02 | 15.11 | diag | free | 0.04 | 0.06 | -0.10 | -0.21 | 0.144 |
| e1_mul_s1 | 9.02 | 15.11 | diag | fixed | -0.64 | -0.26 | 0.16 | 0.26 | 0.144 |
| e1_mul_s1 | 9.02 | 15.11 | off | free | 0.08 | 0.07 | -0.10 | -0.18 | 0.088 |
| e1_mul_s1 | 9.02 | 15.11 | off | fixed | -0.68 | -0.26 | 0.16 | 0.25 | 0.088 |
| e1_mul_s2 | 9.89 | 14.52 | diag | free | 0.21 | 0.16 | -0.21 | -0.41 | 0.157 |
| e1_mul_s2 | 9.89 | 14.52 | diag | fixed | -0.55 | -0.21 | 0.10 | 0.14 | 0.157 |
| e1_mul_s2 | 9.89 | 14.52 | off | free | 0.02 | 0.04 | -0.07 | -0.14 | 0.089 |
| e1_mul_s2 | 9.89 | 14.52 | off | fixed | -0.64 | -0.25 | 0.15 | 0.24 | 0.089 |
| diag_control_s0 | 10.65 | 14.71 | diag | free | 0.42 | 0.28 | -0.36 | -0.74 | 0.176 |
| diag_control_s0 | 10.65 | 14.71 | diag | fixed | -0.48 | -0.16 | 0.02 | -0.06 | 0.176 |
| diag_control_s0 | 10.65 | 14.71 | off | free | 0.22 | 0.12 | -0.12 | -0.23 | 0.091 |
| diag_control_s0 | 10.65 | 14.71 | off | fixed | -0.48 | -0.19 | 0.12 | 0.19 | 0.091 |
| diag_control_s1 | 10.27 | 14.92 | diag | free | -0.06 | 0.06 | -0.16 | -0.36 | 0.160 |
| diag_control_s1 | 10.27 | 14.92 | diag | fixed | -0.85 | -0.33 | 0.17 | 0.24 | 0.160 |
| diag_control_s1 | 10.27 | 14.92 | off | free | -0.03 | 0.04 | -0.08 | -0.16 | 0.097 |
| diag_control_s1 | 10.27 | 14.92 | off | fixed | -0.84 | -0.33 | 0.20 | 0.32 | 0.097 |
| diag_control_s2 | 9.78 | 14.50 | diag | free | -0.06 | 0.00 | -0.04 | -0.10 | 0.165 |
| diag_control_s2 | 9.78 | 14.50 | diag | fixed | -0.66 | -0.29 | 0.21 | 0.34 | 0.165 |
| diag_control_s2 | 9.78 | 14.50 | off | free | -0.05 | 0.02 | -0.07 | -0.14 | 0.125 |
| diag_control_s2 | 9.78 | 14.50 | off | fixed | -0.77 | -0.31 | 0.20 | 0.32 | 0.125 |
| diag_sepops_s0 | 16.22 | 15.97 | diag | free | -0.62 | -0.21 | 0.11 | 0.16 | 0.095 |
| diag_sepops_s0 | 16.22 | 15.97 | diag | fixed | -3.05 | -1.35 | 1.01 | 1.72 | 0.095 |
| diag_sepops_s0 | 16.22 | 15.97 | off | free | -0.30 | -0.10 | 0.06 | 0.08 | 0.065 |
| diag_sepops_s0 | 16.22 | 15.97 | off | fixed | -1.98 | -0.86 | 0.62 | 1.05 | 0.065 |
| diag_sepops_s1 | 16.39 | 16.98 | diag | free | -0.07 | -0.02 | 0.01 | 0.02 | 0.026 |
| diag_sepops_s1 | 16.39 | 16.98 | diag | fixed | -0.98 | -0.40 | 0.27 | 0.45 | 0.026 |
| diag_sepops_s1 | 16.39 | 16.98 | off | free | -0.18 | -0.06 | 0.03 | 0.05 | 0.041 |
| diag_sepops_s1 | 16.39 | 16.98 | off | fixed | -1.61 | -0.69 | 0.48 | 0.81 | 0.041 |
| diag_sepops_s2 | 15.99 | 16.33 | diag | free | -0.02 | 0.01 | -0.03 | -0.05 | 0.088 |
| diag_sepops_s2 | 15.99 | 16.33 | diag | fixed | -2.01 | -0.88 | 0.64 | 1.08 | 0.088 |
| diag_sepops_s2 | 15.99 | 16.33 | off | free | -0.17 | -0.05 | 0.01 | 0.01 | 0.085 |
| diag_sepops_s2 | 15.99 | 16.33 | off | fixed | -2.16 | -0.94 | 0.69 | 1.16 | 0.085 |

## T3. Offset δ = +1, one layer-1 head at a time, norm fixed

| run | target | head 0 | head 1 | head 2 | head 3 | all heads |
|---|---|---|---|---|---|---|
| e1_add_s0 | diag | 0.13 | 0.08 | -0.10 | 0.02 | 0.15 |
| e1_add_s0 | off | 0.06 | 0.03 | 0.03 | 0.04 | 0.14 |
| e1_add_s1 | diag | 0.13 | 0.04 | 0.07 | 0.15 | 0.39 |
| e1_add_s1 | off | 0.06 | 0.09 | 0.15 | 0.09 | 0.39 |
| e1_add_s2 | diag | 0.08 | 0.05 | 0.15 | 0.11 | 0.38 |
| e1_add_s2 | off | 0.10 | 0.07 | 0.09 | 0.11 | 0.37 |
| e1_mul_s0 | diag | 0.00 | 0.00 | -0.03 | 0.02 | -0.01 |
| e1_mul_s0 | off | 0.04 | 0.05 | -0.00 | 0.04 | 0.13 |
| e1_mul_s1 | diag | 0.06 | -0.01 | 0.06 | 0.16 | 0.26 |
| e1_mul_s1 | off | 0.07 | 0.04 | 0.06 | 0.08 | 0.25 |
| e1_mul_s2 | diag | 0.09 | 0.02 | -0.01 | 0.05 | 0.14 |
| e1_mul_s2 | off | 0.07 | 0.06 | 0.05 | 0.06 | 0.24 |
| diag_control_s0 | diag | 0.01 | -0.07 | 0.01 | -0.01 | -0.06 |
| diag_control_s0 | off | 0.06 | 0.03 | 0.04 | 0.05 | 0.19 |
| diag_control_s1 | diag | 0.11 | 0.09 | 0.03 | 0.03 | 0.24 |
| diag_control_s1 | off | 0.04 | 0.11 | 0.10 | 0.09 | 0.32 |
| diag_control_s2 | diag | 0.09 | 0.06 | 0.07 | 0.13 | 0.34 |
| diag_control_s2 | off | 0.11 | 0.07 | 0.05 | 0.08 | 0.32 |
| diag_sepops_s0 | diag | 0.44 | 0.47 | 0.39 | 0.43 | 1.72 |
| diag_sepops_s0 | off | 0.27 | 0.28 | 0.27 | 0.25 | 1.05 |
| diag_sepops_s1 | diag | 0.11 | 0.06 | 0.19 | 0.09 | 0.45 |
| diag_sepops_s1 | off | 0.21 | 0.12 | 0.29 | 0.20 | 0.81 |
| diag_sepops_s2 | diag | 0.29 | 0.22 | 0.26 | 0.31 | 1.08 |
| diag_sepops_s2 | off | 0.31 | 0.25 | 0.30 | 0.30 | 1.16 |

## T4. Group-mean swaps (the first version's tests, rerun with the fixed code)

mass up diag: diagonal examples get each layer-1 head's off-diagonal train mean operand mass (a/b ratio kept).
mass down off: off-diagonal examples get the diagonal train mean. norm up diag: diagonal examples get the off-diagonal
train mean final inverse RMS. norm down off: the reverse. norm flatten diag: diagonal examples get the diagonal mean.

| run | base gap_train | mass up diag | mass down off | norm up diag | norm down off | both up diag | norm flatten diag: Δ diag_train | inv RMS diag / off | L1 mass diag / off (mean over heads) |
|---|---|---|---|---|---|---|---|---|---|
| e1_add_s0 | 3.70 | 4.37 | 3.53 | 2.94 | 2.82 | 2.84 | 0.01 | 4.995 / 5.310 | 0.596 / 0.817 |
| e1_add_s1 | 4.98 | 4.85 | 4.83 | 3.47 | 3.14 | 3.11 | 0.03 | 4.885 / 5.548 | 0.753 / 0.812 |
| e1_add_s2 | 6.17 | 6.25 | 6.06 | 4.93 | 4.45 | 4.59 | 0.01 | 4.821 / 5.390 | 0.751 / 0.843 |
| e1_mul_s0 | 4.56 | 5.15 | 4.38 | 3.63 | 3.45 | 3.63 | 0.01 | 4.565 / 4.908 | 0.631 / 0.799 |
| e1_mul_s1 | 6.09 | 6.28 | 5.98 | 5.07 | 4.67 | 4.85 | 0.04 | 4.807 / 5.285 | 0.722 / 0.843 |
| e1_mul_s2 | 4.64 | 5.09 | 4.47 | 3.25 | 2.97 | 3.17 | 0.02 | 4.911 / 5.512 | 0.692 / 0.840 |
| diag_control_s0 | 4.06 | 5.37 | 4.13 | 3.39 | 3.31 | 3.98 | 0.05 | 4.973 / 5.236 | 0.620 / 0.828 |
| diag_control_s1 | 4.64 | 5.11 | 4.37 | 3.93 | 3.74 | 3.82 | 0.01 | 4.936 / 5.245 | 0.668 / 0.822 |
| diag_control_s2 | 4.72 | 4.67 | 4.59 | 3.56 | 3.29 | 3.25 | 0.04 | 4.579 / 5.058 | 0.681 / 0.770 |
| diag_sepops_s0 | -0.25 | -0.37 | -0.31 | 0.84 | 1.47 | -0.02 | 0.30 | 3.406 / 3.133 | 0.831 / 0.887 |
| diag_sepops_s1 | 0.59 | 0.61 | 0.64 | 0.27 | 0.51 | 0.79 | 0.10 | 3.030 / 3.067 | 0.956 / 0.930 |
| diag_sepops_s2 | 0.34 | 0.32 | 0.35 | 0.43 | 0.76 | 0.34 | 0.13 | 3.447 / 3.403 | 0.846 / 0.852 |

## T5. Accuracy under norm-only swaps (train groups)

| run | base acc diag / off | norm up diag | norm down off | norm flatten diag |
|---|---|---|---|---|
| e1_add_s0 | 0.981 / 1.000 | 0.981 / 1.000 | 0.981 / 1.000 | 0.981 / 1.000 |
| e1_add_s1 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| e1_add_s2 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| e1_mul_s0 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| e1_mul_s1 | 0.979 / 1.000 | 0.979 / 1.000 | 0.979 / 1.000 | 0.979 / 1.000 |
| e1_mul_s2 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| diag_control_s0 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| diag_control_s1 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| diag_control_s2 | 0.980 / 1.000 | 0.980 / 1.000 | 0.980 / 1.000 | 0.980 / 1.000 |
| diag_sepops_s0 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| diag_sepops_s1 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| diag_sepops_s2 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
