# Residual tests (H7) results

Final residual at '=' (pre-norm, fp32), train groups. Checks: component sum error, and the change in centered
pre-norm answer logits from stripping the invisible part (must be ~0).

## T1. Component RMS ratio (diagonal / off-diagonal) and share of the diagonal's excess squared norm

| run | emb ratio | attn0 ratio | mlp0 ratio | attn1 ratio | mlp1 ratio | total ratio | emb share | attn0 share | mlp0 share | attn1 share | mlp1 share | checks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e1_add_s0 | 1.000 | 1.431 | 0.882 | 1.018 | 1.038 | 1.060 | -0.07 | 1.67 | -0.50 | 0.28 | -0.39 | 0e+00 / 2e-04 |
| e1_add_s1 | 1.000 | 1.430 | 0.771 | 1.267 | 1.075 | 1.132 | 0.00 | 0.88 | -0.44 | 0.59 | -0.04 | 0e+00 / 2e-04 |
| e1_add_s2 | 1.000 | 1.424 | 0.844 | 1.208 | 1.004 | 1.114 | -0.01 | 0.97 | -0.36 | 0.54 | -0.14 | 0e+00 / 2e-04 |
| e1_mul_s0 | 1.000 | 1.418 | 0.862 | 1.090 | 0.893 | 1.074 | -0.04 | 1.63 | -0.36 | 0.49 | -0.72 | 0e+00 / 3e-04 |
| e1_mul_s1 | 1.000 | 1.418 | 0.763 | 1.169 | 1.074 | 1.095 | -0.01 | 1.15 | -0.50 | 0.51 | -0.14 | 0e+00 / 3e-04 |
| e1_mul_s2 | 1.000 | 1.413 | 0.802 | 1.156 | 0.993 | 1.119 | -0.01 | 1.06 | -0.27 | 0.40 | -0.18 | 0e+00 / 2e-04 |
| diag_control_s0 | 1.000 | 1.435 | 0.794 | 1.020 | 0.897 | 1.050 | -0.07 | 2.18 | -0.57 | 0.46 | -1.00 | 0e+00 / 2e-04 |
| diag_control_s1 | 1.000 | 1.415 | 0.735 | 1.088 | 1.046 | 1.059 | -0.03 | 1.85 | -0.88 | 0.48 | -0.41 | 0e+00 / 3e-04 |
| diag_control_s2 | 1.000 | 1.409 | 0.751 | 1.186 | 1.196 | 1.100 | -0.01 | 1.00 | -0.58 | 0.46 | 0.12 | 0e+00 / 2e-04 |
| diag_sepops_s0 | 1.000 | 1.080 | 0.982 | 0.890 | 1.064 | 0.919 | -0.00 | 0.02 | 0.34 | 0.65 | -0.00 | 0e+00 / 5e-04 |
| diag_sepops_s1 | 1.000 | 0.929 | 0.953 | 1.023 | 1.116 | 1.009 | 0.03 | -0.05 | -0.73 | 1.76 | -0.00 | 0e+00 / 3e-04 |
| diag_sepops_s2 | 1.000 | 0.978 | 0.973 | 0.975 | 0.934 | 0.983 | 0.03 | -0.05 | 0.65 | 0.38 | 0.00 | 0e+00 / 4e-04 |

## T2. Visible vs invisible energy, and stripping

Readout rank 98 of d = 128. scale gap = |1 − r_diag / r_off| (r = final inverse RMS).

| run | invisible share diag / off | scale gap base | scale gap stripped | gap_train base | gap_train stripped | closed | offset +1 response (free norm): plain / stripped |
|---|---|---|---|---|---|---|---|
| e1_add_s0 | 0.175 / 0.122 | 0.059 | 0.029 | 3.70 | 3.53 | 5% | -0.48 / -0.46 |
| e1_add_s1 | 0.156 / 0.109 | 0.119 | 0.095 | 4.98 | 4.93 | 1% | -0.05 / -0.02 |
| e1_add_s2 | 0.186 / 0.123 | 0.106 | 0.071 | 6.17 | 6.16 | 0% | -0.15 / -0.13 |
| e1_mul_s0 | 0.185 / 0.125 | 0.070 | 0.036 | 4.56 | 4.39 | 4% | -0.62 / -0.61 |
| e1_mul_s1 | 0.133 / 0.090 | 0.090 | 0.068 | 6.09 | 6.13 | -1% | -0.21 / -0.20 |
| e1_mul_s2 | 0.168 / 0.117 | 0.109 | 0.082 | 4.64 | 4.55 | 2% | -0.41 / -0.42 |
| diag_control_s0 | 0.229 / 0.153 | 0.050 | 0.004 | 4.06 | 3.77 | 7% | -0.74 / -0.74 |
| diag_control_s1 | 0.213 / 0.141 | 0.059 | 0.016 | 4.64 | 4.43 | 5% | -0.36 / -0.35 |
| diag_control_s2 | 0.179 / 0.120 | 0.095 | 0.062 | 4.72 | 4.63 | 2% | -0.10 / -0.07 |
| diag_sepops_s0 | 0.030 / 0.024 | 0.087 | 0.091 | -0.25 | -0.30 | — | 0.16 / 0.13 |
| diag_sepops_s1 | 0.022 / 0.022 | 0.012 | 0.012 | 0.59 | 0.60 | -2% | 0.02 / 0.01 |
| diag_sepops_s2 | 0.021 / 0.023 | 0.013 | 0.012 | 0.34 | 0.37 | — | -0.05 / -0.07 |

## Verdicts against the registration

- H7a (i) attn0 RMS ratio ≥ 1.25 in all 9 shared: 1.431, 1.430, 1.424, 1.418, 1.418, 1.413, 1.435, 1.415, 1.409 → True; separate-token in [0.9, 1.1]: 1.080, 0.929, 0.978 → True
- H7a (ii) attn0 + mlp0 share ≥ 0.5 in ≥ 7/9: 1.17, 0.45, 0.61, 1.27, 0.65, 0.78, 1.61, 0.96, 0.42 → 7/9
- H7b (i) invisible share diag > off in ≥ 7/9: 9/9
- H7b (ii) scale gap at most half after stripping in ≥ 7/9: 3/9
- H7b (iii) offset response at least 50% less negative after stripping: 0/8 qualifying runs (need 7)
