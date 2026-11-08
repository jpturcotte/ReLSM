# H9: Claude vs blind Codex

Files: `claude_patch_analysis.md` (written first), `codex_patch_analysis.md` (blind; given the tables without my unregistered T2b).

## Agreement
- **No implementation bug.** The checks hold: forward ≤ 3.1e-5, P_both = donor ≤ 2.3e-5, self-patch ≤ 1.9e-5.
- **P_op0 ≈ P_eq is expected, not a bug.** The '=' state after layer 0 is close to a function of its operand write; the largest gap between them is 0.0078 nats.
- **Verdicts:** C1 0/9, C2 0/9, C3 5/9, alternative 0/9, so mixed by the registration.
- **The design can't localize the deficit.** Healthy off-diagonal pairs with the same answer lose 3.3–7.2 nats under any partial patch. That is as large as the deficit itself.
- **Size alone closes at most ~15 % (P_size),** in line with H8.

## Codex added
- **The interaction term, quantified.** I = Δ_both − Δ_eq − Δ_ops is 10.9–20.7 nats on diagonals and 6.7–14.4 nats on healthy off-diagonals. P_both's 100 % is a reconstruction check, not evidence about location.
- **P_ops includes the op position**, so it does not isolate the operand positions. True. It doesn't change any verdict.
- **Explicit limits:** a failed transplant does not refute an '=' mechanism, and relative transplant damage does not establish an operand-read mechanism. This also limits my T2b reading, which I had marked as suggestive.

## Claude added
- **T2b (unregistered difference-in-differences):** P_ops hurts diagonals more than off-diagonals in 7/9 runs. Given Codex's point, I downgrade this from "suggestive" to "not interpretable without a path-defined method".

## Next design: Codex's is adopted
**Route-resolved integrated-gradient attribution:**
- Interpolate the '=' state, the operand-position K/V and the op-position K/V jointly from recipient to donor, at 32 points.
- Integrate gradient × displacement per route.
- The routes must sum to the measured margin change within 0.05 nats.
- Run off-diagonal controls, and two interpolation choices (before and after the prefix RMSNorm).

**Registered outcomes:**
- '=' route ≥ 80 % → the '=' account;
- operand K/V ≥ 80 % → the operand-read account;
- a large op-position share → a third route;
- cancellation, or more than 0.5 nats of sensitivity to the interpolation choice → coupled / unresolved.

My small-step slope design is the first point of this integral, so Codex's design contains it.
