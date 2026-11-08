# Grokking line: where it stands (2026-11-06)

One page on the grokking studies of 2026-10-28 to 11-06: what they established for train.py, what stays harness-only, and what is retired. Study files are under `grok_runs\<study>\`. Sources: each study's `ANALYSIS.md` or `RESULTS.md` and `score.json`, `consolidation_2026-11-03/CONSOLIDATED_REPORT.md`, and the program review `PROGRAM_REVIEW_fable_2026-11-06.md`, whose claims were checked against those files. Codex checked this page against the same sources (`consolidation_2026-11-06/codex_check.md`, ACCURATE AFTER FIXES; the fixes are applied). The studies left 341 harness run records (`run_record.json`, archived attempts included) and 18 bridge records. None of the code behind them is merged.

## 1. train.py's own path groks modular addition

The line started because nothing in the project had ever run a regime where grokking is documented (`docs/review-2026-10-26/audit-2026-10-27.md` §1). The bridge study (`bridge_2026-11-04`) answers the first question.

- **Setup.** train.py with the new `--mod_table_p 97` path (`tools/grok_bridge.py`, now in PR #291). It uses train.py's llemma digit-string tokens, the four mod_add templates with BOS/EOS, the loss over the answer digits plus EOS, 4 × 128 accumulation, `build_optimizer` (wd 1.0, decay exemptions equal to `repo`) and bf16 autocast. The table and split are the harness's: a + b mod 97, a 50 % paired split, one task, a finite training set.
- **b1** (harness-matched: tied head, dropout 0.1, constant lr 6e-4): accuracy on the test off-diagonal pairs first reaches 0.99 at 1,400 / 1,600 / 1,900 updates and holds from 4,500 / 2,700 / 10,000, in 3/3 seeds. The 10,000 is set by a single dip, to 0.9887 at 9,900, after 8,000 updates at ≥ 0.99 (corrected 2026-11-08, misread audit).
- **b2** (train.py's defaults: untied, dropout 0, wsd): it also groks in 3/3 seeds. Test off-diagonal accuracy first reaches 0.99 at 2,400 / 3,100 / 1,700 and holds from 4,700 / 5,000 / 7,500. Its diagonal pairs hold 0.99 in only 1/3 seeds by 10k. By the same rule b1's diagonals hold only from the last few evals (9,700–10,000) or not at all, so the b1-vs-b2 diagonal contrast rests on the end of the run (corrected 2026-11-08, misread audit). b2 changes three settings at once, so the cause isn't identified.
- **Same seed, same code gives bit-identical results** on this path (102 eval rows and 100 loss records), as in the harness (47 pairs). So same-seed paired comparisons on train.py are exact.
- **What it does not show:** that train.py's real configuration groks. The one surviving real run, DAG-v12 (started Dec 29), used ten tasks, a curriculum, freshly generated examples, peak lr 1.5e-3 with wd 0.1, and 6,458 updates. Ninety-eight percent of its examples went to four easy tasks (audit findings, item 28). The bridge isolates one task on a finite table, the regime the literature describes.

## 2. Harness results not yet tested on train.py's path

These rest on `tools/grok_lab.py` at its base settings (lr 6e-4, `repo` scope, dropout 0.1, one token per number). The bridge shows that a harness effect can keep its sign and lose most of its size on train.py's path: the diagonal deficit, relative to the off-diagonal margin on the train side, shrank about 2–8× under b1 and 3–19× under b2, which changes three settings. The two studies also define the margin differently: the bridge's is the minimum over the answer digits and EOS, the harness's a single answer token (corrected 2026-11-08, misread audit; this said 2–20×). So treat these numbers as harness facts until they are bridged.

- **Training add with mul slows add's grokking by 8.0×** (90 % CI 6.2–10.3; 8 seeds, all slower; registered, `e2_2026-11-04`). That is add's sustained time on unseen off-diagonal pairs, against add trained beside a copy of itself (null-dup). It covers one task pair. add+sub was exploratory, with 3 seeds and a ratio of 1.00× (CI 0.49–2.04). That neither shows a cost nor rules out a doubling.
- **A task that joins after a solo phase generalises poorly** (`staged_2026-11-05`, 3 seeds, one join time at 2,500, a screen). The first task's off-diagonal accuracy survives the join: it is back above 0.99 by 2,800 in every run. Its diagonal does not fully: add's 20–25k mean diagonal accuracy is 0.37 / 0.74 / 0.82 when mul joins it, against 0.84–0.95 in the controls (corrected 2026-11-08, misread audit). The joiner memorises its training pairs within about 300 updates and then generalises only slowly. By 25k, mul joining add reaches 0.37 / 0.21 / 0.53 unseen accuracy and add joining mul 0.05 / 0.02 / 0.03. Trained together from the start, both tasks hold 0.99 by 8,100–14,800. The cause is untested. This is the same shape as train.py's DAG curriculum, which unlocks tasks one after another.

## 3. The layer-1 MLP can disengage without hurting training accuracy

In the harness under `repo` scope, the layer-1 MLP can stop contributing and stop receiving gradient while the model still groks (`b2_2026-11-03`, `gpu_followups_2026-11-04`). In some runs its weights then shrink at the passive AdamW rate (1 − lr·wd) per update. The tested cells fit a wd/lr pattern and rule out lr alone, but they don't establish wd/lr as the deciding variable. At wd/lr ≈ 600–667 the MLP disengaged in 3/3 seeds at both lr 6e-4 and 1.5e-3. At wd/lr 3,000 it stayed engaged (3/3 at 6e-4, 2/3 at 1.5e-3). The bridge's b1 sits at wd/lr ≈ 1,667, and there it stays engaged by the gain readout. Gain and dependence can disagree (`gpu_followups_2026-11-04`), so a dependence check on the bridge models is still open (corrected 2026-11-08, misread audit).

DAG-v12 ran at a peak wd/lr of about 67 (wd 0.1, lr 1.5e-3), far outside the tested range. At that wd, the passive decay over its schedule multiplies a weight by about 0.53 (audit NF-02). So passive decay alone could not have driven its MLP weights to zero. Whether the MLP disengaged there is unknown. Logging each layer's weight norm in train.py is the way to see it. (B2's `RESULTS.md` calls the risk for train.py "concrete". That holds for disengagement, not for decay to zero.)

## 4. Two premises of the fix plan were wrong (C1 walk)

- **bf16 is fine.** The bf16 loss cliff is in the CPU kernel only, and CUDA autocast doesn't have it.
- **The embedding doesn't need weight decay to grok.** `repo` scope grokked in 3/3 seeds. It was slower to its first crossing and had about twice the logit scale.
- **Dropout is the open regimen question.** It was the only C1 setting that removed collapses after grokking, and train.py now defaults to 0.

## 5. Retired: settled for the harness or unresolved, and not to be pursued

- **The size and mechanism of the diagonal (a, a) deficit.** Under full decay (C1's settings) it is 0.3–0.6 nats, against 3.7–6.6 at the base, but that contrast is in nats, which follow the logit scale. As a share of the off-diagonal margin it is 0.34 / 0.11 / 0.08 under full decay against 0.27–0.39 at the base (E1): one seed matches the base, two are about 3× smaller, none 10×. C1's full-decay arm also changes lr, batch, precision, dropout and the horizon, and with dropout 0.1 restored its test gap is 6.9–7.8 nats. So it does not show the deficit is an artefact of the base (corrected 2026-11-08, misread audit; this said "a property of the harness's base"). The line is retired on other grounds: on train.py's path the relative deficit is 2–8× smaller (b1) and shows as instability. The format study found that the harness's own answer rows halve it and that digit inputs don't shrink it.
- **The lr dependence of the add+mul cost.** The 3-seed slope (−0.25) didn't replicate (+0.006 [−0.39, +0.40]).
- **The comparison with Xu's 4–5× ratio.** Xu's control, criterion and model (a bidirectional encoder with separate heads) differ from ours, so the gap isn't a target.
- **The 2^(4−1) factorial.** Seven of its 42 runs hit the 25k cap, and all four registered main effects are unreadable. The one complete split, in late add off-diagonal accuracy, is post hoc and aliased (AC = BD). The proposed 24-run single-factor follow-up is dropped.

## 6. What the studies never touched

The audit tied several train.py bugs causally to "why nothing grokked" in the real runs, and no harness study addresses them:

- difficulty stuck at the easiest phase when workers are on (#141);
- curriculum unlock gates that can't be reached (#157; NF-15, NF-47, NF-49);
- truncated answers (#139);
- token accounting, where warmup ran 4–9× what was asked (#217, #248);
- evaluation reseeding the RNGs (#143).

## 7. Next

1. **Bridge the two harness results in §2 on train.py's path.** This is a screen, not a test of the real curriculum. If the add+mul cost and the joiner's slow generalisation shrink there the way the diagonal deficit did, they are mostly harness properties. If they hold, the one-after-another unlock becomes the first suspect for the curriculum, and curriculum against balanced sampling at matched compute (avenue A4 in `audit-2026-10-27-avenues.md`) becomes the experiment to run. Its design is pre-registered separately (`grok_runs/bridge_mt_2026-11-06/`).
2. **Land the code that exists.** In order:
   - the bridge path (`--mod_table_p`, the table dataset and scorer, the run record), now open as PR #291, which also adds exact resume for it. It covers part of fix-plan PRs 5–7 and #214, with #187's run record. Not yet: the compact tokenizer, full-batch mode and update-based stopping;
   - a decision on the open PRs #277 and #279 (#278, #280 and #281 have merged since this page was written);
   - the small flags #186 (`--wd_scope`), #246 (betas/eps), #267 (`--grad_clip 0` zeroes gradients), #268 (precision);
   - then the issues in §6.
3. **Method, for whatever runs next.** Three seeds pin a ratio only to about ×/÷1.5–2 (E2's one 3-seed ratio, add+sub, came out at ×/÷2.04). Use them to screen large effects, not to test whether one effect depends on another. Before a study, write down which train.py decision changes with each outcome. The pre-registration and Codex checks caught errors in every analysis, but nobody asked that question, and the back half of the line went to harness-only questions.
