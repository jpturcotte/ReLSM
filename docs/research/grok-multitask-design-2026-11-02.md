# Grokking beyond a+b: more tasks, and several tasks in one model

Design for extending `tools/grok_lab.py`, 2026-11-02. It builds on PR #281 (branch `feat/grok-lab-diag-arms`, commit b04662c). A prototype `--tasks` option and its tests come with this doc. A Fable red-team reviewed the first draft; §6 lists its findings and the changes made.

**Corrections from the 2026-11-03 consolidation** (full analysis: `grok_runs\consolidation_2026-11-03\CONSOLIDATED_REPORT.md`; each edit below is marked "corrected 2026-11-03, consolidation"):
- **Base:** every E1, diagonal and mechanism result is at lr 6e-4, `--wd_scope repo`, dropout 0.1. That differs from train.py (dropout 0, wsd, batch 64 × 4). No-training checks on older checkpoints (2026-11-03, `grok_runs\checks_2026-11-03\RESULTS.md`) have since tested part of this:
  - **full decay:** the diagonal deficit exists under `--wd_scope all` (3/3 seeds), but at C1's settings it is small, 0.3–0.6 nats; most of its size at this base comes from the training stack;
  - **lr 1e-3:** the gap is about half the 6e-4 gap at 1e5 updates, the mechanism's route mix differs, and the layer-1 MLP decays to zero in 2/3 seeds;
  - **dropout:** at C1's settings, dropout 0.1 makes every unseen error diagonal (3/3 seeds). Dropout 0 at this base is still untested.
  - train.py itself can't pose this task yet (random modulus per example, no held-out split; `checks_2026-11-03\objective_audit.md`), so nothing here transfers to it without a bridge run.
- **E1:** "order costs nothing" was a diagonal artefact (off-diagonal, sub is later than add in 3/3 seeds); s5_s2 never sustains 0.99, so the 100k rule fires; "all five grok" holds for first crossings only.
- **Pitfalls 4, 5 and 10:** starvation does not come from an exact zero past the cliff; "needs both tokens and commutativity" is a strong interaction in the tested tasks, not a shown necessity; same-seed CUDA reruns are bit-identical, so a rerun measures no noise.
- **E2:** the readout moves to unseen off-diagonal crossings (diagonals are reported separately). The range rule's power is low (0.18–0.40 at ×1.5), so it is replaced by a paired-t interval rule with 8 seeds for the headline arms (§4.3, §5 E2).
- **E3:** the small-weight control is restored, with a sham arm or a small `--task_replay` change to deal with its sampling confound; a matched no-switch continuation, a sub-from-scratch baseline and `--op_classes own` are added; the decay row names its tensor (under `repo` the table is not decayed). See §5 E3.
- **Logging:** per-layer MLP and attention weight norms at every eval, and diagonal readouts through training in dropout arms (§4.4).
- **Mechanism summary:** H5, H6c, H8 and H10 restated as the corrected README states them.

## Summary

- **Recommended first experiment.**
  - **Arms:** five single-task arms, `add`, `sub`, `mul`, `div` and `s5`, 3 seeds each, at the diagonal arms' settings (`stack_lr6e4`, 25k updates).
  - **Split:** every arm uses `--pairs paired`, so no arm learns a test pair from its swapped twin. add would get that for free; sub and div get the negated or inverted answer from it.
  - **Extra run:** one same-seed CUDA rerun of add, to size the run-to-run noise. (Historical: it was bit-identical, so it measured determinism, not noise; corrected 2026-11-03, consolidation.)
  - **What it answers:** whether this model groks each task, and how fast. Every multi-task result needs these grokking times as its baseline.
  - **Cost:** 16 runs, about 1 h 05 min of GPU time.
  - **Long reruns:** s5 is likely to need 100k updates (about 16 min per seed), so that rerun is budgeted from the start.
  - Exact flags are in [E1](#e1-single-task-catalogue).
- **Then:**
  - E2: add+sub and add+mul, read against two null arms, add alone at the same batch and add paired with a copy of itself;
  - E3: a switch from add to sub, with two controls that separate "unused and decayed" from "overwritten". Revised 2026-11-03: a matched no-switch continuation, a small-weight replay arm with its exposure control, and a sub-from-scratch baseline are added (§5 E3).
- **What looks new.** The November 2026 search found none of the following:
  - a single-task catalogue in a RoPE-only, tied-head decoder;
  - op-token multi-task training against explicit "sharing cost" nulls;
  - sequential training with the first task's retention measured;
  - the diagonal (a, a) weakness.

  For the last two this was mostly an abstract-level search, so read it as "not found", not "absent". Simultaneous op-token multi-task grokking itself has been done (Furuta et al. 2024; Xu 2026, with separate heads).
- **Prototype:** `--tasks add,add_dup,add2b,sub,mul,div,sq,sqab,cubeab,s5,gcd`, plus options:
  - per-task answer tokens;
  - shared or independent splits;
  - per-task train fractions and loss weights;
  - start and stop updates;
  - idle (decay-only) phases;
  - `--wd_until`.

  Readouts are logged per task with the suffix `@task`. `--tasks add` is the same run as `--task add`, bit for bit on CPU; a test checks this. Test status is in [Prototype](#prototype).

## 1. How the harness works today (PR #281)

- **Task.** Every ordered pair (a, b) with a, b in 0..p−1 (p = 97) is one example. The input is the four tokens `[a, op, b, =]` and the answer is read at `=`.
  - Tasks: `add` is (a + b) mod p; `add2b` is (a + 2b) mod p.
  - Token ids: numbers 0..p−1, op = p, '=' = p + 1.
  - `--operands separate` gives the second operand its own tokens p..2p−1. op and '=' then shift to 2p and 2p + 1.
- **Split.** Seeded, at `--frac` (0.5), over the p² pair ids.
  - `ordered` splits pairs one by one; `paired` keeps (a, b) and (b, a) together.
  - A test pair is tagged "transpose seen" when (b, a) is in train.
  - `--diagonals` and `--holdout_line` move whole families of pairs to one side of the split.
- **Answer classes.** `answer_logits` scores the p answers plus op and '=' (p + 2 classes). Under separate operands, the second-operand tokens are dropped from the outputs.
- **Model.** `BaselineTransformer`: d 128, 2 layers, 4 heads, SwiGLU (d_ff 512), RMSNorm, tied head, `max_seq_len` 4.
  - Position enters only through RoPE on queries and keys, and through the causal mask. There is no position embedding.
- **Readouts** (`evaluate`, every `--eval_every` updates):
  - accuracy and fp64 loss on train and test;
  - test accuracy for seen, unseen, diagonal and unseen off-diagonal pairs;
  - margins by group;
  - train margins with cliff fractions;
  - norms and Adam state.
- **Run record.** Grokking times (`first_update_unseen[_offdiag]_ge_0.99`) are read from the whole eval trace. `final_errors.json` lists every wrong pair.

## 2. Prior work (search to November 2026)

A search agent read the primary sources: arXiv abstracts and HTML pages, and for Power et al. the PDF text. Anything it could not confirm is marked unverified. The October review in `docs/research/grokking-lit-2026-10-28/` had found Xu 2026 already.

**Single tasks:**
- **Power et al. 2022** (arXiv 2201.02177) used our shape: 2 layers, width 128, 4 heads, AdamW lr 1e-3, wd 1, β2 0.98, batch 512, p = 97, with learned position embeddings.
  - **Operations** (their App. A.1.1):
    - x+y, x−y, x/y;
    - [x/y if y is odd, else x−y];
    - x²+y², x²+xy+y², x²+xy+y²+x, x³+xy, x³+xy²+y;
    - in S5: x·y, x·y·x⁻¹, x·y·x.
  - x−y and x/y need about the same data.
  - Symmetric operations need less data than their asymmetric versions. They suggest a transformer "can ignore the positional embedding" for those, and say it may depend on the architecture.
  - Their split is over ordered pairs, so a symmetric table also gets every swapped training pair as a free test answer. That alone favours symmetric operations.
  - x³+xy²+y did not generalize at any train fraction up to 95 %.
  - The per-operation minimum fractions exist only in their Fig. 2 and were not read.
- **Furuta et al. 2024** (arXiv 2402.16726, TMLR) used 1 layer, d 128, input [a, op, b], p = 97, train fraction 0.3.
  - The search agent read "no positional embedding" in the paper's setup section. The RoPE-only novelty claim leans on this, so re-check it before citing.
  - +, −, × and factorizable polynomials such as (a±b)ⁿ grok.
  - a²+ab+b² generally fails at 0.3 when trained alone.
- **Doshi et al. 2024** (arXiv 2406.03495) used MLPs. Polynomials of the form h(g₁(a)+g₂(b)) are learnable; adding a cross term breaks this.
  - (4a+b²)³+ab reached 2.3 % at p = 97.
  - Prediction for us: a²+b² groks. Whether a³+ab groks is an MLP-based guess, not a known result for transformers.
- **Groups.** Chughtai et al. 2023 (arXiv 2302.03025) and Stander et al. 2023 (arXiv 2312.06581) grok S5 and S6 with MLPs and 1-layer transformers, one group per model. The two papers disagree on the algorithm learned.
- **Algorithmic tasks.**
  - Sparse parity groks: Barak et al. 2022, arXiv 2207.08799; Merrill et al. 2023, arXiv 2303.11873.
  - GCD is learned by transformers (Charton 2023, arXiv 2308.15594), but not framed as grokking.
  - No report was found of grokking on sorting or reversal over a fixed table. Delétang et al. 2022 (arXiv 2207.02098) studies length generalization, not grokking.

**Several tasks:**
- **Furuta et al. 2024 §8.** {add, sub, mul} with an op token and a shared output, trained together. All three "co-grok", but the mixture needs more data than each task alone: 0.3 is enough for each task, not for the mix. Multiplication is slowest.
  - Mixing polynomials can help: a²+ab+b² groks inside {a²+b², a²+ab+b², (a+b)²} but not alone.
  - Embedding transfer works between add and sub. Multiplication shows no synergy with either.
  - This is the closest prior work to E2.
- **Xu 2026** (arXiv 2602.18523, single author) trained a 2-layer bidirectional encoder on input [a, b], with no op token and a **separate linear head per task**, simultaneous only.
  - add+mul grok at about 13k updates at wd 1, against a stated 2–3.7k single-task: a slowdown of about 3.5–6.5×. That single-task configuration is not fully stated, so this is not a controlled comparison.
  - add+mul+sq grok staggered, mul first, then sq, then add: 12–31k updates.
  - No forgetting experiment. Gradient cosines between tasks were 0.15–0.38.
- **Mallinar et al. 2024** (arXiv 2407.20199, a non-neural feature-learning model): x+y and x²+y² together grok at two separate times.
- **Transfer and forgetting.**
  - Jia 2026 (arXiv 2609.18078) warm-starts from a grokked a+b on to other targets. Continued training produces "post-grokking relapse", with drawdowns of 22–81 %.
  - Zhang et al. 2026 (arXiv 2607.29503): forcing random labels after grokking drops the original task below 75 %.
  - Neither measures the first task's retention while a real second task trains, as far as the abstracts and HTML pages show.
- **Diagonal weakness:** not found. Our finding (`diag_arms_report.md`, 2026-11-02) appears new.
- **RoPE versus other position encodings** for grokking non-commutative tables: no comparison found.

**What this design adds:**
- the catalogue in a RoPE-only, tied, 2-layer decoder, with swapped-pair leaks closed for every task;
- op-token pairs read against explicit null arms (same batch, a copy of the same task);
- sequential training with controls for decay and for disuse;
- diagonal readouts for every task.

## 3. Task catalogue

**"4-token"** means the task fits `[a, op, b, =]` with one answer at `=`, so the prototype runs it today. Table sizes are at p = 97. Diagonal answers are given because the diagonal readouts apply to every task.

| Task | Answer | Domain, table size | Diagonal (a, a) → | Commutative | Interest | Expected difficulty (source) | Harness |
|---|---|---|---|---|---|---|---|
| `add` | a+b | 97², 9,409 | 2a | yes | the control | groks, ~300–1,000 updates here | today |
| `add_dup` | a+b, second op token | 9,409 | 2a | yes | the null partner: two tasks that agree completely | as add | prototype |
| `add2b` | a+2b | 9,409 | 3a | no | separates "same token twice" from "symmetric input" | groks (Furuta: linear forms need fraction 0.4–0.5 from scratch, approximate). Here: groks, diagonal gap about 1 with shared tokens (run 2026-11-02, pitfall 5) | today |
| `sub` | a−b | 9,409 | 0 (constant) | no; (b, a) answers −(a−b) | order must come from RoPE and the causal mask; diagonal answer constant | groks, needs more data than add (Power; Furuta) | 4-token, prototype |
| `mul` | a·b | a, b ≠ 0: 96², 9,216 | a² (48 values) | yes | the group (Z/97)* ≅ Z/96. A Fourier-in-log-a number embedding would conflict with add's Fourier in a | groks like add (Power); first to grok in Xu's mixes | 4-token + domain, prototype |
| `div` | a·b⁻¹ | b ≠ 0: 97·96, 9,312 | 1 (constant) | no; (b, a) answers the inverse | mul's non-commutative twin | like sub (Power) | 4-token + domain, prototype |
| `sq` | a²+b² | 9,409 | 2a² | yes | h(g₁(a)+g₂(b)) form | groks (Power; Doshi predicts yes) | 4-token, prototype |
| `sqab` | a²+ab+b² | 9,409 | 3a² | yes | has a cross term. Since 3 divides 96, it factors mod 97 as (a−ωb)(a−ω²b). That is a product of linear forms that each mix a and b: an add-type then mul-type composition, not one of Furuta's factorable forms. The factorization is my inference and says nothing on its own about learnability | uncertain: Power lists it as generalizing with more data; Furuta: fails at 0.3 | 4-token, prototype |
| `cubeab` | a³+ab | 9,409 | a³+a² | no | cross term | unknown for transformers. Doshi's MLP cross-term result suggests no; Power's curve was not read | 4-token, prototype |
| `s5` | g∘h in S5 | 120², 14,400 | g² | no | non-abelian group; representation-theory algorithm (Chughtai) | groks at 25–50 % data, slowly; Power needed up to 1e5 updates | 4-token, own 120 tokens, prototype |
| `gcd` | gcd(a, b) | a, b ≠ 0: 9,216 | a | yes | not a group, no Fourier solution; skewed answers (gcd = 1 for about 61 % of pairs) | learned by transformers (Charton), never framed as grokking | 4-token + domain, prototype. Large gcd values occur almost only on the diagonal, so run with `--diagonals train` or `--allow_missing_residues` |
| parity | xor of n bits | 2ⁿ (n = 12: 4,096) | — | — | full parity is hard for attention | hard. Sparse k-of-n parity groks (Barak; Merrill) but needs a sampled table, since 2ⁿ is huge | needs n+1 tokens and 2 answer classes |
| sort-i | i-th smallest of a length-L sequence | k^L · L (k = 5, L = 5: 15,625) | — | — | rank and count; one answer per query keeps one answer per example | no grokking report found | needs `[x1..xL, op, i, =]` (L+3 tokens) |
| reverse-i | x_{L−1−i} | k^L · L | — | — | pure position lookup: a direct test of RoPE-only addressing | no grokking report found | same as sort-i |

**Changes needed for the sequence tasks** (phase 2, not in the prototype):
- `max_seq_len` must follow the encoding, not stay at 4.
- The split must work over whole sequences, so every query of a test sequence is unseen.
- The tasks need an alphabet of symbols and of position-query tokens.
- "diagonal" is replaced by a "repeated symbol" group (sequences with ties), which is the same identical-token question.
- The answer is still read at the last position, so the loss and metrics code stays the same.

## 4. Several tasks in one model

### 4.1 How the task is chosen and scored

| Option | What it tests | Risk | In the prototype |
|---|---|---|---|
| **A. Op token, shared answers** (`--answers shared`, default): `[a, op_t, b, =]`, and tasks on the same alphabet answer with its number tokens | the full sharing case: input embeddings, trunk and output rows are all shared | answer tokens mean the same thing across tasks only when the answers are numbers mod p. S5 and parity need their own blocks | yes; S5 gets its own 120-token block automatically |
| **B. Separate output ranges** (`--answers per_task`): each task answers with its own output-only tokens | whether interference sits in the readout. It is the tied-head analogue of Xu's separate heads | under a tied head, those rows are trained only as outputs, which changes the single-task baseline. Run `--tasks add --answers per_task` as its own control | yes |
| C. No op token, separate heads (Xu 2026) | a shared trunk only | the trunk cannot know the task | no; B covers the readout question |

**Scoring.** Each task is scored over its own answer classes: its answer block, then every op token and '='. With shared answers and T tasks that is p + T + 1 classes, against p + 2 today. One more competitor class per extra task barely moves the loss, and with one task the classes are exactly today's.

The alternative, one softmax over the whole vocabulary, lets tasks push down each other's answer tokens. That is a confound, not a question, so it is left out.

The same push-down happens, on a smaller scale, through the shared op-token classes: a stopped task's op token stays a wrong class for the task still training, so its row keeps being trained through the tied head. E3 therefore uses a planned `--op_classes own`, under which each task's classes are its answer block, its own op token and '=' (§5 E3; not implemented yet; added 2026-11-03, consolidation).

### 4.2 When each task trains

- **Simultaneous:** every task from update 0. This is the default.
- **Staged addition:** `--task_start sub=5000`. add trains from 0 and sub joins at 5,000; add keeps training. This tests transfer and interference.
  - Run it at `--batch 1024`: two tasks share the batch after 5,000. At 512, sub would get about 256 examples per update, half of what it gets in E1.
- **Switch:** `--task_stop add=5000 --task_start sub=5000`, the forgetting test.
  - A drop in add can come from three causes:
    1. sub's gradients move shared weights (**overwriting**);
    2. weight decay shrinks weights that nothing trains any more (**decay**);
    3. Adam's leftover momentum.
  - The decay timescale 1/(lr·wd), about 1,700 updates at lr 6e-4 and wd 1, applies only to decayed weights. Under `--wd_scope repo`, the embeddings (add's op-token row included) and the head are exempt. And sub's own Adam steps, about lr per coordinate per update, move shared weights much faster than decay. So the timescale alone cannot separate the causes.
  - Two controls can:
    - **`--wd_until 5000`:** the same switch with no weight decay after the switch. If add still collapses, the cause is overwriting.
    - **`--task_stop add=5000 --task_start sub=25000 --allow_idle`:** add, then nothing. sub stays in `--tasks`, so the vocabulary matches the other arms, but its start equals `--updates`, so it never trains (without it, sub would train from update 0). The idle updates are zero-gradient steps, so they apply only weight decay plus Adam's dying momentum. The update ratio decays about (β1/√β2)ᵏ ≈ 0.92ᵏ, roughly 13 lr-sized steps in all (an approximate tail multiplier under simplifying assumptions, not a displacement bound). Under `repo` this momentum tail is the only way the table moves in the idle arm (corrected 2026-11-03, consolidation). This is the decay-only curve.
  - A small-weight control (`--task_weight add=0.01`) is kept as a fourth arm (corrected 2026-11-03, consolidation; it was dropped on a false premise). At the switch (update 5,000) add's zero-target share is ≈ 0 (0.0–0.1 % of pairs), and past the cliff the non-target gradients survive. Whether Adam drowns a 1 % share is what the arm measures. Two design points (both settled in §5 E3):
    - **add must stay active after 5,000:** no `--task_stop add=5000` in this arm. With `--task_weight add=0.01` the add-only phase is unchanged bit for bit, since weights are normalized over the active tasks (the coefficient is exactly 1.0 while add trains alone).
    - **Sampling:** the sampler draws uniformly from the pooled rows of the active tasks and never reads the loss weight. At batch 512 with add active, sub gets 256 ± 11 rows per update instead of exactly 512. sub's expected loss scale barely changes (×0.990), but it rests on half as many rows, so its gradient noise and per-update coverage double. Without a fix, the arm can't separate rehearsal from reduced sub exposure. §5 E3 gives two fixes: a sham-weight arm, available now, or a small `--task_replay` change.
- **Curriculum and mixing ratios.** The loss is a fixed-weight objective over the active tasks:
  - `equal` (default): Σ (uᵢ/U) · meanᵢ;
  - `examples`: Σ uᵢ nᵢ meanᵢ / Σ uⱼ nⱼ;
  - here uᵢ is `--task_weight`, nᵢ the task's train size, and U the sum of the active tasks' weights.
  - Under `--batch`, each task's batch mean is scaled by cᵢN/(Bnᵢ), where cᵢ is its rows in the batch, B the batch size and N the pooled active train size. The minibatch loss is then an unbiased estimate of the full-batch objective.
  - A task with few rows in a batch counts little, and an absent task adds nothing. The weights are never renormalized over the tasks a batch happens to contain, which the first draft did.
- **Per-task train fractions:** `--task_frac sub=0.3`. The most informative use is to put a task below its single-task threshold and see whether a partner carries it over. Furuta saw this for a²+ab+b².
- **Split alignment:** `--task_split`.
  - `aligned` (default): every task splits from one shuffle of all p² pairs, and the domain is applied after the shuffle. A test pair of one task is then a test pair of the others, even for add with mul (a test checks this). Only a few pairs at the edge differ when domains differ.
  - `independent`: each task gets its own shuffle. About half of one task's test pairs are then the other's train pairs, a cross-task leak worth measuring, but only on purpose.
  - Task 0 always gets the single-task split.

### 4.3 Metrics

All existing readouts are kept, per task:
- Every `evaluate()` key is logged with `@task`: `test_acc@sub`, `test_margin_mean_diagonal@mul`, `train_margin_p5_offdiag@s5`, and so on.
- Accuracy and loss are also pooled, weighted by examples.
- Norms and Adam state are model-wide and computed once per eval.
- Each eval line, update 0 included, has `active_tasks`.

Per task, in `run_record.json`:
- **Grokking time.**
  - `first_update_test_ge_0.99@t`: under `--pairs paired`, this is the grokking time for non-commutative tasks. For commutative tasks it is diagonal-limited (E1 add: test 1,200–2,000 vs off-diagonal 800–1,100). The headline is an unseen off-diagonal crossing: pre-register one primary outcome (first or sustained) and report the other separately (corrected 2026-11-03, consolidation). E2 pre-registers the sustained crossing (see the decision rule below). The record has no sustained-time field yet; until it does, compute it from the eval trace.
  - `first_update_unseen_offdiag_ge_0.99@t`.
  - `first_update_unseen_ge_0.99@t` (diagonals included).
- **Forgetting or relapse.**
  - `min_test_acc_after_first_ge_0.99@t`, the lowest test accuracy after the task first grokked. One low eval can be a loss spike.
  - `min_test_acc_last10pct@t`, the lowest over the last 10 % of evals. A low value here, or a run of low evals, is forgetting.
- **Interference (from the analysis, not logged).**
  - **Readout:** a task's grokking time with a partner, against the same task's time in the matching null arm (§5 E2), never against a different batch size.
  - **Seed pairing** (corrected 2026-11-03, consolidation; this said seeds are not paired across arms). Arms with the same vocabulary are init-paired: E1's numeric arms (vocab 99) have identical update-0 norms per seed. A different vocabulary size still changes the init and every random draw after it (s5, null-batch). Before using seed-paired comparisons for E2's two-task arms, verify tensor and split identity, not only summary norms.
  - **Decision rule** (replaced 2026-11-03, consolidation; the analysis is `grok_runs\checks_2026-11-03\e2_rule_power.md`).
    - **Why the old rule was dropped.** It called an effect only when every partner-arm seed fell outside the null arm's range, with 5 seeds per headline arm. Its false-positive rate is low, but so is its power. Simulated with E1 add's spread of test times: ×1.25: 0.04–0.08; ×1.5: 0.18–0.40; ×2: 0.61–0.91. Re-simulated with seed pairing and E1's off-diagonal spread: 0.10–0.32 at ×1.5 with 5 v 5 seeds, and it *loses* power as seeds are added (0.12 at 8 seeds). A sign test can't help at 3 seeds: its smallest one-sided p is 0.125.
    - **Primary comparison:** add+mul against null-dup, one-sided (partner slower).
    - **Primary outcome:** add's *sustained* unseen off-diagonal grokking time: the first eval from which `test_acc_unseen_offdiag@add` stays ≥ 0.99 through update 25,000 (fixed horizon). Evals every 100 updates (50 if the smoke run shows the cost is small). The first crossing is secondary. Sustained was the lower-variance outcome in E1 (pooled log-sd 0.16 vs 0.20), and a relapse caused by the partner counts as interference too.
    - **Estimand:** the geometric-mean ratio of that time, add+mul over null-dup, across seeds: exp(mean over seeds of log(T_add+mul / T_null-dup)).
    - **Rule:** a paired t on the per-seed log ratios (n − 1 df), reported as a two-sided 90 % CI on the ratio.
      - **"Slower":** the CI's lower bound is above 1 (one-sided α = 0.05).
      - **"No slowdown ≥ 1.5×":** only if the CI's upper bound is below 1.5.
      - **Otherwise inconclusive:** report the point estimate and the interval, as "no slowdown detected; slowdowns up to [upper bound]× are not excluded". "Inside the null range" and "no conflict" are retired as conclusions.
    - **Pre-registered companion:** the exact paired sign test. Report both whenever a run is censored (a null-dup run in particular) or a late dip delays the sustained crossing by more than 2× the first. If they disagree, call the result fragile. A run that never sustains by 25k gets T = 25,000 and is flagged. An aborted or missing run is rerun on the same seed (bit-identical on CUDA); if that can't be done, drop the seed from both arms. Never impute.
    - **Seeds:** 8 per arm for null-dup and add+mul (seeds 0–7), fixed before launch and not extended after seeing results. Power at ×1.5: 0.97 at log-sd 0.2 with no within-seed correlation, 0.96 at log-sd 0.3 with correlation 0.5, and 0.78 at log-sd 0.3 with none (the pessimistic corner). Five seeds reach 0.80 only if the sd of the per-seed log ratio is ≤ ~0.29, which E1's 3 seeds can't guarantee. A 1.25× slowdown is below this design's resolution. After the run, report the observed sd of the per-seed log ratio, so the achieved minimum detectable effect is stated.
    - **Exploratory, reported as estimates with 90 % CIs and no verdict:** add+sub (3 seeds, paired with null-dup's seeds 0–2), null-batch (3 seeds), and sub's and mul's own times. At 3 seeds, "add+sub is not slower than 1.5×" is reached only 0.32–0.74 of the time even with no true effect, so the outcome "add+mul slower and add+sub not" needs add+sub at 8 seeds too if it is to be confirmatory (+5 runs, about 30–40 min).
    - **Pairing, from the code** (`e2_rule_power.md` §2; to be confirmed by a CPU check before launch):
      - null-dup, add+sub and add+mul all have vocab 100. With the same seed they get bit-identical initial weights (table and tied head included) and the identical add train/test split.
      - add+sub also gets add's identical minibatch row indices as null-dup (same pool size, 9,408). add+mul does not: its pool is 96 rows smaller, so its batch permutation differs from update 0. That adds data-order noise, so the design assumes no within-seed correlation for the headline pair.
      - null-batch (vocab 99) has a different init from all three. Compare it with null-dup unpaired (Welch t on logs). It is init- and split-paired with E1's add, which gives a free paired estimate of the batch-size effect (1024 vs 512).
      - **CPU check (seconds per seed):** build each arm through grok_lab's own parser for seeds 0–7; assert `torch.equal` on add's split tensors across arms and on every `state_dict` tensor across the three vocab-100 arms; draw the first ~200 batches and compare add's row indices; record a hash of each initial `state_dict` in the run record.
- **Margin interference:** a task's `train_margin_p5@t` and `test_margin_p5_unseen_offdiag@t` before and after a partner joins.
- **Diagonal gap per task:** `test_margin_mean_unseen_offdiag@t − test_margin_mean_diagonal@t`, as in the diagonal report.
- **Errors:** `final_errors.json` has a `task` column, and `n_wrong_final.by_task` in the record.

### 4.4 Diagnosis: readouts that say why, not only whether

The outcome readouts above say whether a task grokked, when, and whether it was lost. These say why. All are logged at every eval and need no extra disk.

| Readout | Measures | Diagnoses |
|---|---|---|
| `emb_fourier_spectrum`, `emb_fourier_top5_frac`, `emb_fourier_top5` | power of the number embedding by frequency along a (k = 1..48 at p = 97). A grokked a+b concentrates it in a few frequencies (Nanda 2023); a random table gives a top-5 share of about 0.12 (Monte Carlo 0.116–0.125; logged initial shares 0.118–0.123; corrected 2026-11-03, consolidation) | which structure the table holds; when it forms relative to the grokking time |
| `emb_logfourier_*` | the same along log a (a = 5^j mod 97, a ≠ 0). (Z/97)* is cyclic, so a Fourier solution for a·b concentrates here instead | the add/mul conflict, measured directly. add alone should be concentrated along a and spread along log a; mul alone the reverse; add+mul shows whether the table holds both, one, or neither |
| `*_b` versions | the second operand's block under `--operands separate` | whether the two operands build the same structure |
| `grad_cos_all@i\|j`, `grad_cos_trunk@i\|j` | cosine between two tasks' full-train-set gradients: all weights, and the trunk without the embedding table and head | raw full-train gradient alignment in eval mode (no Adam preconditioning, clipping or dropout): negative means the raw gradients pull shared weights apart; near zero means separate directions; positive means they agree. With the tied head, all − trunk is the whole table, input and output rows together, so it cannot separate embedding conflict from readout conflict (corrected 2026-11-03, consolidation) |
| `grad_norm_task@t` | each task's gradient norm | starvation: a task whose gradient norm is near zero is no longer defending its weights. Past the fp32 cliff only the target term of the gradient vanishes; the non-target gradients survive (0.10–0.71 of the pre-cliff L2 norm). So the cliff alone doesn't zero a task's gradient; a small norm comes from CE's exponential tail at high margins, cliff or not (pitfall 4). Report the zero-target share (`zero_tgt_grad_frac@t`, train mode when dropout is on) alongside this norm, not in place of it (corrected 2026-11-03, consolidation; this said a task past the cliff has a gradient near zero) |
| `wnorm_mlp@l`, `wnorm_attn@l` (to add) | each layer's MLP and attention weight norms (Frobenius, per matrix or per block), at every eval | a layer dying during training. At lr 1e-3, wd 1, under `repo`, the gated layer-1 MLP decayed to exactly zero within 25–50k updates in 2/3 seeds (`checks_2026-11-03\RESULTS.md`, K2b), and only a checkpoint probe found it. The existing `norm_*` keys are model-wide and can't show it. Required in every arm from 2026-11-03 (consolidation); not yet in the code |
| `--ckpt_at` | weights at a few chosen updates (2.2 MB each; `final.pt` 6.5 MB; corrected 2026-11-03, consolidation) | anything not logged, after the run: attention patterns at '=' for diagonal pairs, Nanda's restricted and excluded loss, probing |

**Diagonal readouts in dropout arms** (added 2026-11-03, consolidation). At C1's settings, dropout 0.1 made every unseen error diagonal (8–14 per seed, 3/3) and multiplied the test-side diagonal gap 5–15× (`RESULTS.md`, K1). C1 logged no diagonal metrics during training, so it can't say whether dropout delays grokking through the diagonals. In every arm with `--dropout > 0`, the per-eval diagonal readouts that `evaluate` already logs (`test_acc_diagonal`, `test_margin_*_diagonal`, `train_margin_*_diagonal`, per task) are outcomes in their own right, read over the whole trace, not only at the end. The train-mode pass (`*_trainmode`) has no diagonal split yet; adding one is a small change, not implemented.

How they combine with the outcome readouts:
- **E1:**
  - **Expected:** add and sub put their top-5 share along a; mul and div along log a.
  - **If a task groks without either concentrating,** it found a different algorithm. Check its checkpoints before comparing it with the others.
  - **If sub is slower than add,** see whether its spectrum forms late (the embedding is the bottleneck) or early with a late grok (the trunk is).
- **E2:**
  - **Embedding conflict:** add+mul is slower than null-dup, the table shares power along both a and log a without concentrating in either, and `grad_cos_all@add|mul` is below `grad_cos_trunk@add|mul`.
  - **Trunk conflict:** the slowdown comes with negative trunk cosines while the table concentrates normally.
  - **Readout conflict:** neither of the above, but `--answers per_task` removes the slowdown.
  - **Caveat** (added 2026-11-03, consolidation): these are candidate mechanisms, not observations. The cosines are raw eval-mode gradients, and under the tied head "all below trunk" can't tell embedding conflict from readout conflict. `--answers per_task` is the only clean readout-conflict test.
- **E3:** after the switch, follow the power at add's own top-5 frequencies (read from `emb_fourier_top5` just before update 5,000) in `emb_fourier_spectrum`.
  - **Decay:** under `repo` the table is exempt from decay, so a shrinking spectrum in `tok_emb` can't be attributed to direct table decay; gradients and leftover momentum can still move it (corrected 2026-11-03, consolidation; this row said all of add's frequencies shrink together while their share stays high, and the idle arm looks the same). Read decay from the decayed (trunk) weights, or run the arm at `--wd_scope all`.
  - **Overwriting:** add's frequencies lose share to new ones, sub's if they differ, and the `--wd_until` arm looks the same.

### 4.5 Decision tree

| After | Result | Next |
|---|---|---|
| E1 | a task under 0.99 at 25k | rerun it at 100k before using it anywhere. If it still fails, drop it from E2 and report it |
| E1 | sub or div slower than its partner, with its spectrum forming late | order costs here. Run sub with `--operands separate` (the two operands then have different tokens) to see whether the cost is about telling positions apart |
| E1 | diagonal gaps follow the prediction: add and mul large, sub, div and s5 small | commutativity plus identical tokens explains the weakness across tasks, not just a+b vs a+2b. Look for the mechanism in checkpoints: compare attention at '=' for (a, a) on add against sub |
| E1 | sub, div or s5 shows a large gap, or mul a small one | commutativity is not the whole story. With sub and div, the constant diagonal answer is the next suspect. Run the task with `--operands separate` to confirm the gap needs shared tokens |
| E2 | add+mul not slower: the 90 % CI's upper bound is below 1.5 | no slowdown ≥ 1.5× at this size; never "no conflict". If the upper bound is 1.5 or above, the result is inconclusive, not a null. Go to E3, and try a harder pair (add+sqab; add+s5 only after s5's 100k rerun) (corrected 2026-11-03, consolidation) |
| E2 | add+mul slower (the CI's lower bound above 1), with embedding-conflict signs | run `--operands separate` and `--answers per_task` arms to locate where the shared table can be split |
| E2 | add+mul slower, trunk-conflict signs | scale the trunk (d 256) or depth before concluding anything about tasks |
| E2 | add+sub faster than null-dup | positive transfer. Test it as a lower data threshold: `--task_frac sub=0.3` |
| E3 | add kept | run the staged-addition arm and longer switches |
| E3 | overwriting | protection is now tested inside E3 (A4, or A4′ against A5; added 2026-11-03, consolidation). If a 1 % loss protects add, see how small the data share can be: a data-share replay, `--task_replay add=5:0.0098` (1 % of rows at sub's per-row weight) |
| E3 | decay only | forgetting here is a weight-decay effect. Repeat with `--wd_scope all` against `repo`, to see which weights carry add |

### 4.6 Pitfalls

1. **Answer-class collisions.** A shared answer token must mean the same thing in every task.
   - Numbers mod the same p: fine.
   - S5 elements, parity bits, or gcd with a different range: not fine. Give them their own block (automatic for S5) or use `--answers per_task`.
   - Under `per_task` the answer rows are output-only, like the second-operand rows under `--operands separate`.
2. **Unequal table sizes.**
   - S5 has 14,400 pairs and 7,200 train pairs at 0.5; mul has 9,216.
   - The weighting above fixes the objective, but a bigger task still fills more of each minibatch.
   - Report per task. Pooled numbers are example-weighted and hide a small task.
3. **Compute and optimizer share per task.**
   - Two tasks at `--batch 1024` keep each task at about 512 examples per update, but AdamW normalizes the summed gradient. Each task's share of a shared weight's step therefore shrinks, so a slowdown of up to about 2× is expected with no conflict at all.
   - Batch 1024 also changes the gradient noise and how often clipping binds, against E1's 512.
   - Hence the null arms in E2: add alone at batch 1024, and add + add_dup. The second separates "two tasks share the model" from "these two tasks conflict".
4. **One task groks and starves another.**
   - On CUDA fp32 the target gradient can vanish below 16.6 at realistic logit spreads. Past it the non-target gradients survive (0.10–0.71 of the pre-cliff L2 norm). In effect, starvation comes from CE's exponential tail: at margin ~15 the exact gradient is already about e^-15 of a margin-0 example, cliff or not. Under `repo`, decay shrinks only the trunk (corrected 2026-11-03, consolidation; this said the target gradient is exactly zero past 16.6 while weight decay keeps shrinking).
   - The partner's gradient then owns the shared weights, and the first task can relapse. Jia 2026 reports relapse in transfer.
   - Watch `zero_tgt_grad_frac@t` in train mode (with dropout on), alongside each task's gradient norm (`grad_norm_task@t`): the zero-target share is not a complete starvation measure, since non-target gradients survive. Keep `margin_frac_cliff@t` as a logit-scale descriptor only. Also watch both forgetting readouts.
   - The opposite failure: a hard task whose loss never falls dominates the gradient. Equal weighting caps its share of the loss, not of the gradient.
5. **Shared operand tokens and the diagonal.** Revised after the add2b runs (2026-11-02, `diag_arms_report.md` evening addendum).
   - **The layer-1 story is necessary, not sufficient.** With shared tokens, (a, a) puts the same token at both operand positions, so layer 1 reads identical value vectors there, whatever the attention weights (RoPE touches only queries and keys). By layer 2 the positions differ through the causal mask.
   - **What the runs showed** (ordered split, 3 seeds each, test / train gap = off-diagonal minus diagonal mean margin):
     - a+b, shared tokens: 7.1–8.1 / 4.8–5.5;
     - a+2b, shared tokens: 0.9–1.3 / 0.7–0.8, with no diagonal errors, and the diagonals learned together with the other pairs;
     - separate tokens: about 0 for both tasks.
   - **In the tested tasks (a+b vs a+2b, ordered split, lr 6e-4, `repo`), commutativity and shared tokens interact strongly; necessity is not shown** (corrected 2026-11-03, consolidation; this said the large weakness needs both). E1's non-commutative tasks keep 0.6–2.5 nats (train, late) and up to 4.1 (test), so "about 1 point" is task-dependent. Untested at lr 1e-3. Why the combination matters is not identified. a+b and a+2b also differ in more than symmetry, so this is Moderate confidence.
   - **What this means here.**
     - Commutative tasks, add, add_dup, mul, sq, sqab and gcd, should carry the large gap.
     - Non-commutative ones, sub, div, s5, add2b and cubeab, should show a smaller gap; E1 found 0.6–2.5 nats on train, not "about 1 point at most" (corrected 2026-11-03, consolidation).
     - E1 pre-registers this test.
   - Read multi-task conclusions on unseen off-diagonal pairs, or check them under `--operands separate`. This matters most when a commutative task is in the mix.
6. **Swapped pairs leak more than the commutative tasks.** sub's (b, a) answer is the negated answer; div's is the inverse. The first draft compared sub with add under an `ordered` split. There add gets about 75 % of its table for free through swapped pairs, against 50 % for sub, so "sub is slower" would show up whatever the architecture. Every arm now uses `--pairs paired`, which removes swapped-pair information for all tasks.
7. **Order comes only from RoPE and the causal mask.** A non-commutative task must tell a from b without a position embedding. Whether that costs anything is what E1's sub/add and div/mul comparison measures, once swapped pairs are closed (pitfall 6).
8. **Missing answers.**
   - The start-up check uses each task's reachable answers.
   - mul over nonzero operands never answers 0.
   - gcd's large values sit almost only on the diagonal: use `--diagonals train`, and note that this puts every diagonal in train.
9. **bf16 readouts.** Under `--bf16 autocast`, logged margins are bf16-rounded, with a step of 0.125 near 16. Rescore checkpoints in fp32 for fine margin comparisons, as the diagonal report did.
10. **Run-to-run noise.** (Corrected 2026-11-03, consolidation.) Same-seed CUDA reruns are bit-identical on this machine (47 pairs, through spikes and collapses). A same-seed rerun checks determinism; it measures no noise. All noise is between seeds. The two-task path is untested: check it once.
11. **The dropout readout under `--tasks`.** It runs per task, so its masks are not the next update's. That holds only for one task under full batch.

## 5. First experiments

**Base settings.** These are the diagonal arms' and the add2b runs' (2026-11-02) `stack_lr6e4` settings, plus a paired split:

```
--device cuda --updates 25000 --ckpt_every 0 --wd_scope repo --dropout 0.1 --beta2 0.95 --clip 1 --batch 512 --bf16 autocast --lr 6e-4 --pairs paired
```

This base differs from train.py on dropout (0.1 vs 0), schedule (constant vs wsd) and batch (512 vs 64 × 4), and from the harness default in seven settings. It had the worst late accuracy of the three long-run LR settings, and no reason for choosing lr 6e-4 was recorded. Results are statements about this base (added 2026-11-03, consolidation). On the same base at lr 1e-3, the long runs have about half the diagonal gap at 1e5 updates, no unseen diagonal errors at the end, a different mechanism route mix, and a layer-1 MLP that decays to zero in 2/3 seeds (`RESULTS.md`, K2; lr and lr × wd are confounded there). So the lr choice is itself a result-shaping setting.

**Checkpoints** (`--ckpt_at`, 2.2 MB each, `final.pt` 6.5 MB; corrected 2026-11-03, consolidation). The original 41 runs came to about 0.8 GB; the revised plan's 56–59 runs (E2 at 22, E3 at 18–21) come to about 1.1 GB, inside the disk margin:
- E1: `--ckpt_at 500,1000,2500,5000,10000`, around and after the expected grokking times;
- E2: `--ckpt_at 1000,2500,5000,10000,15000`;
- E3: `--ckpt_at 4900,5000,5100,5500,7000,10000`, just before and after the switch. The 5,000 checkpoint is the branch-identity gate (§5 E3).

`final.pt` holds the last update in every run.

Seeds 0, 1, 2. About 4 min per single-task run on the RTX 4060 (measured 2026-11-02); two-task runs evaluate each task and take a bigger batch, so expect about 1.5–2× that. Time a 1,000-update smoke run of one two-task arm first. Run through the guard with `MAX_PROCS=1`. The per-worker RAM is not established: the memlog shows 3.3 (unit unlabelled), not the 1.4 GB stated here before; re-measure (corrected 2026-11-03, consolidation). Coordinate the GPU through `gpu_handoff`.

Under `--pairs paired`, the diagonal arms' add control and the add2b runs are not directly comparable to E1, since they used `ordered`. E1 has its own add arm. In particular, the diagonal has no swapped twin under either split, but under `ordered` many off-diagonal test pairs did. So test-side diagonal gaps are not comparable across the two splits; see E1.

### E1. Single-task catalogue

5 arms × 3 seeds, plus 1 rerun = 16 runs, about 1 h 05 min.

| Arm | Extra flags | Pre-registered prediction |
|---|---|---|
| add | `--tasks add` | groks (≥ 0.99 test) by update 2,000 in 3/3 seeds |
| sub | `--tasks sub` | groks by 25k in 3/3 seeds, later than add |
| mul | `--tasks mul` | groks by 25k in 3/3 seeds |
| div | `--tasks div` | groks by 25k in 3/3 seeds, later than mul |
| s5 | `--tasks s5` | may not finish by 25k. If not, rerun at `--updates 100000` (about 16 min per seed, budgeted) |
| add rerun | `--tasks add --seed 0` again | same-seed CUDA spread (historical: the rerun was bit-identical, so the spread is 0; corrected 2026-11-03, consolidation) |

**Questions it answers:**
- **Does each task grok here, and when?** Read `first_update_test_ge_0.99@t` and `test_acc@t` at 25k. A task under 0.99 at 25k gets a 100k run before anything is concluded (Power's budget).
- **Does operand order cost anything in a RoPE-only model?**
  - Compare sub with add and div with mul, all under `--pairs paired`.
  - "Order costs" if all 3 sub seeds are slower than all 3 add seeds and also outside the same-seed spread, and the same holds for div against mul. (Historical: the rerun showed bit-identity, so the same-seed spread is 0; corrected 2026-11-03, consolidation.)
  - The null (no cost) is a real possibility: Furuta's no-position-embedding model learned sub.
- **Diagonal gap by task (pre-registered 2026-11-02, before any E1 run).**
  - **Readouts:**
    - test gap = `test_margin_mean_unseen_offdiag@t − test_margin_mean_diagonal@t`;
    - train gap = `train_margin_mean_offdiag@t − train_margin_mean_diagonal@t`;
    - both are means over the last 10 % of evals, one value per seed, as in the diagonal report.
  - **Prediction, from pitfall 5:**
    - add and mul (commutative): **large**, every seed's train gap ≥ 3;
    - sub, div and s5 (not commutative): **small**, every seed's train and test gap ≤ 1.5;
    - anything in between counts as inconclusive, not as a hit.
  - **Why train is the main readout.** `--pairs paired` changes the test side compared with the earlier `ordered` runs:
    - Under `ordered`, about half of a commutative task's test off-diagonal pairs had their swapped twin (same answer) in train. The diagonal has no twin, so this raised the off-diagonal side of the test gap.
    - Under `paired` no test pair has its twin in train, so add's test gap may come out below 7–8 without contradicting anything.
    - The train gap does not depend on twins being split. E1's own add arm is the reference for both.
  - **mul's diagonal** answers a², which takes only 48 values: a and −a share a square. Read its diagonal accuracy alongside the margin.
  - **s5 has a few commuting pairs** (7/120 of ordered pairs, including the diagonal). That is too few to change the prediction.

**E1 results (run 2026-11-02, 21:13–22:21, 16/16 done; one RAM shed, rerun cleanly).** Full table: `grok_runs\e1_2026-11-02\e1_report.md`.
- **Grokking:** all five tasks reach a *first* crossing of test accuracy 0.99 by 25k (corrected 2026-11-03, consolidation). First update at 0.99:
  - add: 1,200–2,000 (sustained 1,500/2,300/2,000; add_s1 holds only from 2,300). Off-diagonal: 800/1,100/1,100 first, 1,200/1,300/1,100 sustained;
  - sub: 900–1,300;
  - mul: 2,000–3,000;
  - div: 1,000–1,500;
  - s5: 1,500, 3,700 and 14,600. s5_s2 never sustains 0.99 (final 0.9864; 2/26 late evals ≥ 0.99), so the 100k rule (§4.5, E1 table) fires. "s5 groks" holds in 2/3 seeds until that rerun.
- **Order** (corrected 2026-11-03, consolidation; this said "order costs nothing"): no order cost large enough for 3 v 3 to detect. On off-diagonal first crossings, sub is later than add in 3/3 seeds (+17 %, p = 0.15) and div later than mul in 2/3. Sustained off-diagonal crossings tie for add/sub on average (ratio 0.97: sub earlier in s0, equal in s1, later in s2). "sub/div faster" was a diagonal artefact of the commutative partners' test times. The order branch (§4.5) was neither triggered nor excluded.
- **Diagonal: the commutative half is a hit, the non-commutative half is inconclusive.**
  - add and mul have train gaps of 4.5–5.8 in every seed, and all the diagonal errors (apart from unstable s5 seed 2).
  - sub, div and s5 have train gaps of 0.6–2.5 and test gaps of 0.8–4.1. That is above the pre-registered 1.5 in several seeds.
  - The two groups still separate cleanly: the lowest commutative train gap (4.46) is above the highest non-commutative one (2.54). This is post hoc. It holds on means and at the final eval (a 1.08-nat margin there), not on single-eval maxima, and is untested at lr 1e-3 (added 2026-11-03, consolidation).
- **Determinism:** the same-seed CUDA rerun of add matched seed 0 bit for bit (251 evals, 2,542 logged spikes). This measures determinism, not noise (corrected 2026-11-03, consolidation).
- **Spectra:** add and sub put 0.57–0.85 of the embedding's power in their top-5 Fourier frequencies along a; mul and div put 0.54–0.89 in their top 5 along log a. Each other share is 0.107–0.120, at or below chance (about 0.12, Monte Carlo 0.116–0.125; corrected 2026-11-03, consolidation). So the readout tells the two algorithm families apart, as §4.4 intended.

**Why the diagonal is weak: mechanism study (2026-11-03).**

- **Files:**
  - probe: `tools/grok_mech_probe.py`;
  - the whole study (hypotheses, reports, blind Codex analyses, code, raw results): `docs/research/grok-diagonal-mechanism-2026-11-03/`, with an index in its `README.md`. The files below are there under `reports/`;
  - hypotheses, written before each run: `hypotheses.md`;
  - tables: `results_tables.md`;
  - a blind Codex review (gpt-6.1-sol) and its comparison with this session's analysis: `codex_results_review.md`, `comparison.md`;
  - the mirror test: `mirror_summary.md`.
- **Setup:** 27 runs were probed at final weights, plus E1's checkpoints at updates 500 to 10,000:
  - E1 ×15;
  - diag-arms control and separate-token ×6;
  - a+2b shared and separate-token ×6.

  The probe's margins match the logged ones within 0.03. Its margin split is exact to 4e-5.
- **Ruled out** (each pre-registered prediction failed at its threshold, the same verdicts in the blind review):
  - the operand token competing with the answer: it is the runner-up on 0–2 % of diagonals, and removing operand classes changes nothing;
  - layer-0 attention computing in its a/b ratio: forcing it to 50/50 changes the gap by ≤ 2 %;
  - one component carrying the deficit;
  - the gap appearing only with generalization: it is already 2.0–2.5 at update 500, and ≥ 1 at update 100.
- **Two phases:**
  - **Early:** the gap sits in the last MLP, and attention interventions do nothing.
  - **Late:** layer-1 attention adds to it. On diagonals, head-mean layer-1 operand mass drops 8–25 % (median 18 %), onto `op` and `=`. Equalizing the a/b ratio lowers off-diagonal margins by 1.6–2.3 nats (32–49 % of the gap) and moves diagonals ≤ 0.006 (corrected 2026-11-03, consolidation; this said heads move "about a quarter" of their weight and the ratio "removes 32–49 % of the final gap").
  - **Correction (H6c, below):** raising layer-1 operand attention helps diagonal and off-diagonal pairs about equally (registered MIXED). Mass and ratio are different quantities: the ratio test never moved diagonals, and the mass test shows no diagonal-specific effect at final weights. Neither shows nor rules out a layer-1 cause (narrowed 2026-11-03, consolidation).
- **Mirror test** (ordered split: off-diagonal train pairs with vs without their swapped twin in train; R = (lone − diagonal) / (mirror − diagonal)):
  - **Final weights:** R = 0.96 / 0.89 / 0.91.
  - **Over training** (CPU fp32 reruns with checkpoints, which act as new seeds): R is about 0.22 at update 100, 0.56–0.68 at 200, 0.82–0.88 at 300, and 0.87–0.95 afterwards.
  - **Control:** separate tokens show no twin effect (≤ 0.11).
  - **Reading:**
    - during memorization, the model treats (a, b) and (b, a) as one fact, so twinned facts learn faster and the twinless diagonal lags;
    - exposure explains most of the gap at update 100 and 32–44 % at 200. It fades by about 500 (R 0.87–0.90) but leaves a 5–15 % residual through 10k (corrected 2026-11-03, consolidation; this said the effect is gone from about update 300). The diagonal falls further behind (gap about 1 → 3.5–5.3) mostly because of how the symmetric (a, a) input is handled.
  - **By the registered scoring the verdict is mixed across seeds,** because its noise gate (gap ≥ 1) hides different early checkpoints in each seed; the three curves match.
- **Follow-up tests H6–H10** (final weights of 12 runs: E1 add and mul, diag-arms control and separate-token, 3 seeds each; each pre-registered, ambiguous results reviewed blind by Codex):
  - **H6, H6c (restore):**
    - raising layer-1 operand attention with the final norm fixed helps diagonals and off-diagonals about equally (≤ 0.39 nats; registered MIXED): no diagonal-specific effect at final weights (corrected 2026-11-03, consolidation; this said "so it is not the limiter");
    - with the norm free, it hurts diagonals more;
    - swapping the final-norm scale closes 15–30 % of the gap, accuracy unchanged. This is under `repo`, which leaves the norm gains undecayed (added 2026-11-03, consolidation).
  - **H7 (residual):**
    - layer-0 attention writes √2 more on (a, a) in all 9 shared-token runs (1.41–1.44; separate tokens 0.93–1.08): the two identical operand values add coherently;
    - diagonals carry more energy the answer readout can't see (9/9), but removing it closes ≤ 7 % of the gap.
  - **H8 (gain):** shrinking that operand write to off-diagonal size *lowers* diagonal margins (0/9 for the doubling-cause account; the alternative holds 9/9). But off-diagonals pay a matched cost: net of it, the gap narrows in 6/9 (≤ ~12 %), and with the norm free the diagonal margin rises by 1–16 % of the gap in 9/9, but net of the off-diagonal change the gap narrows in only 2/9 (−10.8 % to +5.5 %; corrected 2026-11-08: the 1–16 % counted the diagonal change alone). Size alone doesn't selectively rescue diagonals; at final weights it is a minor factor, not ruled out (corrected 2026-11-03, consolidation; this said "its size is not the cause").
  - **H9 (transplants):** swapping parts of the computation between pairs with the same answer costs 3–7 nats even between healthy off-diagonal pairs, so transplants cannot localize the deficit. Verdict mixed; design flaw recorded.
  - **H10 (route-resolved integrated gradients):** the deficit splits about 2/3 through the '=' stream after layer 0 (60–85 % in 8 of 9 runs) and 1/3 through layer 1's reads of the operand positions; the op position carries 0. mul_s0 is the exception (57–59 % operand reads). Registered verdict mixed (7 "unresolved", 2 "'='"); the registration had no "split" category. Per pair the routes are noisy; the split is an average.
  - **Scope of H10** (added 2026-11-03, consolidation): these shares are for lr 6e-4 at 25k updates. On long-stack runs that differ only in lr (1e-3, scored at 1e5), the operand route carries 103–131 % of the deficit and the '=' route is negative (−3 to −31 %) in 3/3 seeds, and the layer-1 MLP has decayed to zero in 2/3. Even at 6e-4, by 1e5 updates the '=' share is 48–70 %. The routes cancel heavily per pair, so these are path averages (`RESULTS.md`, K2b; Codex agreed blind).
  - **Reading (inferred, path-averaged; corrected 2026-11-03, consolidation):** averaged along one integration path, about 2/3 of the deficit flows through the '=' state after layer 0. That state includes the layer-1 query and the '=' self K/V, so "layer-0 content" is not localized. Exposure fades early but leaves a residual. Layer-1 attention and size show no diagonal-specific effect at final weights. Invisible energy accounts for ≤ 7 % at final weights. Training-time roles are untested, and every model is at lr 6e-4, `repo`, dropout 0.1. (This said: what (a, a) writes into '=' at layer 0 is the wrong content, not the wrong size.)
- **Open:**
  - finer attribution: the '=' route into residual / query / self K-V, the operand route into K / V. Run it on the 6e-4 and 1e-3 models together, after a per-pair path replay on the saved checkpoints (added 2026-11-03, consolidation);
  - a 2× diagonal-weight training arm, to test whether the deficit is a learned trade-off or a structural limit. It waits for the scope and lr checks, and needs a matched off-diagonal control (added 2026-11-03, consolidation). Its lower priority in the consolidation rested partly on "content, not size" looking weakened; that weakening used the misread norm-free figure (see H8, corrected 2026-11-08), so the size account is weaker, and this arm's motivation less weakened, than the consolidation said.

**E1b,** after E1: `sq`, `sqab`, `cubeab` × 3 seeds, 9 runs.
- **Predictions:** sq groks; sqab is open (Power yes with more data, Furuta no at 0.3); cubeab does not reach 0.99 in 25k.
- **Diagonal predictions** (same rule as E1): sq and sqab large, since they are commutative. cubeab small, if it learns at all.
- **If cubeab groks,** the MLP-based prediction is wrong for this model. Report that as a result, after the usual record checks.
- **Also:** `--tasks add --answers per_task`, the control for option B.

### E2. Two tasks at once

All arms use `--batch 1024`, so each task keeps about 512 examples per update. The split is aligned (the default).

| Arm | Flags | Seeds | Role |
|---|---|---|---|
| null-batch | `--tasks add --batch 1024` | 3 | add alone at E2's batch size; exploratory (unpaired with the two-task arms) |
| null-dup | `--tasks add,add_dup --batch 1024` | 8 | the cost of sharing when the tasks agree |
| add+sub | `--tasks add,sub --batch 1024` | 3 | exploratory |
| add+mul | `--tasks add,mul --batch 1024` | 8 | headline |

22 runs (corrected 2026-11-03, consolidation: null-dup and add+mul go from 5 to 8 seeds for the paired rule in §4.3). At about 6–8 min per two-task run and 4–5 min per single-task run at batch 1024, that is about 2.1–2.8 h, against 1.5–2.0 h for the old 16-run plan. Before launch: the CPU pairing check (§4.3), and the 1,000-update smoke run, run twice and compared bitwise, which also times the two-task arms.

**Hypothesis:**
- add and sub share Fourier number embeddings (Furuta's add↔sub embedding transfer), so add+sub should cost no more than null-dup.
- mul's natural embedding is periodic in log a, so add+mul should cost more than null-dup.
- Xu saw add+mul slow down about 3.5–6.5×, but with separate heads and an uncontrolled baseline.

**How to read it** (corrected 2026-11-03, consolidation):
- add's unseen off-diagonal grokking time in add+mul and add+sub, against null-dup. The primary outcome is the sustained crossing; the first crossing is reported separately (§4.3). add's test time is set by its diagonals: every post-grok dip in E1's add/mul runs is diagonal-only (12/12), and the diagonals delay add's whole unseen-test crossing by 400–900 updates past the off-diagonal one (E1 add: 400 / 600 / 900), the size of the effect E2 looked for. (Corrected 2026-11-05: this was called "the diagonal lag", but the diagonals' own crossing lags the off-diagonal one by 2,900–13,900 updates in E1 add, which is what E2's registered "diagonal lag" measures. The mislabel fed my 1.3–3× guess for E2, which came out at 8×.) Report diagonal accuracy, lag and gap as separate outcomes.
- sub's and mul's times against E1, scaled by the null-batch/E1 ratio for add. This assumes the batch effect is the same for every task; read them against E1 directly too, with the ratio as a sensitivity check.
- add+sub and null-batch keep 3 seeds, so they are exploratory: estimates with 90 % CIs, no verdicts (added 2026-11-03, consolidation).
- The decision rule is the paired t with a 90 % CI on the geometric-mean ratio, with the sign test alongside (§4.3).

**Outcomes** (corrected 2026-11-03, consolidation; these were read with the range rule):
- add+mul slower than null-dup (the 90 % CI's lower bound above 1) and add+sub not: the conflict is task-specific, which is the embedding hypothesis. "add+sub not slower" is confirmatory only if add+sub also runs 8 seeds; at 3 it is an estimate.
- add+mul not slower: "no slowdown ≥ 1.5×" only if the CI's upper bound is below 1.5. Otherwise, "no slowdown detected; slowdowns up to the upper bound are not excluded". Never "sharing costs no more than any second task" (the old wording).
- Both slower: a generic two-different-tasks cost. The add+mul vs add+sub contrast isn't powered at any seed count considered, so keep it descriptive.

**Follow-up:** `--answers per_task` on add+mul. If that removes the extra cost, it sat in the shared output rows.

### E3. Sequential training and forgetting

The first design was 3 arms × 3 seeds = 9 runs at `--batch 512`, about 36 min, with one task training at a time in every arm:

```
--tasks add,sub --task_stop add=5000 --task_start sub=5000
--tasks add,sub --task_stop add=5000 --task_start sub=5000 --wd_until 5000
--tasks add,sub --task_stop add=5000 --task_start sub=25000 --allow_idle
```

**Revised 2026-11-03 (consolidation): controls added.** The specification, with code sketches and tests, is `grok_runs\checks_2026-11-03\e3_controls_spec.md`. **Its two code changes (`--task_replay` and `--op_classes`) and the op-row logging are sketched there, not implemented.**

- **The small-weight control and its sampling confound.** The sampler draws uniformly from the pooled train rows of the active tasks; `--task_weight` changes only the loss coefficients. With add kept active at 0.01 after 5,000 (no `--task_stop add=5000`), the pool is 9,408 rows, half of them sub's, so sub gets 256 ± 11 rows per update (95 % of updates in 234–278) instead of exactly 512. sub's expected loss scale barely changes (×0.990), but each sub row carries about twice the weight, the estimate rests on half as many rows, and each sub example is visited once per 18.4 updates instead of 9.2. So the confound is exposure and gradient noise, not a smaller sub loss. Before 5,000 the arm is unchanged bit for bit. Two ways to deal with it:
  - **Available now: a sham-weight arm.** `--task_start sub=5000 --task_weight add=1e-12` draws the same rows as the 0.01 arm every update (same pool and sampler seeds, same batch shapes, so the same dropout masks), with add's gradient effectively zero. The 0.01 arm against the sham isolates rehearsal at half of sub's exposure; the sham against the plain switch measures what halving sub's rows does on its own. The remaining confound: the rehearsal effect is measured at half exposure, and need not carry over to a 512-row switch.
  - **Exact match, a small code change: `--task_replay name=rows:weight`** (about 35 lines plus tests). After the task's `--task_stop`, it stays in every update with its own epoch-shuffled rows at an absolute loss weight, outside the active tasks' batch, pool and weights. The replay forward runs under a forked RNG. So sub gets the switch arm's identical row sequence at coefficient exactly 1, and the arm differs from the switch only by the term 0.01 · CE_add(512 rows). Recommended arm: `--task_stop add=5000 --task_start sub=5000 --task_replay add=512:0.01`, about 1.3–1.8× the cost per update. Key test: weight 0 reproduces the switch arm bit for bit on CPU.
- **Matched no-switch continuation: `--task_start sub=25000`.** sub stays in the layout (vocab 100, the same answer columns and scoring as every E3 arm) but never trains, so add trains alone for the whole run. E1's add (vocab 99) is not a valid baseline: the vocabulary changes the init and the answer competitors.
  - **A shared branch at 5,000, by determinism.** The harness has no resume. But every add-first E3 arm (A0–A5; A6 trains sub from scratch) runs the identical computation through update 4,999 (same seed, vocab, init, optimizer and phase-1 sampler; the flags first act at 5,000 or later), and same-seed CUDA runs are bit-identical. So every add-first arm reaches update 5,000 with bitwise-identical weights, Adam state, sampler position and RNG state. **Gate, per seed, before reading any arm:** `torch.equal` on `ckpt_005000.pt` across the add-first arms, and every trace and metric value at or before 5,000 identical (timings and `active_tasks` aside). A6 can't share this branch: compare it by initialization only, as the transfer readout does (corrected 2026-11-08). All arms run on the same machine, torch/CUDA build and commit, on CUDA; never mix CPU and CUDA arms.
  - **So seeds are paired across E3 arms,** and seed-paired differences against the continuation are legitimate.
- **A sub-from-scratch baseline: `--task_start add=25000`.** sub trains alone from update 0 in E3's layout. It replaces "sub in E1" (vocab 99) as the transfer baseline.
- **Op rows: `--op_classes own`** (a small code change; default `all` keeps every existing run unchanged). Today each task's answer columns include every op token, so after the switch sub's softmax keeps pushing add's op row down through the tied head (never decayed under `repo`), and before the switch add's softmax shapes sub's op row. Under `own`, each task's classes are its answer block, its own op token and '=' (99 classes, the same count as E1). Single-task runs are unchanged by construction. Log `op_row_norm@t`, `op_row_grad@t|from_s` and the op row's Adam moments at every eval, with the number and '=' row norms as a scale reference.

**Arms** (base: the E-series flags plus `--tasks add,sub --op_classes own --ckpt_at 4900,5000,5100,5500,7000,10000`; seeds 0–2):

| # | Arm | Arm flags | Runs | ≈ GPU min | Decides |
|---|---|---|---|---|---|
| A0 | continuation | `--task_start sub=25000` | 3 | 13 | add's own retention in E3's layout; every "add lost" is called against it, seed-paired |
| A1 | switch | `--task_stop add=5000 --task_start sub=5000` | 3 | 13 | does add drop when sub replaces it |
| A2 | switch, no decay | A1 + `--wd_until 5000` | 3 | 13 | does the drop need post-switch decay (read sub's own learning too) |
| A3 | idle | `--task_stop add=5000 --task_start sub=25000 --allow_idle` | 3 | 8 | does decay plus the momentum tail alone remove add (under `repo`: trunk matrices only) |
| A4 | replay, exact (needs `--task_replay`) | A1 + `--task_replay add=512:0.01` | 3 | 18 | does a 1 % add loss protect add at exactly A1's sub exposure |
| A4′ | replay today (if no code change) | `--task_start sub=5000 --task_weight add=0.01` | 3 | 13 | the same at half of sub's exposure; read only against A5 |
| A5 | sham (only with A4′) | `--task_start sub=5000 --task_weight add=1e-12` | 3 | 13 | the exposure effect (A5 vs A1); pairs with A4′ |
| A6 | sub from scratch | `--task_start add=25000` | 3 | 13 | sub's grokking time in E3's layout without add first: the transfer baseline |

- **With `--task_replay`** (A0–A4, A6): 18 runs, about 80 min. **Without it** (A0–A3, A4′, A5, A6): 21 runs, about 90 min. The first design was 9 runs, about 36 min. No two-task run has been timed yet.
- **`--op_classes own` is in every arm's base,** so it has to be implemented and tested before E3 launches. The fallback is to run with the current behaviour (`all`) and read the op rows through the op-row logging (also still to be added), accepting that sub's loss trains add's op row after the switch.
- **Boundary smoke run first:** seed 0, every arm with the switch moved to 500 and `--updates 1000`, plus a second run of A1. Check the add-first arms bitwise through update 500 (A6 at initialization only), the A1 rerun bitwise everywhere, and the readouts after the switch. About 2–3 min; it also times the arms.
- **Follow-ups, not in the table:** A1 and A3 at `--wd_scope all` and `embeddings`, if A3 shows any loss.

**Read** (revised 2026-11-03, consolidation):
- **Primary:** add's unseen off-diagonal accuracy (`test_acc_unseen_offdiag@add`). **"add lost":** its minimum over the evals in 22,500–25,000 is below 0.99 while the same seed's A0 is at or above 0.99. Report the paired difference to A0.
- **Secondary:** the first eval after 5,000 below 0.99 (censored at 25,000); add's margins; the diagonals, reported separately, since add's diagonals dip to 0.83–0.89 when it trains alone.
- **Transfer:** sub's first and sustained off-diagonal crossing, measured from its own start, against A6 (paired by init only, so read it as a distribution).
- **Logged in every arm, not outcomes:** add's top-5 Fourier power, `grad_cos_*@add|sub`, `grad_norm_task@*`, `zero_tgt_grad_frac@add`, the op-row keys, and the per-layer weight norms (§4.4).
- Three seeds per arm detect conspicuous, consistent effects only. "Kept" or "no protection" is not a null claim without an interval or a power statement.

**Pre-registered reading of the first three arms.** The "Reading" column lists candidate mechanisms, not observations: add retained does not show that sub reused add's circuit, and removing decay can change sub's learning as well as add's retention (noted 2026-11-03, consolidation). Each "kept" and "lost" is now judged against A0. The spec's reading table (§5 there) supersedes this one: it keeps observed outcomes apart from candidate mechanisms, adds the A0, replay, sham and A6 rows, and lists the readouts that tell the mechanisms apart.

| switch | switch, no decay after 5,000 | idle (decay only) | Reading |
|---|---|---|---|
| add kept | — | — | sub reuses add's circuit without destroying it |
| add lost | add lost | any | overwriting by sub's gradients |
| add lost | add kept | add lost | disuse plus decay; sub does not overwrite add |
| add lost | add kept | add kept | sub's training and decay together erode add; neither alone does |

**Staged addition** (add kept, sub joining at 5,000) runs after E3 at `--batch 1024`, with null-dup's staged twin as its null.

**Total:** the first design was E1 + E2 + E3 = 41 runs, about 3.5–4 h of GPU time. As revised on 2026-11-03 (consolidation): E1 16 (done) + E2 22 + E3 18–21 = 56–59 runs, about 4.5–5.5 h with E1's 1 h 05 min (about 3.5–4.5 h still to run), plus the smoke runs. E1 alone answers question (1) for five tasks and is the baseline for (2).

## 6. Red-team review and changes

A Fable reviewer read the draft, the code and the tests. Its findings, and what changed:

| # | Finding | Severity | Change |
|---|---|---|---|
| 1 | sub against add was confounded by swapped-pair leaks (add gets about 75 % of its table under `ordered`); the decision rule nearly could not fail | high | every arm uses `--pairs paired`; same-seed rerun; pre-registered predictions (§5 E1, pitfall 6) |
| 2 | E2 had no null, so a ~2× slowdown from sharing Adam's step would read as interference; E1 used batch 512 against E2's 1024 | high | null-batch and null-dup arms; new `add_dup` task (§5 E2) |
| 3 | E3's decay-timescale argument could not separate disuse from overwriting; `--wd_scope repo` exempts add's op row; the small-weight control was empty | high | `--wd_until` and `--allow_idle` (idle, decay-only phases) in the code; three-arm E3 with a reading table |
| 4 | same-seed runs are not paired across arms with different vocabularies | medium | unpaired comparison rule (§4.3); superseded 2026-11-03 (consolidation): equal-vocab arms are init-paired |
| 5 | 3 against 3 seeds gives about 10 % false positives for "either direction" | medium | stated; 5 seeds for the headline arms. Superseded 2026-11-03 (consolidation): 8 seeds and a paired interval rule (§4.3) |
| 6 | the staged arm at batch 512 halves sub's data per update | medium | staged arm at 1024 (§4.2, §5 E3) |
| 7 | rules that cover every outcome (cubeab "re-check the encoding") | low | predictions written before the runs |
| 8 | `aligned` did not align add with mul, since the shuffle ran over each domain | medium (bug) | `make_split` shuffles all p² pairs, then applies the domain; test added |
| 9 | the sqab factorization is inference and does not imply learnability | low | relabelled |
| 11 | literature overstatements: Xu's slowdown, "pairwise matrix", Furuta's no position embedding, negatives from abstracts | medium | corrected (§2, Summary) |
| 12 | the test status pointed to a note that did not exist | low | recorded under Prototype |
| 13 | no bug found in the `--tasks add` ≡ `--task add` path | — | none |
| 14 | `equal` weighting renormalized over the tasks present in a minibatch, giving a 3-row task half the loss | medium (bug) | unbiased fixed-weight estimator (§4.2); test added |
| 15 | default `--task_split independent` was the leaky mode | low | default is now `aligned` |
| 16 | model-wide readouts recomputed once per task | low | computed once (`model_stats`) |
| 17 | `active_tasks` missing at update 0; the dropout-readout claim no longer holds under `--tasks`; one spike could read as forgetting | low | added at update 0; docstring note; `min_test_acc_last10pct@t` |
| 10 | 1/(lr·wd) arithmetic | — | correct; see #3 for why it was misapplied |

## Prototype

Landed together with this doc.

- **`tools/grok_lab.py`:**
  - a task registry (`TASK_SPECS`), with domains and S5;
  - a vocabulary layout (`make_layout`);
  - `make_split(..., valid=)`, which shuffles all pairs and then applies the domain; the old split is unchanged when every pair is valid;
  - `setup_tasks`, which builds per-task views;
  - `TaskSchedule`, which handles start and stop, idle phases and pooled minibatches;
  - `multi_loss`, a fixed-weight objective with an unbiased minibatch estimate;
  - `evaluate_tasks` with `model_stats`;
  - `summarize_tasks`, which adds grokking time and the two forgetting readouts per task;
  - `--wd_until` (any run);
  - the diagnosis readouts: `embedding_spectra` (along a and along log a, through `primitive_root`) in `model_stats` for every run, `task_grad_cosines` for runs with two or more tasks, and `--ckpt_at`.
- **Single-task code is unchanged in behaviour.** `answer_logits` and `operands_of` each gained one check for a task view; `train_step` takes an optional loss function; `evaluate` can skip the model-wide readouts.
- **`tests/test_grok_lab_tasks.py`:**
  - `--tasks add` is bitwise equal to `--task add`: weights, every metric, the record and the error table. Variants: shared and separate operands, minibatch with clipping, dropout with test diagonals, paired split.
  - every task's answers brute-forced at p = 7;
  - S5 is a group table and is not commutative;
  - the vocabulary layout;
  - domain-respecting splits;
  - aligned splits holding across add and mul;
  - aligned versus independent;
  - a three-task run (add, div, s5);
  - schedules, full batch and minibatch;
  - the loss weighting, full batch, minibatch and an absent task;
  - idle phases;
  - `--wd_until`;
  - per-task answer tokens;
  - gcd's start-up check;
  - spectra recover planted add-like (frequency along a) and mul-like (frequency along log a) tables, and are logged with and without separate operands;
  - the gradient cosine matches a direct computation and leaves training bit-identical;
  - `--ckpt_at`;
  - the forgetting summary;
  - bad-flag errors.
- **Test status:**
  - First draft: 71 tests passed (43 existing, 28 new) in two runs, CPU only, through the memory guard with a 7.5 GB disk floor. The owner approved 7.5 GB for CPU test runs in this design session; GPU runs keep the normal 10 GB. The first run was shed once for low RAM and rerun.
  - After the red-team fixes: 77 passed (43 existing, 34 new) in 3 min 27 s, CPU only, through the guard on the same settings.
  - After adding the diagnosis readouts: 85 passed (43 existing, 42 new) in 4 min 56 s, same settings.
  - Each test run writes about 200 MB of temporary output. Delete it afterwards: with about 8 GB free, one earlier run hit the guard's 7.5 GB disk floor and was stopped. All test outputs have been deleted. On 2026-11-02 at about 17:00, C: had 7.2 GB free, below both floors, so no further guard run of any kind until space is freed.

**Not in the prototype:**
- sequence tasks;
- an oversampling knob;
- a per-task batch size (`--batch` is a total);
- a "pair seen in another task's train" tag;
- `--holdout_line` with `--tasks`;
- planned on 2026-11-03 (consolidation), not yet written: `--task_replay` and `--op_classes own` with the op-row keys (sketched in `e3_controls_spec.md`), per-layer MLP and attention weight norms, a train-mode diagonal split, and a sustained grokking-time field in the record.

## Open questions for the owner

1. ~~Run E1 now, or wait for the add2b result?~~ Answered: add2b finished on 2026-11-02, and its result is built into pitfall 5 and E1's pre-registered diagonal predictions. E1 can run whenever the GPU and disk allow.
2. Should the prototype go into PR #281 or a new stacked PR? It touches the same file.
3. Phase 2 (sequence tasks: parity, sort-i, reverse-i) needs `max_seq_len` and a sequence split. Is it worth building before E1 and E2 have answered the 4-token questions?
