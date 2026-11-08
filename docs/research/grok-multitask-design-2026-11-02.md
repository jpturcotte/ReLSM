# Grokking beyond a+b: more tasks, and several tasks in one model

Design for extending `tools/grok_lab.py`, 2026-11-02. It builds on PR #281 (branch `feat/grok-lab-diag-arms`, commit b04662c). A prototype `--tasks` option and its tests come with this doc. A Fable red-team reviewed the first draft; §6 lists its findings and the changes made.

## Summary

- **Recommended first experiment.**
  - **Arms:** five single-task arms, `add`, `sub`, `mul`, `div` and `s5`, 3 seeds each, at the diagonal arms' settings (`stack_lr6e4`, 25k updates).
  - **Split:** every arm uses `--pairs paired`, so no arm learns a test pair from its swapped twin. add would get that for free; sub and div get the negated or inverted answer from it.
  - **Extra run:** one same-seed CUDA rerun of add, to size the run-to-run noise.
  - **What it answers:** whether this model groks each task, and how fast. Every multi-task result needs these grokking times as its baseline.
  - **Cost:** 16 runs, about 1 h 05 min of GPU time.
  - **Long reruns:** s5 is likely to need 100k updates (about 16 min per seed), so that rerun is budgeted from the start.
  - Exact flags are in [E1](#e1-single-task-catalogue).
- **Then:**
  - E2: add+sub and add+mul, read against two null arms, add alone at the same batch and add paired with a copy of itself;
  - E3: a switch from add to sub, with two controls that separate "unused and decayed" from "overwritten".
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
    - **`--task_stop add=5000 --task_start sub=25000 --allow_idle`:** add, then nothing. sub stays in `--tasks`, so the vocabulary matches the other arms, but its start equals `--updates`, so it never trains (without it, sub would train from update 0). The idle updates are zero-gradient steps, so they apply only weight decay plus Adam's dying momentum (about 0.9ᵏ for β1 = 0.9). This is the decay-only curve.
  - A small-weight control (`--task_weight add=0.01`) was dropped. Past the fp32 cliff add's target gradient is exactly zero anyway, and Adam's normalization drowns a 1 % loss share.
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
  - `first_update_test_ge_0.99@t`: under `--pairs paired`, and for any non-commutative task, this is its grokking time.
  - `first_update_unseen_offdiag_ge_0.99@t`.
  - `first_update_unseen_ge_0.99@t` (diagonals included).
- **Forgetting or relapse.**
  - `min_test_acc_after_first_ge_0.99@t`, the lowest test accuracy after the task first grokked. One low eval can be a loss spike.
  - `min_test_acc_last10pct@t`, the lowest over the last 10 % of evals. A low value here, or a run of low evals, is forgetting.
- **Interference (from the analysis, not logged).**
  - **Readout:** a task's grokking time with a partner, against the same task's time in the matching null arm (§5 E2), never against a different batch size.
  - **Seeds are not paired across arms.** A different vocabulary size changes the embedding init and every random draw after it. So compare the seed distributions, and do not divide seed by seed.
  - **Decision rule.** Call an effect only when every partner-arm seed falls outside the null arm's range. With 3 against 3 seeds, that happens by chance with probability 1/20 for each direction, about 10 % for "either direction". So the headline comparison (add+mul against add+add_dup) uses 5 seeds per arm, where the chance falls to 1/252 for each direction.
- **Margin interference:** a task's `train_margin_p5@t` and `test_margin_p5_unseen_offdiag@t` before and after a partner joins.
- **Diagonal gap per task:** `test_margin_mean_unseen_offdiag@t − test_margin_mean_diagonal@t`, as in the diagonal report.
- **Errors:** `final_errors.json` has a `task` column, and `n_wrong_final.by_task` in the record.

### 4.4 Diagnosis: readouts that say why, not only whether

The outcome readouts above say whether a task grokked, when, and whether it was lost. These say why. All are logged at every eval and need no extra disk.

| Readout | Measures | Diagnoses |
|---|---|---|
| `emb_fourier_spectrum`, `emb_fourier_top5_frac`, `emb_fourier_top5` | power of the number embedding by frequency along a (k = 1..48 at p = 97). A grokked a+b concentrates it in a few frequencies (Nanda 2023); a random table gives a top-5 share of about 0.10 | which structure the table holds; when it forms relative to the grokking time |
| `emb_logfourier_*` | the same along log a (a = 5^j mod 97, a ≠ 0). (Z/97)* is cyclic, so a Fourier solution for a·b concentrates here instead | the add/mul conflict, measured directly. add alone should be concentrated along a and spread along log a; mul alone the reverse; add+mul shows whether the table holds both, one, or neither |
| `*_b` versions | the second operand's block under `--operands separate` | whether the two operands build the same structure |
| `grad_cos_all@i\|j`, `grad_cos_trunk@i\|j` | cosine between two tasks' full-train-set gradients: all weights, and the trunk without the embedding table and head | conflict, directly: negative means the tasks pull shared weights apart; near zero means separate directions; positive means they help each other. Trunk against all says whether the conflict sits in the table or in the trunk |
| `grad_norm_task@t` | each task's gradient norm | starvation: a task whose gradient is near zero (past the fp32 cliff) is no longer defending its weights |
| `--ckpt_at` | weights at a few chosen updates (about 2 MB each) | anything not logged, after the run: attention patterns at '=' for diagonal pairs, Nanda's restricted and excluded loss, probing |

How they combine with the outcome readouts:
- **E1:**
  - **Expected:** add and sub put their top-5 share along a; mul and div along log a.
  - **If a task groks without either concentrating,** it found a different algorithm. Check its checkpoints before comparing it with the others.
  - **If sub is slower than add,** see whether its spectrum forms late (the embedding is the bottleneck) or early with a late grok (the trunk is).
- **E2:**
  - **Embedding conflict:** add+mul is slower than null-dup, the table shares power along both a and log a without concentrating in either, and `grad_cos_all@add|mul` is below `grad_cos_trunk@add|mul`.
  - **Trunk conflict:** the slowdown comes with negative trunk cosines while the table concentrates normally.
  - **Readout conflict:** neither of the above, but `--answers per_task` removes the slowdown.
- **E3:** after the switch, follow the power at add's own top-5 frequencies (read from `emb_fourier_top5` just before update 5,000) in `emb_fourier_spectrum`.
  - **Decay:** all of add's frequencies shrink together while their share stays high, and the idle arm looks the same.
  - **Overwriting:** add's frequencies lose share to new ones, sub's if they differ, and the `--wd_until` arm looks the same.

### 4.5 Decision tree

| After | Result | Next |
|---|---|---|
| E1 | a task under 0.99 at 25k | rerun it at 100k before using it anywhere. If it still fails, drop it from E2 and report it |
| E1 | sub or div slower than its partner, with its spectrum forming late | order costs here. Run sub with `--operands separate` (the two operands then have different tokens) to see whether the cost is about telling positions apart |
| E1 | diagonal gaps follow the prediction: add and mul large, sub, div and s5 small | commutativity plus identical tokens explains the weakness across tasks, not just a+b vs a+2b. Look for the mechanism in checkpoints: compare attention at '=' for (a, a) on add against sub |
| E1 | sub, div or s5 shows a large gap, or mul a small one | commutativity is not the whole story. With sub and div, the constant diagonal answer is the next suspect. Run the task with `--operands separate` to confirm the gap needs shared tokens |
| E2 | add+mul within null-dup | no task-specific conflict at this size. Go to E3, and try a harder pair (add+s5, add+sqab) |
| E2 | add+mul slower, with embedding-conflict signs | run `--operands separate` and `--answers per_task` arms to locate where the shared table can be split |
| E2 | add+mul slower, trunk-conflict signs | scale the trunk (d 256) or depth before concluding anything about tasks |
| E2 | add+sub faster than null-dup | positive transfer. Test it as a lower data threshold: `--task_frac sub=0.3` |
| E3 | add kept | run the staged-addition arm and longer switches |
| E3 | overwriting | test protection: keep add at a small share of the batch (a replay arm), and see how small it can be |
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
   - Once a task's margins pass the fp32 cliff (16.6), its target gradient is exactly zero, while weight decay keeps shrinking.
   - The partner's gradient then owns the shared weights, and the first task can relapse. Jia 2026 reports relapse in transfer.
   - Watch `margin_frac_cliff@t`, `zero_tgt_grad_frac@t` and both forgetting readouts.
   - The opposite failure: a hard task whose loss never falls dominates the gradient. Equal weighting caps its share of the loss, not of the gradient.
5. **Shared operand tokens and the diagonal.** Revised after the add2b runs (2026-11-02, `diag_arms_report.md` evening addendum).
   - **The layer-1 story is necessary, not sufficient.** With shared tokens, (a, a) puts the same token at both operand positions, so layer 1 reads identical value vectors there, whatever the attention weights (RoPE touches only queries and keys). By layer 2 the positions differ through the causal mask.
   - **What the runs showed** (ordered split, 3 seeds each, test / train gap = off-diagonal minus diagonal mean margin):
     - a+b, shared tokens: 7.1–8.1 / 4.8–5.5;
     - a+2b, shared tokens: 0.9–1.3 / 0.7–0.8, with no diagonal errors, and the diagonals learned together with the other pairs;
     - separate tokens: about 0 for both tasks.
   - **So the large weakness needs both** identical operand tokens **and** a commutative task. Identical tokens alone leave about 1 margin point. Why the combination matters is not identified. a+b and a+2b also differ in more than symmetry, so this is Moderate confidence.
   - **What this means here.**
     - Commutative tasks, add, add_dup, mul, sq, sqab and gcd, should carry the large gap.
     - Non-commutative ones, sub, div, s5, add2b and cubeab, should show about 1 point at most.
     - E1 pre-registers this test.
   - Read multi-task conclusions on unseen off-diagonal pairs, or check them under `--operands separate`. This matters most when a commutative task is in the mix.
6. **Swapped pairs leak more than the commutative tasks.** sub's (b, a) answer is the negated answer; div's is the inverse. The first draft compared sub with add under an `ordered` split. There add gets about 75 % of its table for free through swapped pairs, against 50 % for sub, so "sub is slower" would show up whatever the architecture. Every arm now uses `--pairs paired`, which removes swapped-pair information for all tasks.
7. **Order comes only from RoPE and the causal mask.** A non-commutative task must tell a from b without a position embedding. Whether that costs anything is what E1's sub/add and div/mul comparison measures, once swapped pairs are closed (pitfall 6).
8. **Missing answers.**
   - The start-up check uses each task's reachable answers.
   - mul over nonzero operands never answers 0.
   - gcd's large values sit almost only on the diagonal: use `--diagonals train`, and note that this puts every diagonal in train.
9. **bf16 readouts.** Under `--bf16 autocast`, logged margins are bf16-rounded, with a step of 0.125 near 16. Rescore checkpoints in fp32 for fine margin comparisons, as the diagonal report did.
10. **Run-to-run noise.** CUDA runs are not bitwise reproducible. One same-seed rerun in E1 sizes the noise within a seed.
11. **The dropout readout under `--tasks`.** It runs per task, so its masks are not the next update's. That holds only for one task under full batch.

## 5. First experiments

**Base settings.** These are the diagonal arms' and the add2b runs' (2026-11-02) `stack_lr6e4` settings, plus a paired split:

```
--device cuda --updates 25000 --ckpt_every 0 --wd_scope repo --dropout 0.1 --beta2 0.95 --clip 1 --batch 512 --bf16 autocast --lr 6e-4 --pairs paired
```

**Checkpoints** (`--ckpt_at`, about 2 MB each; all 41 runs together about 0.5 GB, inside the disk margin):
- E1: `--ckpt_at 500,1000,2500,5000,10000`, around and after the expected grokking times;
- E2: `--ckpt_at 1000,2500,5000,10000,15000`;
- E3: `--ckpt_at 4900,5100,5500,7000,10000`, just before and after the switch.

`final.pt` holds the last update in every run.

Seeds 0, 1, 2. About 4 min per single-task run on the RTX 4060 (measured 2026-11-02); two-task runs evaluate each task and take a bigger batch, so expect about 1.5–2× that. Time a 1,000-update smoke run of one two-task arm first. Run through the guard with `MAX_PROCS=1`; each GPU worker takes about 1.4 GB of RAM. Coordinate the GPU through `gpu_handoff`.

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
| add rerun | `--tasks add --seed 0` again | same-seed CUDA spread |

**Questions it answers:**
- **Does each task grok here, and when?** Read `first_update_test_ge_0.99@t` and `test_acc@t` at 25k. A task under 0.99 at 25k gets a 100k run before anything is concluded (Power's budget).
- **Does operand order cost anything in a RoPE-only model?**
  - Compare sub with add and div with mul, all under `--pairs paired`.
  - "Order costs" if all 3 sub seeds are slower than all 3 add seeds and also outside the same-seed spread, and the same holds for div against mul.
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
- **Grokking:** all five tasks reach test accuracy ≥ 0.99 by 25k. First update at 0.99:
  - add: 1,200–2,000;
  - sub: 900–1,300;
  - mul: 2,000–3,000;
  - div: 1,000–1,500;
  - s5: 1,500, 3,700 and 14,600. The last seed ends at 0.986: late and unstable.
- **Order costs nothing.** Both "later than" predictions fail: sub is no slower than add, and div is faster than mul in every seed.
- **Diagonal: the commutative half is a hit, the non-commutative half is inconclusive.**
  - add and mul have train gaps of 4.5–5.8 in every seed, and all the diagonal errors (apart from unstable s5 seed 2).
  - sub, div and s5 have train gaps of 0.6–2.5 and test gaps of 0.8–4.1. That is above the pre-registered 1.5 in several seeds.
  - The two groups still separate cleanly: the lowest commutative train gap (4.46) is above the highest non-commutative one (2.54).
- **Noise:** the same-seed CUDA rerun of add matched seed 0 to every printed digit.
- **Spectra:** add and sub put 0.57–0.85 of the embedding's power in their top-5 Fourier frequencies along a; mul and div put 0.54–0.89 in their top 5 along log a. Each other share sits at chance, about 0.11. So the readout tells the two algorithm families apart, as §4.4 intended.

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
  - **Late:** layer-1 attention adds to it. On diagonals, layer-1 heads move about a quarter of their weight off the operand positions onto `op` and `=`. Equalizing their a/b ratio removes 32–49 % of the final gap.
  - **Correction (H6c, below):** raising layer-1 operand attention helps diagonal and off-diagonal pairs about equally. So the weight shift is not a diagonal-specific limiter, and the 32–49 % is not the diagonal's own cause.
- **Mirror test** (ordered split: off-diagonal train pairs with vs without their swapped twin in train; R = (lone − diagonal) / (mirror − diagonal)):
  - **Final weights:** R = 0.96 / 0.89 / 0.91.
  - **Over training** (CPU reruns with checkpoints): R is about 0.22 at update 100, 0.56–0.68 at 200, 0.82–0.88 at 300, and 0.87–0.95 afterwards.
  - **Control:** separate tokens show no twin effect (≤ 0.11).
  - **Reading:**
    - during memorization, the model treats (a, b) and (b, a) as one fact, so twinned facts learn faster and the twinless diagonal lags;
    - from about update 300, that exposure effect is gone. The diagonal falls further behind (gap about 1 → 3.5–5.3) because of how the symmetric (a, a) input is handled.
  - **By the registered scoring the verdict is mixed across seeds,** because its noise gate (gap ≥ 1) hides different early checkpoints in each seed; the three curves match.
- **Follow-up tests H6–H10** (final weights of 12 runs: E1 add and mul, diag-arms control and separate-token, 3 seeds each; each pre-registered, ambiguous results reviewed blind by Codex):
  - **H6, H6c (restore):**
    - raising layer-1 operand attention with the final norm fixed helps diagonals and off-diagonals about equally (≤ 0.39 nats), so it is not the limiter;
    - with the norm free, it hurts diagonals more;
    - swapping the final-norm scale closes 15–30 % of the gap, accuracy unchanged.
  - **H7 (residual):**
    - layer-0 attention writes √2 more on (a, a) in all 9 shared-token runs (1.41–1.44; separate tokens 0.93–1.08): the two identical operand values add coherently;
    - diagonals carry more energy the answer readout can't see (9/9), but removing it closes ≤ 7 % of the gap.
  - **H8 (gain):** shrinking that operand write to off-diagonal size *lowers* diagonal margins (0/9 for the doubling-cause account; the alternative holds 9/9). Its size is not the cause.
  - **H9 (transplants):** swapping parts of the computation between pairs with the same answer costs 3–7 nats even between healthy off-diagonal pairs, so transplants cannot localize the deficit. Verdict mixed; design flaw recorded.
  - **H10 (route-resolved integrated gradients):** the deficit splits about 2/3 through the '=' stream after layer 0 (60–85 % in 8 of 9 runs) and 1/3 through layer 1's reads of the operand positions; the op position carries 0. mul_s0 is the exception (57–59 % operand reads). Registered verdict mixed (7 "unresolved", 2 "'='"); the registration had no "split" category. Per pair the routes are noisy; the split is an average.
  - **Reading:** what (a, a) writes into '=' at layer 0 is the wrong content for the answer, not the wrong size. Layer 1's operand reads add a smaller share.
- **Open:**
  - finer attribution: the '=' route into residual / query / self K-V, the operand route into K / V;
  - a 2× diagonal-weight training arm, to test whether the deficit is a learned trade-off or a structural limit.

**E1b,** after E1: `sq`, `sqab`, `cubeab` × 3 seeds, 9 runs.
- **Predictions:** sq groks; sqab is open (Power yes with more data, Furuta no at 0.3); cubeab does not reach 0.99 in 25k.
- **Diagonal predictions** (same rule as E1): sq and sqab large, since they are commutative. cubeab small, if it learns at all.
- **If cubeab groks,** the MLP-based prediction is wrong for this model. Report that as a result, after the usual record checks.
- **Also:** `--tasks add --answers per_task`, the control for option B.

### E2. Two tasks at once

All arms use `--batch 1024`, so each task keeps about 512 examples per update. The split is aligned (the default).

| Arm | Flags | Seeds | Role |
|---|---|---|---|
| null-batch | `--tasks add --batch 1024` | 3 | add alone at E2's batch size |
| null-dup | `--tasks add,add_dup --batch 1024` | 5 | the cost of sharing when the tasks agree |
| add+sub | `--tasks add,sub --batch 1024` | 3 | |
| add+mul | `--tasks add,mul --batch 1024` | 5 | headline |

16 runs. With the 1.5–2× factor (assumed until the smoke run, since the batch is doubled) that is about 1.6–2.1 h.

**Hypothesis:**
- add and sub share Fourier number embeddings (Furuta's add↔sub embedding transfer), so add+sub should cost no more than null-dup.
- mul's natural embedding is periodic in log a, so add+mul should cost more than null-dup.
- Xu saw add+mul slow down about 3.5–6.5×, but with separate heads and an uncontrolled baseline.

**How to read it:**
- add's grokking time in add+mul and add+sub, against null-dup.
- sub's and mul's times against E1, scaled by the null-batch/E1 ratio for add.

**Outcomes:**
- add+mul slower than null-dup (all 5 seeds outside its range) and add+sub not: the conflict is task-specific, which is the embedding hypothesis.
- Both inside null-dup's range: sharing costs no more than any second task.
- Both slower: a generic two-different-tasks cost.

**Follow-up:** `--answers per_task` on add+mul. If that removes the extra cost, it sat in the shared output rows.

### E3. Sequential training and forgetting

3 arms × 3 seeds = 9 runs at `--batch 512`. Only one task trains at a time in every arm.

```
--tasks add,sub --task_stop add=5000 --task_start sub=5000
--tasks add,sub --task_stop add=5000 --task_start sub=5000 --wd_until 5000
--tasks add,sub --task_stop add=5000 --task_start sub=25000 --allow_idle
```

**Read:**
- `min_test_acc_last10pct@add`;
- the add test-accuracy and margin traces after update 5,000;
- `first_update_test_ge_0.99@sub` minus 5,000, against sub in E1 (transfer).

**Pre-registered reading:**

| switch | switch, no decay after 5,000 | idle (decay only) | Reading |
|---|---|---|---|
| add kept | — | — | sub reuses add's circuit without destroying it |
| add lost | add lost | any | overwriting by sub's gradients |
| add lost | add kept | add lost | disuse plus decay; sub does not overwrite add |
| add lost | add kept | add kept | sub's training and decay together erode add; neither alone does |

**Staged addition** (add kept, sub joining at 5,000) runs after E3 at `--batch 1024`, with null-dup's staged twin as its null.

**Total:** E1 + E2 + E3 = 41 runs, about 3.5–4 h of GPU time. E1 alone answers question (1) for five tasks and is the baseline for (2).

## 6. Red-team review and changes

A Fable reviewer read the draft, the code and the tests. Its findings, and what changed:

| # | Finding | Severity | Change |
|---|---|---|---|
| 1 | sub against add was confounded by swapped-pair leaks (add gets about 75 % of its table under `ordered`); the decision rule nearly could not fail | high | every arm uses `--pairs paired`; same-seed rerun; pre-registered predictions (§5 E1, pitfall 6) |
| 2 | E2 had no null, so a ~2× slowdown from sharing Adam's step would read as interference; E1 used batch 512 against E2's 1024 | high | null-batch and null-dup arms; new `add_dup` task (§5 E2) |
| 3 | E3's decay-timescale argument could not separate disuse from overwriting; `--wd_scope repo` exempts add's op row; the small-weight control was empty | high | `--wd_until` and `--allow_idle` (idle, decay-only phases) in the code; three-arm E3 with a reading table |
| 4 | same-seed runs are not paired across arms with different vocabularies | medium | unpaired comparison rule (§4.3) |
| 5 | 3 against 3 seeds gives about 10 % false positives for "either direction" | medium | stated; 5 seeds for the headline arms |
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
- `--holdout_line` with `--tasks`.

## Open questions for the owner

1. ~~Run E1 now, or wait for the add2b result?~~ Answered: add2b finished on 2026-11-02, and its result is built into pitfall 5 and E1's pre-registered diagonal predictions. E1 can run whenever the GPU and disk allow.
2. Should the prototype go into PR #281 or a new stacked PR? It touches the same file.
3. Phase 2 (sequence tasks: parity, sort-i, reverse-i) needs `max_seq_len` and a sequence split. Is it worth building before E1 and E2 have answered the 4-token questions?
