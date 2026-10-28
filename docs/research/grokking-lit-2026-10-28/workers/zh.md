# Grokking literature sweep: Chinese-language lane (zh), as of 2026-10-28

Worker lane: Simplified and Traditional Chinese sources only. An English paper by Chinese authors counts here only as a pointer; what is recorded is what the Chinese-language page adds.

**Counts:** about 200 search results seen; 21 unique sources opened (5 more attempts failed); 12 retained as evidence-bearing, of which 4 report the author's own runs with numbers (zh-01, zh-05, zh-11, zh-13).

**Overall yield:** the Chinese material is mostly news coverage and summaries. The new information is a small number of community replications (Zhihu, CSDN, a university course report). No peer-reviewed Chinese journal paper or thesis on grokking was found (see Gaps).

---

## Takeaways (most decision-relevant for ReLSM's first run, issue #249)

1. **Stay in fp32, and treat Adam ε as a grokking hyperparameter.** This comes from the Chinese coverage of *A Fix for Softmax and Cross-Entropy* (arXiv 2610.31642, posted 2026-10-22; zh-02, zh-04); check it against the paper before relying on it.
   - **The cliff.** The standard softmax/CE gradient for the target logit rounds to zero once the logit margin passes a "precision cliff": about 5.5 in bf16, 16.6 in fp32 and 36.7 in fp64.
   - **The equilibrium.** With AdamW and weight decay after memorization, the margin settles near m ≈ ln(1/ε). That is 18.4 for ε = 1e-8, which is past the fp32 cliff. So the recipe ReLSM is about to run (AdamW, wd = 1, default ε) sits right at the failure point.
   - **The reported effect.** A 1-layer Transformer on mod-31 addition (fp32, AdamW, wd = 1, ε = 1e-8) gave 94.3% ± 5.4% test accuracy over 5 seeds, with repeated loss spikes (the "Slingshot"). The fixed CE gave 100% on all 5 seeds; in bf16 the figures were 93.3% vs about 100%.
   - **Mitigation.** ε ≥ 1e-7 keeps the equilibrium inside the fp32 cliff.
   - **For ReLSM:**
     - keep the logits and the loss in fp32, not bf16 autocast;
     - log the logit margin and the fraction of training rows whose target gradient is exactly zero;
     - run a second arm with ε = 1e-7, or a stable CE / StableMax.
2. **Do not put Grokfast in the first run.** The only Chinese controlled A/B found (zh-01) is a 16-arm single-seed experiment: mod 97 add/mul, 10% split, 3.18M-parameter Transformer, AdamW lr 1e-3 with warmup + cosine, batch 256, 30k steps, wd ∈ {0, 0.03, 0.1, 0.3}.
   - **Result.** Grokfast-EMA (α = 0.98, λ = 2.0, the official defaults) produced **0/8 grokking transitions**. It suppressed memorization: train accuracy stayed at 4–35% in most arms. The best baseline arm fell from 86.2% to 8.3%.
   - **Shared bug.** The author also hit a bug of the same family as ReLSM #137. Writing the amplification as an in-place `buf.mul_(lam)` doubles the EMA state every step, and the gradients blow up within about 50 steps.
   - **Initialisation.** The author's EMA buffer is initialized with `zeros_like`, the same zero-init defect ReLSM has.
   - **For ReLSM:** get a baseline that groks first. Then add Grokfast as a separate arm, filtering once per optimizer step and seeding the EMA from the first gradient.
3. **Weight decay strength is the switch, and wd ≤ 0.1 often fails within 1e5 steps.** This also argues for fixing #186 before the run, not after.
   - **Near-identical recipe (zh-13).** A CSDN replication nearly matches ReLSM's: 2 layers, d = 128, 4 heads, FFN 512, AdamW lr 1e-3 with β = (0.9, 0.98), wd = 1.0, full batch, p = 97, 30% split. Results:

     | wd | Train reaches 100% | Test first > 50% | Test at 100k steps |
     |---|---|---|---|
     | 1.0 | step ~800 | ~21k (90% at ~26k) | 98.7% |
     | 0.1 | step ~700 | never | 1.4% |
     | 0 | step ~600 | never | 1.2% |

   - **16-arm run (zh-01).** At 30k steps, wd = 0.3 reached 86.2% and wd = 0.1 only 14.7%.
   - **Silent config bug (zh-13).** The author once lost hours because `weight_decay` was silently not applied. Their rule is to print `optimizer.param_groups` before every run, which maps directly onto ReLSM's exempt-embedding bug.
4. **Instrument the run for a messy, non-monotone transition.**
   - **What the replications saw.**
     - Baseline wd = 0.3 test accuracy climbed to 60–86% five times between steps 10.6k and 21.3k, fell back four times (to 3.9%, 23.5%, 23.5% and 42.7%), and only held from step 21.4k (zh-01).
     - zh-13 reports staged jumps (about 60%, then more than 95%) and occasional un-grokking later in training.
     - The grok step varies 2–3× across seeds (zh-13).
   - **What to log.**
     - Evaluate every 250 steps or less.
     - Log val loss and mean confidence; zh-13 says they move before accuracy does.
     - Log weight norm; its inflection coincided with grokking onset (zh-13).
     - Save checkpoints every ~2k steps.
     - Run at least 3 seeds.
     - Check that every residue appears in the training split, or the test ceiling stays below 100% (zh-13).
5. **The weight norm and the LR matter more than they look. Both are pointers to English papers via Chinese pages.**
   - **Norm (zh-10, arXiv 2606.13753).** Clamping the weight norm above the critical W_c slows grokking roughly as T ∝ exp(7.5·ρ), and a final LayerNorm removes the dependence. This bears on #218: 31k unused embedding rows inflate the norm.
   - **LR (zh-08, Tian's Li₂ paper).** On mod-127 addition with 20% training data, Adam at lr 1e-3 or 2e-3 found the Fourier solution, while lr 5e-3 found a memorizing solution.
   - **For ReLSM:** keep a constant lr ≤ 1e-3.
   - **Split caveat (zh-01).** At a 40% split with wd = 0.1, the model memorized and generalized together (train 97.7%, val 90.3% at step 400), with no delay. So at ReLSM's 0.5 split, a short or absent plateau is a valid outcome, not a failed run.

Practical aside (zh-13): 1e5 full-batch steps of a ~0.45M-parameter model took 1–2 hours, and the author blames CUDA kernel-launch overhead, not FLOPs. On the RTX 4060, CUDA graphs or `torch.compile`, and sparser evaluation, are the main levers.

---

## Sources (retained, evidence-bearing)

### zh-01 — 16-arm Grokfast × weight-decay A/B (Zhihu)
- **Identity:** 权重衰减与 Grokfast 梯度滤波对小模型模运算顿悟的影响：16 臂对照实验 ("Effect of weight decay and Grokfast gradient filtering on small-model modular-arithmetic grokking: a 16-arm controlled experiment"). Author: Zhihu user 和谐和谐和. Venue: Zhihu column. Posted 2026-09-13. URL: https://zhuanlan.zhihu.com/p/2082568126093325501
- **Type:** blog (experiment log with code).
- **Setup:**
  - RTX 5070 Ti Laptop, PyTorch 2.11.
  - Task: mod 97 a+b and a·b, all 9409 pairs, first 10% (940 pairs) as train.
  - Model: char-level Transformer, d = 256, 4 layers, 4 heads, vocab 17, 3.18M params.
  - Optimizer: AdamW lr 1e-3, warmup 300 + cosine, batch 256, 30k steps.
  - Arms: wd ∈ {0, 0.03, 0.1, 0.3} × {none, Grokfast-EMA α = 0.98, λ = 2.0}. Single seed.
- **Results (val accuracy on addition at 30k steps):**

  | wd | Baseline | + Grokfast |
  |---|---|---|
  | 0 | 0.354 | 0.013 |
  | 0.03 | 0.474 | 0.019 |
  | 0.1 | 0.147 | 0.014 |
  | 0.3 | 0.862 | 0.083 |

  - Multiplication failed in every arm (val ≤ 2.7%, against 1.03% chance).
  - Grokfast train accuracy was 4–35% in most arms; only the wd = 0.3 arms reached 99.4% / 90.9%.
  - Baseline jumps: +46.9 pp at step 11.7k (wd = 0.3) and +40.5 pp at step 23.2k (wd = 0.03).
- **Other findings:**
  - A first attempt with a 40% split did not grok at all: train 97.7%, val 90.3% at step 400.
  - The in-place `buf.mul_(lam)` bug doubled the EMA state each step and destroyed the model within about 50 steps.
  - The EMA buffer is initialized with `torch.zeros_like`.
- **Quote:** "滤波器把顿悟整个没收了" (导读 / lead paragraph); "buf.mul_(lam) 是原地操作，等于把 EMA 状态每步翻倍" (§4.2).
- **Grade:** C.
- **Novelty:** new negative result and an independent replication. Limits: single seed, no tuning of Grokfast α/λ, cosine schedule, minibatch 256 rather than full batch, wd ≤ 0.3. The author says it bounds Grokfast's domain and does not refute the paper.

### zh-02 (mirror zh-03) — Jiqizhixin report on "A Fix for Softmax and Cross-Entropy"
- **Identity:** PyTorch、JAX都被点名！Softmax曝出隐藏Bug，Gemini立功 ("PyTorch and JAX named: hidden Softmax bug found, Gemini helped"). Author: 机器之心 (Jiqizhixin) editorial team, reposted on Sina and Zhihu. Dated 2026-10-25. URL: http://k.sina.com.cn/article_5953189932_162d6782c067051yhi.html (mirror https://zhuanlan.zhihu.com/p/2097768316077187086)
- **Type:** news (detailed summary of arXiv 2610.31642 by Barron, Gnanapragasam and Tschannen).
- **Findings:**
  - **Precision cliffs (logit margin):** bf16 ≈ 5.5, fp32 ≈ 16.6, fp64 ≈ 36.7.
  - **Mechanism for the Slingshot:**
    1. The target gradient is zeroed past the cliff.
    2. Adam's second-moment estimate decays.
    3. Some samples fall back inside the cliff and their gradients reappear.
    4. Divided by the shrunken second moment, the update is huge, and the loss spikes.
  - **Mod-31 addition, 1-layer Transformer (fp32, AdamW, wd = 1, ε = 1e-8):** standard CE gave 94.3% ± 5.4% over 5 seeds; fixed CE gave 100% on all 5. In bf16, standard CE gave 93.3% and the fix about 100%.
  - **Mod-97 sum-of-squares MLP (no wd, ε = 1e-16):** with standard CE the model hit the fp32 cliff at about step 1250 and test accuracy collapsed from 61% to 23%. The fix kept climbing to about 85%.
  - **Equilibrium:** m ≈ ln(1/ε), so 18.4 for ε = 1e-8. ε ≥ 1e-7 avoids the spikes in that setting.
  - **Slingshot and generalization:** with LayerNorm and no wd, the norm growth the Slingshot causes can act as a lower effective LR. Its apparent link to grokking may therefore be incidental.
- **Quote:** "当 ε=10⁻⁸ 时，这个值约为 18.4。而 fp32 的精度悬崖恰好位于 16.6。" (section 「Adam 的默认参数，恰好踩在临界点附近」).
- **Grade:** D (news summary; the numbers are attributed to the paper).
- **Novelty:** summary only, but it was the only Chinese source that reached the bf16/fp32 failure mode. Very recent (3 days old), so check against the paper.

### zh-04 — BAAI Hub page for arXiv 2610.31642
- **Identity:** A Fix for Softmax and Cross-Entropy (智源社区论文页 / BAAI Hub paper page). Authors: Barron, Gnanapragasam, Tschannen. 2026-10-22. URL: https://hub.baai.ac.cn/paper/9ad18609-7625-46ca-b09b-fde2f4b023e0
- **Type:** paper (index page with Chinese abstract).
- **Findings (abstract):** the fix removes the Slingshot, speeds up grokking with weight decay, prevents accuracy collapse without it, and shows that post-memorization logit growth is bounded by a weight-decay/ε equilibrium. The page shows a social-media discussion score of 3427.
- **Quote:** "在引入权重衰减（weight decay）时加速了“顿悟”（grokking）过程" (中文摘要).
- **Grade:** D (abstract only).
- **Novelty:** none; it confirms the claims in zh-02.

### zh-05 — "Why does AI suddenly 'get it'" (Zhihu, with partial own reproduction)
- **Identity:** 为什么AI 会突然开窍 ("Why does AI suddenly get it"). Author: Zhihu user Bill. Column 有趣的 AI. Edited 2026-07-21; 226 upvotes. URL: https://zhuanlan.zhihu.com/p/2063047573246567123
- **Type:** blog.
- **Findings:**
  - **Setup.** It walks through the Nanda et al. recipe: mod 113, ~30% (~3800 pairs) train, 1 layer, d = 128, 4 heads, MLP 512 with ReLU, no LayerNorm, AdamW wd λ = 1, full batch, ~40k epochs.
  - **Own run.** The author's run only reached ~6100 epochs. Excluded loss had begun to lift off train loss, but the cleanup phase had not been reached.
  - **Tally reported** (Nanda's numbers, re-derived in the post):
    - key frequencies split between attention (35, 42, 52) and ReLU (14, 41);
    - about 85% of neurons are approximated by a single frequency;
    - the correct answer scores 5.0 against about 2.6 for the luckiest wrong answer.
  - **Pointer.** It cites Xu, Vardi & Safran (ICML 2026, arXiv 2601.19791): the grokking wait scales as t₂ ∝ 1/λ.
- **Quote:** "完整的清理相位需要更长训练，见 Nanda 等原论文。" (§5.2, caption of the progress-measure figure); "等待时长 t₂ ∝ 1/λ：正则强度减半，等待时间翻倍。" (§6.5).
- **Grade:** C (a partial reproduction with plots; mostly exposition).
- **Novelty:** a partial replication. Its practical point is that 6k epochs at 30% data are not enough to see cleanup.

### zh-06 — Interview with Yuandong Tian at NeurIPS 2025 (Zhihu)
- **Identity:** 对谈田渊栋：AI走向何方？我们需要怎样的深度学习理论？ ("Conversation with Yuandong Tian: where is AI going?"). Interviewer 卢R (Tsinghua; Zhihu author 宜城漫士). Zhihu column. About December 2025; the exact posting date was not captured. URL: https://zhuanlan.zhihu.com/p/1982024166560518507
- **Type:** blog (interview transcript).
- **Findings:**
  - Tian says he updated his grokking paper (arXiv 2509.21519) to remove the assumption that feature learning depends on weight decay, and that this fits experiments better.
  - He says grokking also occurs without weight decay.
  - He describes an energy (Lyapunov) function whose local optima are the learnable features. Each hidden neuron initially climbs it independently, so independent initialisation covers the optima and different runs perform alike.
- **Quote:** "grokking 也会在没有 weight decay 的时候出现" (Tian's answer on the saddle-point view).
- **Grade:** D (opinion from an author interview).
- **Novelty:** adds an author statement that is not in the abstract: a paper revision removing the weight-decay dependence.

### zh-07 — Tencent Tech interview with Tian (36Kr repost)
- **Identity:** AI研究员田渊栋：“AI顿悟”的真相、大模型如何学会压缩世界 ("Yuandong Tian on the truth about AI grokking"). Interviewer 课代表立正, Tencent Tech; repost on 36Kr. 2025-10-31. URL: https://m.36kr.com/p/3532949078842243
- **Type:** news (interview).
- **Findings:**
  - O(M log M) samples suffice for group tasks of order M. For example, about 7000 rather than about 1e6 at M = 1000.
  - Grokking is a collective jump once the "generalization peak" in a multi-peak non-convex landscape rises above the "memorization peak".
  - Tian says parts of the paper came from discussions with GPT-5.
- **Quote:** "数据不足时“记忆峰”占优，数据增多时“泛化峰”升高" (核心观点 list).
- **Grade:** D.
- **Novelty:** author framing only; no new numbers.

### zh-08 — Jiqizhixin on Tian's Li₂ framework (Zhihu)
- **Identity:** Meta FAIR田渊栋唯一作者发文：拆解模型「顿悟时刻」 ("Tian, sole author, dissects the 'aha moment'"). 机器之心 (editor 冷猫). Zhihu, edited 2025-10-05. URL: https://zhuanlan.zhihu.com/p/1958241331680875100
- **Type:** news (summary of arXiv 2509.21519).
- **Findings:**
  - **Three stages:** lazy, then independent feature learning (needs η > 0 in the original version), then interactive feature learning.
  - **LR effect:** on mod-127 addition with n = 3225 (20%), Adam at lr 0.001 or 0.002 found the generalizing Fourier solution, while lr 0.005 found a memorizing one.
  - **Muon:** it beat Adam when hidden units were limited.
  - **Data-dependent LR:** the paper argues limited data may need a *smaller* LR to stay in the generalizing basin.
- **Quote:** "当使用较大学习率（0.005，对应右图）时，Adam 则会找到不可泛化的解" (阶段 II section).
- **Grade:** D.
- **Novelty:** summary; the LR numbers are useful.

### zh-10 — Auto-translation of "The Weight Norm Sets the Grokking Timescale" (arXiv 2606.13753)
- **Identity:** 权重范数决定了“顿悟”（Grokking）的时间尺度：一种因果延迟定律 ("The weight norm sets the grokking timescale: a causal delay law"). AI News Daily (auto-generated). 2026-06-16. URL: https://ai-news-9hr.pages.dev/news/2026-06-16/7f7d7226686b
- **Type:** news (machine translation of the abstract).
- **Findings:**
  - With wd, grokking happens when the norm reaches W_c. W_c varies only 1–2% (CV) across seeds and LRs, and grows with the modulus as a power law.
  - Clamping the norm at ρ·W_c gives T_grok ∝ exp(αρ) with α ≈ 7.5 (R² = 0.996).
  - Over the swept ranges, the held norm moved the delay about 19× and the LR only about 2×.
  - A final LayerNorm removes the dependence.
- **Quote:** "将范数保持在 $W\_c$ 以上会减缓顿悟，而不是阻止它" (abstract translation).
- **Grade:** D.
- **Novelty:** none (translation). Retained as a pointer that matters for #218.

### zh-11 — Peking University course report: reproducing grokking and probing its factors
- **Identity:** Grokking 现象的复现与影响因素探究：模加法任务 实验报告 ("Reproducing grokking and probing its factors: modular-addition experiment report"). Three student authors; the student IDs look like PKU's format, but the affiliation is not confirmed. Hosted on maozy-leo.github.io. PDF created 2026-01-10. URL: https://maozy-leo.github.io/assets/documents/grokking-report.pdf
- **Type:** thesis-like course report (10 pp., full method appendix).
- **Readability:** the Chinese body text did not extract (font-encoding issue), so only numbers, tables and English terms are readable.
- **Setup (readable):**
  - p = 97; α (train fraction) swept over [0.15, 0.85].
  - Decoder-only Transformer: d_model = 128, 2 layers, 4 heads, d_ff = 512, sinusoidal PE, causal mask.
  - Batch 512 (full batch if smaller); 1e5–1e6 steps; 10-step warmup.
  - 5 seeds {0, 1, 2, 4, 5}, reporting the median. A100 rented on AutoDL.
  - Grokking threshold: val accuracy > 99.5%.
- **12-optimizer table.** Baseline AdamW lr 1e-3, wd 1.0, β = (0.9, 0.98). The other 11 arms:
  - Adam with no WD;
  - AdamW with WD toward the initial weights;
  - AdamW with high WD = 3.0;
  - AdamW with low LR = 1e-4;
  - AdamW with β2 = 0.999;
  - Adam at lr 5e-3;
  - SGD with heavy-ball momentum 0.9;
  - RMSprop;
  - SGD with Nesterov momentum 0.99;
  - full-batch GD;
  - AdamW with weight noise σ = 0.01.
- **Architecture sweep.** Simple MLP, Gated MLP, SIREN, 1D-CNN, LSTM, vanilla RNN, ResNet MLP, Deep MLP.
- **Readable results:**
  - Train accuracy reached 99% at step 2400 for the simple MLP and step 350 for the Transformer.
  - Val accuracy reached 99% at step 5150 for the MLP and step 2050 for the Transformer.
  - The figure labels show K = 2–5 summands, with "PE+Mask" needed at K ≥ 4.
- **Quote** (readable table text, not Chinese): "AdamW (WD=1.0) [Baseline] AdamW 10−3 1.0 - β = (0.9, 0.98)" (Appendix A.3, Table 2).
- **Grade:** C. The method is fully specified, but the prose and per-optimizer results could not be read.
- **Novelty:** an independent replication with a broad optimizer ablation. It is worth a human reading the rendered PDF: the 12-optimizer outcome table is the single most useful item in this lane that could not be extracted.

### zh-12 — Student repo with a Chinese README (sant-liustu/grokking-phenomena)
- **Identity:** Grokking 现象研究项目 ("Grokking phenomenon research project"). Author: GitHub user sant-liustu. Venue: GitHub. Date unknown; 6 commits. URL: https://github.com/sant-liustu/grokking-phenomena
- **Type:** code.
- **Findings:**
  - It has the same subtask structure as zh-11 (likely the same course assignment): split fraction, architectures, optimizers, K-ary addition.
  - It adds `slingshot.ipynb` and `Sharpness代码.ipynb`.
  - README claims: AdamW is best in most cases; too-strong regularization destabilizes training; larger K delays grokking. The README gives no numbers.
- **Quote:** "过强的正则化项会导致训练不稳定" (README §实验内容 3).
- **Grade:** D (code but no reported numbers).
- **Novelty:** weak; it corroborates zh-11.

### zh-13 — CSDN "Grokking reproduction" with a near-identical recipe
- **Identity:** Grokking复现：权重衰减驱动延迟泛化与训练相变 ("Grokking reproduction: weight decay drives delayed generalization and a training phase transition"). Author: CSDN user weixin_34246529. Published 2026-10-25. URL: https://blog.csdn.net/weixin_34246529/article/details/169096490
- **Type:** blog with full code.
- **Setup:**
  - p = 97, 30% split (2822 train pairs).
  - `nn.TransformerEncoder`: 2 layers, d = 128, 4 heads, FF 512, GELU, post-LN, dropout 0, learned 2-position embedding, about 0.45M params.
  - AdamW lr 1e-3, β = (0.9, 0.98), wd 1.0 on `model.parameters()` (the embedding included), full batch, no schedule, eval every 250 steps.
- **Results (one seed, 100k steps):**

  | wd | Train reaches 100% | Test first > 50% | Test at 100k | Weight norm |
  |---|---|---|---|---|
  | 0 | ~600 | never | 1.2% | rising |
  | 0.1 | ~700 | never | 1.4% | flattens |
  | 1.0 | ~800 | ~21k (90% at ~26k) | 98.7% | inflects near grok onset, then falls |

- **Troubleshooting notes:**
  - print `param_groups` (a mistyped wd keyword was once silently ignored);
  - check that the split covers all 97 residues;
  - grok step varies 2–3× across seeds;
  - un-grokking after reaching 99%;
  - 1e5 steps took 1–2 h, bound by kernel-launch overhead;
  - lr 3e-4 doubles the steps needed, while lr 3e-3 may be unstable.
- **Quote:** "权重衰减是主开关" (§2.4); "第一件事就是打印一遍优化器的 param_groups" (§5.1).
- **Grade:** C. Code plus numbers, but a single seed, partly vague prose, and possibly AI-assisted writing.
- **Novelty:** a replication on almost exactly ReLSM's #249 configuration. Its timings (grok at ~2e4 steps, 98.7% by 1e5) are the best available prior for the expected run length.

### zh-17 — Bilibili explainer: "Grokking then collapse" with Muon
- **Identity:** Grokking then Collapse? The Muon Optimizer Makes Transformers Learn First, Then Go Crazy. Uploader wkeeli. Bilibili. 2026-08-09. URL: https://www.bilibili.com/video/BV1MMuR6fE8e/
- **Viewing:** the page rendered in Bilibili's English UI, and the video itself was not watched; only the description was read.
- **Type:** video.
- **Findings:** it covers arXiv 2608.07436. On modular addition, Transformers trained with Muon first grok and then suddenly lose generalization. The video discusses which layer collapses and why.
- **Quote** (description, as displayed in English): "first exhibit Grokking, then a sudden collapse of generalization ability".
- **Grade:** D.
- **Novelty:** pointer only. It is a failure mode to keep in mind if ReLSM later tries Muon as an accelerator.

---

## Opened but not evidence-bearing (kept in the JSONL)

- **zh-03:** Zhihu mirror of zh-02.
- **zh-09:** BAAI Talk 368 page announcing Mingyue Xu's (Purdue) talk on the ICML 2026 Outstanding Paper Honorable Mention (arXiv 2601.19791). Announcement only; no talk content was obtained.
- **zh-14:** Zhihu survey 深度学习中的 Grokking 现象 (2026-01-25), marked "包含 AI 辅助创作". It is **unreliable**:
  - a leftover chat-assistant line in the body;
  - references with placeholder IDs ("arXiv:2501.XXXXX") and wrong authors (NeuralGrok attributed to Power & Loshchilov);
  - unsourced tables claiming wd = 1e-4 groks at 300k steps while wd = 1e-2 does not.
- **zh-15:** CSDN content-farm piece on the spectral theory of weight decay. Factual errors: it says DeepMind named grokking, and it describes an invented "mod 3 of length-12 sequences" experiment.
- **zh-16:** NJU student forum thread. An anecdote that perfect fit took about 150 epochs and grokking about 7000–12000. The rest of the thread is an LLM-generated answer.
- **zh-18:** general Muon explainer, not about grokking.
- **zh-19:** general explainer on mixed precision, not about grokking.
- **zh-20:** CSDN URL whose snippet promised "Grokfast integration in Axolotl" but whose content had been replaced with an unrelated tutorial.
- **zh-21:** 2024 Zhihu summary of the Grokfast paper. It repeats the paper's λ = 5, w = 100 and 13.57× figures and adds nothing.

---

## Contradictions with the standard English account

1. **Grokfast's "50× faster grokking" vs the Chinese replication.** The 16-arm A/B (zh-01) found Grokfast-EMA at the official defaults blocked grokking: 0/8 arms, and 86.2% → 8.3% in the best arm. It did so by suppressing memorization. The setup differs from the paper (cosine LR, batch 256, 10% split, wd ≤ 0.3), so this narrows Grokfast's domain rather than refuting it. It does match the view that Grokfast's gains are setup-specific.
2. **The Slingshot as an optimizer phenomenon vs as a numerical bug.** The standard account (Thilak et al., 2022) treats the Slingshot as an adaptive-optimizer dynamic tied to grokking. The coverage of arXiv 2610.31642 (zh-02) attributes it to softmax-collapse gradient zeroing interacting with Adam's second moment, and says the fix removes it while still speeding up grokking.
3. **Is weight decay necessary?** Tian (zh-06) says grokking occurs without weight decay and that his revised theory drops the wd assumption. Yet every Chinese replication with numbers (zh-01, zh-13) saw no grokking at wd = 0 within 3e4–1e5 steps. Both can hold: without wd, the delay may simply exceed those budgets, or depend on other mechanisms such as loss type or initialisation scale.
4. **Which wd values work.** The unreliable survey zh-14 claims wd = 1e-4 groks at 300k steps and wd = 1e-2 never groks. The replications and the standard recipe show wd = 1.0 works and 0.1 fails within 1e5 steps. Treat zh-14 as noise.
5. **A clean cliff vs a ragged transition.** Popular summaries describe a single sudden jump. Both community runs saw repeated rise-and-fall cycles before stabilizing (zh-01), and staged jumps or later un-grokking (zh-13).
6. **Transformer vs MLP.** The PKU report (zh-11) finds the Transformer much faster than the MLP (val 99% at step 2050 vs 5150). The 2026 systematic study, cited only via the LLM answer in the NJU forum, reportedly finds the gap vanishes once hyperparameters are matched. This is weak evidence on both sides.

---

## Gaps (expected but not found, or not reachable)

- **CNKI, Wanfang and VIP theses and journals** (计算机学报, 软件学报, 自动化学报): none found. The engines used do not index CNKI full text, and no Chinese-language peer-reviewed grokking paper surfaced in 2025–2026. CNKI itself was not accessible; it needs an institutional login.
- **CCF conference papers in Chinese:** none found. Chinese authors publish grokking work in English venues (Tian; Xu, Vardi & Safran; Li, Fan & Zhou's OLMoE work; GrokTransfer by Zhiwei Xu et al.). The Chinese pages about them are machine translations (alphaXiv, chatpaper, HyperAI) with no added detail.
- **StableMax / ⊥Grad:** no Chinese replication found. The only Chinese coverage is via the 2026-10 Softmax-fix news.
- **bf16/fp16 grokking experiments:** none in Chinese beyond the numbers in that news piece. Generic mixed-precision explainers do not cover grokking.
- **Grokfast with gradient accumulation / per-micro-batch filtering (ReLSM #137), or EMA initialisation:** no Chinese discussion found. zh-01's in-place bug is the closest relative.
- **Embedding weight-decay exemption (#186) or unused embedding rows (#218):** no Chinese discussion of either.
- **Grokking in LLM pretraining:** no original Chinese work or substantive discussion found beyond pointers to Li, Fan & Zhou (arXiv 2506.21551).
- **Blocked or unreachable pages:**
  - Zhihu Q&A pages (/question/…) hit a login/"安全验证" wall; only column articles were readable.
  - wiki.swarma.org (集智百科) is behind a Cloudflare challenge.
  - ashcreek.cc ("grokking 论文阅读复现", 2026-09-02) has an expired TLS certificate.
  - CSDN 167116997 returned HTTP 521.
  - The WeChat (微信公众号) mirrors and Bilibili video content (beyond descriptions) were not accessed.
  - These were not circumvented.
- **Search infrastructure:** SearXNG stopped returning results mid-session, so later searches went through Bing (cn.bing.com in the browser pane) and WebSearch. Coverage of 稀土掘金, 博客园, 量子位, 新智元 and HyperAI is therefore thin. No 2025–2026 grokking-specific posts surfaced on those sites.

---

## Method

**Engines:**
- SearXNG (language zh / zh-TW; time_range year), until it failed;
- WebSearch, as a second engine;
- Bing China (cn.bing.com, mkt = zh-CN) through the browser pane.

**Pages were opened with:**
- `web_url_read`;
- WebFetch;
- the browser pane, whose `get_page_text` / page scripts were the only way to read Zhihu and CSDN.

The PDF zh-11 was downloaded by WebFetch and its text extracted locally with pypdf, in isolated mode (`python -I`).

**Queries used:**
- grokking 顿悟 模加法 权重衰减
- Grokking 顿悟现象 延迟泛化 复现
- Grokfast 加速 顿悟 知乎
- "Grokfast" 16 臂全因子 顿悟 权重衰减 模运算
- StableMax 数值稳定性 顿悟 Softmax Collapse
- 田渊栋 Li2 grokking 知乎 自己 解读 群论 表示学习
- 顿悟 grokking 硕士论文 延迟泛化 研究
- 大模型预训练 顿悟 grokking (+ 机器之心 OLMoE variant)
- Grokking 弹弓机制 slingshot Adam 损失尖峰 顿悟
- "顿悟现象" 神经网络 期刊 论文 知网
- Grokking 頓悟 延遲泛化 神經網路 權重衰減 (zh-TW)
- grokking 软件学报 OR 计算机学报 OR 自动化学报 顿悟
- grokking 复现 实验; Grokfast 复现 效果 不稳定; 顿悟 grokking 知乎 实验记录
- WebSearch: 顿悟现象 grokking 知网 学位论文 延迟泛化 模运算; Grokking 頓悟 延遲泛化 權重衰減 模加法; grokking 复现 模97 加法 … 博客园 OR 掘金 OR CSDN; site:bilibili.com grokking 顿悟 复现; grokking 顿悟 预训练 大模型 记忆到泛化 路由 机器之心
- Bing: grokking 顿悟 复现 权重衰减 知乎; StableMax 顿悟 数值稳定性 Softmax坍缩 复现; Muon 优化器 顿悟 grokking 加速; GrokTransfer 嵌入迁移 顿悟 加速; grokking bf16 半精度 顿悟 失败 损失尖峰; Grokfast 复现 梯度累积 EMA 顿悟 没有加速

**Counts:**
- About 200 result entries seen across all engines.
- 21 unique sources opened (zh-01 to zh-21).
- 5 more opens failed: CSDN 167116997, ashcreek.cc, wiki.swarma.org, a Zhihu question page, and a first 36Kr attempt that hit a bot check, later opened in the browser as zh-07.
- 12 retained as evidence-bearing:
  - grade C: zh-01, zh-05, zh-11, zh-13 (4 sources);
  - grade D: zh-02, zh-04, zh-06, zh-07, zh-08, zh-10, zh-12, zh-17 (8 sources).
- zh-03, the Zhihu mirror of zh-02, is grouped with zh-02 above and is not counted as a separate evidence source.
- Grades: none A or B; 4 C; the rest D.

**Treatment of observed content:** all quotes are under 25 words, copied from the opened pages. Page content was treated as data. No instructions found on any page were acted on.
