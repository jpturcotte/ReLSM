# Grokking literature sweep: Japanese-language lane (as of 2026-10-28)

Lane: Japanese-language sources only. English papers by Japanese authors are treated as pointers only.
Worker: `ja`. Opened 27 unique sources; 12 are evidence-bearing (see Method).

## Takeaways (most decision-relevant for ReLSM issue #249)

1. **Softmax and cross-entropy numerics will probably cause loss spikes ("slingshot") under the planned recipe. Compute the loss in fp32 and log the margin.**
   - Mechanism (Barron et al., arXiv:2610.31642, 2026-10-22; relayed and replicated in Japanese, ja-09 and ja-10):
     - Standard CE rounds the correct-class gradient to exactly 0 once the logit margin m exceeds about 16.6 in fp32. In bf16 the threshold is about 5.5.
     - Under AdamW with weight decay, the post-memorization margin settles near log(1/ε). For ε = 1e-8 that is 18.4, just past the fp32 threshold.
   - Replication (Takuya, ja-09): a 15-seed PyTorch run of a+b mod 31 with full-batch AdamW (lr 1e-3, β 0.9/0.98, wd 1) produced an average of 21.3 loss spikes per 30k steps with standard CE, and 0 with the margin-form fix.
   - Setting ε = 1e-6 also removed the spikes: 0 spikes and 2/2 seeds at 100%.
   - Computing the loss in bf16 badly hurts grokking. Standard CE gave 4/15 seeds at ≥95% and mean margin stuck at 4.6, against 10/15 seeds with the fixed CE.
   - Actions for #249:
     - Keep logits and CE in fp32.
     - Either use a `softplus(-m)` CE, or set AdamW ε ≥ 1e-7.
     - Log the mean train margin each eval. The run should finish at fp32 or fp64 (not bf16) unless that is the experiment.
2. **Use decoupled AdamW and check the order of decay and update. Coupled L2 at wd = 1 destroys training.**
   - Shake AI Lab (ja-25) first implemented weight decay as `g += wd*p` inside Adam. At wd = 1.0 the weights collapsed and train accuracy stayed at 2.6% for 40k steps.
   - Their own "fixed" code still decays after the Adam step, not before it. They flag that this order is not exactly standard AdamW.
   - This is cheap to unit-test before a 1e5-step run.
3. **Expect a quick result with the planned recipe. A run with no test rise by about 1e4 steps more likely has a bug than needs more patience.**
   - The closest Japanese replication to #249 is Kanda (ja-13): (a+b) mod 97, 50/50 split, 2 layers, 4 heads, d = 128 (0.42M parameters), AdamW lr 1e-3, wd 1.0.
     - Train accuracy reached 100% at step 400. Test accuracy plateaued at 30–50%, passed 90% at about 1,200–1,300 steps, and held 100% after step 1,700.
   - Shintani lab, Chubu University (ja-04): p = 31, frac 0.5, 2 layers, d = 128, lr 1e-3, wd 1.0, full batch.
     - Train ≥ 99% at epoch 130; test ≥ 90% at epoch 3,140.
     - With lr = 3e-4 the run stalled at about 52% test accuracy after 6,000 epochs. Keep lr = 1e-3 for the first run.
   - The 1e5-update budget is a generous ceiling, not the expected grokking time.
4. **Train fraction and weight-decay strength set the delay monotonically. Keep frac ≥ 0.4–0.5 and wd at 1.0 for the first run.**
   - disassemble-channel (ja-02), MLP, p = 59, AdamW wd 1.0, full batch. Steps to 90% test accuracy:
     - frac 0.6: 2,000.
     - frac 0.5: 4,150.
     - frac 0.4: 9,150.
     - frac 0.3: not reached in 50k steps; test accuracy 50.6% at the end.
   - Shake AI Lab (ja-25), p = 53, frac 0.4, MLP. Steps to 90% test accuracy:
     - wd 0 and wd 0.03: never within 60k steps.
     - wd 0.1: final test accuracy 0.317.
     - wd 0.3: 27,600 steps.
     - wd 1.0: 10,400 steps.
   - Lower effective decay slows or blocks grokking. ReLSM's embedding exemption (#186) and 31k unused rows (#218) both lower the effective decay on the embedding, where the Fourier structure forms.
     - Every Japanese reproduction decays all parameters, including embeddings, via `model.parameters()`. None tested an exemption.
     - Fix #186 and #218 before the first run, or treat them as a planned ablation, not a default.
5. **Accelerators: ⊥Grad / OrthoGrad is the only one with a Japanese replication showing a large speed-up. StableMax alone was slower. No Japanese Grokfast evidence exists.**
   - 50s_zerotohero (ja-03), 1-layer Transformer, p = 113, frac 0.3: AdamW + wd 1.0 took about 10,000 epochs.
     - StableMax with wd 0 took about 60,000 epochs.
     - ⊥AdamW (OrthoGrad) with wd 0 took about 200 epochs.
     - Plain AdamW with wd 0 never generalized.
   - Barron et al. (via ja-10) report that StableMax margins balloon to 10³–10⁴, with lower test accuracy than the fixed CE at every ε.
   - Japanese Grokfast material is limited to a machine-translation mirror (ja-26). It adds nothing on the per-micro-batch filter (#137) or EMA initialisation issues.

## Sources

Grades: A = peer-reviewed or thesis with full method; B = preprint, domestic conference or detailed report; C = blog or repo with code and results; D = opinion, summary or translation.

### ja-01: 宝くじ仮説の観点からの Grokking の理解 ("Understanding Grokking through the Lens of the Lottery Ticket Hypothesis")
- **Identity:** 峰岸剛基, 岩澤有祐, 松尾豊 (U. Tokyo, Matsuo–Iwasawa Lab). JSAI2024, 1B4-GS-2-03, published 2024-06-11. https://www.jstage.jst.go.jp/article/pjsai/JSAI2024/0/JSAI2024_1B4GS203/_article/-char/ja (PDF `_pdf`). DOI 10.11517/pjsai.JSAI2024.0_1B4GS203.
- **Type:** domestic conference paper (4 pages). Japanese counterpart of arXiv 2310.19470.
- **Setup and findings:**
  - Setup: (a+b) % p with p = 67. The train:test split reads as 4:6 in the PDF, but the text layer is partly garbled.
  - Model: a Liu et al.-style MLP, `softmax(σ((E_a+E_b)W_in) W_out W_unemb)` with ReLU.
  - Grokking tickets are found by magnitude pruning of the generalized solution and rewinding to initial values.
    - Per the Fig. 3 caption, the periodic (Fourier-like) neuron structure appears at about 2,000 epochs with the ticket, against about 30,000 epochs for the dense base model.
  - A dense network matched to the ticket's L1/L2 norm ("controlled dense") still takes much longer to generalize.
  - Edge-popup (structure-only optimisation, no weight updates) applied to the memorized solution at 400 epochs (train accuracy 100%) reaches the generalizing solution. Applied from initialisation, it mostly fails.
  - Layer-wise Jaccard similarity to the final mask tracks the rise in test accuracy.
- **Quote** (J-STAGE abstract, item 2): 「等しい重みノルムであっても、密なネットワークは完全な汎化を到達するためにかなり長い訓練が必要である」
- **Grade:** B.
- **Novelty:** a 2024 Japanese venue version of an English paper. It adds little beyond 2310.19470.
- **Use for ReLSM:** weight norm alone is not a sufficient progress measure. Mask-overlap (Jaccard) to a pruned solution is a possible diagnostic for later runs.

### ja-02: Grokking（グロッキング）— 過学習のはるか先で突然訪れる汎化を実測する ("Measuring grokking: generalization far past overfitting")
- **Identity:** 「機械学習と情報技術」 (disassemble-channel.com). Blog, 2026-07-04. https://disassemble-channel.com/grokking-delayed-generalization/
- **Type:** blog with full PyTorch code and printed outputs (seed 0).
- **Setup:** (a+b) mod 59; embedding 64 → concat → 256-unit ReLU MLP (about 50k parameters); full batch; AdamW lr 1e-3, betas (0.9, 0.98), wd 1.0 on all parameters including the embedding; 15k steps on CPU.
- **Findings:**
  - Train accuracy 100% at step 150. Test accuracy ≥ 50% at step 7,300 and ≥ 90% at 9,150; final 1.0.
  - With wd = 0: train 100% at step 150, but test accuracy is 0.05% at 15k steps, below chance (1.7%).
  - Weight norm peaks at 69.4 and falls to 53.8 with wd = 1. With wd = 0 it ends at 210.0.
  - Steps to 90% test by train fraction: 0.6 → 2,000; 0.5 → 4,150; 0.4 → 9,150; 0.3 → not reached in 50k steps (final 0.506).
  - The top-6 embedding Fourier frequencies take 42.1% of non-DC spectral mass after grokking.
- **Quote** (§weight decay): 「15,000ステップ訓練し続けてもテスト精度は0.05%」
- **Grade:** C.
- **Novelty:** an independent replication with a train-fraction sweep and weight-norm numbers.

### ja-03: Grokkingを10分で再現する — モデルが"突然目覚める"瞬間を可視化で追う ("Reproduce grokking in 10 minutes")
- **Identity:** ZeroToHero (Zenn). Published 2026-03-08, updated 2026-03-13. https://zenn.dev/50s_zerotohero/articles/b4481d2e97f2a8
- **Type:** blog. The experiments were run with Claude Code on an RTX 5090, and the reproduction instructions are attached.
- **Setup:** 1-layer Transformer, d_model 128, 4 heads, d_mlp 512 (about 400k parameters); addition mod 113 with a 30/70 split; AdamW lr 1e-3, wd 1.0; full batch; 40k epochs.
- **Findings:**
  - Phases: memorization ends at about 1,400 epochs, circuit formation runs from 1,400 to 9,400, and cleanup from 9,400 to 14,000.
  - The key frequencies were {14, 25, 42, 52}, compared with 5 frequencies in Nanda et al. The frequencies depend on the seed.
  - With wd = 0:
    - fp16: Softmax Collapse occurs very early and test accuracy never rises.
    - fp32: oscillating Softmax Collapse with slight generalization.
    - fp64: Softmax Collapse is delayed.
  - Epochs to generalize by method:

    | Method | Epochs to generalize |
    |---|---|
    | AdamW + wd 1.0 | ~10,000 |
    | StableMax, wd 0 | ~60,000 |
    | ⊥AdamW (OrthoGrad), wd 0 | ~200 |
    | AdamW, wd 0 | never |

- **Quote** (§5, float precision): 「float16（左）：SCが極めて早期に発生し、test accuracyは全く上がりません。」
- **Grade:** C.
- **Novelty:** a replication of Prieto et al. with a head-to-head speed table. StableMax alone was about 6× slower than AdamW + wd.

### ja-04: シンプルな microGPT を十分学習させると…modular addition で grokking を観察する ("Observing grokking in modular addition with a simple microGPT")
- **Identity:** 新谷正嶺 (Shintani Lab, Chubu University). Qiita, posted 2026-04-10, updated 2026-04-14. https://qiita.com/Seine_A_Shintani/items/21d707e31ce8ec670a78
- **Type:** blog with full Colab code.
- **Setup:** p = 31, train fraction 0.5 (480 train / 481 test); 2-layer decoder-only, dim 128, 1 head, RMSNorm, SiLU FFN; input `[a,+,b,=]`; AdamW lr 1e-3, betas (0.9, 0.98), wd 1.0 on all parameters; full batch; 6,000 epochs; seed 0.
- **Findings:**
  - Train ≥ 99% at epoch 130. Test ≥ 50% at 1,550, ≥ 80% at 2,860 and ≥ 90% at 3,140; best test accuracy 1.0 at epoch 4,750.
  - Negative result: with lr = 3e-4, test accuracy stayed at about 0.52 after 6,000 epochs.
- **Quote** (§3 「grokking は設定依存」): 「grokking は必ず同じように出るわけではなく、ハイパーパラメータ依存が強い」
- **Grade:** C.
- **Novelty:** a replication plus a negative LR result.

### ja-06: lutelute/test_NN — Grokking研究: Transformerによるモジュラー加算タスクの遅延汎化現象の解析 ("Delayed generalization on modular addition with a Transformer")
- **Identity:** GitHub user lutelute. Repository; date not stated. https://github.com/lutelute/test_NN (README read via raw.githubusercontent.com)
- **Type:** code repository.
- **Setup:** p = 67; 1-layer Transformer, d_model 128, 4 heads; wd 1.0; 10k epochs (2-token task) and 15k epochs (3-token (a+b+c) mod p). Optimizer, lr, train fraction and batch are not stated.
- **Findings:** for the 2-token task, memorization completes at epoch 83 and grokking at epoch 221; final train and test accuracy are both 100%.
- **Quote** (README): 「強い正則化（Weight Decay）がGrokkingを促進」
- **Grade:** C, low weight because the setup is incomplete.
- **Novelty:** a minor replication. The very short delay is probably due to an undisclosed large train fraction.

### ja-09: Softmax と Cross-Entropy に潜む 2 つの数値バグを Mac で再現してみた ("Reproducing two numerical bugs in softmax and cross-entropy on a Mac")
- **Identity:** Takuya (@Takuya__). Qiita, posted 2026-10-26 (also on takuya-genai.com). https://qiita.com/Takuya__/items/05bbe2c9e09c41382ebc
- **Type:** detailed reproduction report with code.
  - Environment: Apple M5 Max, PyTorch 2.14.1 (CPU and MPS), JAX 0.12.0.1.
  - Seeds: 15 for the fp32 and bf16 comparisons, 2–3 for the other conditions.
- **Setup:** a+b mod 31 with half for training; 1-layer, width-64 ReLU Transformer with no LayerNorm (4 heads and MLP 256 are the author's guess); full-batch AdamW lr 1e-3, β 0.9/0.98, ε 1e-8, wd 1; 30k steps (the paper ran 10k).
- **Measured precision cliffs** (smallest margin where the correct-class gradient becomes exactly 0):

  | Precision | Standard CE | Fixed CE |
  |---|---|---|
  | bf16 | 5.5625 | 89 (underflow) |
  | fp32 | 16.64 | 88.73 (underflow) |
  | fp64 | 36.74 | none |

  - On MPS, bf16 CE is computed internally in fp32, so the cliff is 16.75.
  - HF transformers upcasts logits to fp32 before the loss, so its cliff is 16.6, not 5.5.
- **Grokking results** (15 seeds unless stated):

  | Condition | Mean final test acc | Seeds ending ≥ 95% | Seeds reaching 95% | Loss spikes (mean) | Final margin |
  |---|---|---|---|---|---|
  | fp32, standard CE | 84% | 8/15 | 11/15 | 21.3 | 12.6 |
  | fp32, fixed CE | 80% | 8/15 | 8/15 | 0 | 17.0 |
  | bf16 loss, standard CE | 68% | 4/15 | 5/15 | not counted | 4.6 |
  | bf16 loss, fixed CE | 83% | 10/15 | 10/15 | 0 | 17.1 |

  - In fp32, standard CE crossed 95% first on 6 seeds and fixed CE on 6 seeds. With standard CE, 3 runs that reached 95% oscillated and ended below it.
  - ε sweep with the fixed CE: final margins 12.4, 17.0, 22.1 and 26.8 for ε = 1e-6, 1e-8, 1e-10 and 1e-12. Theory gives 13.8, 18.4, 23.0 and 27.6.
  - With ε = 1e-6, standard CE stops at margin 12.4: 0 spikes, and 2/2 seeds at 100%.
  - Without wd and with ε = 1e-16: standard CE averaged 22.3 spikes and the fixed CE 0, but neither generalized in 30k steps (35–46% test accuracy).
  - In LLMs with an fp32 loss, only 0–0.03% (Qwen3-0.6B) to 0.2–0.5% (8B) of tokens are past the cliff. LoRA over 600 steps showed no spikes.
- **Quote** (§はじめに, conclusions): 「論文の「グロッキング（遅れてやってくる汎化）が速くなる」は、手元の fp32 の実験では再現せず」
- **Grade:** C. This is the strongest Japanese source in the sweep.
- **Novelty:** a new replication with a partial negative result. The slingshot is confirmed; faster grokking in fp32 is not.
  - The paper reports 94 ± 5% for standard CE and 100 ± 0% for the fix, over 5 seeds at 10k epochs.

### ja-10: 【文献メモ】A Fix for Softmax and Cross-Entropy… ("Literature note: A Fix for Softmax and Cross-Entropy")
- **Identity:** Chihiro Mihara (@CookieBox26). Qiita, 2026-10-26. https://qiita.com/CookieBox26/items/48b302211800bb458a1b
- **Type:** detailed literature summary of arXiv:2610.31642 (Barron, Gnanapragasam, Tschannen, Google DeepMind).
- **Findings (relayed from the paper):**
  - Cliffs: 5.5 (bf16), 16.6 (fp32), 36.7 (fp64).
  - Slingshot mechanism: the gradient vanishes, Adam's second moment decays, wd pulls the margin back below the cliff, and a huge step follows.
  - Margin equilibrium m* ≈ log(1/ε), 18.4 at ε = 1e-8. This was shown with LaProp and the fixed CE, not measured with the standard CE.
  - 1-layer Transformer on a+b mod 31: the fixed CE reaches 99–100% at bf16, fp32 and fp64, against 93–97% for standard CE at bf16 and fp32. The fix also beats zero-sum projection and StableMax.
  - StableMax margins grow to 10³–10⁴.
  - ε values tested: 1e-8, 1e-12, 1e-16.
- **Quote** (実験結果): 「標準実装は bf16 と fp32 で 93〜97% にとどまりました」
- **Grade:** D (summary).
- **Novelty:** a pointer to a 6-day-old English paper. The Japanese note adds the comparison with Prieto et al., including that Barron et al. do not cite ⊥Grad.

### ja-13: TransformerでGrokkingを観察する ("Observing grokking in a Transformer")
- **Identity:** Kanda (@ka_kan). Zenn, 2026-03-07. https://zenn.dev/ka_kan/articles/39090a66a8c904
- **Type:** blog with reported results and figures. No code is shown, and the batch setting is not stated.
- **Setup:** (a+b) mod 97, 9,409 pairs, 50/50 split, input `[a,+,b,=]`; 2 layers, 4 heads, embedding 128, 422,656 parameters; AdamW lr 1e-3, wd 1.0.
- **Findings:**
  - Train accuracy 100% at step 400. Test accuracy plateaued at 30–50%, jumped past 90% at about 1,200–1,300 steps, and held 100% from step 1,700.
  - With wd = 0, grokking also occurred.
  - Negative transfer: a 54.5M-parameter model (4 layers, d 256, lr 3e-4, wd 0.1) trained 10k steps on a capital-city corpus never produced held-out capitals (0%). Known-capital probability peaked at about 60% at step 2k, then destabilised.
- **Quote** (結果): 「重み減衰なし（λ=0）でもGrokkingは起きました。」
- **Grade:** C.
- **Novelty:** the closest public replication to #249's exact recipe. It also gives a wd = 0 data point that contradicts several other Japanese posts (see Contradictions).

### ja-14: モジュロ演算でのTransformerのgrokking現象の可視化してみた ("Visualizing grokking of a Transformer on modular arithmetic")
- **Identity:** shirohige (廣重, Institute of Science Tokyo, mathematics). note, 2026-01-06. https://note.com/shirohige11/n/n622aa414af43 (code: github.com/hiroshige1/grok_modulo)
- **Type:** blog with a code link. JAX on a Colab T4; training took about 3 minutes.
- **Setup:** p = 113, after Nanda et al. Hyperparameters are not restated.
- **Findings:**
  - Step 1,000: train accuracy 1.000, train loss 0.0182, validation accuracy 0.034.
  - Step 2,000: train loss rose to 0.0352, validation accuracy 0.111.
  - Step 3,000: validation accuracy 0.9981. Step 4,000: 1.000.
  - The PCA ring and a TDA H1 persistence point appear only after grokking.
- **Quote** (結果): 「valid Lossについては大体2300ステップ目に学習が達成されていました」
- **Grade:** C.
- **Novelty:** a replication. The TDA diagnostic is a small addition, and the train-loss rise during the transition is worth logging.

### ja-17: 網羅的統制実験によるTransformerの加算能力獲得機序の解明 ("Elucidating how Transformers acquire addition via exhaustive controlled experiments")
- **Identity:** 熊懐祐太, 吉永直樹 (U. Tokyo, IIS). NLP2026 (32nd Annual Meeting of the Association for NLP), TS4-7, pp. 1473–1478, March 2026; not peer-reviewed. https://anlp.jp/proceedings/annual_meeting/2026/pdf_dir/TS4-7.pdf
- **Type:** domestic conference paper. The PDF's Japanese text layer is garbled in extraction; numbers come from the tables.
- **Setup:** 1–3-digit + 1–3-digit addition, 1M problems, 80/10/10 per digit-combination; L2R greedy number tokenization (0–99); Transformer encoder with a causal mask; AdamW lr 1e-4, per-GPU batch 512 (2,048 total), dropout 0.1, GeLU, gradient clipping 1.0, 1,000 epochs.
- **Findings:**
  - Capacity threshold: at d = 32, 8 layers stall, while 16 layers show a sudden grokking.
  - Accuracy by configuration: (64, 4) 95.63%; (32, 16) 99.93%; (64, 8) 99.99%.
  - Data composition matters. Mixed 1-vs-3 and 2-vs-3-digit data is the strongest trigger, and those structures fit a T = 10 Fourier helix (R² 0.325).
  - The proposed two-phase "Grokking-inducing" curriculum (phase 1 on structure-promoting data, phase 2 adapt and refine) raised the (64, 4) model from 95.63% to 99.97% and R² from 0.261 to 0.372. Accuracy on 1d+1d fell to 0.00%.
- **Quote** (abstract or conclusion, as indexed by the search engine): 「これを意図的に誘導する Grokking 誘導型学習戦略を提案した」
- **Grade:** B.
- **Novelty:** new domestic-only work. Depth is a trigger, and data mixture acts as a grokking accelerator. It is not the finite-set modular recipe.

### ja-24 / ja-25: Grokkingとは？ 訓練99%から27,400ステップ遅れて汎化した実験 ("Grokking: test generalization arrived 27,400 steps after 99% train accuracy")
- **Identity:** Shake AI Lab. Blog post 2026-07-21, updated 2026-09-26: https://shakeailab.com/blog/grokking-sudden-generalization/ (code: github.com/shake-ai-lab/experiments). Companion note post 2026-08-02: https://note.com/shakeworks_ai/n/nd65d560cf56f
- **Type:** blog with code; single seed. The note post (ja-24) is a summary of the blog (ja-25).
- **Setup:** (a+b) mod 53, 2,809 pairs, 40% train (1,123 / 1,686); embedding 96 → concat → 256-unit ReLU → 53; NumPy Adam; full batch; lr 1e-3; up to 60k steps; logged every 200 steps.
- **Findings:**
  - Train accuracy ≥ 99% at step 200 in every condition.
  - Test ≥ 90% by weight decay: wd 0 never (final 0.000); wd 0.03 never (0.000); wd 0.1 never (0.317); wd 0.3 at 27,600; wd 1.0 at 10,400.
  - The weight norm peaks at about 74 near step 1,400, then falls.
  - Failure: coupled L2 (`g = g + wd*p` before Adam's moments) at wd = 1.0 gave 2.6% train accuracy for 40k steps.
  - The published code decays after the Adam update. The author notes this is not exactly standard AdamW.
- **Quote** (失敗談): 「この実装のwd=1.0では重みが極端に小さくなり、訓練精度2.6%(ほぼランダム)のまま4万ステップが終わった。」
- **Grade:** C.
- **Novelty:** a replication with a weight-decay sweep and a documented implementation failure mode that is directly relevant to ReLSM.

### Opened but low-value or non-evidence sources (D or method)

- **ja-05:** 7B MoEで探るLLMのGrokking現象 ("Exploring grokking in LLMs with a 7B MoE").
  - Identity: ke-suke-Soft, Qiita, 2025-06-29. https://qiita.com/ke-suke-Soft/items/d1ca3fe6c751ec25b126
  - A D summary of arXiv 2506.21551 (Li, Fan, Zhou), claiming grokking in one-pass OLMoE pretraining with test-free pathway metrics.
  - **Unreliable:** it contains `:contentReference[oaicite]` artefacts, and its implementation table (LoRA r = 32, A100×32, etc.) and numbers look machine-generated and unverified. Do not cite its numbers.
  - Quote: 「テスト不要の新指標（Pathway Similarity／Consistency）でリアルタイム可視化」
- **ja-07:** 【DL輪読会】"OMNIGROK: GROKKING BEYOND ALGORITHMIC DATA" ICRL2023 ("DL reading group: Omnigrok").
  - Identity: 若杉健介 (Panasonic HD), Deep Learning JP. Docswell, 2024-02-09. https://www.docswell.com/s/DeepLearning2023/ZQ8PDN-2024-02-09-133325
  - A D summary of Omnigrok, noting that a large initial norm induces grokking (Levi et al.).
  - Quote (p. 3): 「現象発現には、weight decayが最も効果的」
- **ja-08:** 深層学習における正則化手法が Grokking に及ぼす効果の検討 ("Effects of regularization methods on grokking in deep learning").
  - Identity: 美齊津諒, 渡部航史, 前田圭介, 小川貴弘, ジェプカ ラファウ, 伊藤敏彦, 長谷山美紀 (Hokkaido University, Media Dynamics Lab).
  - Venue: 令和7年度 電気・情報関係学会北海道支部連合大会 (2025 joint convention of the Hokkaido sections of the electrical and information societies), pp. 45–46. Also listed as a MIRU 2026 poster on a lab member page.
  - Lab news page, posted 2026-06-02: https://www-lmd.ist.hokudai.ac.jp/news/category/researches-achievement/4defense/
  - **Title only.** The paper body is not online, so its regularizer comparison is a lead with no evidence. ja-27 (lab member page, https://www-lmd.ist.hokudai.ac.jp/member/koshi-watanabe/) was opened for the MIRU listing.
- **ja-11:** a post by 岡野原大輔 (Preferred Networks) on X, 2026. https://x.com/hillbig/status/2114136106971337120
  - A D summary of Barron et al. Quote: 「FP32ではmが約16.6を超えた時点でこの問題が発生する」
- **ja-12:** 大規模言語モデルと機械論的解釈性：その展開と現在地 ("LLMs and mechanistic interpretability: development and current state").
  - Identity: 峰岸剛基, Speaker Deck, 2026-03-01. https://speakerdeck.com/gouki/da-gui-mo-yan-yu-moderutoji-jie-lun-de-jie-shi-xing-sonozhan-kai-toxian-zai-di
  - The slide text could not be extracted. The search snippet shows only a history timeline that mentions grokking.
- **ja-15:** 「わたしのべんきょうノート」 Grokking tag (AkihikoWATANABE), last updated 2025-12-09. https://akihikowatanabe.github.io/paper_notes/articles/Grokking-1.html
  - D: GPT-generated summaries of 7 papers.
- **ja-16:** 【深層学習】過学習の「向こう側」にある汎化：Grokking（頓悟）現象の数理メカニズム ("Generalization beyond overfitting: the mathematics of grokking").
  - Identity: lixian, Zenn, 2026-01-22. https://zenn.dev/lixian/articles/bad35e0cbbeeb2
  - D summary. It asserts that grokking does not occur at λ = 0, which conflicts with ja-13.
- **ja-18:** 文脈内構造切替下での大規模言語モデルにおける文脈内表現学習の幾何学的解析 ("Geometric analysis of in-context representation learning in LLMs under in-context structure switching").
  - Identity: 堀場幸也, 矢向高弘 (Keio). NLP2026 B5-15. https://www.anlp.jp/proceedings/annual_meeting/2026/pdf_dir/B5-15.pdf
  - Cites grokking only in passing.
- **ja-19:** J-STAGE full-text search for "grokking" (9 hits), plus searches for 「グロッキング」 (0 relevant) and 「遅延汎化」 (13 hits, none relevant). https://www.jstage.jst.go.jp/result/global/-char/ja?globalSearchKey=grokking
  - Method source. The only JSAI conference paper with grokking in the title is ja-01.
- **ja-20:** Mechanistic Interpretability : 解釈可能性研究の新たな潮流 ("Mechanistic interpretability: a new current in interpretability research").
  - Identity: 青木洸士郎, 高槻瞭大, 峰岸剛基, 趙羽風. 人工知能学会論文誌 41(4), AN40-D, 2026-07-01. https://www.jstage.jst.go.jp/article/tjsai/41/4/41_41-4_AN40-D/_html/-char/ja
  - A review article that cites grokking (Nanda 23) only in the references.
- **ja-21:** CiNii Research OpenSearch for "grokking" (48 hits). https://cir.nii.ac.jp/opensearch/all?q=grokking
  - The only Japanese-language research item is ja-01. It also lists the English TMLR 2024 paper "Towards Empirical Interpretation of Internal Circuits and Properties in Grokked Transformers on Modular Polynomials" (U. Tokyo group) as a pointer. No IPSJ 研究報告 (technical report) or thesis appears.
- **ja-22:** NLP2026 programme page, https://anlp.jp/proceedings/annual_meeting/2026/. No title contains grokking; TS4-7 is the closest.
- **ja-23:** NLP2025 programme page, https://www.anlp.jp/proceedings/annual_meeting/2025/. Zero grokking matches.
- **ja-26:** Fugu-MT, "Grokfast: Slow Gradientを増幅することで、グローキングを加速する" ("Grokfast: accelerating grokking by amplifying slow gradients"). https://fugumt.com/fugumt/paper_check/2405.20233v2
  - D: a machine-translation mirror and the only Japanese Grokfast material found.

## Contradictions with the standard English account (and inside the Japanese corpus)

1. **Does wd = 0 ever grok?**
   - Japanese MLP replications say no: ja-02 (0.05% test accuracy at 15k steps) and ja-25 (0% at 60k steps).
   - Plain AdamW with wd = 0 in a 1-layer Transformer also never grokked (ja-03).
   - Kanda's 2-layer, 4-head, d = 128 Transformer on mod 97 at frac 0.5 did grok with λ = 0 (ja-13).
   - This matches the English nuance (Prieto et al.: generalization without decay is possible until Softmax Collapse halts it, more so at a larger train fraction). It contradicts the popular summaries that say λ = 0 never groks (ja-16).
   - For ReLSM, a wd = 0 control at frac 0.5 may grok slowly, so it is not a clean "no-grok" negative control.
2. **Does fixing softmax numerics speed up grokking?**
   - Barron et al. claim 94 ± 5% for standard CE against 100 ± 0% for the fix (5 seeds, 10k epochs; ja-10).
   - A 15-seed replication (ja-09) finds no fp32 speed-up: both reach 8/15 seeds ending ≥ 95%, and standard CE reaches 95% on more seeds (11 against 8).
   - The benefits that did replicate: the slingshots disappear, accuracy stays stable once reached, and a bf16 loss improves sharply.
3. **StableMax as an accelerator.** Prieto et al. present StableMax as enabling grokking without regularization. In ja-03 it was about 6× slower than AdamW + wd, and Barron et al. (ja-10) report StableMax margins of 10³–10⁴ and lower accuracy than the fixed CE.
4. **Weight norm as the explanation.** ja-01 argues that weight-norm reduction is insufficient: a norm-matched dense network groks much later than the ticket. This is the same as the English version and adds no new data.
5. **Grokking delay magnitudes.**
   - With wd 1, lr 1e-3, p = 97 and frac 0.5, Japanese replications see test accuracy above 90% within about 1.3k–4k steps (ja-13, ja-04).
   - This is far shorter than the 10⁵-step delays popularised by Power et al., which came from other operations and smaller fractions.
   - It is not a contradiction, but the "10⁵ steps" folklore should not set ReLSM's expectations.

## Gaps (searched, nothing found)

- **Grokfast in Japanese:** no replication, critique or tips. Only the Fugu-MT mirror (ja-26) and crypto-token noise turned up. Nothing on the per-micro-batch filter or EMA initialisation (#137).
- **Embedding-specific weight decay / unused vocabulary rows:** no Japanese discussion of exempting embeddings from decay, of unused rows inflating the norm, or of GrokTransfer replication. Only an alphaXiv-ja mirror appeared in snippets; it was not opened.
- **JSAI2025 / JSAI2026 conference papers:** none found.
  - J-STAGE full-text search for grokking gives only JSAI2024 (ja-01) among JSAI conference papers.
  - Searches by session or confit page returned nothing.
  - JSAI2026 proceedings may be incompletely indexed.
- **NLP2025:** zero matches. **NLP2026:** only TS4-7 (arithmetic) and a passing citation (B5-15).
- **IBIS 2024–2025, IPSJ 研究報告 (technical reports), CiNii theses (修士論文 master's / 卒業論文 bachelor's):** nothing found.
  - The Hokkaido paper (ja-08) exists only as a title.
  - A Wantedly profile mentions a kosen-era grokking project with no output.
- **Tanaka Hidenori (NTT / Harvard) Japanese talks on grokking:** none found.
- **DL輪読会 (Deep Learning JP reading-group) decks on grokking from 2025–2026:** none found; the latest is the 2024 Omnigrok deck.
- **Japanese experiments on grokking in LLM pretraining:** none. Only summaries of English work exist (ja-05, which is unreliable).
- **bf16 / fp16 failure reports outside the numerics posts:** none beyond ja-03 and ja-09.
- **Expected but not found:** a Japanese replication of ⊥Grad other than ja-03, Muon-for-grokking posts, and a published version of the Hokkaido regularizer study.

## Method

- **Tools:** SearXNG (`language: ja`, some queries with `time_range: year`); WebSearch as a second engine; `web_url_read` and WebFetch to open pages. PDFs from J-STAGE and anlp.jp were downloaded to the scratchpad and text-extracted with pypdf (`python -I`). Direct J-STAGE and CiNii API queries were made with curl.
- **Queries (main):**
  - Grokking 宝くじ仮説 人工知能学会; grokking 遅延汎化 モジュラー加算; Grokking 人工知能学会全国大会 2025 論文 (WebSearch).
  - "grokking" 人工知能学会全国大会論文集 JSAI2025; "Grokking" JSAI2026 J-STAGE 汎化 (0 results); jstage pjsai JSAI2025 Grokking (WebSearch).
  - "正則化手法が Grokking に及ぼす効果"; 美齊津 Grokking 正則化 MIRU2026.
  - Grokking 情報処理学会 研究報告 遅延汎化; Grokking 言語処理学会 NLP2025; NLP2026 grokking; IBIS grokking.
  - Grokfast 実装 試してみた (0); Grokfast (ja); Grokfast グロッキング 加速 EMA (WebSearch, no Japanese hits).
  - Softmax Collapse StableMax grokking 解説; grokking 埋め込み層 weight decay (0); GrokTransfer 解説 (no relevant hits).
  - grokking 修士論文 OR 卒業論文; 田中秀宣 grokking (0); grokking 再現 起きない 失敗 weight decay; grokking bf16 fp16; Grokking ticket 峰岸.
  - J-STAGE full-text: grokking, グロッキング, 遅延汎化. CiNii: grokking. Programme pages: NLP2025 and NLP2026 grepped.
- **Counts:**
  - Results seen: about 600 search-result entries across 24 queries, plus 9 + 13 J-STAGE hits, 48 CiNii hits and 2 programme pages.
  - Unique sources opened: 27 (ja-01 … ja-27). Two fetch failures are not counted: the IEICE Hokkaido author index returned empty content, and Zenn shakeailab returned 403.
  - Retained evidence-bearing: 12 (ja-01, 02, 03, 04, 06, 09, 10, 13, 14, 17, 24, 25).
  - Non-evidence or method: 15.
- **Caveats:**
  - Japanese quotes from ja-01 and ja-17 come from J-STAGE abstract text and the search-index text layer, because pypdf garbled the Japanese glyphs.
  - Numeric claims from ja-01 and ja-17 come from English captions, tables and digits in the PDFs.
  - Blog numbers are mostly single-seed (except ja-09).
