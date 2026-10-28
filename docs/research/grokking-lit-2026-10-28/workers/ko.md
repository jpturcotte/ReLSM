# Grokking literature sweep: Korean-language lane (ko), as of 2026-10-28

Worker lane: Korean-language sources only. English papers by Korean authors are listed only as pointers. The GitHub threads of the Grokfast authors (Seoul National University) are in English. They are kept because the brief asks for the group's own tuning and failure advice, and they are flagged as English-language.

Bottom line: Korean-language grokking work is thin. RISS lists one domestic conference paper (KIISE 2025) and two 2026 theses that touch grokking. The rest is reviews, auto-translated summaries, and two Korean-language GitHub write-ups with real runs. I found no Korean-language talk, blog post or interview by the Grokfast authors, and no Korean press coverage of Grokfast. The most decision-relevant material is (a) the official Grokfast code and the author's issue replies, and (b) two small Korean replications that give concrete step counts and failure signatures.

## Takeaways (for ReLSM's first grokking run, issue #249)

1. **Run the baseline without Grokfast, and check ReLSM's filter against the official code before reusing it.**
   - The reference `gradfilter_ema` initializes its EMA from the first gradient, not from zeros. Its defaults are alpha=0.98, lamb=2.0. The moving-average variant uses window 100, lamb 5.0, with warmup. It is called once per optimizer step, between `backward()` and `step()` (ko-15).
   - ReLSM's zero-init EMA and its per-micro-batch application (#137) both depart from this reference.
   - The author's own advice (ko-10): fix the task-default weight decay, tune the filter first, then try weight decay at 1×, 2×, 5× and 10×. 100× did not help. The learning rate can be raised about 1.5–2× over the baseline.
   - Known failures, all acknowledged by the author:
     - gradient norms can grow exponentially (ko-11);
     - fp16 AMP plus GradScaler is unsolved, and the loss scale explodes or collapses (ko-12);
     - an independent user found Grokfast+Adam failed on tasks where plain AdamW grokked, and the author replied that the filter parameters are task-sensitive (ko-13).
   - Grokfast's README tunes the algorithmic task with weight decay 0.005–0.01 (ko-15). That is far below #249's 1.0, and none of these sources reports Grokfast combined with weight decay 1.0.
   - **Action:** keep Grokfast as a separate, later arm. Run it in fp32, log the gradient norm, use the reference defaults, and apply the filter once per optimizer step.

2. **Expected timeline and signatures from two Korean-language replications of the #249 recipe (both full batch, AdamW, lr 1e-3, weight decay 1.0).**
   - **ko-08, MLP 256 hidden, p=97, 30% train:**
     - validation accuracy 0.00 at 4k steps, 0.16 at 8k, 0.54 at 12k, 0.84 at 16k, 0.97 at 20k;
     - with weight decay 0, it stays at 0.00 for all 20k steps.
   - **ko-20, 1-layer transformer, d=64, 4 heads, 38,016 parameters, p=37, 35% train, loss on the answer position only:**
     - train accuracy passes 99% at step 122, validation passes 90% at step 5,400 and 99% at step 6,200;
     - during the plateau, validation loss *rises* from 4.2 to 5.0, a textbook early-stopping trigger;
     - recurring training-loss spikes of about 100× (e.g. 0.0003 → 0.0169) drop validation accuracy for one report interval, then recover. The author reads them as decay fighting the loss, not as a bug.
   - **Action:** do not early-stop or checkpoint-select on validation loss; track validation accuracy. Treat loss spikes as expected. With train fraction 0.5, these numbers suggest grokking well inside 1e5 steps. If validation accuracy is still at chance after about 3–5e4 steps, suspect a defect (#186, #218) before blaming the budget. This is my extrapolation, not a measured bound for d=128 and 2 layers.

3. **Fix the embedding weight-decay exemption (#186) before the run, or shrink the vocabulary to the task.**
   - The Korean replication uses a task-sized vocabulary (38 tokens) and stresses that decoupled decay keeps acting when the loss gradient is about zero (ko-20).
   - My inference, not a source claim: under AdamW, a row that gets exactly zero gradient decays as (1 − lr·λ)^t, which is about e^−5 after 5k steps at lr=1e-3 and λ=1. So the 31k unused rows (#218) would shrink by themselves, but only if the embedding is *not* exempt (#186).
   - If the output head is tied to the embedding, unused rows also take softmax gradient, and they then affect logits and the weight-norm dynamics.
   - Simplest path for #249: restrict the vocabulary or logits to the 97 residues plus delimiter tokens, and score only the answer position, as ko-20 does.

4. **Budget for seeds and the transition; never read a short-run negative as "no grokking".**
   - ko-21 is a pre-registered toy screen in a Korean repo (d=64, L=6, binding/XOR task). Accuracy is at chance (0.50, 0.44) at 3.6k and 7.2k steps, jumps to 0.875 at 14.4k, and saturates at 1.0 by 28.8k.
   - At 60k steps, three seeds gave 1.00, 0.75 and 0.78. A higher weight-decay floor (0.1) gave 0.81, against 1.00 at the default.
   - The author concludes that every earlier negative measured at 3.6k steps cannot be interpreted.
   - The ridge-regression theory reviewed in Korean (ko-02; ICML 2026 honorable mention) gives a grokking time t₂ ∝ 1/(η·λ). Halving lr × weight decay should roughly double the delay.
   - **Action:** use at least 3 seeds; scale the step budget by 1/(lr·wd) when either changes.

5. **Cheap progress measures to log.**
   - Taehun Cha (Korea University PhD, 2026, ko-05) reports that Target Linearity, the R² of a linear regression fitted to the learned representations, correlates with the onset of grokking.
   - A KIISE 2025 paper (ko-03) reports the same delayed pattern in shortcut learning and points to an NTK lens.
   - These are abstract-level claims only, with no numbers. Still, a linear-probe R² on the residual stream per checkpoint is cheap and could sit next to the weight norm.

## Sources (retained, evidence-bearing)

### ko-02: [논문리뷰] 외운 뒤에야 비로소 이해한다 — To Grok Grokking: Provable Grokking in Ridge Regression (ICML 2026)
- **English title:** "[Paper review] Only after memorizing does it understand — To Grok Grokking…"
- **Identity:** 카이스토, tistory blog "인공지능과 주식이야기", 2026-10-01. https://bluediary8.tistory.com/1012
- **Type:** blog (paper review).
- **Findings:** Korean walkthrough of arXiv:2601.19791 (Xu, Vardi, Safran):
  - the weight splits into θ∥ (the data span) and θ⊥ (its complement, decayed only by weight decay);
  - three phases: overfit, plateau, generalize;
  - t₁ (end of overfitting) is roughly ∝ n/(η·λ_min), and t₂ (onset of generalization) is roughly ∝ 1/(η·λ). The reviewer flags these forms as simplified.
  - Relevance to ReLSM: an unused embedding row is exactly a "θ⊥" direction, which the data never informs and only weight decay removes.
- **Quote:** "가중치 감쇠 λ가 작을수록(정규화가 약할수록) 일반화가 시작되는 시점이 뒤로 크게 밀린다" (section "그로킹 시점을 정량화하는 경계식").
- **Grade:** D.
- **Novelty:** summary only.

### ko-03: 지름길 학습과 그로킹의 연관 관계에 대하여
- **English title:** "On the Relationship between Shortcut Learning and Grokking"
- **Identity:** 임진우 (Jinwoo Lim) and 문수묵 (Soo-Mook Moon). 한국정보과학회 학술발표논문집 (KIISE conference proceedings, likely KCC 2025), Vol. 2025 No. 7, pp. 1060–1062. https://www.riss.kr/link?id=A109886547 (full text paywalled on DBpia).
- **Type:** domestic conference paper.
- **Findings:**
  - Defines grokking as mid-training high train and low test performance.
  - Notes NTK-related theories proposed to explain it.
  - Claims the same delayed pattern appears in shortcut learning, so NTK theory may transfer to shortcut-learning research.
  - No numbers in the abstract; I could not read the full text.
- **Quote:** "본 논문에서는 지름길 학습(shortcut learning)에서도 동일한 현상이 일어남을 밝히고" (RISS abstract).
- **Grade:** B (3-page domestic proceedings; only the abstract was seen).
- **Novelty:** new framing (grokking ↔ shortcut learning) that I did not see in the English pointers; the evidence is unverified.

### ko-04: 수학 문제에 대한 신경망의 가능성: 그로킹과 물리정보신경망을 중심으로
- **English title:** "The potential of neural networks for mathematical problems: Focusing on Grokking and PINN"
- **Identity:** 김민석 (Kim Min-seok), advisor 김연응 (Kim Yeon-eung). 서울과학기술대학교 인공지능응용학과 (SeoulTech, Dept. of AI Applications). Master's thesis, 2026-02, 56 pp. https://www.riss.kr/link?id=T17391114
- **Type:** thesis (석사).
- **Findings:**
  - Chapter 2, "Arithmetic Generalization: Grokking", covers the arithmetic manifold, the Kolmogorov–Arnold representation and modular addition, and the problems and model.
  - The abstract says the author's experiments concern representation geometry and transfer effects. No numbers are given.
  - Likely the thesis form of ko-06.
  - Full text not accessible.
- **Quote:** "수학 문제에 대한 신경망의 가능성: 그로킹과 물리정보신경망을 중심으로" (title; the record has no Korean abstract text).
- **Grade:** B (thesis, but only the abstract and table of contents were seen).
- **Novelty:** domestic thesis; its content probably overlaps ko-06.

### ko-05: Two Faces of One Gram: Understanding Feature Learning with Virtual Covariance and Target Linearity (가중치 그람 행렬에 대한 두 가지 관점: 가상 공분산과 타겟 선형성)
- **English title:** as given (Korean title: "Two perspectives on the weight Gram matrix: virtual covariance and target linearity").
- **Identity:** Taehun Cha (차태훈), advisor Donghun Lee. 고려대학교 대학원 수학과 (Korea University, Applied Mathematics). PhD thesis, 2026-08, 114 pp. https://www.riss.kr/link?id=T17574313
- **Type:** thesis (박사).
- **Findings:**
  - "Virtual Covariance" tracks weight-Gram updates with Pearson about 0.98, against about 0.71 for AGOP.
  - Target Linearity (TL) is the R² of a linear fit on learned representations. TL provably rises under gradient descent and with depth.
  - Neural Collapse is an extreme case of TL maximization.
  - TL correlates with the generalization gap and with the onset of grokking. No grokking numbers are in the abstract.
- **Quote:** "그로킹（grokking）의 발현과도 상관관계가 있음을 확인한다" (Korean abstract on RISS).
- **Grade:** B (abstract only).
- **Novelty:** new candidate progress measure, a linear-probe R².

### ko-06 (pointer): Acceleration of Grokking in Learning Arithmetic Operations via Kolmogorov-Arnold Representation
- **Identity:** Yeachan Park, Minseok Kim, Yeoneung Kim (SeoulTech group). arXiv:2405.16658, 2024-05-26. https://arxiv.org/abs/2405.16658
- **Type:** paper (English, pointer only).
- **Findings:** proposes data augmentation for commutative operations, plus transfer learning that uses shared Kolmogorov–Arnold structure across binary operations. With embedding transfer, operations can be learned from fewer tokens. No numbers in the abstract.
- **Quote:** n/a (English pointer).
- **Grade:** B.
- **Novelty:** pointer, included only because ko-04 appears to be its domestic thesis form.

### ko-08: Grokking (그로킹) — Ultralytics 용어집 (Korean edition)
- **English title:** "Grokking — Ultralytics glossary"
- **Identity:** Ultralytics (non-Korean company; this is the localized Korean page). Undated. https://www.ultralytics.com/ko/glossary/grokking
- **Type:** code with reported results (localized glossary).
- **Findings:**
  - Runnable PyTorch script: one-hot (a,b) → MLP(194→256→97), p=97, 30% of 9,409 pairs for training, AdamW lr 1e-3, weight decay 1.0, full batch, 20k steps, seed 0, CPU in under a minute.
  - At step 4,000, train accuracy is 1.00 and validation 0.00. Validation then reaches 0.16 at 8k, 0.54 at 12k, 0.84 at 16k and 0.97 at 20k.
  - With weight decay 0.0, validation stays at 0.00 for all 20k steps.
  - Also mentions the Softmax Collapse and StableMax/⊥Grad account.
  - It warns that early stopping (e.g. YOLO `patience`) ends grokking runs mid-plateau.
- **Quote:** "`weight_decay=0.0`을 사용하면 같은 실행에서도 20,000단계 내내 검증 정확도가 0.00에 머뭅니다." (section "그로킹이 진행되는 과정").
- **Grade:** C.
- **Novelty:** a small replication with numbers. It is a translation of an English page, but the numbers make it a useful 1-minute sanity check for the data split, optimizer and weight-decay path.

### ko-10: ironjr/grokfast issue #1 "Choosing weight decay?" (English)
- **Identity:** question by timothykassis, replies by ironjr (Jaerin Lee, SNU, Grokfast first author), 2024-06-17 and 06-18, follow-up 06-23. https://github.com/ironjr/grokfast/issues/1
- **Type:** forum (GitHub issue).
- **Findings:**
  - Author's rule of thumb, stated as not rigorous: (1) start from the task's default weight decay; (2) tune the filter (momentum or window, amplitude) first; (3) raise weight decay to 2×, 5× and 10×. 100× gave no improvement.
  - The default learning rate stays at about 1e-3 to 1e-4. The maximum stable learning rate is similar with and without Grokfast, and the author tested raising it about 1.5–2.0×.
- **Quote:** "Although weight decay do affect the values of the optimal filter parameters, its effect seems to be insignificant in my experiences."
- **Grade:** C.
- **Novelty:** first-party tuning advice that is not in the paper.

### ko-11: ironjr/grokfast issue #9 "Exploding Gradients" (English)
- **Identity:** opened 2024-06-30 by DustinEwan; author reply 2024-07-01; closed 07-10. https://github.com/ironjr/grokfast/issues/9
- **Type:** forum.
- **Findings:**
  - Gradient norm grew from about 3.8e12 to about 7e17 while the loss stayed near 11.
  - Proposed fix: normalize to (g + λ·ḡ)/(1+λ).
  - The author says this is equivalent to a lower learning rate. He frames Grokfast as a mix of slow, large-batch-like gradients and fast gradients. That view breaks down when parameters move fast, because the old filtered gradients were taken at stale parameters.
  - He suggests plotting the mean and standard deviation of both components.
- **Quote:** "Grokfast can be regarded as a mixture of slow and fast gradients."
- **Grade:** C.
- **Novelty:** first-party failure mode. It matches the exponential-feedback risk in ReLSM #137.

### ko-12: ironjr/grokfast issue #10 "How to use Grokfast with FP16 mixed precision training?" (English)
- **Identity:** opened 2024-07-03 by peterjc123; replies 07-08 (damian0815) and 07-10 (ironjr); still open. https://github.com/ironjr/grokfast/issues/10
- **Type:** forum.
- **Findings:**
  - Calling the filter between `scale` and `unscale_` made the loss scale explode.
  - Calling it after `unscale_` drove the scale to 0 and gave NaN losses.
  - The author calls the code a proof of concept, says mixed precision needs more careful step-size control, and promised a later version. No fix is recorded.
- **Quote:** "increased gradient norm due to the added low-pass filtered gradient"
- **Grade:** C.
- **Novelty:** first-party, unresolved AMP incompatibility.

### ko-13: ironjr/grokfast issue #12 "AdamW better than grokfast + Adam?" (English)
- **Identity:** Zhi0467 and ironjr, 2024-07-10. https://github.com/ironjr/grokfast/issues/12
- **Type:** forum.
- **Findings:**
  - Multiplication task, weight decay 0.01: AdamW grokked, while EMA Grokfast + Adam failed. That run used α 0.99 and λ 5.0, not the suggested 0.98 and 2.0.
  - x²+xy+y² mod 113, weight decay 5e-3: AdamW reached the expected accuracy, while Grokfast (0.98, 2.0) + Adam failed to learn.
  - The author answered that filter parameters seem task-sensitive and that filter design needs more study. The issue was closed without a resolution.
- **Quote:** "Your precious report seems to imply that the filter parameters are currently sensitive to the tasks in use."
- **Grade:** C.
- **Novelty:** negative replication, acknowledged by the author.

### ko-15: ironjr/grokfast repository (README and `grokfast.py`)
- **Identity:** Jaerin Lee et al., SNU Computer Vision Lab, 2024. https://github.com/ironjr/grokfast (source read verbatim via raw.githubusercontent).
- **Type:** code.
- **Findings, from the source (verified):**
  - `gradfilter_ema(m, grads=None, alpha=0.98, lamb=2.0)`.
  - On the first call, `grads = {n: p.grad.data.detach() ...}`, so the EMA starts at the first gradient. The first filtered gradient is therefore g + 2g = 3g.
  - It then sets `p.grad.data = p.grad.data + grads[n] * lamb`.
  - `gradfilter_ma` uses window_size=100, lamb=5.0, filter_type 'mean', warmup=True (no amplification until the window is full).
- **Findings, from the README (via fetch):**
  - Algorithmic transformer settings: MA λ 5.0, window 100, weight decay 0.01; EMA α 0.98, λ 2.0, weight decay 0.005.
  - Iterations to 95% validation accuracy: 39,890 (baseline) → 790 (MA) and 910 (EMA).
  - MA raised per-iteration latency from 0.15 to 0.37 s and VRAM from 290 to 458 MB.
- **Quote:** `grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}` (`grokfast.py`, line 43).
- **Grade:** C.
- **Novelty:** the reference behavior that ReLSM's implementation should match: EMA starts at the first gradient, applied once per step.

### ko-17: Muon Optimizer Accelerates Grokking — Daily Arxiv (haebom)
- **English title:** as given.
- **Identity:** 안광섭 (haebom), slashpage "Daily Arxiv". Undated; the paper is arXiv:2504.16041 (2025). https://slashpage.com/haebom/d367nxm3xd55k2j98pv1
- **Type:** blog (Korean summary).
- **Findings:**
  - 7 numerical tasks, mainly modular arithmetic.
  - Compares Muon against AdamW, and softmax against stablemax and sparsemax.
  - Muon cut mean grokking epoch from 153.09 to 102.89 (t=5.0175, p=6.33e-08).
- **Quote:** "Muon은 모든 구성에서 평균 그로킹 에포크를 153.09에서 102.89로 줄였으며" (개요).
- **Grade:** D.
- **Novelty:** summary only.

### ko-19: ICML 2026에서 우수 논문상과 Test of Time상을 수상한 10편의 논문 소개 (PyTorchKR)
- **English title:** "Introducing the 10 ICML 2026 outstanding-paper and Test-of-Time award papers"
- **Identity:** 9bow, discuss.pytorch.kr topic 11112, 2026-07-06. https://discuss.pytorch.kr/t/11112
- **Type:** forum (Korean summary).
- **Findings:**
  - Korean summaries of two grokking-related award papers.
  - "How much do language models memorize?": about 3.6 bits per parameter. Models memorize until capacity fills, then "grokking" begins and unintended memorization drops. Hundreds of transformers from 500K to 1.5B parameters were trained. This is the grokking-in-LM-pretraining topic.
  - "To Grok Grokking" (ridge regression), as in ko-02.
- **Quote:** "용량이 가득 차는 시점에 \" 그로킹(grokking) \"이 시작되고" (memorization-paper section).
- **Grade:** D.
- **Novelty:** summary only.

### ko-20: D8. 그로킹 재현 (codechosun/hangul-engine)
- **English title:** "D8. Reproducing grokking"
- **Identity:** codechosun, GitHub `hangul-engine`, chapter `ch-D08-grokking`, commit 2026-09-15. https://github.com/codechosun/hangul-engine/tree/main/ch-D08-grokking (a C++ from-scratch transformer textbook; the commit carries a Claude co-author trailer).
- **Type:** code with reported results.
- **Setup:**
  - task (a+b) mod 37, input `[a, b, =]`, vocabulary 38;
  - 1369 pairs: 479 train (35%), 890 held out;
  - 1 layer, d_model 64, 4 heads, FFN 128, 38,016 parameters;
  - AdamW lr 1e-3, weight decay 1.0 (100× its earlier chapter), full batch, 8,000 steps, seed fixed;
  - cross-entropy on the last position only, with upstream gradient zeroed elsewhere;
  - softmax computed in double with max subtraction.
- **Results:**
  - train accuracy over 99% at step 122; validation over 90% at step 5,400 and 99% at step 6,200 (a gap of 5,278 steps);
  - plateau from step 200 to 3,000: train loss about 0.0003, validation accuracy about 31%, validation loss rising 4.2 → 5.0;
  - recurring spikes, e.g. at step 1,800 train loss 0.0169 and validation accuracy 26%; spikes also at 3,800, 5,200 and 6,000;
  - lowering the learning rate calms the spikes but delays grokking;
  - 306.8 s for 8,000 steps on CPU;
  - the code was verified beforehand by gradcheck (max relative error 6.787e-07) and a NumPy curve match (6.661e-15).
- **Claimed regimes:** weight decay 0 gives no grokking; 10.0 gives no learning; a narrow middle window works. Train fraction also has a window. These are stated as expectations, not runs.
- **Quote:** "흔들림(손실이 100배 튀는 자리)은 버그가 아니라 감쇠와 손실이 맞서는 자리" (정리).
- **Grade:** C.
- **Novelty:** an independent replication with step-level numbers and practical tips: answer-only loss masking, no early stopping on validation loss, spikes are expected, verify the code first.

### ko-21: H_9817 — 재조합 벽은 표현력 벽인가 최적화 벽인가 (그로킹 레짐 스크린 · 사전등록)
- **English title:** "Is the recombination wall a capacity wall or an optimization wall? (grokking-regime screen, pre-registered)"
- **Identity:** dancinlab, GitHub `anima`, `HYPOTHESES/cards/H_9817_grokking_regime_screen.md`, last commit 2026-07-20. https://github.com/dancinlab/anima/blob/main/HYPOTHESES/cards/H_9817_grokking_regime_screen.md
- **Type:** code with reported results (research log).
- **Setup:** toy XOR-binding task (d=64, L=6, seq 96, batch 8, seed 7), a 2×2 grid of steps × weight-decay floor, with a pre-registered decision table.
- **Results:**
  - at 3.6k steps, accuracy 0.50 with either weight decay;
  - 60k steps at default weight decay: 1.00; 60k steps at weight-decay floor 0.1: 0.8125;
  - curve: 0.50 (3.6k), 0.44 (7.2k), 0.875 (14.4k), 1.0 (28.8k), 0.97 (43.2k), 1.0 (60k);
  - seeds 7, 11 and 4302 at 60k: 1.00, 0.75, 0.78 (torch arm);
  - held-out-vocabulary accuracy 0.69.
- **Author's conclusion:** every earlier negative at 3.6k steps cannot be interpreted. The effect size varies by seed.
- **Quote:** "7200 까지 완전히 우연에 눌려 있다가 14400 에서 급격히 올라 28800 부터 포화." (굳히기 ①).
- **Grade:** C (single toy setup, small n; the author self-reports the statistical power).
- **Novelty:** a practical lesson about negative results under budget and about seed variance. The task is not modular arithmetic.

## Opened but not evidence-bearing (kept in JSONL)
- **ko-01** velog pnuaid1020 "Grokking 현상 탐구" (2024-05-31): a generic student-club column with no numbers. It lists dropout, batch norm and dataset size as triggers without evidence. Grade D.
- **ko-07** GeekNews 10292 (2023-08-11): a Korean summary of the PAIR "grokking explorable" plus GN⁺ translations of Hacker News comments. One comment corrects that weight decay (L2) does not produce sparsity, while L1 does. Grade D.
- **ko-09** jaerinlee.com: the Grokfast first author's homepage. It lists no Grokfast talk, blog post or follow-up, only a 2022 Korea Military Academy HPC talk and 2026 optimizer and edge-of-stability preprints. Grade D.
- **ko-14** grokfast issue #16 (2025-04-06): asks for a none/MA/EMA comparison; no replies. Grade D.
- **ko-16** SlideShare "220109 DL paper review: Grokking" (딥러닝논문읽기모임, presenter 이근배, 2022-01-09): reviews the original Power et al. paper; no replication. Grade D.
- **ko-18** HyperAI Korean wiki "그로킹": a machine-translated summary of the lazy→rich account. Grade D.

## Contradictions with the standard English account
- **Coupled L2 vs AdamW (ko-20).** The author claims that with coupled (non-decoupled) decay, "the decay dies along with the gradient" once the loss gradient is about zero. By my analysis this is wrong for Adam: with coupled L2, the λw term passes through the adaptive normalization, so the update is about lr·sign(w). The decay is strong, not dead. The claim holds for neither SGD nor Adam. The practical recommendation (use AdamW with decoupled decay) is still standard. Flagged as an unreliable mechanism claim.
- **Grokfast's ×50 speed-up vs user reports (ko-13, ko-11, ko-12).** The README reports 39,890 → 910 iterations with EMA. Independent users reported failure or instability, and the author conceded task sensitivity. These sources record no resolution.
- **Grokfast's weight-decay regime vs #249.** The README's algorithmic recommendations use weight decay 0.005–0.01 (ko-15). The #249 recipe uses 1.0 on all weights. Results from one regime may not transfer to the other, and no source tests Grokfast together with weight decay 1.0.
- **"Grokking is solved" (lead only, not verified).** Chester Roh (X, January 2025) reportedly wrote that StableMax, a stronger weight decay and a gradient filter would make grokking "disappear". I could not open the post (HTTP 402). The English account (Prieto et al. 2025) says StableMax and ⊥Grad *induce* grokking without weight decay by avoiding Softmax Collapse, which is not the same as removing it.
- **Large models (ko-01) vs pretraining evidence (ko-19).** ko-01 asserts that grokking is hard to observe in large models. ko-19's summary of the memorization-capacity paper describes grokking setting in once capacity fills, in LMs up to 1.5B parameters. The disagreement is about framing, not data.

## Gaps
- **Grokfast in Korean.** No Korean-language talk, blog post, interview, slides or press coverage by the SNU group or anyone else. No Korean discussion of accumulation, AMP, EMA initialization or clipping order. The only first-party material is the English GitHub threads (ko-10 to ko-14). Searches for 그록패스트, 그로크패스트, "Grokfast 리뷰" and "Grokfast 서울대" found nothing Korean.
- **Domestic venues.**
  - DBpia's search page is JS-rendered and returned 0 results through fetch.
  - The KCI portal did not render results.
  - RISS gave 2 articles for "grokking" (one relevant, ko-03; the other a Muon/ResNet paper that does not cover grokking), 1 for "그로킹", 0 for "Grokfast", and none for "지연된 일반화" in the ML sense.
  - RISS theses gave 2 domestic hits (ko-04, ko-05).
  - ScienceON gave 35 mostly international records; the only Korean-authored one was the SSRN version of ko-06.
  - Expected but not found: KCC or KSC papers on modular-arithmetic grokking, StableMax, or weight-decay schedules.
- **Not found at all:** Korean replications of StableMax or ⊥Grad, of bf16/fp16 effects on grokking, of slingshot dynamics, or of weight-decay schedules. Also no Korean YouTube talks (PR12, 모두의연구소, 서울대) from 2025–2026; the only Korean study-group material found is from 2022 (ko-16).
- **Inaccessible:**
  - Naver Blog (WebSearch refuses the domain; SearXNG returned spam);
  - aitrends.kr article 34183 "Clip to Grok… 39~249배 가속" (HTTP 410 Gone; it covers a self-reported Reddit result on weight-norm clipping, which the English lane should check);
  - the aitrends.kr glossary (410);
  - Chester Roh's X post (402);
  - the full texts of ko-03, ko-04 and ko-05.
- **Pointers for the English lane (Korean authors, English papers, not summarized here):**
  - Taeyoung Kim (KIAS), "Grokking on the Weight-Decay Clock" (arXiv:2607.23967). Search snippets suggest a (1−β)/(ηλ) time scaling under heavy-ball optimization; not verified.
  - Park, Kim and Kim, arXiv:2405.16658 (ko-06).
  - Jaerin Lee's 2026 optimizer and edge-of-stability preprints (arXiv:2512.06370, 2608.18415, 2609.32290).

## Method
- **Tools:** SearXNG (`language: "ko"`, some with `time_range: "year"`), WebSearch (standard and extended, including domain-restricted runs), `web_url_read` and WebFetch to open pages, curl for RISS and PyTorchKR JSON and raw GitHub files, and `gh` for GitHub code search and file reads.
- **SearXNG queries:**
  - 그로킹 grokking 일반화 딥러닝; Grokfast 그로킹 가속;
  - 그로킹 현상 논문 dbpia OR kci 지연된 일반화;
  - "grokking" KCC 2025 OR KSC 2025 OR "학술발표논문집" 그로킹;
  - Grokfast 논문 리뷰 EMA 필터 그래디언트 (0 results);
  - discuss.pytorch.kr grokking OR grokfast; velog grokfast; news.hada.io grokking;
  - 이재린 Grokfast 서울대 이경무 그로킹 (0 results);
  - Softmax Collapse StableMax 그로킹 수치 안정성 리뷰;
  - 그로킹 grokking 실험 재현 모듈러 연산 트랜스포머 블로그 (year);
  - blog.naver.com 그로킹 …; brunch.co.kr 그로킹 …;
  - "Progress measures for grokking" 리뷰 …; Grokked Transformers … 리뷰 그로킹 추론;
  - 그로킹 float32 float64 정밀도 소프트맥스 붕괴;
  - 그로킹 grokking 유튜브 논문 리뷰 지연된 일반화 설명.
- **WebSearch queries:**
  - grokking 그로킹 velog OR tistory 모듈러 덧셈 weight decay;
  - "그로킹" 한국정보과학회 학술발표 일반화 트랜스포머;
  - Grokfast 느린 그래디언트 증폭 서울대 … 그로킹 가속;
  - Grokfast 논문리뷰 저주파 그래디언트 그로킹 50배;
  - 이재린 Jaerin Lee 그록패스트 … 인터뷰 OR 강연;
  - youtube 딥러닝논문읽기모임 Grokking 리뷰;
  - "그로킹" 신경망 블로그 가중치 감쇠 실험 … (extended);
  - two domain-restricted runs (velog, tistory, brunch, medium, github.io, github.com, discuss.pytorch.kr);
  - KIAS grokking heavy-ball …;
  - grokking 그로킹 논문 학술대회 (restricted to dbpia, kci, riss, scienceon, kiss);
  - 서울대 연구팀 그로킹 가속 기술 Grokfast 뉴스;
  - "그로킹" 현상 KAIST OR 서울대 OR 포스텍 … (extended);
  - 모두의연구소 grokking …; 강태웅 김기남 김도현 KAIST …;
  - "지름길 학습과 그로킹의 연관 관계에 대하여";
  - "Clip to Grok" weight norm clipping grokking 249;
  - dbpia grokking transformer modular arithmetic 한국정보과학회 (extended).
- **Database and API queries:**
  - RISS articles: grokking, 그로킹, Grokfast, delayed generalization, 지연된 일반화;
  - RISS theses: grokking, 그로킹, modular addition transformer;
  - DBpia: grokking, 그로킹;
  - ScienceON: grokking; KCI: grokking;
  - PyTorchKR `search.json`: grokking, grokfast, 그로킹;
  - GitHub code search: "그로킹" (30 hits, 3 relevant ML repos).
- **Counts:** about 550 search results seen. 21 unique sources opened (ko-01 to ko-21). 5 more fetches failed: X 402; aitrends 410 ×2; the Grokfast project page, a redirect stub; one GeekNews 403, later fetched successfully. 15 evidence-bearing sources retained: ko-02 to ko-06, ko-08, ko-10 to ko-13, ko-15, ko-17, and ko-19 to ko-21.
- **Grades of the 15 retained:** 0 A, 4 B, 8 C, 3 D. The theses are graded B because only their abstracts could be read.
