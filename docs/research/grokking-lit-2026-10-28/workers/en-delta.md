# English delta lane: grokking work from 2025-06 to 2026-10 not in the existing maps

Lane: `en-delta`. Swept 2026-10-28. This report lists only papers that are absent from the coverage files (res-codex-lit, res-chatgpt-2026, res-chatgpt-dec, res-claude, res-new-exports, res-gemini-perplexity, literature_register_2026-10-24.json, lit-lexical-grokking, lit-puzzles-concepts, research_grokfast.md, and audit-2026-10-27-avenues.md). The merged inventory `grokking-research-and-ideas-2026-10-28.md` was not on disk when this lane finished, so it was not checked.

Quotes come from the arXiv abs or HTML pages, or the venue pages, as returned by WebFetch. Grades: A = peer-reviewed or accepted; B = preprint with a solid method; C = repo or blog with code and results; D = opinion or post.

## Takeaways (the 5 most decision-relevant)

1. **The #249 recipe sits on the fp32 softmax cliff. #255 needs more arms than StableMax.**
   - Barron, Gnanapragasam and Tschannen (en-01, Oct 2026) find that under AdamW with weight decay, the post-memorisation logit margin settles near log(1/ε). That is ≈18.4 at the default ε=1e-8.
   - The fp32 cross-entropy cliff is ≈16.6, and the bf16 cliff is ≈5.5. So ReLSM's planned run (fp32, wd=1, default ε) is expected to cross the cliff.
   - On a+b mod 31 (5 seeds), standard fp32 cross-entropy reached 94.3±5.4% test accuracy. Their fixed log-space cross-entropy reached 100.0±0.0%.
   - "Grokking or Glitching?" (en-02, accepted to NeurIPS 2026 according to the arXiv comments) shows that the Slingshot spikes come from precision. They vanish with an fp64 loss or Adam ε=1e-5.
   - **Action:**
     - Log the maximum logit margin and Adam's ε in #249.
     - In #255, add four arms: (a) loss computed in fp64; (b) ε=1e-5; (c) the Barron-style fixed cross-entropy; (d) a bf16 arm with the loss in fp32 versus bf16.
     - Read any loss spikes as numerical, not as phase structure.
2. **#251 should sweep and plot the product η·λ, not λ alone.**
   - Four groups find that, to leading order, the grokking delay scales as **1/(η·λ)**:
     - Pracher et al. (en-04): MLP and Transformer phase grids;
     - Luo et al. (en-05): a stochastic-geometric theory, 10 seeds per cell;
     - Truong et al. (en-06): 293 runs, R²=0.97 for weight decay and 0.92 for learning rate;
     - Boursier et al. (en-10, NeurIPS 2025): a theory of slow norm minimisation.
   - Strong decay can stop the model fitting at all, which gives the sweep an upper cutoff.
   - So the λ thresholds the maps quote (for example a λc boundary, or a "Goldilocks" band) only transfer at the same learning rate.
   - **Action:**
     - Add a small learning-rate cross-check.
     - Add one cheap causal control: freeze the norm, or turn weight decay off, at memorisation. In en-07 this stopped grokking in 0/6 runs, against 3/3 for the baseline.
     - Caveats: en-06 and en-07 share all four authors, and most of the theory assumes MSE or SGD.
3. **Sampling order can be a first-order variable, which matters for #252.**
   - LeDoux (en-32) uses p=9973 and 0.3% training data. Two fixed orderings reached 99.5% test accuracy at epochs 487 and 659. IID order reached 0.30% after 5,000 epochs, and an adversarial order prevented learning.
   - In shared-trunk multi-task training, Xu (en-14) finds a consistent grokking order: multiplication, then squaring, then addition.
   - Both are single-author preprints.
   - **Action:**
     - In #252, hold the example set fixed across arms and log the presentation order and seed.
     - For any multi-task variant of #253, expect tasks to grok one after another, not together.
4. **#254 (Grokfast) has no new independent transformer replication. Its competitors claim larger speedups, but against weak baselines.**
   - The only new Grokfast test is a single-seed community run on sparse parity (en-28). There, Grokfast was 17× slower because the task has no memorisation plateau.
   - The new accelerators also compare against weak baselines:
     - **Clip-to-Grok** (en-26): per-row ℓ2 clipping plus Lion, a 550-step median over 200 seeds versus 35,040 steps for AdamW with wd=0.01 (repo only);
     - **spherical residual stream** (en-24): over 20× faster without weight decay, but no gain on S5;
     - **GrokAlign** (en-22);
     - **predator-prey optimiser** (en-23, IEEE TAI);
     - **Low-Rank Decay** (en-25).
   - **Action:**
     - Benchmark every #254 arm against the tuned wd=1.0 baseline from #249.
     - Clip-to-Grok is the cheapest extra arm with real seed statistics.
5. **Diagnostics: train-only signals exist, but metrics read at the grok step are biased.** This affects #249 logging and #253.
   - Two candidate train-only signals:
     - the commutator defect (en-15), tested on SCAN and Dyck only;
     - the local learning coefficient, estimated from training data (en-19).
   - The Xu eigengap detector (en-17) fired in 15/15 grokking seeds and 0/15 controls, but only with an x² activation and MSE loss. With ReLU it never fired.
   - The "At-Grok" audit (en-20) finds that effective rank read at the grok step overstates the converged value by 3–5× on an MLP. Compression continues ≥10k steps after grokking.
   - **Action:** keep #249 running well past the grok step, and do not report representation metrics at it.
   - For #258, the only new loop result is Niu (en-33): a looped transformer and shared-memory rereading both cross an out-of-distribution grokking barrier that a dense model does not. That was on synthetic knowledge-graph QA. **No looped-transformer study of modular-arithmetic grokking was found**, so #258 would be filling a gap.

## Sources

Format: identity; findings; evidence; grade; effect on ReLSM. "In map" means present in the coverage files listed above.

### Precision and numerics (#255, #249)

**en-01: A Fix for Softmax and Cross-Entropy**
- **Identity:** Barron, Gnanapragasam, Tschannen. arXiv preprint 2610.31642 (v1 22 Oct 2026, v2 26 Oct 2026). https://arxiv.org/abs/2610.31642. Not in map; flagged by the other lanes.
- **Findings:**
  - Names two numerical bugs: "Softmax Collapse" (the target gradient rounds to 0) and "Softmax Cancellation".
  - Rewrites log-softmax in log space (softplus, expm1, log1p), leaving the largest logit out of the log-sum-exp.
  - Reports cliffs at ≈5.5 (bf16), 16.6 (fp32) and 36.7 (fp64).
  - With weight decay, the margin equilibrium is ≈log(1/ε): 18.4 at ε=1e-8, 27.6 at 1e-12, 36.8 at 1e-16.
  - Grokking setup: 1-layer ReLU transformer, width 64; a+b mod 31; 50% train; full-batch AdamW with lr 1e-3, λ=1 and β2=0.98; 10k epochs.
  - With the fix, the model groks faster on all 5 seeds: 100.0±0.0% versus 94.3±5.4% test accuracy.
  - Without weight decay (a²+b² mod 97, ε=1e-16), standard cross-entropy fell from 61% to 23% after a spike, while the fix rose to 85%. The fix also removes the Slingshot spikes.
- **Evidence:** "a precision cliff (≈5.5 in bf16, 16.6 in fp32, and 36.7 in fp64)" (Fig. 1 caption, HTML v1).
- **Grade:** B (strong group; 5 seeds; one small task).
- **Effect:**
  - #255: add the fixed cross-entropy and ε as arms.
  - #249: the default ε=1e-8 combined with wd=1 lands above the fp32 cliff.

**en-02: Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes**
- **Identity:** Liu, Cao, Li, Zhou. arXiv 2605.06152 (v4 7 Oct 2026). The arXiv comments say "accepted to NeurIPS 2026"; this was not checked on neurips.cc. https://arxiv.org/abs/2605.06152. Not in map.
- **Findings:**
  - Slingshot spikes come from the correct-class gradient rounding to 0, which breaks the zero-sum property and drives "Numerical Feature Inflation".
  - Tested on mod-97 division with a 2-layer transformer, CIFAR-10 models, and a 110M nanoGPT.
  - Every run used weight decay = 0.
  - An fp64 loss removes the spikes. So does ε=1e-5, while ε=1e-6 only makes them rarer.
  - Label smoothing removes the spikes but adds edge-of-stability instability.
- **Evidence:** "the gradient of the correct class is rounded exactly to zero" (abstract).
- **Grade:** A (accepted, per arXiv comments).
- **Effect:**
  - #255: ε and fp64-loss arms.
  - Label smoothing is not a clean control.

**en-03: What Does the Weight Norm Control in Grokking? Logit-Scale Mediation under Cross-Entropy**
- **Identity:** Truong Xuan Khanh. arXiv 2606.18465 (16 Jun 2026); code on GitHub (ClevixLab). Not in map.
- **Findings:**
  - With the norm clamped, varying only the output temperature spans the full norm-induced delay range.
  - The delay tracks the logit scale (R²=0.97); the norm adds only 1–2% on top.
  - Restoring the logit scale recovers ≈85% of the delay at two moduli.
- **Evidence:** "The proximal variable is the logit scale and the softmax saturation it drives" (abstract).
- **Grade:** B (single author; this author has four related papers, en-06, en-07 and en-20).
- **Effect:** #249: log the logit scale. #255: links norm-based explanations to precision-based ones.

### Weight-decay and learning-rate laws (#251, #249)

**en-04: A Spectral Theory of Grokking: Weight Decay Induces Feature Learning**
- **Identity:** Pracher, de Jong, Lieshaus, Jeffares, Rulands. arXiv 2609.26679 (22 Sep 2026). Not in map.
- **Findings:**
  - Training moves from a fixed NTK to an evolving kernel.
  - The grokking timescale depends on η·λ.
  - Feature learning slows logarithmically near a critical decay, and strong decay prevents fitting.
  - Tested with an 84×90 (η, λ) grid on an MLP and a 42×45 grid on a one-block transformer (p=23).
  - The theory assumes squared loss, coupled L2 decay and gradient flow. It does not cover AdamW or cross-entropy.
- **Evidence:** "the grokking timescale is controlled by the product of learning rate and weight decay" (abstract).
- **Grade:** B.
- **Effect:** #251: use an η·λ axis and expect an upper failure cutoff.

**en-05: A Stochastic–Geometric Theory of Scaling Laws in Grokking**
- **Identity:** Luo, Gagné, Ngnawé, Ullah, Morrissey. arXiv 2606.30388 (v3 8 Jul 2026). Not in map; flagged by the other lanes.
- **Findings:**
  - Under Adam with weight shrinkage, solutions form a shell–core structure.
  - The delay is τ ≈ (s̄/(ηλ))·log(ρM/ρG) + c/(bλ) + c₂/λ², so batch size has only a weak O(b^-1/2) effect.
  - Experiments: a 2-layer MLP on S5 and Z127, 10 seeds per cell. At the base cell, τ = 12,471 ± 1,072.
- **Evidence:** "recovers the inverse dependence of the grokking delay on the regularization coefficient" (Sec. 5.2).
- **Grade:** B.
- **Effect:**
  - #251: η·λ axis.
  - #249: full batch versus minibatch should matter little for the delay.
  - Ngnawé is also a co-author of en-31.

**en-06: The Norm-Separation Delay Law of Grokking**
- **Identity:** Truong, Truong, Luu, Phan. arXiv 2603.13331 (v2 2 May 2026). Not in map.
- **Findings:**
  - 293 runs on modular addition, modular multiplication and sparse parity.
  - The delay scales inversely with weight decay (R²=0.97) and with learning rate (R²=0.92).
  - It depends on the log of the norm ratio (r=0.91).
  - SGD fails where AdamW groks.
  - At memorisation time, the delay can be predicted with 34.6% mean absolute error (N=60 seeds).
- **Evidence:** "SGD fails entirely at the same hyperparameters where AdamW reliably groks." (abstract).
- **Grade:** B.
- **Effect:** #251: keep AdamW fixed and read the delay against log norm.

**en-07: First-Passage Prediction of Grokking Delay: A Calibrated Law under AdamW with Causal Validation**
- **Identity:** Truong, Truong, Luu, Phan. arXiv 2605.18845 (13 May 2026). Not in map.
- **Findings:**
  - Closed-form first-passage law: 17.7% mean absolute percentage error (MAPE) on 26 held-out runs over a 41× delay range, 18.0% on MLPs, and 23.3% across tasks.
  - The angle predicted at p=97 is 47.2°, against 47.8° observed.
  - Freezing the norm or removing weight decay at memorisation stops grokking: 0/6 versus 3/3.
  - The correction factor is measured per architecture and varies about 2× between variants.
- **Evidence:** "We give the first quantitative prediction of grokking delay under AdamW." (abstract).
- **Grade:** B (same authors as en-06; small causal n).
- **Effect:** #251: a cheap causal control.

**en-08: To Grok Grokking: Provable Grokking in Ridge Regression**
- **Identity:** Mingyue Xu, Vardi, Safran. arXiv 2601.19791 (v4 16 Jul 2026). ICML 2026, with an Outstanding Paper Honorable Mention (en-09). Not in map.
- **Findings:**
  - Over-parameterised ridge regression trained with gradient descent and weight decay goes through three provable stages.
  - Gives quantitative bounds on the grokking time in terms of the hyperparameters, so the delay can be made larger or removed.
- **Evidence:** "grokking is not an inherent failure mode of deep learning" (abstract).
- **Grade:** A.
- **Effect:** #251: theory support for treating the delay as tunable; a linear-model baseline to sanity-check the sweep logic.

**en-10: A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation**
- **Identity:** Boursier, Pesme, Dragomir. NeurIPS 2025 poster; arXiv 2505.20172. https://neurips.cc/virtual/2025/poster/116559. Not in map.
- **Findings:**
  - As weight decay goes to 0, gradient flow first interpolates quickly onto a critical manifold.
  - It then drifts slowly along a Riemannian ℓ2-norm-minimising flow, and that drift causes the generalisation jump.
  - Validated on synthetic regression only.
- **Evidence:** "the training loss rapidly reaches zero while the test loss plateaus for an extended period before suddenly improving" (abstract).
- **Grade:** A.
- **Effect:** #251: the delay is roughly 1/λ, because grokking comes from slow norm reduction.

**en-11: On the Dynamics of Delayed Generalization in Linear Models**
- **Identity:** Vedantam, Das, Lakshminarayanan. arXiv 2602.08302 (v3 14 Oct 2026). Not in map; flagged by the other lanes.
- **Findings:** A linear model trained with gradient descent and an exponential-tail loss on separable data shows a delay. The delay comes from a support-vector-dominated "unlearning" phase.
- **Evidence:** "Delayed generalization, where training accuracy becomes nearly perfect substantially before test accuracy" (abstract).
- **Grade:** B.
- **Effect:** Minor. Background for reading cross-entropy delays.

**en-12: Scaling and Renormalization in Neural Networks: An Effective RG Description of Grokking**
- **Identity:** Chen, Wang, Wu, Huang, Liu. arXiv 2610.36113 (26 Oct 2026). Not in map; flagged by the other lanes.
- **Findings:**
  - On a 2-layer network for modular addition, the squared weight norm serves as the order parameter, with epoch and weight decay as control variables.
  - Block-spin renormalisation gives five critical exponents that match empirical estimates within 1–10%.
- **Evidence:** "Grokking is a delayed transition from memorization to generalization in deep neural networks" (abstract).
- **Grade:** B (theory-heavy; seed count not stated).
- **Effect:** Low. It suggests reading #251 as a phase diagram, with heat-capacity-style fluctuation proxies.

**en-13: Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking (Li₂)**
- **Identity:** Yuandong Tian. arXiv 2509.21519 (v5 2 Dec 2025). Not in map; flagged by the other lanes. One search snippet said ICLR 2026, but this was not verified.
- **Findings:**
  - Three stages: lazy learning, independent feature learning, then interactive feature learning.
  - Weight decay acts like a learning rate for feature learning. It is sufficient for grokking but not necessary.
  - Zero-initialising the top layer speeds up feature learning, by up to 10× with MSE loss in multi-layer networks (100 versus 1,000 epochs).
  - Muon rebalances hidden nodes.
- **Evidence:** "the independent dynamics follows exactly the gradient ascent of an energy function" (abstract).
- **Grade:** B.
- **Effect:**
  - #249: zero-initialising the output layer is a cheap arm.
  - #251: consistent with the η·λ coupling.

### Multi-task, curriculum and ordering (#252, #253)

**en-14: The Geometry of Multi-Task Grokking: Transverse Instability, Superposition, and Weight Decay Phase Structure**
- **Identity:** Yongzhong Xu. arXiv 2602.18523 (v3 3 Apr 2026). Not in map.
- **Findings:**
  - Shared-trunk transformers trained on mod-add with mod-mul, with or without mod-square.
  - The tasks grok in a staggered order across seeds.
  - Without weight decay the model fails sharply.
  - Removing under 10% of the orthogonal gradient components eliminates grokking.
  - Model size and seed count were not stated in the abstract.
- **Evidence:** "multiplication generalizes first, followed by squaring, then addition, with consistent delays across seeds." (abstract).
- **Grade:** B (a single author with four papers in this sweep: en-14 to en-17).
- **Effect:** #253 and #252: expect staggered per-task grokking, so report grok steps per task.

**en-32: The Order Is The Message**
- **Identity:** Jordan LeDoux. arXiv 2603.25047 (24 Mar 2026). Not in map.
- **Findings:**
  - p=9973 with 0.3% of the input space as training data.
  - Two fixed orderings reached 99.5% test accuracy at epochs 487 and 659. IID order reached 0.30% after 5,000 epochs.
  - An adversarial ordering suppressed learning.
  - The Fourier fundamental of the learned representation matches the Fourier dual of the ordering.
- **Evidence:** "An adversarially structured ordering suppresses learning entirely." (abstract).
- **Grade:** B (single author; seed count not given in the abstract).
- **Effect:** #252: sampling order can dominate, so treat it as a first-class factor with a fixed-data IID control.

### Acceleration methods (#254)

**en-22: GrokAlign: Geometric Characterisation and Acceleration of Grokking**
- **Identity:** Walker, Humayun, Balestriero, Baraniuk. arXiv 2506.12284 (v2 31 Jul 2025); code at ThomasWalker1/grokalign. Not in map.
- **Findings:**
  - A Jacobian-alignment regulariser that groks much sooner than weight decay.
  - Centroid alignment tracks the training stages.
  - A secondary summary says the transformer modular-addition gains need a modified task. This was not verified.
- **Evidence:** "Here, we explain how grokking is realised in the Jacobian of a deep network" (abstract).
- **Grade:** B.
- **Effect:** #254: a candidate arm, but the Jacobian cost is high.

**en-23: Predator-Prey Model: Driven Hunt for Accelerated Grokking**
- **Identity:** Lopatin, Kozyrev, Pechen. IEEE Transactions on Artificial Intelligence 7(10), 2026; arXiv 2509.10562 (v2 20 Oct 2026). Not in map.
- **Findings:** Two coupled optimiser agents. Up to 100× fewer gradient evaluations and up to 60× less wall-clock time on grokking tasks.
- **Evidence:** "to avoid optimization getting stuck in a ravine" (abstract).
- **Grade:** A (journal). The baseline strength was not checked.
- **Effect:** #254: low priority, because it needs two model copies.

**en-24: The Geometric Inductive Bias of Grokking: Bypassing Phase Transitions via Architectural Topology**
- **Identity:** Alper Yıldırım. arXiv 2603.05228 (v3 4 May 2026); code on GitHub. Not in map.
- **Findings:**
  - L2-normalising the residual stream, with a fixed-temperature unembedding, groks over 20× faster without weight decay (≈54,160 to ≈2,480 epochs, per a search snippet).
  - Uniform attention skips the delay on modular addition.
  - There is no gain on S5, a negative control.
- **Evidence:** "reducing grokking onset time by over 20x without weight decay" (abstract).
- **Grade:** B.
- **Effect:**
  - #254: an architectural arm.
  - #258: normalisation choices change the delay, so hold them fixed across loop and non-loop models.

**en-25: Low-Rank Decay for Grokking in Scale-Invariant Transformers**
- **Identity:** Mingyu Li. arXiv 2606.04405 (3 Jun 2026). Not in map.
- **Findings:**
  - With RMSNorm or QK-norm, L2 decay mostly changes the weight scale.
  - A nuclear-norm-style decay keeps compressing the spectrum after the task gradients vanish.
  - It collapses the Q/K rank and shifts the data-fraction threshold for grokking.
  - Described as a preliminary draft.
- **Evidence:** "standard Frobenius-norm weight decay acts purely along the radial direction of the weight space" (abstract).
- **Grade:** B (preliminary).
- **Effect:** #251 and #258: if ReLSM uses RMSNorm or QK-norm, then "weight decay on all weights" may act mainly through the scale.

**en-26: Clip to Grok (GitHub, NiftyliuS/cliptogrok)**
- **Identity:** Repo and PDF, Mar 2026. https://github.com/NiftyliuS/cliptogrok. Not in map; flagged by the ko lane.
- **Findings:**
  - Per-row ℓ2 clipping of decoder weights (max_norm 2.0) after every step, with no weight decay.
  - 2-layer model on mul mod 97, n=200 seeds: Lion with clipping took a median of 550 steps, Adam with clipping 1,200, and SignSGD with clipping 1,140, all with 0 failures.
  - The AdamW baseline used wd=0.01 and took a median of 35,040 steps.
  - 8-layer model, n=100: 1,570 steps (18×) versus 28,905.
- **Evidence:** "This is applied only to decoder layer weights (attention projections, MLP layers, and LayerNorm parameters)." (README).
- **Grade:** C (code and many seeds, but a weak baseline at wd=0.01).
- **Effect:** #254: the cheapest strong-claim arm. Compare it against the wd=1.0 baseline.

**en-27: r/MachineLearning post announcing Clip to Grok (snapshot, 20 Mar 2026)**
- **Identity:** reddit.sentinel-team.org snapshot. Not in map.
- **Findings:**
  - Self-reported speedups: 66× (2-layer) and 18× (8-layer, 300 seeds).
  - Commenters asked whether this is still grokking, since there is no overfit phase, and asked for a plain Lion control.
- **Evidence:** "The method: per-row ℓ₂ clipping on decoder weights after every optimizer step." (post).
- **Grade:** D.
- **Effect:** It records the open objections to en-26.

**en-28: SutroYaro Experiment 4: GrokFast (low-pass gradient filter)**
- **Identity:** cybertronai.github.io, 2026-03-03. Not in map.
- **Findings:**
  - 20-bit sparse parity (k=3), width 1000, SGD with lr 0.1, wd 0.01, batch 1, Grokfast α=0.98 and λ=2, seed 42 only.
  - Grokfast took ~12 epochs to pass 90% versus ~5 for the baseline, and 383.7 s versus 22.7 s.
  - It also moved the weights 83× further.
- **Evidence:** "GrokFast was designed for settings where grokking takes thousands of epochs (modular arithmetic, algorithmic tasks)." (page).
- **Grade:** C (single seed, out of regime).
- **Effect:** #254: this is not a failed replication in the intended regime, but it shows that Grokfast hurts when there is no plateau.

### Diagnostics and early warning (#249 logging, #253)

**en-15: Early-Warning Signals of Grokking via Loss-Landscape Geometry**
- **Identity:** Yongzhong Xu. arXiv 2602.16967 (v3 3 Apr 2026). Not in map.
- **Findings:**
  - The commutator defect, computed from training gradients, rises before generalisation on SCAN and Dyck-1.
  - Its lead time follows a superlinear power law (α≈1.18 and 1.13).
  - Spectral concentration is not a universal precursor.
  - Amplifying non-commutativity speeds grokking.
  - The grok labels themselves still use test accuracy.
- **Evidence:** the commutator defect "rises well before generalization" (abstract).
- **Grade:** B.
- **Effect:** #249: a train-only signal worth logging, but it has not been tested on mod-p.

**en-16: The Lifecycle of the Spectral Edge**
- **Identity:** Yongzhong Xu. arXiv 2604.07380 (8 Apr 2026). Not in map.
- **Findings:**
  - At grokking, the gradient and weight-decay components of the dominant update direction align.
  - Ablating that direction hurts more than 4,000× as much as ablating a random one.
  - Removing weight decay after grokking reverses the compression but keeps the algorithm.
- **Evidence:** "the edge becomes a compression axis that is perturbation-flat yet ablation-critical" (abstract).
- **Grade:** B.
- **Effect:** #251: removing weight decay after the grok step is a useful arm.

**en-17: Feature Repulsion and Spectral Lock-in**
- **Identity:** Yongzhong Xu. arXiv 2605.08119 (28 Apr 2026). Not in map.
- **Findings:**
  - Tests Tian's repulsion theorem with M=71, K=2048 and MSE loss.
  - An eigengap slope detector fired in 15/15 grokking seeds at epoch 174 (IQR 173–174) and in 0/15 controls with an x² activation.
  - With ReLU it never fired.
- **Evidence:** "the theorem does not specify when this mechanism becomes empirically observable" (abstract).
- **Grade:** B.
- **Effect:** Do not adopt eigengap detectors for a ReLU or GELU model without checking them first.

**en-18: ILDR: Geometric Early Detection of Grokking**
- **Identity:** Shreel Golwala. arXiv 2604.20923 (22 Apr 2026). Not in map.
- **Findings:**
  - The inter/intra-class distance ratio crosses 2.5× its baseline before the validation transition.
  - Lead time is 9–73% of the budget, and 950 ± 250 steps over 8 seeds.
  - Used for early stopping, it saves 18.6% of training.
  - It needs held-out data.
- **Evidence:** "ILDR provides an early detection signal" (abstract).
- **Grade:** B.
- **Effect:** #249: a cheap logging metric, but not held-out-free.

**en-19: A Basin-Selection Perspective on Grokking via Singular Learning Theory**
- **Identity:** Cullen, Estan-Ruiz, Danait, Li. arXiv 2603.01192 (v3 7 May 2026). Not in map.
- **Findings:**
  - Grokking is a move from a higher-LLC (local learning coefficient) memorising basin to a lower-LLC generalising basin.
  - Gives analytic LLC formulas for quadratic networks.
  - LLC trajectories estimated from training data track the onset.
- **Evidence:** "the local learning coefficient (LLC) which quantifies the local degeneracy of the loss surface" (abstract).
- **Grade:** B.
- **Effect:** #249 and #253: an optional train-only diagnostic. SGLD-based estimation costs extra compute.

**en-20: At-Grok Is Not Converged: A Measurement-Validity Audit for Grokking Representation Metrics**
- **Identity:** Truong Xuan Khanh. arXiv 2607.06639 (7 Jul 2026); toolkit released. Not in map.
- **Findings:**
  - Effective rank read at the grok step overstates the converged value by 3–5× on an MLP and 1.3–1.5× on a transformer.
  - Compression continues ≥10k steps after grokking.
  - Adding LayerNorm cuts the share of compression completed by the grok step from 0.87 to 0.25.
- **Evidence:** "Reading effective rank at the grokking transition overstates the converged value by 3-5x on an MLP" (abstract).
- **Grade:** B.
- **Effect:** #249, #253 and #258: train long after the grok step, and report converged metrics.

**en-21: An Exploratory Replica-Overlap Probe of the Grokking Transition**
- **Identity:** "A. C. Opus", J. Q. Lu. arXiv 2609.25634 (22 Sep 2026); pre-registered. The author name suggests possible AI authorship. Not in map.
- **Findings:**
  - 64 seeds across 4 configurations.
  - The registered verdict is UNDETERMINED, because the alignment step broke function preservation.
  - Grok rates were 0/16, 11/16 and 16/16, but the training fraction was confounded with the split.
- **Evidence:** "It is the alignment step, not the overlap statistic, that determines what this registered probe can report." (abstract).
- **Grade:** B (null or instrument-failure result).
- **Effect:** #249: confirms that grok rates vary with the split, so report the split ID with every seed.

**en-31: Margin-based Progress Measures for Grokking**
- **Identity:** Ngnawé, Notsawo. NeurIPS 2026 workshop poster ("Neural Network Artifacts as a New Data Modality"). https://neurips.cc/virtual/2026/162993. Not in map.
- **Findings:** The mean test logit margin stays low during memorisation and rises at generalisation, across MLPs and LSTMs on mod-p and ResNet-18 on shapes.
- **Evidence:** "Grokking is the phenomenon of delayed generalization" (abstract).
- **Grade:** B (workshop).
- **Effect:** #249: log margins. Combined with en-01, the train margin also signals that the precision cliff is near.

**en-29: Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking**
- **Identity:** Han, Adilova, Petzka, Kleesiek, Kamp. NeurIPS 2025; arXiv 2509.17738 (v3 4 Feb 2026). Not in map.
- **Findings:**
  - Relative flatness predicts the onset; neural collapse does not.
  - Regularising away from flat solutions delays generalisation.
- **Evidence:** "only flatness consistently predicts it." (abstract).
- **Grade:** A.
- **Effect:** #249: a relative-flatness probe is a candidate train-side signal.

**en-30: Explaining Grokking and Information Bottleneck through Neural Collapse Emergence**
- **Identity:** Sakamoto, Sato. ICLR 2026 poster; arXiv 2509.20829; code at keitaroskmt/collapse-dynamics. Not in map.
- **Findings:** The contraction of within-class variance, and the gap between the fit timescale and the collapse timescale, explain the delay.
- **Evidence:** "the contraction of population within-class variance is a key factor underlying both grokking and information bottleneck" (ICLR abstract).
- **Grade:** A.
- **Effect:** It partly conflicts with en-29 (see Contradictions). Low practical impact.

### Recursion and loops (#258)

**en-33: Repeated Shared Access Enables Grokking, but Edit Propagation Depends on an Addressable Memory**
- **Identity:** Yanan Niu. arXiv 2606.20737 (v2 23 Jun 2026). Not in map.
- **Findings:**
  - A 2×2 design on synthetic knowledge-graph QA: dense, loop, dense with memory, and loop with memory (LMC).
  - Both looping and memory rereading cross the out-of-distribution grokking barrier that the dense model fails.
  - Edit propagation follows the memory axis: LMC 0.78–0.92 versus Loop 0.04–0.30.
- **Evidence:** "the propagating ingredient is an addressable site that an edit can write to and later computation rereads" (abstract).
- **Grade:** B (single author).
- **Effect:** #258: direct support for the shared-loop hypothesis on composition tasks, though not on mod-p.

### Other opened sources

**en-09: AIhub, "Congratulations to the #ICML2026 award winners"** (20 Jul 2026). Confirms that en-08 received an honourable mention. The article mislabels the conference year in its body. Grade C. Not evidence-bearing.

**en-34: Grokking From Abstraction to Intelligence**
- **Identity:** Zhang, Shen, Xiong, Dong. arXiv 2603.29262 (31 Mar 2026). Its comments say "under review for ICML 2026", not accepted, although a notes blog lists it as ICML 2026.
- **Findings:** A qualitative account of grokking as simplification driven by parsimony. No numbers in the abstract.
- **Grade:** C. Not evidence-bearing for the recipe.

## Contradictions

1. **Does weight decay keep logits below the cliff?**
   - en-02 argues that weight decay bounds logit growth, so the absorption threshold is rarely reached. All of its experiments used wd=0.
   - en-01 measures the with-decay equilibrium at ≈log(1/ε) ≈ 18.4 for ε=1e-8, which is above the fp32 cliff of 16.6.
   - en-01 is the direct measurement. It means the existing maps' framing, with StableMax as the main precision control, is incomplete for the wd=1 recipe.
2. **λ thresholds versus η·λ laws.**
   - The maps quote weight-decay boundaries without the learning rate: Verma's λc ≈ 0.016 and Manir and Rupa's narrow "Goldilocks" band.
   - en-04 to en-07 and en-10 put the delay at ≈1/(η·λ). Those thresholds are only comparable at the same learning rate and optimiser.
3. **Order versus norm.**
   - en-32 claims that the delay "is entirely explained by data ordering", per a search snippet; the abstract shows IID at 0.30% versus 99.5% for fixed orders.
   - The norm and logit-scale papers (en-03, en-06, en-07) explain the same delay through norm or logit dynamics under IID sampling.
   - These are not formally incompatible, since en-32 uses 0.3% data at p=9973. But they disagree on what drives the delay, which matters for how #252 is read.
4. **Is weight decay necessary?**
   - en-07 finds that removing decay at memorisation kills grokking (0/6).
   - en-24 and en-26 grok fast with no decay at all.
   - These fit together if the operative variable is a norm or logit-scale constraint: clipping and spherical normalisation impose one without decay.
5. **Flatness versus collapse.** en-29 (NeurIPS 2025) says neural collapse is not necessary. en-30 (ICLR 2026) treats within-class variance contraction as the key factor. They differ in emphasis.
6. **Internal tension in one author's work.** Xu's en-15 says spectral concentration is not a universal precursor, while Xu's en-17 detector fires 15/15, but only with x² and not with ReLU.
7. **Zero-init.** Tian (en-13) says zero-initialising the top layer accelerates grokking. This agrees in direction with the older large-init-delays findings (Omnigrok, Kumar), so it is not a real contradiction. It is flagged because the maps record only the large-init side.

## Gaps (looked for, not found)

- An independent, multi-seed replication of Grokfast on transformer modular arithmetic in 2025–2026 against a tuned wd=1.0 baseline. The only new test, en-28, is out of regime.
- A grokking study of a looped or weight-tied transformer on modular arithmetic, for example how the delay changes with loop count. en-33 uses knowledge-graph QA, and "Loop, Think & Generalize" is already in the map.
- A controlled curriculum-versus-IID grokking study for #252 beyond en-32's ordering result.
- An empirical study of label smoothing and grokking. Only en-02's remark about edge-of-stability instability was found.
- An empirical batch-size study of grokking delay. Only en-05's theory, with a weak b^-1/2 effect, was found.
- A COLM 2025 or 2026 grokking paper. None was confirmed.
- New LessWrong or Alignment Forum modular-arithmetic replications with numbers from 2025–2026. Only older Nanda walkthroughs surfaced.
- TMLR 2025–2026 grokking papers beyond Minegishi et al. (already partly in the map) were not confirmed.
- A reported step count for p=97 with wd=1.0 at d=128 and 2 layers in any repo or blog. A blog used this configuration but reported no grok step.
- The fp32-versus-bf16 grokking comparison on modular arithmetic itself. en-01 gives the bf16 cliff value, but its bf16 grokking runs were not extracted.

## Method

**Queries (WebSearch, 23 run; 1 in extended mode):**
1. grokking arXiv 2026 modular arithmetic weight decay new mechanism
2. accelerating grokking method 2025 arXiv transformer modular addition
3. grokking early warning signal without validation set 2025 2026
4. GrokAlign Jacobian alignment grokking
5. grokking failed replication negative result
6. looped transformer grokking recurrent depth 2026
7. grokking multi-task modular arithmetic transformer
8. grokking bfloat16 precision softmax collapse float32 2026
9. NeurIPS 2025 grokking paper poster
10. ICLR 2026 grokking openreview
11. TMLR grokking 2025 2026
12. LessWrong grokking modular addition replication
13. "Flatness is Necessary, Neural Collapse is Not"
14. Grokfast replication independent evaluation 2026
15. grokking optimizer comparison Muon AdamW SGD 2026
16. grokking learning-rate warmup batch size
17. weight norm clipping accelerates grokking
18. weight-tied looped transformer modular addition grokking
19. grokking initialization scale zero init 2026
20. Sakamoto Sato neural collapse grokking
21. grokking label smoothing
22. local learning coefficient grokking; github grokking reproduction p=97 wd 1.0; COLM 2025 grokking; ICML 2026 grokking (extended); grokking LLM pretraining checkpoints 2026; grokking universal transformer depth recurrence

**Counts:**
- Seen: about 200 result links, roughly 140 unique.
- Opened: 34 unique sources, listed in en-delta.sources.jsonl.
- Retained as evidence-bearing: 31. en-09, en-27 and en-34 are supporting sources only, and en-21 is retained as a null result.

**Coverage checks:**
- Every candidate's arXiv ID and title was searched for with grep in a concatenation of all the coverage files.
- "Not in map" means zero hits outside the worker lane reports.
- The worker-cited papers already in the map are ExPLAIND (2505.20076), "Where to find Grokking in LLM Pretraining" (2506.21551) and Manir and Rupa (2603.25009).
- The worker-cited papers not in the map are 2610.31642, 2610.36113, 2601.19791, 2602.08302, 2606.30388, Tian's Li₂ (2509.21519) and Boursier et al. (2505.20172). All seven are included above.

**Skipped as already covered in the maps (not re-researched):**
- Prieto et al., StableMax (2501.04697)
- Grokfast (2405.20233)
- Muon accelerates grokking (2504.16041)
- EGD (2510.04930)
- Li, Fan and Zhou (2506.21551)
- Manir and Rupa (2603.25009)
- Verma, weight-decay regimes (2605.20441)
- Weight norm as the timing variable (2606.13753)
- Circuit synchronisation (2606.12966)
- Anti-grokking (2506.04434 and 2602.02859)
- ExPLAIND (2505.20076)
- GrokTransfer (2504.13292)
- Embedding-layer grokking (2505.15624)
- Ootani, 12K-parameter study (2607.05104)
- Loop, Think & Generalize (2604.07822)
- Song and Ye, capacity (2605.09724)
- Howe, priors (2607.04333 and 2607.12735)
- Carvalho (2502.01774)
- Furuta et al. (2402.16726)
- NeuralGrok (2504.17243)
- Glass relaxation (2505.11411)
- Grokking in the Wild (2504.20752)
- TRM (2510.04871)
- The grammatical-generalisation analogue (2606.00230)

**Limits:**
- WebFetch's model limits verbatim quoting, so the quotes are short abstract or caption excerpts.
- Several seed counts sit in full texts that were not read and are marked "not stated".
- Author clusters, which are self-consistent but not independent replications:
  - Yongzhong Xu: en-14, en-15, en-16, en-17;
  - Truong Xuan Khanh and co-authors: en-03, en-06, en-07, en-20;
  - Ngnawé: en-05, en-31.
- No training or GPU code was run.
