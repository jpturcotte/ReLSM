# Grokking: what five non-English literatures add, and what ReLSM's first run needs

As of 2026-10-28. A synthesis of the zh, ja, ko, fr and de lanes, with the decision-driving claims checked against primary sources.

## Answer

The standard recipe in issue #249 is sound. Every non-English replication that matches it groks, and most do so well inside 1e5 updates. The first run will fail for ReLSM-specific reasons before it fails for literature reasons. Fix those first, and add two numerical guards that the recipe does not mention.

1. **Put weight decay on the embedding (and so on the tied head) before the run.** Every replication that groks decays the embedding. Weight decay 0.1 or less does not grok within 3e4–1e5 steps. One ICML 2026 case study finds that late-phase weight decay acts mainly on the embedding and output layers [5][19][20][26][28][29]. Confidence: high.
2. **Keep logits and the loss in fp32, log the margin, and run the softmax fix as an arm.** Standard cross-entropy (CE) zeroes the target gradient once the logit margin passes about 16.6 in fp32 (5.5 in bf16) [1]. Under AdamW with weight decay the margin stops growing at a value of order ln(1/ε), which is 18.4 at ε = 1e-8, but that formula drops the weight decay, the parameter size and the loss's 1/N, each of which shifts it by nats; Barron et al. measured a mean margin of 17.9 with their fixed loss on a 1-layer mod-31 model, and 13.0 with standard CE [1]. Whether a 2-layer p = 97 run crosses the cliff is therefore not known in advance; log the per-example margin and the fraction of zero target gradients. Keep standard CE at ε = 1e-8 as the primary arm (it is the recipe). Run the margin-form CE [1] and an fp64 loss [1][3] beside it; both have direct evidence (Table 1 of [1]: 100 ± 0 for each, against 94.3 ± 5.4). A larger ε is a swept factor, not a guard: no source runs ε = 1e-7, and the ε = 1e-5 result is from runs without weight decay [3]. Too large an ε can halt learning, with no threshold given [2]. Confidence: high for the mechanism, low for any speed benefit in fp32 [21].
3. **Treat loss spikes as a diagnosable event, not divergence.** Do not roll back or early-stop on them. Log the logit margin, so a spike can be classified as a precision-cliff event or as something else [1][4][38]. Confidence: medium.
4. **Fix the split's commutation leak.** A random 50% split of ordered pairs puts about half the held-out pairs' transposes in training. Report held-out accuracy separately for "transpose seen" and "transpose unseen", or split on unordered pairs [4][5][32]. Confidence: high.
5. **Budget 1e5 steps; inspect at 2e4 and 5e4; do not call a bug on elapsed steps alone.** Every source with a stated batch size and model that groks within a few thousand steps at weight decay 1 is a **1-layer** transformer: 1.6–2.2k at a 0.5 split with batch 512 [56], 2–4k at a 0.4 split full batch [57], about 10k at a 0.3 split (Nanda). The only 2-layer, 0.5-split number is a single blog run with no batch size stated, whose weight decay 0 arm also grokked [24]; the 2-layer, 0.3-split number is one seed, first above 50 % near 21k [20]. Power et al.'s recipe is minibatch 512, not full batch, and its only quantified p = 97, 0.5-split timing is division at 1e5–1e6 steps. A 15-seed 1-layer mod-31 run had only 8 of 15 seeds above 95 % at 30k steps [21]. Five author groups put the delay at a scale of order 1/(ηλ) with unstated prefactors [7][8][54][55][56][57]; 1e3 is a lower scale here, not a prediction. So at 2e4 and 5e4 steps inspect the margin trace, the optimizer groups, the split and the transpose-seen curve, then finish the budget. Call a defect on a positive sign (train accuracy never reaches 99 %, transpose-unseen high from the start, an empty decayed group), not on a step count.
6. **Leave Grokfast out of the first run.** The README's baseline is plain Adam with **no** weight decay and minibatch 512; the Grokfast arms add coupled L2 (0.005 or 0.01) inside `torch.optim.Adam` [13][14]. So the 39,890 → 910 figure confounds the filter with adding weight decay to an unregularized baseline. No source tests it against AdamW at weight decay 1.0. Independent users report failures, and ReLSM's filter departs from the reference code in three ways [15][16][17][18][19]. Confidence: high.
7. **Add one cheap causal control, after A1 has grokked once.** Turn weight decay off at memorization. At p = 97, η = 1e-3, λ = 1, full batch, this stopped grokking in 0/3 runs against 3/3 for the baseline (a further 0/3 is a norm-freeze arm) [57]. That study used a 1-layer model with LayerNorm, a 0.4 split and β2 = 0.999, so it is close to #249, not identical. Confidence: medium (small n).

The section "Parameters to test or sweep" lists what the planned experiments (#249, #251–#255) do not yet cover.

What could change this answer: a seed-matched run showing that ReLSM's d = 128 stack groks with the embedding exempt, or a replication of the Barron fix showing a real speed-up in fp32. The only 15-seed replication found no fp32 speed-up [21].

## Scope and assumptions

- **Question:** what French, German, Chinese, Japanese and Korean work adds to the grokking picture as of October 2026. The focus is algorithmic grokking: mechanisms, the finite-set recipe, acceleration, and failure modes.
- **Decision served:** which recipe details, guards and diagnostics ReLSM's first finite-set run (#249, #214) needs, and what to avoid.
- **Tier and profile:** standard / report.
- **Included:** non-English sources from 2024 to October 2026. English papers by authors in those language communities count only as pointers, unless the native-language record adds something. The synthesis also opened English primary sources wherever a recommendation depends on them.
- **English work:** the English core literature is handled separately and is used only to judge novelty (see "Cross-check against the English map"). An English delta lane covers English work from mid-2025 to October 2026 that the existing maps miss; it is merged here (see "Recent English work (delta lane)").
- **Assumptions about ReLSM** come from the brief and from the owner's audit notes:
  - the head is tied to the embedding;
  - the model ends in a final RMSNorm;
  - AdamW uses β2 = 0.95 at HEAD;
  - training runs in bf16.
  All four were confirmed in code at `be08cde` on 2026-10-28: `model.py:704` (tied head), `model.py:700` and `899` (final RMSNorm before the head), `train.py:1844` (betas 0.9, 0.95), `train.py:2356-2358` (autocast forced on CUDA, bf16 on the 4060).
- **Two departures from the reference recipes that #249 should state:** Power et al. train with minibatch `min(512, N/2)` = 512, not full batch (full batch is Nanda's setting); and both Power (untied, post-norm LayerNorm) and Nanda (untied, no norm) use an untied head, while ReLSM's is tied under a final RMSNorm.
- **Terms used in this report:**
  - **Logit margin:** the target logit minus the log-sum-exp of the other logits. It measures how confident the model is in the correct class.
  - **Precision cliff:** the margin past which the floating-point format rounds the target gradient to zero.
  - **Slingshot:** a recurring spike in training loss under Adam.
  - **Grokking time:** see Ambiguities; the lanes use at least five definitions.

## Cross-lane parallels

These findings were reached in two or more lanes. Shared upstream sources are merged into one evidence cluster.

### 1. Weight decay near 1.0 is the switch, and it must reach the embedding

This is the strongest convergence in the sweep. It comes from independent code, in four lanes.

- **zh:** a CSDN replication of almost exactly #249 (2 layers, d = 128, p = 97, 30% split, full batch) groks at weight decay 1.0. It reaches 98.7% at 100k steps. At weight decay 0.1 and 0 it stays at 1.4% and 1.2% [20]. A Zhihu 16-arm run finds that only its weight decay 0.3 arm groks within 30k steps [19].
- **ja:** an MLP sweep and a NumPy sweep both show slower or no grokking as weight decay falls. Shake AI Lab reaches 90% at 10,400 steps with weight decay 1.0, 27,600 with 0.3, and never with 0.1 or below [26][28].
- **fr:** three French replications decay the embedding table, and all three grok. Their weight decay 0 controls fail [30][33][35].
- **One cluster, not three:** the Ultralytics glossary script is a single source translated into de, fr and ko [29][30][31]. Two lanes re-ran it on CPU and got 0.98 at 20k steps, against 0.00 throughout with weight decay 0. That is one source reproduced twice, not three confirmations.
- **de:** the ExPLAIND case study (LMU, ICML 2026) attributes the regularization influence in grokking mainly to the embedding and decoder layers [5][6].

The ko and fr lanes each derived the same consequence independently, and I agree with it (inference). Under decoupled AdamW, a row that gets zero gradient shrinks by a factor of (1 − ηλ) per step. So putting the embedding under decay also drains the 31k unused rows (#218). But this only happens if the embedding is not exempt (#186).

### 2. The softmax precision cliff and the slingshot

The zh and ja lanes reached the Barron et al. paper (arXiv 2610.31642) through secondary write-ups: a Jiqizhixin news report, a BAAI index page and a Qiita note [22][23]. Those are **one cluster**.

**Independent evidence from separate code:**
- **ja:** a 15-seed Qiita replication, run on a Mac [21];
- **fr:** a Julia port, which found the same zero-gradient and Adam-v decay mechanism months earlier and without the paper [36];
- **ja:** a 10-minute reproduction of softmax collapse in fp16, fp32 and fp64 [27].

**Primary source check:** I read the paper and checked every number the lanes quote (see Verification). The English delta lane confirms it at the source.

**A second, separate group:** Liu et al. (NeurIPS 2026, per arXiv) reach the same rounding mechanism from runs without weight decay. In their runs an fp64 loss removes the spikes, and so does ε = 1e-5, while ε = 1e-6 only makes them rarer [3].

### 3. Early stopping kills the run

Five lanes say this, from different sources:
- **ko:** validation loss rose from 4.2 to 5.0 during the plateau, a textbook early-stopping trigger [38];
- **fr:** a news article and the glossary both make the point [30][44];
- **zh:** the CSDN author's first rule [20];
- **ja:** several runs show the plateau;
- **de:** a governance blog warns that standard metrics give no early sign.

### 4. Seeds matter, and the transition is ragged

- The grok step varies 2–3× across seeds [20].
- Only 8 of 15 fp32 seeds ended at 95% or more in the ja replication [21].
- Three seeds of a ko toy screen gave 1.00, 0.75 and 0.78 [39].
- Test accuracy rose and fell five times before holding in the zh 16-arm baseline [19].
- The ETH thesis saw two-step jumps and slow rises [4].

### 5. Constant learning rate near 1e-3

- At lr 3e-4, a ja run stalls at about 52% test accuracy after 6,000 epochs [25], and the CSDN author says it doubles the steps needed [20].
- At lr 5e-3, Adam finds a memorizing solution on mod-127 addition, in Tian's Li₂ paper as reported in Chinese [41].
- The Grokfast author keeps lr at 1e-3 to 1e-4 [15].

### 6. The delay scales as 1/(ηλ)

- **fr:** Boursier et al. (NeurIPS 2025) put the norm-minimizing phase at time of order 1/λ in gradient flow [7].
- **ko and zh:** Korean and Chinese reviews of Xu et al. (ICML 2026) give t₂ ∝ 1/(ηλ) in ridge regression [8][49].
- **en-delta:**
  - Pracher et al. fit inverse-product scaling on an 84×90 (η, λ) grid for an MLP and a 42×45 grid for a one-block transformer [54];
  - Luo et al. derive a shell–core law under Adam, with a weak batch-size term [55];
  - Truong et al. report 293 runs, with R² = 0.97 for weight decay and 0.92 for learning rate [56], and a calibrated first-passage law [57].
- **Independence:** the two Truong papers share all four authors, so they count as one group. That leaves five groups.
- **Limits:** most of the theory assumes squared loss, coupled L2 or gradient flow [54]. Strong decay can stop the model fitting at all, which gives any sweep an upper cutoff [54].
- **Consequence:** a weight-decay threshold quoted without its learning rate does not transfer.

### 7. Decoupled weight decay, not L2 inside Adam

- Shake AI Lab's first implementation added `wd*p` to the gradient before Adam's moments. At weight decay 1.0, train accuracy stayed at 2.6% for 40k steps [28].
- A French student README stresses that "Adam + weight_decay" and AdamW are different [37].
- Truong et al. find that SGD fails at hyperparameters where AdamW reliably groks [56].

## Contradictions

| # | Topic | Side A | Side B | Resolution | Confidence |
|---|---|---|---|---|---|
| 1 | Loss spikes | Benign: every grok in the ETH thesis was tied to a spike of more than 10⁶× [4]; the ko replication reads spikes as decay fighting the loss [38] | Pathology: the fp32 cliff causes spikes that slow grokking at weight decay 1 [1]; spikes are precision-driven [3] | Both are true in different regimes; see the prose below | Medium |
| 2 | Adam ε | ε = 1e-7 (zh, via [23]); ε = 1e-6 (ja, [21]); ε = 1e-5 (Liu et al., no weight decay, [3]); ε = 1e-4 plus an fp64 CE (fr Julia port, [36]) | Too large an ε halts learning and grokking [2] | No source runs ε = 1e-7; it is one untested sentence in [1] ("for ε ≥ 1e-7, log(1/ε) ≤ 16.1 < c32"), and the ln(1/ε) formula drops terms worth a few nats. 1e-6 removed spikes on 2/2 seeds at mod 31 [21]. 1e-5 removed spikes in runs without weight decay, with the task of that figure unstated [3]. 1e-4 is outside every tested grokking range, and its run never grokked [36]; [2] gives no threshold. Keep ε = 1e-8 in the primary arm, run the margin-form CE and an fp64 loss as arms, and sweep ε in #255 | Medium |
| 3 | StableMax speed | About 60k epochs for StableMax against about 10k for AdamW with weight decay 1, read off figures in [27] | The original paper's claim is that StableMax enables grokking *without* regularization [2] | No contradiction with the paper, which makes no speed claim against weight decay. The two [27] numbers come from different setups (a [200, 200] MLP at lr 1e-2, wd 0 for StableMax; a 1-layer transformer at wd 1 for AdamW), so no ratio should be quoted. Barron et al. measure StableMax margins of 3,200–21,252 and lower accuracy than the fixed CE [1]. ⊥Grad is the faster one: it beats tuned weight decay [2], at about 200 epochs in [27] | Medium |
| 4 | Grokfast weight decay | README presets: weight decay 0.005 (EMA) and 0.01 (MA) [13] | ReLSM: AdamW weight decay 1.0 | Untested combination. The README baseline has **no** weight decay (`main.py` default 0, `torch.optim.Adam`); the Grokfast arms add coupled L2 inside Adam, a different quantity from torch AdamW's lr·wd. Do not transfer the values | High |
| 5 | Grokfast effect | Paper and README: 39,890 → 910 iterations [13] | Zhihu A/B: 0/8 arms grok with Grokfast [19]; issue #12: AdamW beats Grokfast + Adam [18] | This narrows Grokfast's domain rather than refuting it. The Zhihu run used cosine LR, batch 256, a 10% split and one seed, and is labelled AI-assisted | Medium |
| 6 | Timescale | About 1.3k steps (ja, 50% split, 2 layers, one run, batch size not stated, [24]) | About 21k steps to first >50% (zh CSDN, 30% split, 2 layers, one seed, full batch, [20]); 10⁵ steps (Power et al., batch 512) | The split fraction, the batch size and the number of layers all move it, and the commutation leak inflates early numbers (see #8). 1/(ηλ) = 1e3 is a lower scale, not a prediction [7][8]. No 2-layer, 0.5-split, full-batch number with a stated setup exists | Low |
| 7 | Weight norm as trigger | Norm inflection coincides with grok onset [20][26][28] | ℓ2 is not a reliable proxy [11]; a norm-matched dense net still groks late [45]; a final LayerNorm removes the norm dependence [42] | Log the norm per layer, but never gate on it | Medium |
| 8 | Does weight decay 0 grok? | Yes: Kanda, 2 layers, 50% split [24]; Tian says grokking occurs without weight decay [40] | No: ja MLP runs, fr runs, the Ultralytics script and the CSDN run [20][26][29][35] | Grokking without regularization is possible until softmax collapse stops it, more so at large splits [2]. Kanda's 30–50% plateau at a 0.5 split matches the transpose-seen fraction (inference). A weight decay 0 arm is not a clean negative control at f = 0.5 | Medium |
| 9 | Un-grokking | "There is no un-grokking" up to 2·10⁶ steps [4] | Anti-grokking past 10⁷ iterations [47]; zh runs see late un-grokking [20] | Different horizons and models. For a 1e5-step run, keep evaluating to the end | Low-medium |
| 10 | Faster grokking from the CE fix | 94.3 ± 5.4% vs 100 ± 0% (5 seeds) [1] | 15 seeds: 8/15 vs 8/15 ending ≥95%; standard CE reaches 95% on *more* seeds [21] | The spike removal and stability replicate; a speed-up in fp32 does not. The bf16-loss benefit replicates strongly | Medium |
| 11 | Does weight decay keep the margin below the cliff? | Yes: weight decay bounds the weights, so the margin "rarely exceeds" about 16 in fp32 (Liu et al., App. A.3.1; all their runs used weight decay 0) [3] | No: at λ = 1 the margin stops near ln(1/ε), 18.4 at ε = 1e-8, above the cliff; standard CE "repeatedly drops and rebounds" at the cliff on a+b mod 31 (Fig. 2) [1] | Neither side measures the #249 regime. Barron et al.'s measurement is a 1-layer mod-31 model (N ≈ 480): mean margin 17.9 with their fixed loss, 13.0 with standard CE (Table 1). Their formula drops λ, the parameter size and the loss's 1/N; at N = 4,704 the 1/N term alone is worth about 2.3 nats, more than the gap to the cliff. Liu et al. argue from a bound on the weights, without a weight-decay run. Whether the cliff is crossed must be measured in the run | Low |
| 12 | What drives the delay: order or norm? | Presentation order: two fixed orders reach 99.5% at epochs 487 and 659, IID reaches 0.30% after 5,000 epochs [59] | Norm and logit scale under IID sampling [56][58] | Different regimes: p = 9973 with 0.3% data, against dense mod-p splits. Under full batch, order inside a step cannot matter. It matters for #252's minibatch curricula | Low |
| 13 | Is weight decay necessary? | Removing it at memorization stops grokking: 0/3 (decay off) and 0/3 (norm frozen) vs 3/3, 1-layer model [57] | Per-row clipping and a spherical residual stream grok fast with no decay [62][63]; Tian says decay is sufficient, not necessary [69] | Compatible if the operative variable is a norm or logit-scale constraint, which clipping and normalization impose without decay (inference) | Medium |

### The loss-spike tension, resolved as far as the evidence allows

The ETH thesis replicated the original `openai/grok` code. In that code, training loss falls near zero and then jumps by more than 10⁶×. The author calls the spikes "a fundamental part of grokking behaviour" [4]. That run used weight decay 0.01 and needed up to 2·10⁶ steps. The same thesis also shows a case where the spikes "are not essential to get to better accuracy" (§4.4, Fig. 4.9) [4].

Barron et al. give the mechanism, and say when spikes help:

1. Above the cliff, the target gradient is exactly zero, while the non-target gradients stay positive.
2. Adam's second moment v then decays.
3. Weight decay pulls a few examples back under the cliff.
4. Their restored gradient is divided by a tiny √v, which gives a huge step [1].

Whether this helps depends on how the weight norm is controlled:
- With LayerNorm before the output and no weight decay, each spike inflates the weight norm and "acts as an accidental learning-rate decay". It can look helpful.
- When weight decay controls the norm, spikes "only slow down and hurt grokking" [1].
- With neither, a single spike can be destructive: test accuracy fell from 61% to 23% on a²+b² mod 97 [1].

Liu et al. (NeurIPS 2026, per arXiv) independently attribute slingshot spikes to floating-point absorption, not to optimization dynamics [3]. Prieto et al. had already proposed this link as a hypothesis (App. H) [2].

**Reconciliation.** The ETH regime has low weight decay and the original model. That is the regime where spikes are the main way the model moves. ReLSM's #249 regime has weight decay 1. In that regime, if the margin reaches the cliff, the spikes are a numerical artefact with a small and uncertain cost [1][21]. Whether it reaches the cliff in a 2-layer p = 97 run is not known: Barron et al.'s evidence is a 1-layer mod-31 model (Contradictions, row 11). The harm is not established in fp32: the 15-seed replication found no speed difference [21].

Two inferences follow for ReLSM. First, it ends in a final RMSNorm, so part of the "LayerNorm" regime applies. A spike there may change the effective step size rather than the logits. Second, the practical rule is the same either way:
- do not treat spikes as divergence (no rollback, no "loss went up" guard);
- remove their numerical cause in the default arm;
- keep a standard-CE arm if spike dynamics are themselves of interest;
- log the margin, so a spike that happens below the cliff stands out as a different phenomenon.

## Ambiguities

- **Steps vs epochs.** Under full batch these are the same thing. Lanes that say "epochs" for full-batch runs [25][27] can be compared directly with "steps". They are not the same in the Science Étonnante notebook, where an "epoch" is 22 mini-batch steps of 64 [33].
- **Full batch vs minibatch.** Power et al.'s recipe is minibatch `min(512, N/2)` = 512 at p = 97, about 9 updates per epoch; full batch appears in their paper only as a variant. Nanda et al. and most of the replications here use full batch. The Grokfast baseline [13], Liu et al. [3] and Truong et al. [56] use batch 512. Step counts from the two settings are not comparable, and #249's "full batch" is a departure from Power that should be recorded.
- **"Grokking time".** The sources use at least six definitions:
  - test accuracy first above 50% [20];
  - 90% [26][28][34];
  - 95% [13][21];
  - 99.5% [43];
  - 100% [5];
  - in theory, the gap t₂ − t₁ between overfitting and a generalization-error bound [8].

  Report several thresholds, plus a sustained one: the first step after which held-out accuracy stays at or above 95%.
- **Split-fraction conventions.** Most sources take a fraction of all p² ordered pairs. ExPLAIND samples 4,000 train and 2,000 test pairs from 6,328 unordered pairs with a ≥ b [5]. The zh 16-arm run takes 10% of each task's 9,409 pairs [19]. A "0.5 split" therefore means different amounts of non-redundant data.
- **Commutative-pair handling.**
  - The French reader's critique, acknowledged by the author but never followed up [32];
  - the ETH observation of high early validation accuracy on symmetric operations [4];
  - ExPLAIND's a ≥ b constraint [5].

  These are three independent recognitions of one issue. A SeoulTech paper turns the same symmetry into a data-augmentation accelerator (ko pointer, [51]). So symmetry can be either a leak or a lever, and the first run should make the choice explicit.
- **Decoupled vs coupled weight decay.** The same number means different things across implementations:
  - torch AdamW applies lr·wd·θ per step;
  - the Grokfast baseline is coupled L2 inside Adam [13];
  - some ports apply wd per step without lr.

  Quote a weight-decay value only with its optimizer and learning rate.
- **Which layers weight decay must reach.** The replications decay everything, or everything except biases. Bias decay does not decide the outcome: one run decays biases and one does not, and both grok [33][35]. ExPLAIND points at the embedding and the output head [5]. No source tested an embedding-exempt run. So #186's effect is inferred, not measured.
- **fp32 vs bf16: logits only, or the whole model?**
  - Barron et al. vary only the precision of the logits at the loss [1].
  - The ja replication notes that HF transformers upcast logits to fp32, which puts the cliff at 16.6 even in a bf16 model [21].
  - fp16 for the whole model at weight decay 0 never generalized in [27].

  No source isolates bf16 body weights at weight decay 1. For a 0.4M-parameter model, all-fp32 costs nothing, so the first run should use it (inference).

## Cross-check against the English map

The owner's audit corpus holds a consolidated English literature map (res-codex-lit.md, 2026-10-28), earlier AI reports, and lexical-session notes. They were read as data, to place each lane's findings.

**Already known in English; the lanes only repeat it:**
- weight decay as the switch;
- Prieto et al.'s softmax collapse, StableMax and ⊥Grad;
- the Grokfast issues (#9, #10, #12) and the EMA settings (λ 2 for EMA, λ 5 for MA);
- Truong's weight-norm delay law and the final-LayerNorm caveat;
- ExPLAIND's outer-layer alignment;
- grokking fragility across seeds;
- no early stopping.

No lane repeats the known errors in the earlier AI reports. GrokTransfer is attributed correctly (Xu et al.) where it appears. The EMA λ is 2 wherever it is stated. No lane passes off plain softmax as StableMax.

**New relative to the English map:**
- **The Barron et al. paper** (arXiv 2610.31642, 2026-10-22), and two independent replications of its mechanism [21][36]. The map predates the paper. Its ε guidance came only from Prieto App. I ("too large ε halts learning") and from a lexical-session eps result.
- **The commutation leak** in random splits [4][32].
- **Concrete step counts** from near-#249 replications [20][24][25].
- **The coupled-L2 failure signature** [28].
- **The ETH spike thesis** [4].
- **Four pointers:**
  - the Boursier 1/λ timescale [7];
  - the Mila ℓ2 critique [11];
  - the HAL epistemic-uncertainty diagnostic [12];
  - the Korean Target Linearity thesis [48].
- **The negative Grokfast A/B** [19].

**Contradicting or refining the English map:**
- The map reads Prieto App. I as "arguing against the large-eps fix". Prieto gives no ε values and no experiment there; it reports that "in some cases" a too-large ε ended learning [2]. Barron's equilibrium says a larger ε lowers the margin plateau by about ln(ε_new/ε_old), but tests no ε above 1e-8 [1]. So the map's caution stands and the ε question is open; it is a sweep, not a fix.
- The map's "typical grokking needs 10⁴–10⁵ updates" (a counter-review heuristic) is too pessimistic for the #249 recipe at a 0.5 split. 10⁵ is a budget ceiling.

The owner's screened literature register (2026-10-24) covers generated-parameter and adjacent work, and contains no grokking entries to reconcile.

The English delta lane checked every candidate against these maps by arXiv ID and title. It confirms that Barron et al., Liu et al., Boursier et al., Xu et al. and the η·λ papers are all absent from them.

## Recent English work (delta lane)

An English worker searched for work from mid-2025 to October 2026 that the existing maps miss. It opened 34 sources, 31 of them evidence-bearing (`workers/en-delta.md`). Most were read at abstract level. I re-checked the five that change recommendations (see Verification). What it adds:

1. **Precision.** It confirms Barron et al. at the source [1]. It adds Liu et al.: slingshot spikes come from the correct-class gradient rounding to zero, and an fp64 loss or ε = 1e-5 removes them [3]. Liu et al. ran Adam (β2 0.999) without weight decay throughout, single seed, on modular division with batch 512; the task behind the ε figure is not stated. So their claim that weight decay keeps margins below the cliff is argued, not measured. Barron's measurement at λ = 1 is on a 1-layer mod-31 model and does not settle it for #249 either (Contradictions, row 11).
2. **The η·λ law.** Five groups put the delay at about 1/(ηλ) [7][8][54][55][56][57]. Two of the papers share all authors [56][57]. The practical point is that #251 should plot results against η·λ, not λ alone.
3. **A causal control.** Turning weight decay off at memorization stopped grokking, 0/3 vs 3/3, and so did freezing the norm, 0/3, at p = 97, η = 1e-3, λ = 1, full batch, on a 1-layer model with LayerNorm, a 0.4 split and β2 = 0.999 [57]. A related paper finds that removing decay *after* grokking reverses the compression but keeps the algorithm [71].
4. **Logit scale.** One author finds the delay tracks the logit scale (R² = 0.97), with the norm adding 1–2% [58]. This links the norm story to the precision story, since the logit scale sets the margin.
5. **Ordering and curriculum.**
   - Presentation order can dominate at very low data: p = 9973, 0.3% data, fixed orders at epochs 487 and 659, IID at 0.30% after 5,000 epochs [59].
   - In shared-trunk multi-task training, tasks grok one after another, in a consistent order [60].
   - Both are single-author preprints. No controlled curriculum-versus-IID grokking study was found.
6. **Grokfast.**
   - No independent multi-seed replication on transformer modular arithmetic exists. The only new test is a single-seed sparse-parity run with no plateau, where Grokfast was 17× slower [61].
   - Newer accelerators claim large gains, but against weak baselines. Clip-to-Grok, for example, runs 200 seeds against AdamW at weight decay 0.01 [62]; a spherical residual stream gains over 20× without decay [63].
7. **Train-only diagnostics.** Several candidates exist, none validated on a mod-p transformer:
   - the commutator defect, tested on SCAN and Dyck [64];
   - the local learning coefficient [65];
   - the train logit margin [67].

   One audit finds that effective rank read at the grok step overstates the converged value 3–5× on an MLP, and that compression continues for 10k or more steps after grokking [66]. An eigengap detector that fired in 15/15 seeds with an x² activation never fired with ReLU [72].
8. **Loops (#258).** No looped or weight-tied transformer study of modular-arithmetic grokking was found. The nearest result is on knowledge-graph QA, where looping and memory rereading both cross an out-of-distribution grokking barrier [68].

**Caveats.** Three author groups each contribute several papers: Yongzhong Xu (four), the Truong group (four), and Ngnawé (two). Each group counts as one cluster. Several seed counts were not stated in the abstracts.

## Verification of decision-driving claims

I opened each primary source myself on 2026-10-28. Exact passages are kept short. The locator follows each passage.

| Claim (as reported by lanes) | Source | Status | Passage / locator |
|---|---|---|---|
| arXiv 2610.31642 authors; called "Barron et al. 2026" by ja | Barron, Gnanapragasam, Tschannen (Google DeepMind) [1] | Supported. v1 2026-10-22, v2 2026-10-26 | Title page of the v2 PDF |
| Precision cliff ≈ 5.5 bf16, 16.6 fp32, 36.7 fp64 | [1] | Supported | "≈5.5 in bf16, 16.6 in fp32, and 36.7 in fp64" (Fig. 1 caption) |
| Margin equilibrium ln(1/ε) ≈ 18.4 at ε = 1e-8; ε ≥ 1e-7 avoids spikes | [1] | Narrowed. Eq. 10 is the exact per-parameter condition `g/(|g|+ε) + λθ = 0`; Eq. 11 drops λ, |θ| and the loss's 1/N inside "≈". The 18.4 is the formula; the measured mean margin is 17.9 with the fixed CE and 13.0 with standard CE (Table 1, mod 31, 1 layer). No ε above 1e-8 was run; the ε ≥ 1e-7 remark is a parenthetical | "for ϵ≥10−7, log(1/ϵ)≤16.1< c32" (§5, Eq. 11); "ϵ ∈ {10−8, 10−12, 10−16}" (Table 1, Fig. 5) |
| Liu et al. settings | [3] | Supported, narrowed: Adam, β (0.9, 0.999), ε 1e-8, weight decay 0, batch 512, modular division mod 97, 2 layers, no LayerNorm, single seed (42); the ε = 1e-5 figure does not state its task | App. D.1 Table 2; §4 "we use the CE loss with zero weight decay, and the Adam optimizer" |
| Mod-31 grokking: 94.3 ± 5.4% standard vs 100% fixed (fp32); bf16 93.3% | [1] | Supported. 1-layer, width-64 ReLU Transformer, no LayerNorm, AdamW lr 1e-3, β 0.9/0.98, wd 1, 10,000 full-batch epochs, 3–5 seeds | Table 1 rows "fp32 Standard CE 94.3±5.4" and "bf16 Standard CE 93.3±7.4" |
| Spikes "slow down and hurt" grokking when wd controls the norm | [1] | Supported | "slingshot spikes at c32 only slow down and hurt grokking" (§4) |
| No-wd collapse 61% → 23% on mod 97 | [1] | Supported. a²+b² mod 97, λ = 0, ε = 1e-16 | Fig. 2 caption |
| StableMax margins 10³–10⁴, lower accuracy than fixed CE | [1] | Supported. Margins 3,200–21,252; accuracy 95.6–97.4% | Table 1, StableMax rows |
| Prieto: StableMax and ⊥Grad grok without regularization | [2] | Supported. ICLR 2025 | "StableMax leads to grokking in commonly studied settings without regularization" (§3.3) |
| Prieto: ⊥Grad faster than tuned weight decay | [2] | Supported | "⊥Grad leads to faster grokking even when compared to a tuned value of weight decay" (App. E) |
| StableMax "6× slower" contradicts Prieto | [2], [27] | Narrowed. Prieto makes no speed claim against weight decay | §3.3 and App. E |
| Large ε can halt grokking | [2] | Supported. Relevant to the fr ε = 1e-4 fix | "grokking could be stopped before SC due to the ϵ parameter in Adam being too large" (App. I) |
| Boursier et al.: onset at time ~1/λ; NeurIPS 2025 | HAL record via API [7] | Supported. Gradient-flow time, so 1/(ηλ) in steps is inference | "at time of order 1/λ, the trajectory enters a slow drift phase" (abstract) |
| ExPLAIND: weight decay acts on embedding and output layers; ICML 2026 | [5][6] | Supported. ICML 2026 poster; mod 113, a ≥ b, 4,000/2,000 pairs, full batch, AdamW lr 1e-3, wd 4.0, β (0.98, 0.99); test 100% at step 1,939; single run | "regularization acts primarily on the embedding and decoder layers" (§5); App. E.2 |
| ETH thesis (Arn 2023): spikes >10⁶× at grokking | [4] | Supported, with a narrowing: wd 0.01, 2·10⁶ steps; author also shows spikes "not essential" | "spikes again after some time by a factor of over 10⁶" (§4.2); "There is no un-grokking." (§4.4) |
| Grokfast README presets: EMA α 0.98, λ 2, wd 0.005; MA w 100, λ 5, wd 0.01; 39,890 → 910 iterations | [13] | Supported, with a correction to this report's earlier reading: the baseline command passes no `--weight_decay`; `main.py` defaults are `--optimizer Adam`, `--weight_decay 0`, `--batch_size 512`. The baseline is unregularized; the Grokfast arms add coupled L2 | README "Algorithmic data" commands and table; `main.py` argparse defaults |
| Grokfast EMA starts from the first gradient | [14] | Supported. The first filtered gradient is 3g | `grads = {n: p.grad.data.detach() ...}` in `gradfilter_ema` |
| Grokfast issues: exploding norms (#9); fp16 unresolved (#10); AdamW beats Grokfast (#12); tuning advice (#1) | [15][16][17][18] | Supported. #10 still open | Author replies dated 2024-06-18 to 2024-07-10 |
| Zhihu 16-arm A/B: 0/8 Grokfast arms grok; best arm 86.2% → 8.3%; in-place `buf.mul_` bug | [19] | Supported, with caveats: page labelled AI-assisted; setup as the lane reported | Lead paragraph; results table; §4.2 |
| CSDN weight-decay sweep (wd 0 / 0.1 / 1.0 → 1.2% / 1.4% / 98.7%) | [20] | Supported. Posted 2026-10-25, single seed | §4.4 table |
| ja 15-seed replication: no fp32 speed-up; ε = 1e-6 removes spikes | [21] | Supported | Results table; ε section |
| Mila ICML 2025: ℓ2 norm not a reliable proxy | [11] | Supported | "the ℓ2 norm is not a reliable proxy for generalization" (abstract) |
| HAL 2026: epistemic uncertainty collapses at grokking | [12] | Supported. HAL report (not peer reviewed), in-context modular arithmetic | "epistemic uncertainty collapses sharply when the model groks" (abstract) |
| Ridge-regression grokking, arXiv 2601.19791, "ICML 2026 best-paper nominee" | [8][9][10] | Narrowed. ICML 2026 oral; Outstanding Paper Honorable Mention per two secondary pages; the ICML page shows no award label. The paper also credits Tikeng Notsawo et al. with a 1/(ηλ) timescale | "time scale proportional to 1/(ηλ)" (§1, related work) |
| Kanda: grok at ~1.2–1.3k steps, mod 97, 50/50 | [24] | Supported as reported; weak as calibration. One run, no code shown, batch size, betas, ε and precision not stated; the λ = 0 arm also grokked | Results section |
| Liu et al.: fp64 loss or ε = 1e-5 removes Slingshot spikes | [3] (PDF read) | Supported, with a caveat: every run used weight decay 0 (App. D); ε = 1e-6 only reduces spikes | "setting ε= 10−5 completely eliminates them" (Fig. 3 caption) |
| Weight decay off at memorization stops grokking (0/6 vs 3/3) | Truong et al. [57] (PDF read) | Narrowed. wd-off alone is 0/3; the other 0/3 is norm-freeze. p = 97, η = 1e-3, λ = 1, full batch match #249; the model is 1 layer with LayerNorm, split 0.4, β2 = 0.999 | "F3 norm-freeze (0/3 grok ...); F4 wd-freeze (0/3 grok ...)" (§6.5); App. E.3-E.4 |
| Delay controlled by the product η·λ | Pracher et al. [54] | Supported. Squared loss and L2 decay assumed; strong decay prevents fitting | "controlled by the product of learning rate and weight decay" (abstract) |
| 293 runs; R² 0.97 (wd) and 0.92 (lr); SGD fails where AdamW groks | Truong et al. [56] | Supported. Same authors as [57] | Abstract |
| Ordering: fixed orders 99.5% at epochs 487 and 659; IID 0.30% after 5,000 | LeDoux [59] | Supported. Seed count not stated | Abstract |

**Tally:** 30 checks: 25 supported (7 of them with caveats or narrowed conditions), 5 narrowed, 0 contradicted, 0 not found. The red-team of 2026-10-28 narrowed three rows (the margin equilibrium, the Grokfast baseline, the weight-decay-off control) and added the Liu settings row. The 5 added checks cover the delta-lane claims that change recommendations. One lane claim is better described as narrowed: the "best-paper nominee" wording for 2601.19791. One lane claim looked like a contradiction but is not: StableMax being 6× slower does not conflict with anything Prieto et al. claim.

## Recommendations for ReLSM's first grokking run

Ranked by expected effect on whether #249 produces an interpretable result. "Inference" marks steps that go beyond the sources.

### Recipe

1. **Decay all weights, embedding and tied head included, with decoupled AdamW at weight decay 1.0.** This fixes #186 for the run. Print `optimizer.param_groups` before training and assert that the embedding is in a decayed group: a silently ignored weight-decay keyword cost the CSDN author hours [20]. Unit-test that decay is decoupled, because coupled L2 at weight decay 1 collapses training [28]. Evidence: [5][20][26][28][30][35]. Confidence: high.
2. **Restrict the output to the task vocabulary**: 97 residues plus a few delimiters, or a logit mask (#218). Score only the answer position [38]. Inference: with a tied 32k-row head, every unused row is a non-target logit. Unused rows then enter the margin and the softmax-collapse arithmetic, and they dominate the weight norm. The ko and fr lanes both recommend a task-sized vocabulary [30][38]. Confidence: medium-high.
3. **Constant lr 1e-3, β = (0.9, 0.98), full batch, warmup of 10 steps, dropout 0.** β2 = 0.98 is the value in nearly every replication and in Barron et al. [1][20][21][26]. Warmup is "not critical" [37]; Power et al. use 10 updates. Full batch is a stated departure from Power's minibatch 512 (see Ambiguities). Confidence: high.
4. **Split: p = 97, f = 0.5, fixed enumerated split, every residue present in training.** Stratify or tag pairs by whether the transpose is in training. As a cleaner alternative arm, split on unordered pairs with a ≥ b [5]. Confidence: high.
5. **Seeds: at least 3, preferably 5.** Report the grok rate, not one curve [20][21][39]. Pin CPU/GPU determinism where possible, since a Julia port's trajectories diverged on BLAS summation order alone [36]. Confidence: high.

### Precision and softmax guards

6. **Train and evaluate the whole model in fp32 for this run**, or at minimum compute logits and the loss in fp32. A bf16 loss with standard CE stalls at a margin of about 5 [1][21]. Inference: the 0.4M-parameter model makes all-fp32 nearly free on the RTX 4060. Confidence: high.
7. **Primary loss: standard CE at ε = 1e-8, the recipe.** Beside it, in the same sweep and on the same seeds: the margin-form CE, L = softplus(−m) with m = −LSE over j ≠ t of (z_j − z_t) (Eq. 6 in [1]), and an fp64 loss [1][3]. Both have direct evidence (Table 1 of [1]: 100 ± 0 each vs 94.3 ± 5.4; 5 seeds; one small task). ε is a swept factor in #255, not a guard: no source runs ε = 1e-7, and a too-large ε can end learning with no threshold given [2]. Confidence: medium.
8. **Log the margin.** Record the mean and minimum train logit margin, and the fraction of training rows whose target gradient is exactly zero. Two numbers mark the danger zones: 16.6 and ln(1/ε) [1][36]. Confidence: high.

### Diagnostics to log, every 100–250 steps

9. **Train and held-out loss on a log axis.** Spikes are invisible on a linear axis [4].
10. **Held-out accuracy split into "transpose seen" and "transpose unseen"**, plus held-out loss and the mean confidence on the correct class. Loss and confidence move before accuracy does [20].
11. **Adam's second moment.** Record the minimum and median of √v per parameter group, which show the v-decay that precedes a spike [1][36]. Also record the measured update size ‖Δθ‖/‖θ‖ per layer from parameter snapshots, which replaces the SGD-based telemetry (#149). Inference: the measured delta is the honest quantity under Adam.
12. **Weight norm per layer**, with the embedding norm over used rows only. This is a descriptive signal, not a trigger [11][42][45].
13. **Optional, cheap:**
    - a linear-probe R² on the residual stream [48];
    - Fourier power in the embedding [26];
    - the logit scale [58] and the train margin [67].

    These are leads, not validated measures. Report representation metrics after convergence, not at the grok step, and train at least 1e4 steps past it [66].
14. **Checkpoints every ~2k steps.** Train to the full budget [20].

### What to avoid

15. **No Grokfast in the first run.** Add it later as a separate arm with these settings:
    - filter once per optimizer step, between `backward()` and `step()`;
    - seed the EMA from the first gradient;
    - α 0.98, λ 2;
    - fp32, with gradient-norm logging;
    - compared against the same tuned AdamW, using a sustained-threshold metric [13][14][16][17][19].

    Evidence for leaving it out: [13][15][16][17][18][19], and the README's own baseline, which is unregularized Adam, so its headline speed-up is not measured against a tuned weight decay. Confidence: high.
16. **No early stopping, no plateau or cosine schedulers, and no best-validation-loss checkpoint selection** [20][30][38]. Confidence: high.
17. **No bf16 or fp16 loss, and no auto-rollback on loss spikes** [1][4][17]. Confidence: high.
18. **Do not read a weight decay 0 arm at f = 0.5 as a clean negative control.** It may grok slowly, or appear to generalize through symmetry [2][24]. Use the weight-decay-off-at-memorization control instead, once A1 has grokked [57] (0/3 vs 3/3 on a 1-layer, 0.4-split, β2 0.999 model). Confidence: medium.
19. **Do not use SGD as the optimizer baseline.** It fails where AdamW groks [56]. Compare every accelerator against the tuned AdamW weight decay 1.0 baseline, not against weak baselines [61][62]. Confidence: medium.

### Pass/fail expectations

The rows below are inspection points, not failure lines. No source with a stated batch size and model gives a 2-layer, 0.5-split, weight-decay-1 step count (see Answer, item 5). The budget is 1e5 updates, as in Power et al.

| Checkpoint | Expected | Action |
|---|---|---|
| ≤ 1k steps | Train accuracy ≥ 99% (replications memorize in 150–1,200 steps [20][24][56]) | If not: data, labels, masking or LR bug; not a grokking question. Stop. |
| Early | Transpose-seen held-out accuracy above chance; transpose-unseen near chance (1/97) | If transpose-unseen is high from the start, the split leaks or the evaluation is wrong. Stop. |
| 2k–8k steps | 1-layer references grok here [56][57]; a 2-layer model may or may not | Nothing to do. Read the margin trace: rising towards the cliff, or flat. |
| ≈ 2e4 steps | Order of magnitude of the 0.3-split grok time [20]; Nanda's 1-layer 0.3-split time | Inspect: `param_groups` and the decayed numel, the margin trace and the fraction of examples past 16.6, the split overlap, train exact match. Then continue. |
| 5e4 steps | — | Inspect again. Only 8 of 15 seeds had grokked by 30k on an easier 1-layer task [21]. Continue to 1e5. |
| 1e5 steps | #249's rule: groks in at least 2 of 3 seeds, or not | If not: #249's own decision tree (#256, #255), then the untied and final-norm-off arms before any new budget. |
| Any time | Loss spikes | Expected under standard CE if the margin reached the cliff; classify each one with the margin log. A spike far below the cliff is a different problem: suspect the LR or Grokfast |

## Parameters to test or sweep

This section lists the parameters the reports suggest testing that the planned experiments do not cover, or cover only in part. The planned experiments are:

- #249 (A1): the standard recipe;
- #251: a weight-decay sweep from 0 to 2.0 on all weights, tied and untied;
- #252: curriculum vs fixed data;
- #253: a single task at nano;
- #254: Grokfast;
- #255: StableMax, and fp32 vs bf16.

**How the table reads:**
- "Where" names the experiment a parameter belongs in. "New" means no planned experiment covers it.
- "Strength" grades the evidence: A = peer-reviewed; B = preprint with a method; C = blog or repo with runs; D = summary only.
- **Costs are rough inference.** They assume about 1–2 h per 1e5 full-batch steps for a ~0.4M-parameter model, which the CSDN author found was bound by kernel-launch overhead [20], so about 15–25 min per 2e4-step run on one RTX 4060.
- Every arm assumes 3 seeds unless stated.
- "Expected effect" is my reading of the evidence (inference) unless a source measured it.

| Parameter | Suggested values | Where | Evidence (strength) | Expected effect | Rough cost |
|---|---|---|---|---|---|
| Adam ε, plus loss form | ε ∈ {1e-8, 1e-7, 1e-6, 1e-5}; plus a margin-form CE arm and an fp64-loss arm, both at ε 1e-8 | #255 (partial: it plans StableMax and precision only) | Barron et al. [1] (B, 3–5 seeds, 1 layer, mod 31); 15-seed replication [21] (C); Liu et al. [3] (B, accepted, no weight decay, single seed); large ε halts learning [2] (A, no values) | Both loss fixes removed spikes and reached 100 % at λ = 1 in [1]. No source runs ε = 1e-7; ε = 1e-6 removed spikes on 2 seeds [21]. Each ×10 in ε lowers the margin plateau by about 2.3 nats (inference from Eq. 11), which may slow late learning. Speed-up in fp32 uncertain [21] | 6 arms × 3 seeds × 2e4 steps ≈ 5–7 h |
| η·λ product, and learning rate | lr ∈ {3e-4, 1e-3, 3e-3}, probe 5e-3; λ chosen so ηλ ∈ {1e-4, 3e-4, 1e-3, 3e-3} | #251 (partial: it sweeps λ at one lr) | Five groups [7][8][54][55][56][57] (A–B); lr 3e-4 stalls [25] and 5e-3 memorizes [41] (C–D) | Delay ≈ 1/(ηλ) until an upper cutoff where fitting fails [54]. Plot delay against ηλ | Diagonal of 6–8 cells × 3 seeds; low-ηλ cells need up to 1e5 steps, so ≈ 8–12 h |
| Train fraction f | f ∈ {0.3, 0.4, 0.5, 0.6} | New (extends #249) | Monotone delay in f [26] (C); no delay at 40% with wd 0.1 [19] (C); grok rates change with the split [73] (B); larger splits help without regularization [2] (A) | Smaller f gives a longer delay or a failure within budget. f ≥ 0.6 may show no visible delay | 4 × 3 seeds; f = 0.3 needs about 1e5 steps, so ≈ 6–10 h |
| Initialization scale | Init multiplier α ∈ {0.5, 1, 2, 4}; plus a zero-init output layer (untied head only) | New | Large init delays grokking (Omnigrok, Kumar; English map); zero-init top layer up to 10× faster with MSE [69] (B); a port's init was 1.4× smaller than reference [36] (C) | Larger α gives a longer delay. Zero-init head may shorten it. With a tied head, zero init would zero the embedding, so it needs the untied arm (inference) | 5 arms × 3 seeds ≈ 4–5 h |
| Weight-decay scope and coupling | {all; all except embedding and head (ReLSM today); embedding and head only; all except norm gains and biases}; plus coupled L2 at λ = 1 as a negative check | #251 (partial: it decays all weights, tied vs untied, but has no scope ablation). Directly tests #186 | ExPLAIND [5] (B, single run); every replication decays the embedding [20][26][30][35] (C); bias decay irrelevant [33][35] (C); coupled L2 collapses [28] (C); with RMSNorm, L2 acts mainly on scale [70] (B, preliminary) | Exempt embedding: slower or no grokking (inference; untested anywhere). Coupled L2: collapse within minutes | 5 arms × 3 seeds ≈ 5 h; the coupled arm fails fast |
| Weight decay off at memorization | At train accuracy ≥ 99%, set λ = 0; variant: freeze the norm; variant: λ = 0 after grokking | New (a control for #249 and #251, after A1 groks) | 0/3 (wd off) and 0/3 (norm freeze) vs 3/3, at p = 97, η = 1e-3, λ = 1, full batch, on a 1-layer, 0.4-split, β2 0.999 model [57] (B, small n); decay off after grokking keeps the algorithm [71] (B) | No grokking if the run's grokking is decay-driven. After-grok removal reverses compression only | 3 arms × 3 seeds × 2e4 steps ≈ 3 h |
| Commutative-pair handling | {random ordered split with transpose tags; unordered split a ≥ b; commutative augmentation, adding (b, a) for every train (a, b)} | New (split design for #249) | Reader critique [32] (C); symmetric operations start high [4] (A); a ≥ b split [5] (B); symmetry as augmentation [51] (B) | Unordered split: lower early held-out accuracy and a cleaner chance-to-100% curve. Augmentation: faster grokking (inference) | Tags are free; 2 extra arms × 3 seeds ≈ 2–3 h |
| Adam β2 | {0.95 (HEAD), 0.98, 0.999} | New (audit avenue C5 has a flag only) | 0.98 in nearly every replication [1][20][21][26] (B–C); β cancels in the margin equilibrium [1] (B); 0.999 arm in an unreadable ablation table [43] (C) | Affects the path, not the equilibrium margin. 0.999 may make spikes larger, because v reacts more slowly (inference) | 3 × 3 seeds ≈ 3 h |
| Precision scope | {all fp32; bf16 body with fp32 logits and loss; bf16 including loss; fp64 loss} | #255 (partial: it plans fp32 vs bf16 but not the scope split) | Loss-precision arms [1] (B); bf16 loss stalls the margin near 5 [21] (C); fp16 whole model fails without decay [27] (C); fp64 loss removes spikes [3] (B) | A bf16 body with an fp32 loss should behave like fp32 (inference). A bf16 loss stalls | 4 × 3 seeds ≈ 4 h |
| Output vocabulary | {task vocabulary of 97 residues plus delimiters; full 32k tied head; full 32k with a logit mask} | New (tests #218; #251's tied/untied arm touches it) | Task-sized vocabularies in the replications [30][38] (C) | The 32k head adds about 31k non-target logits to the margin and dominates the norm (inference) | 3 × 3 seeds ≈ 3 h |
| Final norm before the head | {final RMSNorm on; off} | New (audit avenue C7, no issue) | A final LayerNorm removes the norm dependence [42] (D, translation); LayerNorm changes how spikes act [1] (B); spherical residual stream groks without decay [63] (B) | With the norm on, weight-norm telemetry may show little, and spikes act as learning-rate decay | 2 × 3 seeds ≈ 2 h |
| Data order and curriculum | Under minibatch: {IID; fixed order; easy-to-hard}, with the example set held fixed and the order seed logged; full batch as reference | #252 (partial: it compares curriculum vs fixed, but not fixed order vs IID) | Order can dominate at low data [59] (B, single author); staggered multi-task grokking [60] (B); batch size has a weak effect [55] (B) | No effect under full batch. Possibly large effects under minibatch | Inside #252's budget |
| Alternative accelerators | Per-row ℓ2 clipping, max_norm 2.0, no decay; Grokfast with the reference settings | #254 (partial: Grokfast only) | Clip-to-Grok, 200 seeds against a weak baseline [62] (C); no independent Grokfast transformer replication [61] (C) | Unknown against a tuned λ = 1 baseline | 2 arms × 3 seeds ≈ 2–3 h |
| Number of seeds | 3 for screening; 5 for any claim; report the grok rate | All | 8/15 seeds [21] (C); 2–3× spread [20] (C); 0/16 to 16/16 by split [73] (B); 1.00 / 0.75 / 0.78 [39] (C) | Single-seed negatives are uninterpretable | Multiplies every row; small models may fit several seeds at once on the 8 GB card (inference) |
| Eval cadence | Every 100–250 steps on the full held-out set; checkpoints every ~2k steps; run ≥ 1e4 steps past the grok step | All (#249 first) | 250-step eval [20] (C); 100-step eval caught rise-and-fall [19] (C); compression continues ≥ 1e4 steps after grokking [66] (B) | Catches non-monotone transitions and late un-grokking | Small: the held-out set is about 4.7k examples |

**What the table changes in the plan (inference):**
- Run the weight-decay-scope and weight-decay-off controls next to #249 at the same seeds. They answer #186 directly and cost about one evening.
- Give #251 a learning-rate axis, and report results against η·λ.
- Give #255 the ε and loss-form arms. StableMax alone does not test the cliff that the #249 recipe crosses.

## Limitations and gaps

- **Thin academic coverage in all five languages.**
  - theses.fr returned 0 theses on grokking.
  - HAL holds only English papers by French-affiliated authors.
  - CNKI was not reachable without an institutional login.
  - The ja lane found one JSAI paper (2024) and one NLP2026 paper.
  - The ko lane found one KIISE paper and two theses, readable at abstract level only.
  - The de lane found one English-language ETH bachelor's thesis and no German thesis.

  Most native-language material is blogs, course notebooks and news. Only 12 of the 95 evidence-bearing sources are grade A, 7 of them outside the English delta lane, and many blog results are single-seed.
- **Coverage shortfall on replications of the exact recipe.** No source runs #249's exact configuration with an embedding-exempt optimizer, a 32k tied vocabulary or a final RMSNorm. The effects of #186 and #218 are therefore inferred.
- **Failed fetches and workarounds.** The lanes report:
  - Zhihu question pages behind a login;
  - wiki.swarma.org behind Cloudflare;
  - an expired TLS certificate at ashcreek.cc;
  - HTTP 410 at aitrends.kr;
  - HTTP 402 on X;
  - DBpia and KCI rendered with JavaScript;
  - LIRIS student PDFs at 404;
  - TU Darmstadt thesis PDFs at 404;
  - an ETH GitLab repository behind a login.

  In this synthesis, the arXiv HTML for 2610.31642 returned 404, so I read the v2 PDF. inria.hal.science sat behind Anubis bot protection, so I used the HAL API. Zhihu and CSDN refused WebFetch (403 and 521), so I read them in the browser pane. None of these protections was bypassed. The automated citation checker therefore cannot reach [19], [20], [40] or [41] (HTTP 403 and 521). [19] and [20] were read in the browser pane during this synthesis, and [40] and [41] by the zh lane.
- **Search infrastructure.** SearXNG stopped returning results partway through the lane sweeps, an episode the brief treats as a rate limit. Later lane searches used WebSearch, and the zh lane also used Bing China. Coverage of 稀土掘金, 博客园, 量子位, 新智元 and Naver Blog is thin. Docker was stopped during this synthesis, so it used WebSearch and WebFetch only.
- **Delta-lane depth.** The English delta lane read most of its 34 sources at abstract level, and several of its papers come from three author groups. I re-checked five of its claims in the papers.
- **Recency.** The Barron paper is 6 days old. Its two Japanese replications and summaries are 2 days old. Expect revisions.
- **Machine-assisted sources.** Several sources are agent-assisted or AI-labelled: the Zhihu A/B, the CoursIA notebooks, the NeuroDSL comments, the hangul-engine chapter, and the ja ZeroToHero runs. Their numbers were treated as reported, not as independently checked. The exceptions are the Ultralytics script, which two lanes re-ran, and the Zhihu and CSDN tables, which I read on the pages.
- **ReLSM facts.** ReLSM facts (tied head, final RMSNorm, β2 0.95, bf16) came from the brief and the audit notes, and were confirmed in code at `be08cde` on 2026-10-28 (see Scope).
- **Red-team corrections, 2026-10-28.** An independent review of this report against the primary sources changed: Answer items 2, 5, 6 and 7; Contradictions rows 2, 3, 4 and 11; the Prieto reading in the cross-check; the Verification rows for the margin equilibrium, the Grokfast baseline and the weight-decay-off control; Recommendations 3, 7, 15 and 18; the pass/fail table; and two rows of the sweep table. The mechanism claims did not change.

## Per-lane summary

- **zh:** 21 sources opened, 12 evidence-bearing. The value is in community replications:
  - the CSDN near-#249 weight-decay sweep [20];
  - the negative 16-arm Grokfast A/B [19];
  - a PKU course report with a 12-optimizer ablation, whose prose would not extract [43].

  The lane also carried the first Chinese coverage of the Barron fix [23].
- **ja:** 27 sources opened, 12 evidence-bearing. This is the strongest lane on numerics:
  - a 15-seed replication of the Barron mechanism, with a partial negative result [21];
  - the closest replication to the #249 recipe [24];
  - a coupled-L2 failure signature [28];
  - a StableMax / ⊥Grad speed table [27].
- **ko:** 21 sources opened, 15 evidence-bearing. Strongest on Grokfast first-party behaviour: the reference code and the author's issue replies [13]–[18]. It also adds a C++ replication with spike and validation-loss signatures [38], a lesson on seed variance [39], and a Target Linearity thesis lead [48].
- **fr:** 29 sources opened, 15 evidence-bearing. It adds:
  - the commutation-leak critique [32];
  - a Julia diagnosis of the Adam-ε slingshot, independent of Barron [36];
  - five replications [30][33][34][35][37];
  - pointers to Boursier, Tikeng Notsawo and Qchohi [7][11][12].
- **de:** 24 sources opened, 7 evidence-bearing. It adds:
  - the ETH spike thesis [4];
  - ExPLAIND's layer finding [5];
  - the pretraining-scale caution [46].

  It also supplied the Ultralytics re-run and an anti-grokking pointer [47].
- **en-delta:** 34 sources opened, 31 evidence-bearing. It adds:
  - Liu et al. [3];
  - the η·λ papers [54][55][56][57];
  - the weight-decay-off control [57];
  - ordering results [59][60];
  - accelerators with weak baselines [61][62][63];
  - train-only diagnostics [64][65][66][67];
  - the loop gap [68].

## Reliability assessment

- **Candidate results seen by the lanes:** about 2,660.
- **Unique sources opened:** 160, all visited, in `sources.jsonl`. That is 156 lane rows plus 7 added in synthesis, minus 3 duplicate locators merged as aliases.
- **Evidence-bearing sources:** 95, forming 68 independent clusters. Translations and syndications are merged; the Ultralytics script and the Barron write-ups are each one cluster.
- **Grade mix of evidence-bearing sources:** A 12, B 34, C 30, D 19.
- **Primary sources opened in synthesis:** 2610.31642, 2501.04697, 2601.19791 and 2605.06152, plus the ExPLAIND appendix, the ETH thesis, the HAL records, the Grokfast source and issues, Zhihu, CSDN and two ja replications.
- **Overall confidence:**
  - **High** for the recipe and avoid-list items.
  - **Medium** for the claim that the CE fix helps beyond stability.
  - **Low** for the timing thresholds, and for whether this run's margin crosses the fp32 cliff at ε = 1e-8.
  - **Low** for the newer diagnostics (Target Linearity, epistemic uncertainty).
- **Orchestration:** coordinator–worker. 6 research workers completed (5 language lanes and the English delta lane); 1 synthesis and verification worker (this one); no separate verifier workers.

## Bibliography

[1] Barron, J. T., Gnanapragasam, D., Tschannen, M. (Google DeepMind). (2026-10-22, v2 2026-10-26). "A Fix for Softmax and Cross-Entropy." arXiv:2610.31642. https://arxiv.org/abs/2610.31642

[2] Prieto, L., Barsbey, M., Mediano, P. A. M., Birdal, T. (2025). "Grokking at the Edge of Numerical Stability." ICLR 2025. arXiv:2501.04697. https://arxiv.org/abs/2501.04697

[3] Liu, H., Cao, J., Li, Y., Zhou, Z. (2026-05-07, v4 2026-10-07). "Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes." arXiv:2605.06152. https://arxiv.org/abs/2605.06152

[4] Arn, B. (2023-03-22). "Investigating 'Grokking' – Late Generalisation in Deep Learning." Bachelor's thesis, ETH Zürich. https://pub.tik.ee.ethz.ch/students/2022-HS/BA-2022-45.pdf

[5] Eichin, F., Du, Y., Mondorf, P., Matveev, M., Plank, B., Hedderich, M. A. (2026-06-11, v4). "ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior." arXiv:2505.20076. https://arxiv.org/html/2505.20076v4

[6] LMU München Open Access record for ExPLAIND (ICML 2026 poster). https://epub.ub.uni-muenchen.de/128628/

[7] Boursier, E., Pesme, S., Dragomir, R.-A. (2025-12-02). "A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation." NeurIPS 2025. HAL hal-05425613. https://inria.hal.science/hal-05425613v1

[8] Xu, M., Vardi, G., Safran, I. (2026). "To Grok Grokking: Provable Grokking in Ridge Regression." ICML 2026. arXiv:2601.19791. https://arxiv.org/pdf/2601.19791

[9] ICML 2026 virtual site, oral listing for "To Grok Grokking" (2026-07-08). https://icml.cc/virtual/2026/oral/71134

[10] 9bow. (2026-07-06). "ICML 2026 우수 논문상과 Test of Time상을 수상한 10편의 논문 소개" ("The 10 ICML 2026 award papers"). PyTorchKR forum. https://discuss.pytorch.kr/t/11112

[11] Tikeng Notsawo, P. J., Dumas, G., Rabusseau, G. (2025). "Grokking Beyond the Euclidean Norm of Model Parameters." ICML 2025. https://tikquuss.github.io/publications/grokking-beyond-l2/

[12] Qchohi, A., Rossi, S. (2026-04-14). "A Bayesian Perspective on the Role of Epistemic Uncertainty for Delayed Generalization in In-Context Learning." HAL report hal-05592608. https://hal.science/hal-05592608v1

[13] Lee, J. et al. (2024). Grokfast official repository, README. https://github.com/ironjr/grokfast

[14] Lee, J. et al. (2024). `grokfast.py`, `gradfilter_ema` and `gradfilter_ma`. https://raw.githubusercontent.com/ironjr/grokfast/main/grokfast.py

[15] Grokfast issue #1, "Choosing weight decay?", author replies 2024-06-18 and 2024-06-23. https://github.com/ironjr/grokfast/issues/1

[16] Grokfast issue #9, "Exploding Gradients" (2024-06-30). https://github.com/ironjr/grokfast/issues/9

[17] Grokfast issue #10, "How to use Grokfast with FP16 mixed precision training?" (2024-07-03; open). https://github.com/ironjr/grokfast/issues/10

[18] Grokfast issue #12, "AdamW better than grokfast + Adam?" (2024-07-10). https://github.com/ironjr/grokfast/issues/12

[19] 和谐和谐和. (2026-09-13). "权重衰减与 Grokfast 梯度滤波对小模型模运算顿悟的影响：16 臂对照实验" ("Weight decay and Grokfast on modular-arithmetic grokking: a 16-arm experiment"). Zhihu. https://zhuanlan.zhihu.com/p/2082568126093325501

[20] weixin_34246529. (2026-10-25). "Grokking复现：权重衰减驱动延迟泛化与训练相变" ("Grokking reproduction: weight decay drives delayed generalization"). CSDN. https://blog.csdn.net/weixin_34246529/article/details/169096490

[21] Takuya. (2026-10-26). "Softmax と Cross-Entropy に潜む 2 つの数値バグを Mac で再現してみた" ("Reproducing two numerical bugs in softmax and cross-entropy on a Mac"). Qiita. https://qiita.com/Takuya__/items/05bbe2c9e09c41382ebc

[22] Mihara, C. (2026-10-26). "【文献メモ】A Fix for Softmax and Cross-Entropy" ("Literature note"). Qiita. https://qiita.com/CookieBox26/items/48b302211800bb458a1b

[23] 机器之心 (Jiqizhixin). (2026-10-25). "PyTorch、JAX都被点名！Softmax曝出隐藏Bug" ("Hidden softmax bug"). Sina. http://k.sina.com.cn/article_5953189932_162d6782c067051yhi.html

[24] Kanda. (2026-03-07). "TransformerでGrokkingを観察する" ("Observing grokking in a Transformer"). Zenn. https://zenn.dev/ka_kan/articles/39090a66a8c904

[25] 新谷正嶺 (Shintani Lab, Chubu University). (2026-04-10). "シンプルな microGPT … modular addition で grokking を観察する" ("Observing grokking with a microGPT"). Qiita. https://qiita.com/Seine_A_Shintani/items/21d707e31ce8ec670a78

[26] 機械学習と情報技術. (2026-07-04). "Grokking — 過学習のはるか先で突然訪れる汎化を実測する" ("Measuring grokking"). https://disassemble-channel.com/grokking-delayed-generalization/

[27] ZeroToHero. (2026-03-08). "Grokkingを10分で再現する" ("Reproduce grokking in 10 minutes"). Zenn. https://zenn.dev/50s_zerotohero/articles/b4481d2e97f2a8

[28] Shake AI Lab. (2026-07-21, updated 2026-09-26). "Grokkingとは？ 訓練99%から27,400ステップ遅れて汎化した実験" ("Grokking: generalization 27,400 steps after 99% train"). https://shakeailab.com/blog/grokking-sudden-generalization/

[29] Ultralytics. (undated). "Grokking" glossary, German edition. https://www.ultralytics.com/de/glossary/grokking

[30] Ultralytics. (undated). "Grokking" glossary, French edition. https://www.ultralytics.com/fr/glossary/grokking

[31] Ultralytics. (undated). "Grokking" glossary, Korean edition. https://www.ultralytics.com/ko/glossary/grokking

[32] Louapre, D. (Science Étonnante). (2023-09-11). "« Grokking » : les modèles d'IA sont-ils capables de piger ?" with reader comments. https://scienceetonnante.substack.com/p/grokking-les-modeles-dia-sont-ils

[33] Science Étonnante. (undated). `grokking` notebook repository. https://github.com/scienceetonnante/grokking

[34] jsboige / CoursIA. (2026-10-26). "2.9 — Grokking : la généralisation qui arrive en retard" (notebook). https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/02-ML-Cours/2.9-Grokking-Generalisation.ipynb

[35] jsboige / CoursIA. (undated). "3.5 — Phénomènes de généralisation" (notebook). https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/03-Deep-Learning/3.5-Phenomenes-de-Generalisation.ipynb

[36] nevermind78 / NeuroDSL. (2026-07-28). `grokking_demo_neurodsl.jl`, French-commented Julia port with diagnostics. https://github.com/nevermind78/NeuroDSL/blob/HEAD/notebook/grokking_demo_neurodsl.jl

[37] Nameless0l. (2025-06-20). `grokking-reproduction` README. https://github.com/Nameless0l/grokking-reproduction

[38] codechosun. (2026-09-15). "D8. 그로킹 재현" ("Reproducing grokking"), `hangul-engine`. https://github.com/codechosun/hangul-engine/tree/main/ch-D08-grokking

[39] dancinlab. (2026-07-20). "H_9817 그로킹 레짐 스크린" ("Grokking-regime screen"). https://github.com/dancinlab/anima/blob/main/HYPOTHESES/cards/H_9817_grokking_regime_screen.md

[40] 卢R. (c. 2025-12). "对谈田渊栋" ("Conversation with Yuandong Tian"). Zhihu. https://zhuanlan.zhihu.com/p/1982024166560518507

[41] 机器之心. (2025-10-05). "Meta FAIR田渊栋唯一作者发文：拆解模型「顿悟时刻」" ("Tian dissects the aha moment"). Zhihu. https://zhuanlan.zhihu.com/p/1958241331680875100

[42] AI News Daily. (2026-06-16). Auto-translation of "The Weight Norm Sets the Grokking Timescale" (arXiv 2606.13753). https://ai-news-9hr.pages.dev/news/2026-06-16/7f7d7226686b

[43] Student course report. (2026-01-10). "Grokking 现象的复现与影响因素探究" ("Reproducing grokking and probing its factors"). https://maozy-leo.github.io/assets/documents/grokking-report.pdf

[44] Levy, T. (Brief IA). (2026-09-28). "Grokking : la généralisation tardive d'un réseau de neurones." https://www.briefia.fr/article/grokking-la-generalisation-tardive-d-un-reseau-de-neurones

[45] 峰岸剛基, 岩澤有祐, 松尾豊. (2024-06-11). "宝くじ仮説の観点からの Grokking の理解" ("Grokking through the lottery-ticket lens"). JSAI2024. https://www.jstage.jst.go.jp/article/pjsai/JSAI2024/0/JSAI2024_1B4GS203/_article/-char/ja

[46] Kemper, J. (THE DECODER). (2026-06-07). "Warum große KI-Modelle bestimmte Aufgaben lernen und kleine nicht." https://the-decoder.de/warum-grosse-ki-modelle-bestimmte-aufgaben-lernen-und-kleine-nicht/

[47] alphaXiv (German machine translation). Prakash, H. K., Martin, C. H. (2025-06-04). "Grokking and Generalization Collapse: Insights from HTSR theory." https://www.alphaxiv.org/de/abs/2506.04434

[48] Cha, T. (2026-08). "Two Faces of One Gram: Understanding Feature Learning with Virtual Covariance and Target Linearity." PhD thesis, Korea University. RISS. https://www.riss.kr/link?id=T17574313

[49] 카이스토. (2026-10-01). "[논문리뷰] 외운 뒤에야 비로소 이해한다 — To Grok Grokking" ("Paper review"). Tistory. https://bluediary8.tistory.com/1012

[50] BAAI Talk 368 announcement, Mingyue Xu on 2601.19791 (2026-07-21). https://hub.baai.ac.cn/view/56505

[51] Park, Y., Kim, M., Kim, Y. (2024-05-26). "Acceleration of Grokking in Learning Arithmetic Operations via Kolmogorov-Arnold Representation." arXiv:2405.16658. https://arxiv.org/abs/2405.16658

[52] 熊懐祐太, 吉永直樹. (2026-03). "網羅的統制実験によるTransformerの加算能力獲得機序の解明" ("How Transformers acquire addition"). NLP2026 TS4-7. https://anlp.jp/proceedings/annual_meeting/2026/pdf_dir/TS4-7.pdf

[53] wkeeli. (2026-08-09). "Grokking then Collapse? The Muon Optimizer…" Bilibili. https://www.bilibili.com/video/BV1MMuR6fE8e/

[54] Pracher, L., de Jong, P., Lieshaus, O., Jeffares, A., Rulands, S. (2026-09-22). "A Spectral Theory of Grokking: Weight Decay induces Feature Learning." arXiv:2609.26679. https://arxiv.org/abs/2609.26679

[55] Luo, R., Gagné, C., Ngnawé, J., Ullah, I., Morrissey, K. (2026-06-29). "A Stochastic-Geometric Theory of Scaling Laws in Grokking." arXiv:2606.30388. https://arxiv.org/abs/2606.30388

[56] Truong, X. K., Truong, Q. H., Luu, D. T., Phan, T. D. (2026-03-05, v2 2026-05-02). "The Norm-Separation Delay Law of Grokking." arXiv:2603.13331. https://arxiv.org/abs/2603.13331

[57] Truong, X. K., Truong, Q. H., Luu, D. T., Phan, T. D. (2026-05-13). "First-Passage Prediction of Grokking Delay: A Calibrated Law under AdamW with Causal Validation." arXiv:2605.18845. https://arxiv.org/abs/2605.18845

[58] Truong, X. K. (2026-06-16). "What Does the Weight Norm Control in Grokking? Logit-Scale Mediation under Cross-Entropy." arXiv:2606.18465. https://arxiv.org/abs/2606.18465

[59] LeDoux, J. (2026-03-24). "The Order Is The Message." arXiv:2603.25047. https://arxiv.org/abs/2603.25047

[60] Xu, Y. (2026-02-19, v3 2026-04-03). "The Geometry of Multi-Task Grokking: Transverse Instability, Superposition, and Weight Decay Phase Structure." arXiv:2602.18523. https://arxiv.org/abs/2602.18523

[61] SutroYaro. (2026-03-03). "Experiment 4: GrokFast (Low-Pass Gradient Filter)." https://cybertronai.github.io/SutroYaro/findings/exp4_grokfast

[62] NiftyliuS. (2026-03). "Clip to Grok" (per-row ℓ2 weight clipping), repository. https://github.com/NiftyliuS/cliptogrok

[63] Yıldırım, A. (2026-03-05, v3 2026-05-04). "The Geometric Inductive Bias of Grokking: Bypassing Phase Transitions via Architectural Topology." arXiv:2603.05228. https://arxiv.org/abs/2603.05228

[64] Xu, Y. (2026-02-19, v3 2026-04-03). "Early-Warning Signals of Grokking via Loss-Landscape Geometry." arXiv:2602.16967. https://arxiv.org/abs/2602.16967

[65] Cullen, Estan-Ruiz, Danait, Li. (2026-03-01, v3 2026-05-07). "A Basin-Selection Perspective on Grokking via Singular Learning Theory." arXiv:2603.01192. https://arxiv.org/abs/2603.01192

[66] Truong, X. K. (2026-07-07). "At-Grok Is Not Converged: A Measurement-Validity Audit for Grokking Representation Metrics." arXiv:2607.06639. https://arxiv.org/abs/2607.06639

[67] Ngnawé, J., Notsawo, P. J. T. (2026). "Margin-based Progress Measures for Grokking." NeurIPS 2026 workshop poster. https://neurips.cc/virtual/2026/162993

[68] Niu, Y. (2026-06-17, v2 2026-06-23). "Repeated Shared Access Enables Grokking, but Edit Propagation Depends on an Addressable Memory." arXiv:2606.20737. https://arxiv.org/abs/2606.20737

[69] Tian, Y. (2025-09-25, v5 2025-12-02). "Provable Scaling Laws of Feature Emergence from Learning Dynamics of Grokking." arXiv:2509.21519. https://arxiv.org/abs/2509.21519

[70] Li, M. (2026-06-03). "Low-Rank Decay for Grokking in Scale-Invariant Transformers: A Spectral-Geometric View." arXiv:2606.04405. https://arxiv.org/abs/2606.04405

[71] Xu, Y. (2026-04-08). "The Lifecycle of the Spectral Edge: From Gradient Learning to Weight-Decay Compression." arXiv:2604.07380. https://arxiv.org/abs/2604.07380

[72] Xu, Y. (2026-04-28). "Feature Repulsion and Spectral Lock-in: An Empirical Study of Two-Layer Network Grokking." arXiv:2605.08119. https://arxiv.org/abs/2605.08119

[73] Opus, A. C., Lu, J. Q. (2026-09-22). "An Exploratory Replica-Overlap Probe of the Grokking Transition." arXiv:2609.25634. https://arxiv.org/abs/2609.25634

## Methodology

- **Lanes:** five language workers (zh, ja, ko, fr, de) and one English delta worker each ran native-language searches and produced a report and a source register. This synthesis read all five reports and registers in full. It merged the registers into `sources.jsonl`, adding a `cluster` field for translations, mirrors and shared upstream sources.
- **Verification:** the synthesis opened every decision-driving primary source itself on 2026-10-28:
  - PDFs were text-extracted locally with pypdf, in isolated mode;
  - the HAL API was used for HAL records;
  - `gh api` was used for GitHub issues;
  - the browser pane was used for Zhihu and CSDN.

  The claims ledger is `claims.jsonl`.
- **English cross-check:** the owner's audit corpus was used only to judge novelty and contradiction, not as evidence for recommendations. It comprised res-codex-lit.md, res-gemini-perplexity.md, the avenues document and the lexical-session notes.
- **Additional context:** two sources are cited for context rather than for recommendations. The NLP2026 addition study shows depth and data mixture acting as grokking triggers outside the modular recipe [52]. A Bilibili explainer flags a "grok then collapse" failure mode under Muon, relevant if Muon is tried later [53]. Two further sources back the ridge-regression timescale in Korean and Chinese summaries [49][50].
- **Stopping rule:** every decision-driving claim was checked against its primary source or explicitly narrowed. The last checks (award tier, Kanda's batch size) changed no recommendation.
