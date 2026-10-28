# French-language grokking sweep (worker `fr`), as of 2026-10-28

Lane: French-language sources only. English papers by French-speaking or French-affiliated authors are listed only as pointers (marked **pointer**), with what the French record adds.

**Bottom line.** The French-language academic record on grokking is close to empty:
- theses.fr returns 0 theses for "grokking";
- HAL has no French-language grokking paper, only English papers by French-affiliated authors and passing mentions in theses;
- no CAp / PFIA / TALN / GRETSI paper was found.

What French-language material does add is practitioner evidence: five independent re-implementations with numbers, one detailed failure-mode log, and one unanswered critique of the split design. I re-ran one recipe (fr-05) locally on CPU, and its numbers reproduced.

## Takeaways (decision-relevant for ReLSM #249)

1. **Every French replication that groks decays the embedding table, and every run with weight decay set to zero fails.** This is a strong argument to fix #186 (the embedding is exempt from decay) before #249.
   - **How each replication decays the embedding:**
     - Science Étonnante (fr-02) uses `AdamW(model.parameters(), weight_decay=1)`, which covers the embeddings and the biases.
     - CoursIA 3.5 (fr-07) decays the weights and the embedding table, but not the biases.
     - Ultralytics (fr-05) uses one-hot inputs, so the first `Linear` *is* the embedding, and it is decayed.
   - **What happens with weight decay set to zero:**
     - fr-05: validation accuracy stays at 0.00 for 20k steps. I reproduced this locally: 0.00 at every checkpoint.
     - fr-07: final test accuracy 0.002, against chance 0.019.
     - fr-10: with Adam, test accuracy is 1.2% after 1M steps.
   - Biases are decayed in fr-02 and not in fr-07, and both grok. So bias decay is not the deciding factor.
   - **Corollary for #218 (my inference, not from a source).** Decoupled AdamW decay shrinks rows that get zero gradient by (1−ηλ) per step: about e^−100 after 1e5 steps at η=1e-3, λ=1. Putting the embedding under decay therefore also drains the 31k unused rows over time. Two caveats:
     - Exclude unused rows from the weight-norm *diagnostic* in any case.
     - If the output head is tied, unused rows still receive softmax gradient.
     - A vocabulary of size P plus a few special tokens for the #249 run removes the issue entirely.

2. **Timescale calibration: French MLP replications grok within about 3k–22k optimizer steps, so a 1e5-step budget is ample for an MLP.** Make ηλ the scheduling knob.

   | Source | Setup | Grok point |
   |---|---|---|
   | Nameless0l (fr-10) | (no code) | step 2,950 |
   | CoursIA 2.9 (fr-06) | P=97, frac 0.4, full batch, AdamW lr 1e-3, wd 1 | step 4,200 |
   | CoursIA 3.5 (fr-07) | NumPy, P=53, d=16 | step 15,500 |
   | Ultralytics (fr-05) | P=97, frac 0.3 | val 0.98 at step 20k |
   | Science Étonnante (fr-02) | batch 64, lr 3e-4 | ≈1,000 epochs ≈ 22k mini-batch steps |

   Boursier–Pesme–Dragomir (fr-13, NeurIPS 2025, pointer) predict that the slow norm-drift phase starts at time of order 1/λ. In discrete steps this is about 1/(ηλ), roughly 1e3 steps at ReLSM's settings. A 2-layer transformer may be slower than these MLPs, but 1e5 steps is still well above the observed range.

3. **Failure mode to instrument: an exactly-zero gradient after memorization, then Adam-ε "slingshot" spikes and limit cycles instead of grokking (fr-09, grade C).**
   - **The observation.** In a French-commented Julia port of Nanda's demo, once memorization is perfect the float32 cross-entropy gradient is *exactly* zero. Adam's v then decays geometrically by β2 to about 1e-16, so √v ≈ ε = 1e-8. The next tiny gradient (after weight decay erodes the margin) then yields a step of size ≈ lr, which shows up as a train-loss spike.
   - **The outcome.** A 16k-step run showed a *stationary* oscillation with a period of about 1,300–1,400 steps over 11 cycles, not grokking.
   - **The author's fixes.** A float64 cross-entropy gradient, and ε raised to 1e-4.
   - **Link to the English literature.** This is the softmax-collapse mechanism (StableMax paper) seen from the optimizer side.
   - **For ReLSM:**
     - compute cross-entropy and softmax in fp32 (never bf16);
     - log the fraction of exactly-zero per-example gradients, the minimum and median Adam v, the logit margin, and train-loss spikes;
     - keep the ε option in mind.
   - **Caveat.** This was p=17 with about 86 training examples, a regime known to be marginal, and the logs are not published.

4. **The split design has an unaddressed leak: commutativity.** A Science Étonnante reader pointed out that a+b is symmetric, so a random 50% split of ordered pairs puts many held-out pairs' transposes into the training set. The author acknowledged the point, and nobody followed up.
   - **How big the leak is (my own arithmetic).** With train fraction f, each off-diagonal held-out pair (a,b) has (b,a) in train with probability f. At f=0.5, about half of held-out accuracy could in principle come from symmetry alone.
   - **For ReLSM:** report held-out accuracy separately for the "transpose seen" and "transpose unseen" subsets, or split on unordered pairs. This costs nothing and tells memorization-by-symmetry apart from the Fourier algorithm.

5. **Disable any patience-based early stopping and any plateau LR scheduler, keep LR constant, and do not rely on the ℓ2 norm alone as a progress measure.**
   - Brief IA (fr-03) and Ultralytics (fr-05) both note that standard early-stopping rules would cut the run during the plateau.
   - Tikeng Notsawo–Dumas–Rabusseau (fr-16, Mila/UdeM, ICML 2025, pointer) show that the ℓ2 norm is an unreliable progress proxy when other implicit biases act. Depth alone can make a model grok or ungrok without explicit regularization.
   - Qchohi & Rossi (fr-14, HAL 2026, pointer) report that epistemic uncertainty collapses sharply at the grok point. That makes it a label-free diagnostic worth adding next to train/test accuracy.

## Sources

Grades:
- A: peer-reviewed or thesis
- B: preprint, conference paper or detailed report with numbers
- C: blog or repo with code and reported results
- D: opinion, summary or translation

Novelty is judged against the English literature.

### fr-01: « "Grokking" : les modèles d'IA sont-ils capables de piger ? »
*"Grokking": can AI models "get it"?*

- **Identity:**
  - Author/org: David Louapre (Science Étonnante)
  - Venue: Substack
  - Date: 2023-09-11
  - URL: https://scienceetonnante.substack.com/p/grokking-les-modeles-dia-sont-ils
- **Type:** blog, with code (fr-02) and reader comments.
- **Findings:**
  - Minimal reproduction: addition mod P=53, embedding (d=128) plus one-hidden-layer feedforward net, 50% split.
  - Train reaches 100% after about 250 "iterations", with test at 0; test reaches 100% after about 1,000. The code (fr-02) shows these are *epochs* of 64-example mini-batches.
  - PCA of the embeddings shows circles at several frequencies, including a "53-hour clock" and a period-3 circle.
  - The author states, without numbers, that a large embedding dimension is necessary.
  - No systematic sensitivity study ("sans faire d'étude systématique de sensibilité").
  - Reader comment (Guillaume, 2023-09-11): commutativity makes a random 50% split near-redundant. The author replied (2023-09-12) that he used addition to keep things simple and that Power et al.'s figure used division.
- **Quote:** « Apparemment un ingrédient clé dans cette affaire, c’est la pénalité qui est appliquée sur les valeurs des poids pendant l’apprentissage. » (section "Comment le modèle fait-il ?")
- **Grade:** C
- **Novelty:** replication plus critique. The symmetry-leak critique appears to be unique to this thread among what I opened.

### fr-02: `scienceetonnante/grokking` (README in English, linked to the French post)

- **Identity:**
  - Author/org: Science Étonnante
  - Venue: GitHub
  - Date: undated (the README links the 2023-09-11 post)
  - URL: https://github.com/scienceetonnante/grokking
- **Type:** code (Jupyter notebook).
- **Findings (read from `grokking.ipynb`):**
  - Data: P=53, `TRAIN_FRACTION = 0.5`, `BATCH_SIZE = 64`, so 1,404 training pairs and about 22 steps per epoch.
  - Model: `D_EMBED = 128`, `HIDDEN = 256`, ReLU.
  - Initialization: Xavier-normal for the embedding and linear layers, zero biases.
  - Optimizer: `AdamW(model.parameters(), lr=0.0003, weight_decay=1)`, so decay covers the embedding *and* the biases.
  - Run length: `NB_EPOCHS = 20000`. The grok at about 1,000 epochs is roughly 22k optimizer steps.
  - Mini-batch, not full batch.
  - Device: CPU or CUDA in default fp32.
  - 69 stars, 4 forks.
- **Quote:** none in French; the README is in English. Locator for the key line: notebook cell 5, `optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)`.
- **Grade:** C
- **Novelty:** replication (a mini-batch feedforward net variant).

### fr-03: « Grokking : la généralisation tardive d'un réseau de neurones »
*Grokking: the delayed generalization of a neural network*

- **Identity:**
  - Author/org: Tom Levy (Brief IA)
  - Venue: briefia.fr, Recherche section
  - Date: 2026-09-28
  - URL: https://www.briefia.fr/article/grokking-la-generalisation-tardive-d-un-reseau-de-neurones
- **Type:** news.
- **Findings:**
  - Popular account: train perfect in under 1,000 steps with test at about 10%, then near 100% at about 20,000 steps. Numbers unsourced.
  - Main practical point: a standard early-stopping rule would have cut the transition.
- **Quote:** « Une règle d’arrêt précoce standard aurait presque interrompu ce basculement » (section "Des métriques trompeuses et l'arrêt précoce en question")
- **Grade:** D
- **Novelty:** summary only. The numbers do not match any specific paper (see Contradictions).

### fr-04: « Grokking (apprentissage automatique) »
*Grokking (machine learning)*

- **Identity:**
  - Author/org: Wikipédia contributors
  - Venue: fr.wikipedia.org
  - Date: undated (oldid 240707053)
  - URL: https://fr.wikipedia.org/wiki/Grokking_(apprentissage_automatique)
- **Type:** encyclopedia (partly translated from the English article).
- **Findings:**
  - Gives the French terms « généralisation différée » and « grokking time ».
  - Unlike a plain summary, it foregrounds Softmax Collapse and ⊥Grad as a revision of the "L2 favours the low-norm solution" story.
- **Quote:** « c'est parce que cette dernière limite le risque d'effondrement de la fonction softmax (Softmax Collapse) » (section "Interprétation")
- **Grade:** D
- **Novelty:** summary.

### fr-05: « Grokking » (Ultralytics glossary, French edition)
*Grokking (glossary entry)*

- **Identity:**
  - Author/org: Ultralytics
  - Venue: ultralytics.com/fr
  - Date: undated
  - URL: https://www.ultralytics.com/fr/glossary/grokking
- **Type:** glossary with a runnable script (French translation of corporate content).
- **Findings:**
  - Setup: p=97, 30% of 9,409 pairs, one-hot inputs, MLP 2p→256→p, `AdamW(lr=1e-3, weight_decay=1.0)`, full batch, 20,000 steps, under a minute on CPU.
  - Reported: train 1.00 at step 4,000 with validation 0.00; validation 0.16, 0.54, 0.84 and 0.97 at steps 8k, 12k, 16k and 20k.
  - With wd=0.0, validation stays at 0.00 throughout.
  - **I re-ran the script on CPU with torch 2.8.0 and it reproduced:** validation 0.00 / 0.18 / 0.63 / 0.86 / 0.98, and 0.00 at every checkpoint with wd=0.
  - It also flags that early stopping (patience 100 in YOLO) would kill a grokking run.
- **Quote:** « Avec `weight_decay=0.0`, la précision de validation de la même exécution reste à 0,00 pendant les 20 000 étapes. » (section "Déroulement du grokking")
- **Grade:** C, independently verified here.
- **Novelty:** replication with a verified full-batch baseline, but translated corporate content.

### fr-06: « 2.9 — Grokking : la généralisation qui arrive en retard »
*Grokking: generalization that arrives late*

- **Identity:**
  - Author/org: jsboige / CoursIA (French ML course; notebooks appear agent-assisted)
  - Venue: GitHub
  - Date: last commit 2026-10-26
  - URL: https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/02-ML-Cours/2.9-Grokking-Generalisation.ipynb
- **Type:** code (executed notebook with stored outputs).
- **Findings:**
  - Setup: P=97, `TRAIN_FRAC = 0.4` (3,763 train / 5,646 test), embedding d=64 plus MLP with two hidden layers of 256 (129,953 parameters).
  - Optimizer: `AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)`, decay on all parameters, full batch, 12,000 steps.
  - Results: grok point (test > 0.90 with train saturated) at **step 4,200**, final test 1.000, 188 s on CPU.
  - Dominant Fourier frequency k*=33 (25% of variance), with top frequencies 33, 41, 30 and 6.
  - Honest caveat: the PCA "clock" is not causal evidence; it recommends ablating Fourier components.
- **Quote:** `WEIGHT_DECAY = 1.0   # LE LEVIER : regularisation forte qui declenche le grok` (code cell 1)
- **Grade:** C
- **Novelty:** replication (full-batch MLP at P=97, frac 0.4).

### fr-07: « 3.5 — Phénomènes de généralisation » (grokking section)
*Generalization phenomena*

- **Identity:**
  - Author/org: jsboige / CoursIA
  - Venue: GitHub
  - Date: undated; repository active in Oct 2026
  - URL: https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/03-Deep-Learning/3.5-Phenomenes-de-Generalisation.ipynb
- **Type:** code (pure NumPy, gradient checked to 1.97e-7).
- **Findings:**
  - Setup: P=53, frac 0.4 (1,123 train / 1,686 test), d_emb=16, two hidden layers of 64 (10,565 parameters).
  - Optimizer: hand-written Adam with decoupled wd=1.0 on weights *and the embedding table, not biases*; lr 1e-3, full batch, 40,000 steps.
  - Results: train ≥ 0.95 at step 500; grok (test ≥ 0.85) at **step 15,500** (test 0.624 at step 15,000, 0.856 at step 15,500); final test 1.000.
  - **wd=0: train 1.000, test 0.002** (chance 0.019).
  - Proposes, but does not run, sweeps over wd ∈ {0.1, 0.3, 1, 3} and train fraction ∈ {0.2, 0.3, 0.4, 0.6}.
- **Quote:** `if k.startswith("W") or k == "emb":      # decay : poids et table, pas les biais` (code cell 6)
- **Grade:** C
- **Novelty:** replication. It shows grokking with a tiny d=16 embedding, which contradicts fr-01's "large dimension needed" claim.

### fr-08: « 2.9c — Grokking : le diagramme de phases »
*Grokking: the phase diagram*

- **Identity:**
  - Author/org: jsboige / CoursIA
  - Venue: GitHub
  - Date: undated
  - URL: https://github.com/jsboige/CoursIA/blob/main/MyIA.AI.Notebooks/ML/DataScienceWithAgents/02-ML-Cours/2.9c-Grokking-Diagrammes-Phases.ipynb
- **Type:** code.
- **Findings:**
  - Reproduces the toy model of Liu et al.'s effective theory: non-modular addition 0..9 with a hard-coded decoder.
  - Validation 1.000 for train fractions ≥ 0.35; 0.890 at a fraction of 0.25.
  - Not the finite-set transformer recipe.
- **Quote:** (none retained)
- **Grade:** C
- **Novelty:** replication of the toy model. Not evidence-bearing for #249.

### fr-09: `grokking_demo_neurodsl.jl` and probes (`grokking_diag_instrumented.jl`, `grok_init_probe.jl`, `grokking_fourier_analysis.jl`)
French-commented Julia port of Nanda's grokking demo.

- **Identity:**
  - Author/org: nevermind78 / NeuroDSL (comments appear agent-written)
  - Venue: GitHub
  - Date: commit 2026-07-28
  - URL: https://github.com/nevermind78/NeuroDSL/blob/HEAD/notebook/grokking_demo_neurodsl.jl
- **Type:** code with documented diagnostics.
- **Findings:**
  - **Setup.** 1 layer, 4 heads, d_model=128, MLP 512, no norms, no biases. p=17 (not 113), frac 0.3 (about 86 training examples), AdamW lr 1e-3, wd 1.0, β2=0.98, full batch, 8k–20k steps.
  - **Documented failure: Adam-ε slingshot.** After perfect memorization the float32 gradient is exactly 0. Adam's second moment then decays by β2 per step (0.98^20 = 0.667 measured over about 1,300 steps) down to about 1e-16, so √v ≈ ε. The next small gradient produces an lr-sized step and a train-loss spike.
  - **Documented failure: limit cycle.** The 16k-step run showed a stationary limit cycle with a period of about 1,300–1,400 steps over 11 cycles, not grokking.
  - **Remedies.** Float64 cross-entropy gradient; ε raised to 1e-4.
  - **Other notes:**
    - CPU and GPU runs diverged from the first steps because BLAS summation order differs ("extrêmement sensible à des écarts numériques infimes").
    - The initialization scale was about 1.4× smaller than TransformerLens defaults, probed with reference to Omnigrok.
    - The p17_seed0 checkpoint "n'a PAS grokké".
  - No result logs are published.
- **Quote:** « en phase de mémorisation parfaite le gradient réel est EXACTEMENT nul » (header comment, lines 52–53 of `grokking_demo_neurodsl.jl`)
- **Grade:** C (code plus reported observations; not replicated; tiny p)
- **Novelty:** new negative result and failure-mode diagnosis. It links softmax collapse to Adam ε and slingshot. I saw no equivalent in French elsewhere.

### fr-10: `grokking-reproduction` README (French message to a supervisor)

- **Identity:**
  - Author/org: Nameless0l
  - Venue: GitHub
  - Date: 2025-06-20
  - URL: https://github.com/Nameless0l/grokking-reproduction
- **Type:** code repo, but the README only; no code is published.
- **Findings:**
  - Reproduces Power et al.
  - Adam with wd=0 overfits completely after 1M steps (train 100%, test 1.2%).
  - AdamW with wd=1.0 groks at **step 2,950**, with test jumping from 0 to 100%.
  - AdamW with wd=1 beats the other optimizers across data fractions.
  - Adam+L2 is "not the same at all" as decoupled AdamW.
  - The 10-step warmup helps early stability but is "pas critique".
  - The MLP groks as well as the Transformer.
  - Modular division at 100k epochs took days.
- **Quote:** « AdamW et Adam+weight_decay c'est pas pareil du tout (decoupled weight decay) » (README, "Points importants")
- **Grade:** D (numbers but no code or config)
- **Novelty:** replication (a student-level practical tip on decoupled decay).

### fr-11: « Apprentissage et Graphes » (GdR IASIS / GdR RADIA day, poster by Lionel Gil)
*Learning and Graphs*

- **Identity:**
  - Author/org: GdR IASIS (CNRS)
  - Venue: Institut Henri Poincaré, Paris
  - Date: 2025-03-31
  - URL: https://gdr-iasis.cnrs.fr/reunions/apprentissage-et-graphes/
- **Type:** conference (event page, poster abstract in English).
- **Findings:** links to fr-12.
- **Quote:** n/a; the abstract is in English.
- **Grade:** D
- **Novelty:** pointer.

### fr-12: *Generalization emerges from local optimization in a self-organized learning network*

- **Identity:**
  - Authors: S. Barland, Lionel Gil
  - Venue: HAL preprint, v2
  - Date: 2025-08-26
  - URL: https://hal.science/hal-04758774v2 (read via the HAL API; the HTML page is behind Anubis)
- **Type:** paper (preprint, English).
- **Findings:**
  - A network with no global loss stores examples as a lookup table, then restructures its topology by local rules.
  - The jump to perfect generalization is abrupt with "first order phase transition" characteristics, which the authors identify with grokking.
- **Quote:** n/a (English)
- **Grade:** B
- **Novelty:** **pointer**. A French-lab argument that grokking-like transitions do not need gradient descent or weight decay. Not decision-relevant for #249.

### fr-13: *A Theoretical Framework for Grokking: Interpolation followed by Riemannian Norm Minimisation*

- **Identity:**
  - Authors: Etienne Boursier, Scott Pesme, Radu-Alexandru Dragomir
  - Venue: NeurIPS 2025; Inria HAL record
  - Date: 2025-12-02
  - URL: https://inria.hal.science/hal-05425613v1
- **Type:** conference paper (English).
- **Findings:**
  - With small weight decay λ, gradient flow first converges to the zero-loss manifold.
  - Then, at time of order **1/λ**, it drifts along a Riemannian gradient flow that minimizes the ℓ2 norm.
  - The generalization jump is attributed to this slow norm reduction.
  - Validated on synthetic regression.
- **Quote:** n/a (English)
- **Grade:** A (peer-reviewed; abstract read only)
- **Novelty:** **pointer**. It gives ReLSM a timescale rule (ηλ).

### fr-14: *A Bayesian Perspective on the Role of Epistemic Uncertainty for Delayed Generalization in In-Context Learning*

- **Identity:**
  - Authors: Abdessamed Qchohi, Simone Rossi
  - Venue: HAL report
  - Date: 2026-04-14
  - URL: https://hal.science/hal-05592608v1
- **Type:** paper (report, English).
- **Findings:**
  - Uses modular-arithmetic in-context tasks.
  - Epistemic uncertainty collapses sharply when the model groks, which makes it a "label-free diagnostic".
  - A spectral mechanism links grokking time to uncertainty peaks.
- **Quote:** n/a (English)
- **Grade:** B
- **Novelty:** **pointer** (new diagnostic).

### fr-15: *A Stochastic–Geometric Theory of Scaling Laws in Grokking*

- **Identity:**
  - Authors: Róisín Luo, Christian Gagné, Jonas Ngnawé, Ihsan Ullah, Karyn Morrissey
  - Venue: arXiv 2606.30388, v3
  - Date: 2026-06-29
  - URL: https://arxiv.org/abs/2606.30388
  - Found through Gagné's French-language CV; that CV was seen only as a search snippet.
- **Type:** paper (preprint, English).
- **Findings:**
  - Under Adam with weight shrinkage, the reachable solutions form shells: initialization on an outer shell, then memorization, then a generalizing core.
  - Stopping-time analysis yields grokking scaling laws in learning rate, batch size and ℓ2 coefficient.
  - No numbers were read.
- **Quote:** n/a (English)
- **Grade:** B
- **Novelty:** **pointer**.

### fr-16: *Grokking Beyond the Euclidean Norm of Model Parameters*

- **Identity:**
  - Authors: Pascal Jr Tikeng Notsawo, Guillaume Dumas, Guillaume Rabusseau (Mila / Université de Montréal)
  - Venue: ICML 2025
  - URL: https://tikquuss.github.io/publications/grokking-beyond-l2/
- **Type:** conference paper (English abstract page).
- **Findings:**
  - Small non-zero regularization of *any* generalizing property P (ℓ1, nuclear norm) induces grokking.
  - Depth can grok or ungrok with no explicit regularization.
  - **The ℓ2 norm is not a reliable proxy** when the regularization targets a different property P.
  - Grokking can be amplified by data selection alone.
- **Quote:** n/a (English)
- **Grade:** A
- **Novelty:** **pointer**. No French-language mémoire or thèse was found for this group.

### fr-17: *Abrupt and spontaneous strategy switches emerge in simple regularised neural networks*

- **Identity:**
  - Authors: Löwe, Touzo, Muhle-Karbe, Saxe, Summerfield, Schuck
  - Venue: PLoS Computational Biology
  - Date: 2024
  - URL: https://hal.science/hal-04979935
- **Type:** journal article (English).
- **Findings:**
  - Insight-like abrupt switches after a delay appear in networks with gradual learning and a constant LR.
  - They require regularized gating plus gradient noise, which accumulate "silent knowledge".
- **Quote:** n/a (English)
- **Grade:** A
- **Novelty:** **pointer**, mechanism-adjacent.

### Opened but not evidence-bearing (in the JSONL)

- **Passing mentions in theses:**
  - fr-18: Szatkownik thesis (Inria, 2025), lists grokking among "narrowly contrived" phenomena
  - fr-19: Helvig thesis (2024, French), one-sentence aside on grokking as a "changement de mode propre"
  - fr-20: Mesbah thesis (Rouen, 2026), one citation of Power et al.
- **Machine translation:**
  - fr-21: chatpaper.ai French translation of Li et al. 2025, "grokking in OLMoE pretraining"
- **Irrelevant or no grokking content:**
  - fr-22: Lokoyote blog, irrelevant
  - fr-23: LinuxFr feature on free AI (2025-02-26), no grokking
- **AI-generated synthesis brief:**
  - fr-24: Sandjab/scriptorium grokking theme brief
- **French-affiliated HAL abstracts with no grokking evidence in the abstract:**
  - fr-25: Satariyan–Cousin–Biau (2026), multiplication transformer; carry encoded as an angle on a ring
  - fr-26: Karmim–Marti–Seddah–Barrière (EMNLP 2026)
  - fr-27: Odonnat–Bouaziz–Cabannes (ICASSP 2025)
- **Hits on "Grokking …" book titles:**
  - fr-28: MECEN-TOURS README
  - fr-29: INSEE SSPHub newsletter 27

## Contradictions with the standard English account

- **Embedding dimension.**
  - fr-01 asserts that a large embedding dimension is needed. fr-07 groks at P=53 with d_emb=16.
  - The claim is at best about ease, not necessity.
- **Popular-press numbers.**
  - Brief IA (fr-03) says test is about 10% after 1k steps and grokking happens near 20k steps, with no source given. That matches neither Power et al. (about 1e5 steps for division) nor typical near-chance (≈1/p) plateaus.
  - All replications I opened sit at 0.00 to chance during the plateau. Treat fr-03's numbers as unreliable.
- **"Robust phenomenon" vs. observed fragility.**
  - fr-01 echoes the claim that grokking is robust across conditions.
  - fr-09 fails to grok at p=17 (about 86 examples), shows limit cycles, and finds CPU/GPU numerical non-determinism changes trajectories.
  - This matches the English data-fraction-threshold and slingshot literature better than the "robust" framing.
- **Why weight decay matters.**
  - French Wikipédia (fr-04) presents softmax collapse as the reason L2 helps.
  - Boursier et al. (fr-13) give a pure optimization-geometry account: slow ℓ2 drift at time of order 1/λ.
  - Tikeng Notsawo et al. (fr-16) say ℓ2 is just one instance of "regularize toward a generalizing property".
  - These are complementary rather than contradictory, but the French encyclopedia picks one.
- **Mini-batch vs. full batch.**
  - The canonical recipe is full batch, but fr-02 groks with batch 64 at lr 3e-4.
  - Batch size is not a necessary condition, consistent with the English literature.

## Gaps (searched, nothing found)

- **Theses and HAL:**
  - theses.fr API: **0** theses for `grokking`. The only `"généralisation tardive"` hits are unrelated.
  - HAL full text: 58 records mention "grokking", 8 of them in French. None is a French-language study of grokking.
  - HAL HTML and PDF pages sit behind Anubis bot protection; I accessed them through the API and plain curl.
- **Conferences:** no CAp, PFIA / RFIA, TALN or GRETSI 2025–2026 paper on grokking was found by search. I did not scan the proceedings tables of contents exhaustively, so a conference-proceedings gap remains possible.
- **Student syntheses:** the "IAC 2024 Synthese Grokking" and "IAC-2024-Grokking" PDFs (LIRIS, Université Lyon 1 course workshop) are indexed by search engines but now return **404** and have no Wayback snapshot.
- **Video:** no French-language YouTube video on grokking was found. Science Étonnante covered it as a blog post, not a video; Machine Learnia and Monsieur Phi had no hits. Search engines index YouTube poorly, so this gap is weak.
- **Methods and numerics:** no French-language replication or discussion of Grokfast, StableMax / ⊥Grad, weight-decay schedules, or bf16/fp16 effects on grokking.
- **Mila / Université de Montréal:** grokking output exists (fr-16) but is English-only. No French mémoire was found.
- **Media and community:** no French coverage in Le Monde Informatique, Usine Digitale, Numerama, Pour la Science or Polytechnique Insights; no Medium-fr posts; no r/france or Québec Reddit threads; no French Hugging Face blog post.
- **LLMs:** no French-language work on grokking in LLMs or pretraining, apart from machine-translated abstracts (chatpaper.ai and alphaXiv /fr pages).
- **Prompt phrasings:** "Grokking du Langage" and "Curriculum Étagé" returned nothing, consistent with those reports being AI-generated.

## Method

- **Engines:**
  - SearXNG with `language: fr`, plus `time_range: year` on 2 queries, which returned 0 results;
  - WebSearch (5 queries);
  - HAL Solr API (10 queries, including `fulltext_t:grokking` and `language_s:fr`);
  - theses.fr API (2);
  - GitHub code search through `gh` (5);
  - direct fetches through `web_url_read`, WebFetch and curl.
- **Queries:**
  - grokking généralisation tardive réseaux de neurones
  - "grokking" thèse apprentissage profond généralisation
  - Grokfast accélérer grokking gradients filtre
  - grokking "softmax collapse" OR StableMax effondrement softmax
  - grokking vidéo youtube français réseau de neurones addition modulaire
  - papyrus umontreal grokking mémoire maîtrise
  - linuxfr grokking réseau neurones
  - "grokking" medium français …
  - grokking IA généralisation (year)
  - grokking LLM pré-entraînement mémorisation généralisation article
  - "grokking" Mila Québec chercheurs généralisation tardive
  - youtube grokking "généralisation" …
  - "grokking" site:reddit.com france …
  - "grokking du langage" OR "curriculum étagé"
  - aclanthology jeptalnrecital 2025 grokking
  - PFIA 2025 CAp actes "grokking"
  - WebSearch: weight decay / décroissance des poids blog; Science Étonnante / Machine Learnia / Monsieur Phi; Pour la Science / Numerama / Usine Digitale; GitHub addition modulaire README; CAp / PFIA / TALN / GRETSI 2025; Tikeng Notsawo mémoire
  - GitHub code search: grokking + généralisation / entraînement / modulaire / apprentissage / réseau
  - HAL: grokking, `fulltext_t:grokking`, "généralisation tardive", "generalisation differee", "delayed generalization"
- **Counts:**
  - about 520 result listings seen (about 380 search-engine hits, 70 HAL records, 75 GitHub code hits);
  - **29 unique sources opened**;
  - **17 retained** as entries, of which **15 are evidence-bearing**: 9 French-language, 6 English pointers by French- or Québec-affiliated authors.
- **Local verification:** I re-ran the fr-05 script on CPU (torch 2.8.0+cpu, seed 0) with wd=1.0 and wd=0.0. Results: validation 0.00 / 0.18 / 0.63 / 0.86 / 0.98 at steps 4k–20k with wd=1.0; 0.00 throughout with wd=0. The script is in the session scratchpad and is not committed.
- **Trust handling:** page content was treated as data. No instructions were found in the pages.
