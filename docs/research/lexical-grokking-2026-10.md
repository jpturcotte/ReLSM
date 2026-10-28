# Grokking lexical structure in ReLSM: literature, the token-table question, experiment designs

Date: 2026-10-26. Branch: `claude/research-lexical-grokking-a7d69f`. Base commit: `e6724c3`.

Scope: what has been tried beyond modular arithmetic; whether the token table can be changed after a model groks math; and two to four experiments that fit this repo and one consumer GPU. This is a brainstorm backed by a literature sweep and one small probe. No training run was made.

## 0. Short answer

1. **Lexical grokking is an open niche.** Small transformers do show delayed generalization on grammar-like rules, but the jump is smoother and weaker than on modular arithmetic. No paper we found studies grokking of morphology (word inflection), lexical classes or small synthetic grammars with the modern grokking toolkit (Grokfast, weight-norm tracking, mechanistic probes). The closest prior work: structural grokking on syntax (Murty 2023), grokking in knowledge composition (Wang 2024), and delayed grammatical generalization during pretraining (Muckatira 2026).
2. **Changing the table after grokking math: yes, but only for rows math does not use.**
   - Replacing or periodically resetting the whole table breaks the math circuit, because its inputs and (with a tied head) its outputs live in the table.
   - Resetting only rows the math phase never used is safe for math, and is probably the best thing to do at a phase switch.
   - Our probe found something that matters for issue #186. Under the current optimizer, rows of unseen tokens are not protected. Through the tied output head they grow about 26× and collapse onto one shared direction, whether or not the table gets weight decay. So the weight-decay exemption does not prevent the "lexical collapse" it was added for (§4.1).
   - This collapse is documented in the literature (Gao 2019, Yu 2022, and most directly Lazić 2026). Resetting only dead rows has precedent in VQ codebook "random restarts" (Jukebox 2020). No standard recipe applies it to a token table during pretraining (§4.3).
3. **Best first experiment: "paradigm grokking"** on a synthetic inflection language (§5, E1). It takes minutes per run, tests lexical structure directly, and settles the #186 decision as a side effect. Before it, run a grok-then-change-the-table experiment on modular addition (E2) as a positive control, to make sure the harness can grok at all.

## 1. Terms used here

| Term | Meaning |
|---|---|
| Grokking | A model first memorizes its training set (training accuracy near 100%, test accuracy near chance), then much later suddenly generalizes to unseen examples. |
| Weight decay | A pull of every weight toward zero at each step. In AdamW it is "decoupled": each step multiplies a decayed weight by (1 − lr × wd). Strong weight decay is the standard ingredient that makes grokking happen. |
| Token table (embedding) | The matrix with one learned vector (row) per vocabulary token. ReLSM *ties* it to the output head: the same matrix turns tokens into vectors at the input and scores every token at the output. |
| Body | Everything between the table and the head: attention and feed-forward layers. |
| Circuit | The specific computation a trained network uses for a task, found by looking inside it. |
| Fourier features | The circuit found for modular addition (Nanda 2023): each number token's row encodes sines and cosines of a few "key frequencies", and the body combines them. |
| Grokfast | A gradient filter that amplifies slow-moving gradient components to speed grokking (Lee 2024). |
| Probe | A small test of a model's internals, e.g. a linear classifier that reads a property from hidden vectors, or swapping one activation and watching the output. |
| φ (phi) | In knowledge-composition work, the ratio of inferred facts to atomic (stored) facts in training. |
| Active forgetting | Resetting the token table every K steps during training so the body learns not to depend on any one table (Chen 2023). |
| Model stitching | Connecting part of one network to part of another through a small learned map. |
| Softmax collapse | A floating-point failure in which the softmax saturates and gradients vanish, stalling grokking (Prieto 2025). |
| Adam epsilon | The small constant Adam adds to its gradient normalization. A larger value makes tiny gradients produce tiny updates instead of full-size ones. |

## 2. What this repo does today (facts that shape the designs)

- **Optimizer.** `build_optimizer` ([train.py:1795–1848](../../train.py)) puts `tok_emb` and `lm_head` in the no-decay group. The head is tied to the table (`tie_weights=True`, [model.py:118](../../model.py), tied at [model.py:703–704](../../model.py)), so the 16.4M-row block (39.4% of nano) is never decayed. Issue [#186](https://github.com/jpturcotte/ReLSM/issues/186) traces this to PR #13 (2025-12-11). It was a hedge against "lexical collapse" of tokens unseen in a synthetic phase, written before the grokking setup existed (PR #45, Dec 17).
- **The model ends in a final RMSNorm, then the tied head** ([model.py:899–900](../../model.py)). Every training step therefore sends a softmax gradient to *every* row of the table, including tokens that never appear in the data. §4.1 explains why that matters.
- **Math is written as text with the Llemma tokenizer** (32,016 tokens; Llama-style tokenizers split numbers into single digits, per the LLaMA paper, [arXiv 2302.13971](https://arxiv.org/abs/2302.13971), Feb 2023). `mod_add` prompts look like `Compute (41 + 77) mod 97. Answer:18` ([data.py:344–386](../../data.py)). So ReLSM's math circuits read digit tokens and work out carries in the body. That differs from the one-token-per-number setup in the grokking papers, where much of the structure sits in the table itself.
- **Tasks are registered** in `AlgorithmicGenerator._get_generators` ([data.py:964–976](../../data.py)). `FixedAlgorithmicDataset` ([data.py:1230](../../data.py)) builds the fixed training set used with `--fixed_data`.
- **Model sizes** come from `create_model` presets ([model.py:1004](../../model.py)). The smallest is nano (d_model 512, 6 layers). Dropout defaults to 0.1.
- **Known blockers on the grokking path** (from the 2026-10-26 review):
  - [#136](https://github.com/jpturcotte/ReLSM/issues/136): `--fixed_data` crashes at the first train-eval.
  - [#137](https://github.com/jpturcotte/ReLSM/issues/137): Grokfast diverges with gradient accumulation ≥ 2.
  - [#145](https://github.com/jpturcotte/ReLSM/issues/145): `create_model` overrides raise TypeError.
  - [#143](https://github.com/jpturcotte/ReLSM/issues/143): eval reseeds the global RNGs.
  - [#135](https://github.com/jpturcotte/ReLSM/issues/135): the latent/act/ssm loops leak future tokens.
  - [#187](https://github.com/jpturcotte/ReLSM/issues/187): runs do not record the code version.
  - Also: no fixed-data path guarantees a test set disjoint from the training set.

**Consequence for the designs:** run the first experiments in a small standalone harness (proposed `tools/grok_lab.py`). It would build `BaselineTransformer(TransformerConfig(...))` directly and train full-batch on tensors already on the GPU, with its own small vocabulary and explicit train/test splits. That avoids every blocker above except #135. Once something groks, port the generator into `data.py` and the switch into `build_optimizer`.

## 3. Literature (dated, linked)

How each row was checked:
- **A**: abstract or arXiv page checked by a research agent.
- **F**: full text read by an agent.
- **S**: also spot-checked by me in this session (title, authors, date).
- **U**: seen only in a search snippet; do not lean on it.

Three agents ran the sweep. Their full notes, with more rows and per-claim markers, are outside the repo in the session notes folder (`lit-lexical-grokking.md`, `lit-puzzles-concepts.md`, `lit-embedding-table.md`).

### 3.1 Grokking beyond modular arithmetic: language, grammar, lexical data

| Date | Work | Link | What it shows for us | Check |
|---|---|---|---|---|
| 2022-10 (ICLR'23) | Liu et al., *Omnigrok* | [2210.01117](https://arxiv.org/abs/2210.01117) | IMDb sentiment groks, but "not as sharp" as algorithmic tasks. Setup: LSTM, 1k-word vocab, 1k training samples, init scale ×6, weight decay 1. Explains grokking by the weight norm: train loss is L-shaped and test loss U-shaped in norm. | F |
| 2023-05 (ACL'23) | Murty et al., *Grokking of Hierarchical Structure in Vanilla Transformers* | [2305.18741](https://arxiv.org/abs/2305.18741) | "Structural grokking" on Dyck, question formation and **tense inflection**. Out-of-distribution accuracy keeps rising long after in-distribution accuracy saturates (300–400k steps, hidden 512, lr 1e-4). Depth shows an inverted U. Early stopping caps generalization below 50% where long training reaches 80–90%. A tree-structuredness probe predicts grokking; weight norm does not. No weight-decay value reported. | F |
| 2023-09 (ICLR'24) | Chen (Angelica) et al., *Sudden Drops in the Loss* | [2309.07311](https://arxiv.org/abs/2309.07311) | Syntax-tracking attention heads appear abruptly, together with a loss drop. | A |
| 2024-01 | Zhu et al., *Critical Data Size of Language Models from a Grokking Perspective* | [2401.10463](https://arxiv.org/abs/2401.10463) | Grokking on IMDb/Yelp appears near a critical data size that grows with model size. Transitions are smoother on language. 1 layer, hidden 128, weight decay 1e-2; the jump came about 350k steps in. | F |
| 2024-09 (NAACL'25) | Lv et al., *Language Models "Grok" to Copy* | [2409.09281](https://arxiv.org/abs/2409.09281) | Copying (induction heads) develops in a grokking-like way; regularization speeds it up. | A |
| 2025-06 (ICLR'26) | Li (Ziyue) et al., *Grokking in LLM Pretraining?* | [2506.21551](https://arxiv.org/abs/2506.21551) | Different data groups grok at different times during OLMoE-7B pretraining. Proposes train-only proxies based on expert routing. Not consumer scale. | A |
| 2026-05 (AACL'26) | Muckatira et al., *A Pre-Training Analogue of Grokking in LMs: Tracing Delayed Grammatical Generalization* | [2606.00230](https://arxiv.org/abs/2606.00230) | Delayed generalization on 5 BLiMP grammar phenomena, split by whether the key phrase was seen in training. After generalization, grammar is concentrated in a few attention heads. The closest thing to "grokking grammar". | S |
| 2026-09 | Sabatelli, *The Dynamics of Quasiregular Neural Learning* | [2609.26018](https://arxiv.org/abs/2609.26018) | A rule-plus-exceptions analogue of the English past tense: overregularization, stronger when exceptions are rare. Not framed as grokking, but it is the regular/irregular morphology case. | A |
| — | Searched, **not found** | — | Grokking on SIGMORPHON-style inflection; on SCAN/COGS; on PCFGs other than Dyck; effects of tokenizer or vocabulary size. | — |

### 3.2 Single-task puzzles, knowledge composition, syntax

| Date | Work | Link | What it shows for us | Check |
|---|---|---|---|---|
| 2024-05 (NeurIPS'24) | Wang (Boshi) et al., *Grokked Transformers are Implicit Reasoners* | [2405.15071](https://arxiv.org/abs/2405.15071) | Composition over stored facts appears only through grokking, about 50× after training fits. Speed depends on **φ (inferred/atomic ratio)**, not raw data size. Out-of-distribution composition fails in a vanilla stack; parameter sharing helps. Probes: logit lens for the bridge entity, causal tracing. | F |
| 2025-04 (ICML'25) | Abramov et al., *Grokking in the Wild* | [2504.20752](https://arxiv.org/abs/2504.20752) | Raising φ with synthetic facts, even false ones, induces grokking on real multi-hop QA. | A |
| 2026-01 | He (Kaiyu) et al., *Is Grokking Worthwhile?* | [2601.09049](https://arxiv.org/abs/2601.09049) | Grokked circuits transfer poorly to new knowledge, and accuracy can decouple from circuit formation. Probe circuits, not just accuracy. | A |
| 2026-04 | Kohli et al., *Loop, Think, & Generalize* | [2604.07822](https://arxiv.org/abs/2604.07822) | Recurrent-depth transformers reach out-of-distribution composition through three-stage grokking, and extra loops extrapolate to more hops. Needed zero-init output projections. Relevant to ReLSM's latent/loop variants. | A |
| 2025-06 | Wang (Guan) et al., *Hierarchical Reasoning Model* (HRM) | [2506.21734](https://arxiv.org/abs/2506.21734) | 27M params, about 1K examples per task (Sudoku-Extreme, Maze-Hard, ARC). Deep supervision, 1-step gradient, Q-learning halting, heavy augmentation. Not framed as grokking. | A |
| 2025-08-15 | ARC Prize team, *HRM analysis* | [arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis) | The hierarchy adds little; the outer refinement loop and augmentation drive the result. A per-puzzle ID embedding makes HRM transductive (it looks puzzles up rather than generalizing). | A |
| 2025-10 | Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks* (TRM) | [2510.04871](https://arxiv.org/abs/2510.04871) | One 2-layer network, 5–7M params, recursed. 87.4% on Sudoku-Extreme. **Weight decay 1.0** (Sudoku/Maze), EMA 0.999 (79.9% without it), about 1K examples × 1000 augmentations. 4 layers overfit. Effectively a grokking recipe, though never called one. | A |
| 2025-12 → 2026-10 | HRM/TRM follow-ups (URM, Sapunov, Shakhvalieva, Neri) | [2512.14693](https://arxiv.org/abs/2512.14693) (U), [2604.21999](https://arxiv.org/abs/2604.21999), [2609.39967](https://arxiv.org/abs/2609.39967), [2610.26666](https://arxiv.org/abs/2610.26666) | Gains come from recurrence plus nonlinearity, not hierarchy. Truncated backprop through loops helps. On a single-GPU budget all three reach near-zero exact solves on ARC-1. | A |
| 2024-04 (TACL) | Ahuja et al., *Learning Syntax Without Planting Trees* | [2404.16367](https://arxiv.org/abs/2404.16367) | A full next-token loss yields hierarchical generalization where seq2seq or prefix-LM training often doesn't. | A |
| 2025-10 (ICML'26) | Schulz et al., *Unraveling Syntax* | [2510.02524](https://arxiv.org/abs/2510.02524) | Hierarchical and linear subnetworks coexist. Subgrammars are learned in parallel; the simplest grammar that fits the data wins. | A |

### 3.3 Concept space

| Date | Work | Link | What it shows for us | Check |
|---|---|---|---|---|
| 2023-10 (NeurIPS'23) | Okawa et al., *Compositional Abilities Emerge Multiplicatively* | [2310.09336](https://arxiv.org/abs/2310.09336) | Accuracy on a combination ≈ product of component accuracies. Rare combinations take much longer. | A |
| 2024-06 (NeurIPS'24) | Park, Okawa, Lubana, Tanaka et al., *Emergence of Hidden Capabilities: Exploring Learning Dynamics in Concept Space* | [2406.19370](https://arxiv.org/abs/2406.19370) | "Concept signal" (how much the data changes when a concept flips) sets learning order and speed. A capability exists inside the model about 25% of training before ordinary prompts reveal it; latent interventions reveal it early. The authors distinguish this from grokking. | A |
| 2024-12 (ICLR'25) | Park et al., *Competition Dynamics Shape Algorithmic Phases of ICL* | [2412.01003](https://arxiv.org/abs/2412.01003) | Algorithms compete, and generalization can be *transient*: memorization later overtakes it. Keep evaluating after a peak. | A |
| 2025-01 (ICLR'25) | Park et al., *ICLR: In-Context Learning of Representations* | [2501.00070](https://arxiv.org/abs/2501.00070) | Word representations reorganize suddenly to fit a structure defined in context. | A |

### 3.4 Grokking and the embedding layer

| Date | Work | Link | What it shows for us | Check |
|---|---|---|---|---|
| 2022-05 (NeurIPS'22) | Liu et al., *Towards Understanding Grokking: An Effective Theory of Representation Learning* | [2205.10343](https://arxiv.org/abs/2205.10343) | Generalization coincides with structured embeddings forming. | A |
| 2023-01 (ICLR'23) | Nanda et al., *Progress Measures for Grokking via Mechanistic Interpretability* | [2301.05217](https://arxiv.org/abs/2301.05217) | Three phases: memorize, form circuit, clean up. Fourier features in the table; probes are restricted/excluded loss and the Fourier spectrum. | A |
| 2023-09 | Varma et al., *Explaining Grokking Through Circuit Efficiency* | [2309.02390](https://arxiv.org/abs/2309.02390) | Memorizing and generalizing circuits compete; critical dataset size; "ungrokking". | A |
| 2024-02 (TMLR'24) | Furuta et al., *Interpretation of Grokked Transformers on Modular Polynomials* | [2402.16726](https://arxiv.org/abs/2402.16726) | Transfer between operations works only for some pairs. In the full text: retraining only the table and head of a grokked model ("PG-T") **failed for subtraction even on subtraction**. | S (abstract); F by agent |
| 2024-05 | Lee (Jaerin) et al., *Grokfast* | [2405.20233](https://arxiv.org/abs/2405.20233) | Amplifying slow gradients gives >50× faster grokking, including on IMDb. | A |
| 2024-06 (NeurIPS'24) | Zhou et al., *Pre-trained LLMs Use Fourier Features to Compute Addition* | [2406.03445](https://arxiv.org/abs/2406.03445) | Number-token rows carry the Fourier basis; plugging pretrained number embeddings into a fresh model rescues addition. | A |
| 2025-01 (ICLR'25) | Prieto et al., *Grokking at the Edge of Numerical Stability* | [2501.04697](https://arxiv.org/abs/2501.04697) | Softmax collapse blocks grokking. StableMax (a softmax replacement) and ⊥Grad allow grokking without weight decay. | A |
| 2025-04 (ICLR'25) | Xu et al., *Let Me Grok for You* (GrokTransfer) | [2504.13292](https://arxiv.org/abs/2504.13292) | Copying a weak model's table removes the grokking delay; the table carries much of the "grokkedness". | A |
| 2025-05 | AlQuabeh et al., *Mechanistic Insights into Grokking from the Embedding Layer* | [2505.15624](https://arxiv.org/abs/2505.15624) | MLPs *without* embeddings generalize almost at once; adding embeddings creates the delay. Rare tokens stall. Fixes: frequency-aware sampling, a separate learning rate for the table. | S |
| 2025-06 (ICML'25) | Prakash & Martin, *Grokking and Generalization Collapse* | [2506.04434](https://arxiv.org/abs/2506.04434) | "Anti-grokking": very long training after grokking can collapse test accuracy. | A |
| 2026-06 | Truong et al., *The Weight Norm Sets the Grokking Timescale* | [2606.13753](https://arxiv.org/abs/2606.13753) | Holding the weight norm above its natural transition value makes the delay grow exponentially. A final LayerNorm removes the dependence (ReLSM has a final RMSNorm). | S |
| 2026-07 | Ootani, *Grokking Is Conditional and Fragile* | [2607.05104](https://arxiv.org/abs/2607.05104) | Grokking depends on training-set coverage. Thread count or CPU vs GPU flips single-seed outcomes. Report grok rates over seeds. | A |
| 2026-08 | Ootani, *Small Enough to Know Everything* | [2609.20166](https://arxiv.org/abs/2609.20166) | Laws from tiny enumerable tasks held from 12K to 50M params, except the weight-decay response, which changes with scale. | S |
| 2026-05 | Verma, *Weight Decay Regimes in Grokking Transformers* | [2605.20441](https://arxiv.org/abs/2605.20441) | Three regimes: memorize, grok, collapse. Cheap diagnostics from attention-head similarity and entropy. | A |
| 2026-05 | Manir & Rupa, *A Systematic Empirical Study of Grokking* | [2603.25009](https://arxiv.org/abs/2603.25009) | Weight decay dominates; only a narrow band of values groks. | A |
| 2026-05 | Hidajat et al., *Grokking as Structural Inference* | [2605.15787](https://arxiv.org/abs/2605.15787) | Early MLP memorization starves attention of gradient; weight decay must erode memorization first. | A |
| 2026 (ICML'26 poster) | *The Geometric Origin of Grokking* (R2G loss) | [icml.cc poster 65146](https://icml.cc/virtual/2026/poster/65146) | Angular reorganization of token embeddings drives grokking. A repulsion loss speeds it up, reportedly also on linguistic tasks. | U (no arXiv ID found) |
| 2026-09 | Jia, *Beyond Embedding Transfer: Component Roles in Grokking Transfer and Stability* | [2609.18078](https://arxiv.org/abs/2609.18078) | Transferring donor embeddings matters most. Freezing the transferred table and readout cut relapse after grokking from 19.40% to 0.07%. Single-author preprint. | S |

### 3.5 Changing a table under a trained body

| Date | Work | Link | What it shows for us | Check |
|---|---|---|---|---|
| 2023-07 (NeurIPS'23) | Chen (Yihong) et al., *Improving Language Plasticity via Pretraining with Active Forgetting* | [2307.01163](https://arxiv.org/abs/2307.01163) | Resetting embeddings (with their optimizer state) every 1000 steps cost about 1 point in English. With the body frozen, relearning for a new language reached 57.8 vs 37.2 XNLI after 5K steps. Ordinary bodies relearn slowly and need a lot of data. | F |
| 2024-10 | Aggarwal et al., active forgetting for decoder LLMs | [2410.16168](https://arxiv.org/abs/2410.16168) | Same idea in decoder-only models: better adaptation to unseen languages. | A |
| 2024 (Nature 632) | Dohare et al., *Loss of Plasticity in Deep Continual Learning* | [nature.com/articles/s41586-024-07711-7](https://www.nature.com/articles/s41586-024-07711-7) | Continual backprop reinitializes low-utility units to keep networks able to learn. | A |
| 2019-10 (ACL'20) | Artetxe et al., *On the Cross-lingual Transferability of Monolingual Representations* | [1910.11856](https://arxiv.org/abs/1910.11856) | A new table for a new language under a frozen body is competitive with multilingual BERT. | A |
| 2020-12 (Findings'21) | de Vries & Nissim, *As Good as New* | [2012.05628](https://arxiv.org/abs/2012.05628) | Relearning only GPT-2's lexical layers worked for Italian; Dutch needed full fine-tuning. | A |
| 2021-03 (AAAI'22) | Lu et al., *Pretrained Transformers as Universal Computation Engines* | [2103.05247](https://arxiv.org/abs/2103.05247) | A frozen GPT-2 body with new input/output layers solves numeric, vision and protein tasks. | A |
| 2021-12 → 2025 | Vocabulary transplant: WECHSEL, FOCUS, Zero-Shot Tokenizer Transfer, Trans-tokenization, OMP | [2112.06598](https://arxiv.org/abs/2112.06598), [2305.14481](https://arxiv.org/abs/2305.14481), [2405.07883](https://arxiv.org/abs/2405.07883), [2408.04303](https://arxiv.org/abs/2408.04303), [2506.06607](https://arxiv.org/abs/2506.06607) | Every working method builds new rows from **shared anchor tokens**, then trains further. None works by swapping in an unrelated table. | A |
| 2015 → 2025 | Model stitching: Lenc & Vedaldi; Bansal et al.; Csiszárik et al.; Moschella et al.; Chen & Merullo et al. | [1411.5908](https://arxiv.org/abs/1411.5908), [2106.07682](https://arxiv.org/abs/2106.07682), [2110.14633](https://arxiv.org/abs/2110.14633), [2209.15430](https://arxiv.org/abs/2209.15430), [2506.06609](https://arxiv.org/abs/2506.06609) | Same-architecture networks usually stitch with one affine map at intermediate layers. Language-model residual streams stitch too. | A |
| 2016-08 (EACL'17) | Press & Wolf, *Using the Output Embedding to Improve LMs* | [1608.05859](https://arxiv.org/abs/1608.05859) | A tied table behaves like an output embedding: every row is updated at every step through the softmax. | A |
| 2026-10-07 | Han et al., *A Deafening Silence: Catastrophic Forgetting Lives in the Output Embeddings of Tokens the Data Never Speaks* | [2610.09835](https://arxiv.org/abs/2610.09835) | Forgetting concentrates in the output rows of tokens missing from new data. The cause is one-sided softmax gradients made full-size by Adam's normalization. A larger Adam epsilon on the output projection removes 39–68% of it. | S |
| 2026-10 | Bathula, frozen binary table in a small LM | [2610.22428](https://arxiv.org/abs/2610.22428) | Adding 10k words cost about 3% with a frozen binary table vs 43% with a learned one. | A |

## 4. Can we grok math first and then change the table?

### 4.1 First, a finding: the exemption does not protect unseen tokens

The exemption assumes that rows of tokens absent from the math data receive no gradient, so only weight decay could move them. With a tied head that assumption is false. Every step, the softmax pushes down the score of every token that is not the target. That gradient reaches every row. Adam then rescales each coordinate by its own recent gradient size, so even a tiny, consistent push turns into a full-size step.

Probe ([unseen_token_rows_probe.py](unseen_token_rows_probe.py), CPU, about one minute, toy scale). Setup:
- `BaselineTransformer` with d_model 64, 2 layers, a 1,000-token vocabulary.
- The repo's own `build_optimizer` (lr 1e-3, wd 1.0).
- 2,000 updates on data that uses only tokens 0–49.

We measured rows 50–999. At init they have norm about 0.16 and mean pairwise cosine about 0.

| Arm | Unseen rows: mean norm | Unseen rows: mean pairwise cosine |
|---|---|---|
| Tied, table exempt (repo default) | 4.216 | **0.999** |
| Tied, table decayed (wd 1.0) | 2.980 | 1.000 |
| Tied, exempt, Adam eps 1e-4 on the table | 0.721 | 0.953 |
| Tied, exempt, Adam eps 1e-2 on the table | 0.176 | 0.196 |
| Untied, both exempt: input rows | 0.160 | 0.000 |
| Untied, both exempt: output rows | 3.599 | 0.998 |

Reading:
- Under the current setting, every unseen token ends up as nearly the same vector, about 26× its initial size, pointing away from the hidden states the model produces. That is the lexical collapse the exemption was meant to prevent, and decay does not cause it.
- Untying protects the input rows but not the output rows.
- A large Adam epsilon on the table mostly prevents it, as Han et al. 2026 report for output projections.
- Caveats:
  - Toy scale; not repeated at nano scale or with a later language phase.
  - eps 1e-2 also slows learning of the rows that *are* used; that cost was not measured.

**Implications for #186.**
- Adding `--decay_embeddings` is still right for grokking runs.
- The phase-switch problem has a better fix than the exemption: at the switch to language, **reinitialize the rows the math phase never used** and reset their Adam state. A table-only Adam epsilon is a second option.
- The ADR the review asked for should record this.

### 4.2 Direct answer

**It depends on which rows you change.** In ReLSM the math circuit reads digit and operator tokens plus the word pieces of the prompt templates, a small fraction of the 32,016 rows. With the tied head, the same rows also produce the answer.

| Intervention | Does the grokked math circuit survive? | Why | Cheap test |
|---|---|---|---|
| **Reset only rows math never used**, at the phase switch | **Yes.** Math inputs are untouched. Fresh rows give small output scores, so the answer is unaffected. | Those rows hold no math information (and per §4.1 they are collapsed junk anyway). This is active forgetting applied once to the rows that need it. Precedent: §4.3. | E2 arms B5 and B6: grok, reset unused rows, check math accuracy is unchanged to the last example. |
| **Reinitialize the whole table, relearn it with the body frozen** | **Not guaranteed.** Likely to recover *if math data is in the relearning mix*. With language-only data it does not come back. | A solution exists (the old table, or any relabeling of it), and the body gives a structured gradient. Frozen-body relearning works in NLP (Artetxe, de Vries, Lu). But the tied table sits at both ends, which makes the problem harder (AlQuabeh). The closest prior test, Furuta's PG-T, failed for subtraction even on the same task. Chen: ordinary bodies relearn slowly. | E2 arm B1: steps to 99% test accuracy after reinit with the body frozen, vs from scratch; do the old key frequencies come back? |
| **Swap in a table from another model or vocabulary** | **No, not zero-shot.** | An unrelated table uses an unrelated basis. All transplant methods rely on shared anchor tokens plus further training. For math the anchors are the ten digits: too few to fit a 512-dimensional map without data. If the swap changes how numbers are split into tokens, the task itself changes and the body must re-grok. | E2 arm B4: map seed B's table into seed A's body by least squares on anchor tokens; zero-shot accuracy vs number of anchors. |
| **Periodic resets during the math phase** (active forgetting) | **Probably delays or blocks grokking.** | Each reset wipes the circuit's inputs and, through the tie, its outputs. Meanwhile wd 1.0 keeps shrinking the body. The body is pushed to compute everything from arbitrary codes, a costlier circuit. The Grokfast EMA must be reset with the table, or it pushes the new rows back along the old direction. Resets fit better in a language phase, or on non-math rows only. | E2 arm B3: reset every 200, 1000 or 5000 steps vs never; time to grok, and recovery speed after a final reset. |
| **Stitch the grokked body to a new table and head** | **Only if the stitch can express the map.** | For Fourier-style circuits, different seeds pick different key frequencies. A linear map cannot create frequencies the source table lacks (an argument derived here, not from a paper). Tying forces one map to serve both ends; untying relaxes that. For a new language vocabulary, a stitch reduces to the swap or relearn cases. | E2 arm B4: linear and MLP stitches between seeds; accuracy vs overlap of key frequencies. |

### 4.3 Resetting only dead or collapsed rows: precedent

The selective reset recommended above has precedent in three lines of work. One 2026 paper describes nearly exactly ReLSM's situation.

| Date | Work | Link | What it shows | Check |
|---|---|---|---|---|
| 2019-07 (ICLR'19) | Gao et al., *Representation Degeneration Problem in Training NLG Models* | [1907.12009](https://arxiv.org/abs/1907.12009) | With tied input/output tables, most word vectors are squeezed into a narrow region, driven by the softmax on the output side. Proposes a regularizer. | S |
| 2021-09 (ACL'22) | Yu et al., *Rare Tokens Degenerate All Tokens* (AGG) | [2109.03127](https://arxiv.org/abs/2109.03127) | A specific part of the gradient on rare tokens' rows causes the collapse and degrades all tokens. Adaptive gradient gating blocks that part. It prevents collapse rather than cleaning it up. | S |
| 2024-05 (EMNLP'24) | Land & Bartolo, *Fishing for Magikarp* | [2405.05417](https://arxiv.org/abs/2405.05417) | Under-trained ("glitch") tokens are common in production LLMs and can be detected from the weights. Per a search summary (full text not opened), detection uses closeness to the mean row of known-unused tokens. | S (abstract) |
| 2026-04 | Lazić et al., *To See the Unseen: on the Generalization Ability of Transformers in Symbolic Reasoning* | [2604.21632](https://arxiv.org/abs/2604.21632) | **Closest match.** Unseen tokens' unembeddings "collapse to nearly the same vector during training", worse when the table is shared. Freezing or resetting the (un)embeddings is part of their fix, and they argue the collapse explains why active forgetting works. Gemma 3's 99 reserved unused tokens have correlated rows, a poor starting point for fine-tuning. | S |
| 2020-04 | Dhariwal et al., *Jukebox* | [2005.00341](https://arxiv.org/abs/2005.00341) | **The usage-triggered recipe, for a VQ codebook** (a learned lookup table). When an entry's mean usage drops below a threshold, it is reset to a random encoder output from the current batch ("random restarts"). | Body text via search snippet |
| 2020-05 (IJCNN'20) | Łańcucki et al., *Robust Training of Vector Quantized Bottleneck Models* | [2005.08520](https://arxiv.org/abs/2005.08520) | Periodic data-dependent re-initialization of codewords increases how many are used. | S |
| 2023-02 (ICML'23) | Sokar et al., *The Dormant Neuron Phenomenon in Deep RL* (ReDo) | [2302.12902](https://arxiv.org/abs/2302.12902) | Recycles neurons that have gone inactive during training. Same idea, applied to units instead of rows. | S |
| 2024 (Nature) | Dohare et al., continual backprop | [nature.com](https://www.nature.com/articles/s41586-024-07711-7) | Re-initializes low-utility units to keep a network able to learn. | A |

What we did not find: a standard recipe for *usage-triggered* resets of dead token rows during language-model pretraining, i.e. the codebook trick applied to a token table. Lazić 2026 (reset or freeze) and Chen 2023 (reset the whole table on a timer) come closest. ReLSM could make a small original contribution here.

**Recipe for ReLSM.**
- **Which rows.**
  - Rows whose token has not appeared as input or target in the last N steps (a usage counter), or
  - rows whose cosine similarity to the centroid of collapsed rows is above about 0.9.
  - Never the rows the math phase uses.
- **Reset to what.** Two options:
  - small random noise at the init scale (std 0.02), or
  - the mean of the *used* rows plus noise (data-dependent, like the codebook restarts).
  
  Not the collapsed vector itself: it points away from the hidden states the model produces, which is why Lazić found Gemma's unused rows a poor starting point.
- **Optimizer state.** Clear the Adam moments (and the Grokfast EMA) for the reset rows. Otherwise the stale momentum pushes them straight back.
- **When.**
  - Mainly once, just before those tokens start being used (the switch to language).
  - Periodic resets during a math-only phase are mostly wasted: in §4.1's probe the rows re-collapsed within 2,000 updates.
  - To stop re-collapse between resets, pair the reset with a table-only Adam epsilon (Han 2026) or gradient gating (Yu 2022).

### 4.4 One more risk, whatever is done to the table

In any phase without math data, weight decay 1.0 keeps eroding the body's math circuit. Keep a math replay fraction (`--persistent_alg_frac` already exists, default 0.15). Freezing the math-token rows helped a lot in Jia 2026.

## 5. Experiment designs (ranked)

All four use the proposed standalone harness (§2). Shared rules:
- Full-batch AdamW, gradient accumulation 1.
- Dropout 0.
- fp32 logits and loss.
- A test set disjoint from training, built from the same rule.
- At least 5 seeds; report the fraction of seeds that grok.
- Log every few hundred steps.
- Never stop early.

**Evidence of grokking, for every design:**
1. **Gap timeline.** T_mem = the step where training accuracy first passes 99%. T_grok = the step where test accuracy first passes 95%. Grokking means T_grok ≫ T_mem, with test accuracy flat in between.
2. **Weight norm.** The norm should shrink before and around T_grok. Log the table and the body separately (`weight_norm.embedding` already exists), so a falling total is not hiding a growing table.
3. **A mechanistic probe** (named per design). It should switch on around T_grok and should *cause* the output, not just correlate with it.

Cost figures are estimates for one consumer GPU (RTX 3060–4090 class), not measurements.

### E1 (rank 1). Paradigm grokking: a synthetic inflection language

Why first: it is the most direct test of lexical structure, it is cheap, and its decay arms settle the #186 default.

- **Generator.**
  - A lexicon of 256 nonce stems, each a single token `S0…S255`.
  - Each stem has a hidden inflection class (4 classes).
  - 8 grammatical cells (2 numbers × 4 cases).
  - The answer is one suffix token, chosen from a class × cell table of 32 entries with some suffixes shared between cells, the way real paradigms reuse forms.
  - Example: `<bos> S17 PL.GEN = -ok <eos>`, with loss on the answer only. Also run one arm with a full next-token loss (Ahuja 2024).
  - About 5% of stems are irregular (fixed random suffixes). That tests whether the rule generalizes while exceptions stay memorized (Sabatelli 2026).
  - **E1a (lexical class).** The class is visible only through a stem's other forms. To fill a held-out cell, the model must store the class *in the stem's table row*. This is the lexical analogue of φ in Wang 2024.
  - **E1b (form-based class).** Stems are two syllable tokens, and the final syllable decides the class. That rule lives in the body and should also generalize to brand-new stems.
- **Splits.**
  - 2,048 cells in total. Training fraction 0.3 / 0.5 / 0.7.
  - Every stem keeps at least 2 training cells, and every cell type appears in training (Ootani 2026: coverage decides grokking).
  - Test sets: held-out cells (E1a and E1b) and unseen stems (E1b only).
- **Vocabulary:** about 300 tokens. **Model:** d_model 128, 2 layers, 4 heads, SwiGLU 512. The body is about 0.52M parameters; the tied table is about 38k.
- **Which parameters decay (the main comparison):**
  - (i) wd 1.0 on everything, table included.
  - (ii) The repo's current rule (table exempt).
  - (iii) wd 0 with StableMax, as a control.
  - (iv) Arm (i) plus Grokfast, accumulation 1.
  - Optionally (v): a table-only learning rate ×0.1 (AlQuabeh).
- **Mechanistic probe.**
  - A cross-validated linear classifier from each stem's table row to its hidden class, tracked over training. It should jump near T_grok in E1a.
  - Causal check: paste a stem's row into a stem of another class; the predicted suffix should follow the pasted class.
  - For E1b: patch the final-syllable position.
- **Prediction to falsify.** In E1a, arm (ii) groks later or not at all compared with (i), because nothing compresses the table. If (ii) groks as well as (i), the exemption is harmless for lexical grokking.
- **Cost.**
  - About 5–10 minutes per run for 10⁵ full-batch steps.
  - Main grid: 4 arms × 3 fractions × 5 seeds = 60 runs, about 5–10 GPU-hours. E1b adds about half of that.

### E2 (rank 2). Grok math, then change the table

Why: it answers §4 with measurements and doubles as the harness's **positive control**. Run arm A first: if the harness cannot reproduce Nanda-style modular-addition grokking, nothing else it reports can be trusted.

- **Generator.** `a + b = c (mod 97)`, one token per residue: the published setup, where the structure sits in the table. 9,409 pairs; train on 30–50%.
  - **E2-text variant:** the same task in ReLSM's text format (`Compute (a + b) mod 97. Answer:` with digit tokens and a word-level vocabulary). This variant tells us which result carries over to ReLSM, where the structure sits mostly in the body.
- **Vocabulary:** about 100 tokens (about 140 in the text variant). Add 400 "language" tokens that appear only in phase B. **Model:** 1–2 layers, d_model 128, tied head.
- **Phase A:** grok under wd 1.0 on everything; a second arm keeps the table exempt.
- **Phase B arms:**
  - B0: permute token IDs consistently in both the data and the table. Accuracy must not change; this checks the harness.
  - B1: reinitialize the table, freeze the body, relearn, with Adam and Grokfast state reset. Compare steps to 99% test accuracy against from scratch, at 100% and 30% data. Repeat with the head untied.
  - B2 ("new language" for the same task): add 97 new residue tokens and teach them on a mix of old and new examples with the body frozen. Do the new rows line up with the old ones (cosine or Procrustes fit)? This is a miniature cross-lingual transfer.
  - B3: resets every 200, 1000 or 5000 steps during phase A.
  - B4: put seed B's table into seed A's body through a least-squares map on 5/10/20/all anchor tokens and through a small MLP. Accuracy vs overlap of key frequencies.
  - B5 (the recommended ReLSM recipe): after grokking, reinitialize only the never-used rows, then run a language phase with and without 15% math replay. Track math accuracy and the math-row drift.
  - B6 (selective reset, how and when; precedent in §4.3): compare ways of resetting only the dead rows. Each variant also resets the rows' Adam and Grokfast state.
    - **Trigger:** (a) once at the phase switch; (b) usage-triggered every 1,000 steps for rows unused in the last N steps (the codebook "random restart" rule); (c) collapse-triggered, for rows with cosine > 0.9 to the collapsed centroid.
    - **Reset target:** small random noise vs the mean of the used rows plus noise.
    - **Anti-re-collapse:** none vs table-only Adam eps 1e-2 vs gradient gating.
    - **Measure:** math accuracy (must not move); unused-row norm and mean pairwise cosine over time (the §4.1 metrics); how fast the reset rows learn once their tokens appear in the language phase (loss on those tokens over the first 2,000 steps).
    - **Prediction:** (a) plus an anti-re-collapse measure matches (b) and (c) at lower cost, and data-dependent init learns fastest.
- **Evidence:** test accuracy over time; the table's Fourier spectrum (key frequencies and their share of the norm); restricted and excluded loss (Nanda 2023); weight norm of table and body.
- **Cost.** About 10⁴–4×10⁴ full-batch steps per run (an estimate based on Nanda's p=113 setup), a few minutes each. The full matrix with 5 seeds is about 3–5 GPU-hours, plus about 2–3 for B6 (3 triggers × 2 targets × 3 anti-re-collapse settings, 3 seeds).

### E3 (rank 3). Lexical composition: two-hop word relations

Why: it tests whether ReLSM's loop and latent variants help with the out-of-distribution composition that vanilla stacks fail (Wang 2024, Kohli 2026). That is ReLSM's own research question, transposed to words.

- **Generator.**
  - 1,000 word tokens and 20 lexical relations (plural-of, past-of, antonym-of, translation-into-L2, …). Each word has about 20 atomic facts `w r → w'`.
  - Training includes two-hop inferred facts `w r1 r2 → w''` at φ ∈ {3.6, 7.2, 12}.
  - About 5% of words appear *only* in atomic facts; that is the out-of-distribution test.
  - Answers are single tokens.
- **Vocabulary:** about 1,050 tokens. **Model:** 4 layers × d_model 256, about 3.1M body parameters. Compare with `shared_loop` (2 unique layers looped), or with `latent` only after #135 is fixed.
- **Decay:** wd 0.1 and 1.0, table included; φ is the main knob.
- **Probes:** logit lens for the intermediate (bridge) word at each layer or loop; causal tracing by swapping a word for another of the same type (Wang 2024).
- **Cost.** About 10⁵–3×10⁵ steps, about 0.5–2 hours per run. A grid of 3 φ × 2 architectures × 3 seeds is about 1 GPU-day.

### E4 (rank 4). Agreement grammar: structural grokking on a small vocabulary

Why: it is the closest analogue to real syntax, and it has a published baseline to reproduce (Murty 2023). It ranks last because it is the most expensive.

- **Generator.**
  - A small grammar with about 80 words in 6 categories, center-embedded relative clauses, and subject–verb agreement (Murty's tense-inflection task).
  - Training data is *ambiguous*: a linear rule (agree with the nearest noun) and a hierarchical rule (agree with the subject) both fit it.
  - Test on sentences where the two rules disagree.
  - Full next-token loss.
- **Vocabulary:** about 100 tokens. **Model:** d_model 256–512, depth 2/4/6 (Murty: inverted U).
- **Decay:** Murty gives no value; sweep wd 0.1 and 1.0, table included.
- **Probes:** tree-projection tree-structuredness (Murty); a linear probe for "subject number" at the verb position; patching the subject token.
- **Cost.** About 300k steps per run, roughly 3–8 hours. 3 depths × 3 seeds is about 1–3 GPU-days.

**Order of work.**
1. Build the harness.
2. Run E2 phase A as the positive control.
3. Run E1 and the rest of E2.
4. Decide the #186 default from E1's arms.
5. Then E3 or E4.

## 6. Open risks

1. **The lexical jump may be gradual rather than sharp.** Omnigrok and Zhu both saw smoother transitions on language. Judge success by the gap timeline and the probe, not by how steep the curve is.
2. **Grokking is fragile and seed-dependent.** Thread count and CPU vs GPU flip single-seed outcomes (Ootani 2026). Use at least 5 seeds, report grok rates, fix determinism flags, and record the code version (#187).
3. **The weight-decay band that groks is narrow** (Manir 2026, Verma 2026), and it shifts with scale (Ootani 2026). Sweep it; don't assume 1.0.
4. **Softmax collapse under fp16/bf16** (Prieto 2025). Keep logits and loss in fp32; StableMax is the fallback.
5. **Generalization can be transient**: anti-grokking (Prakash 2025), competing algorithms (Park 2024). Keep evaluating long after the peak.
6. **Accuracy can decouple from circuits** (He 2026). Every claim needs its causal probe.
7. **Repo blockers.** If experiments go through `train.py` before the fixes land, results are invalid: #136, #137, #145, #143, and #135 for the loop variants. The standalone harness sidesteps all but #135.
8. **A synthetic lexicon is not English.** Even if E1–E4 succeed, transfer to a real language phase with the 32k tokenizer is unproven. E2-B5 is the bridge experiment.
9. **The §4.1 probe is toy scale.** It used 2,000 updates, d_model 64, no later language phase. Repeat it at nano scale before writing the ADR. The cost of a large table-only Adam epsilon was not measured.
10. **Literature limits.**
    - Several 2026 sources are single-author preprints.
    - Most rows were checked only at the abstract.
    - The R2G poster has no arXiv ID.
    - The ARC/HRM details partly come from blog posts.
    - The PG-T detail comes from an agent's reading of Furuta's full text; I confirmed only the abstract.
    - Nothing here was reproduced.
11. **Cost figures are estimates, not timings.**
