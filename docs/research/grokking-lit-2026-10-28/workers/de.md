# Grokking literature sweep: German-language lane (de)

Sweep date: 2026-10-28. Lane: German-language sources, plus English theses and papers from German-speaking institutions that are not otherwise visible. These are marked as pointers.

**Bottom line.** The German-language web adds very little primary evidence on grokking. Of the 24 sources opened, 7 bear on the topics:

- One student replication with full method: an ETH Zürich bachelor's thesis, written in English.
- One LMU München conference paper with a grokking case study, in English, used as a pointer.
- One German-translated glossary page with runnable code. This lane re-ran its code locally.
- Two the-decoder.de press reports.
- One German machine translation of an English abstract.

Everything else is a summary, a lexicon entry, or a thesis *offer*. I found no German-language Bachelor's, Master's or doctoral thesis with grokking results. I found no grokking coverage in heise, golem, t3n, c't or iX, and nothing in German on Grokfast, StableMax/⊥Grad, bf16/fp16, or weight-decay schedules.

## Takeaways

The five most decision-relevant points for ReLSM's first grokking run (issue #249):

1. **Expect training-loss spikes, and do not treat them as divergence.** This comes from the ETH replication of the original `openai/grok` code (de-07, grade A).
   - On a log scale, training loss repeatedly jumps by more than 10⁶× out of near-zero plateaus. Every grokking event the author saw was tied to such a spike.
   - Spikes kept occurring after validation accuracy was already perfect. The model never jumped to a minimum that generalized worse ("There is no un-grokking").
   - **Caveat:** that run used weight decay 0.01 (about 100× smaller than ReLSM's planned 1.0) and needed up to 2·10⁶ steps. The slingshot-style spike regime is the low-WD regime.
   - **Practical rule:** log training loss on a log axis, and turn off any auto-rollback or "loss went up" guard. Spikes are diagnostic, not fatal.
2. **The cleanup phase acts mainly on the embedding and unembedding.** This is the LMU ExPLAIND case study (de-17/18, grade B, English pointer).
   - It decomposes a grokking Transformer's training into kernel and regularization influence. Regularization "acts primarily on the embedding and decoder layers".
   - The final phase is described as the input embeddings and the final layer aligning around a representation pipeline learned after memorization.
   - **Inference (not tested by the paper):** fix #186 (token embedding exempt from weight decay) and #218 (31k unused embedding rows dominating the weight norm) *before* the first run. In that case, weight decay's main grokking work would land on the very matrix ReLSM currently exempts.
3. **A one-minute CPU smoke test that this lane reproduced.** The German Ultralytics glossary (de-20) ships a full-batch recipe:
   - Recipe: MLP 194→256→97, (a+b) mod 97, 30% train split, AdamW lr 1e-3, weight decay 1.0, 20k steps.
   - Reported validation accuracy: 0.16 / 0.54 / 0.84 / 0.97 at 8k / 12k / 16k / 20k steps, and 0.00 throughout with wd=0.
   - I re-ran it verbatim on torch 2.8.0+cpu and got 0.18 / 0.63 / 0.86 / 0.98, and 0.00 throughout with wd=0. Training accuracy was 1.00 from step 4k in both runs.
   - Use it as a pipeline check (split, eval, WD wiring) before the d=128 Transformer run.
   - For the Transformer, ExPLAIND (de-18) is a useful reference: full-batch AdamW, lr 1e-3, weight decay 4.0, β=(0.98, 0.99), d_model 64, 1 layer, mod 113. Test accuracy reached 100% by about step 1,939. So 10⁵ updates at WD 1.0 is probably generous, not tight.
4. **Commutative duplicates muddy the curve.** Decide whether to deduplicate.
   - The ETH thesis (de-07, §4.5) found that for symmetric operations (addition, multiplication) validation accuracy starts well above chance. The author attributes this to commuted duplicates: the model can ignore position embeddings and get (b,a) from (a,b).
   - ExPLAIND enforces a ≥ b, drawing 4,000 train and 2,000 test pairs from 6,328 unordered pairs.
   - For a clean "chance → 100%" grokking signature, pick one convention explicitly. Either split on unordered pairs, or accept above-chance early validation and track validation *loss*.
   - The thesis also confirms the data-fraction phase picture: at a fraction where fast generalization occurs, grokking occurs at a smaller fraction of the same dataset. At a 60% split, several training-loss curves oscillated instead of spiking, which the author could not explain.
5. **Do not expect grokking from a mixed or pretraining-style data stream at small scale.** This comes from the-decoder.de's report on Huang et al., arXiv 2505.29548 (de-08, grade D, press summary).
   - Modular addition was injected into OLMo pretraining (4M–4B parameters, up to 210B tokens). Only the larger models grokked it, and only when the task appeared often enough.
   - A 20M model fell into an "update-and-forget" loop.
   - This supports keeping the first ReLSM run single-task, finite-set and full-batch, as #249 plans.

## Sources

### Evidence-bearing sources

#### de-07: ETH Zürich bachelor's thesis (Arn, 2023)

- **Identity:** "Investigating 'Grokking' – Late Generalisation in Deep Learning" (in English; same meaning in German: "Untersuchung von Grokking – späte Generalisierung im Deep Learning").
  - Author: Benjamin Arn. Supervisors: Peter Belcák and Prof. Dr. Roger Wattenhofer, Distributed Computing Group, ETH Zürich.
  - Bachelor's thesis dated 2023-03-22.
  - URL: https://pub.tik.ee.ethz.ch/students/2022-HS/BA-2022-45.pdf
- **Type:** thesis. It is an **English-language thesis at a German-speaking university** that is not otherwise visible: it has no arXiv version, and its code repo `gitlab.ethz.ch/disco-students/hs22/investigating-grokking` is behind an ETH login.
- **Setup:** a replication of Power et al. using the authors' `openai/grok` code, essentially unmodified.
  - Model: 2-layer Transformer (PyTorch Lightning), custom AdamW with **weight_decay = 0.01**.
  - Training length: 2·10⁶ optimization steps, double the original 10⁶.
  - Operations: x+y, x−y, x·y, x/y, x²+y² mod 97 (9,409 equations), plus 3 abstract groups.
  - x³+xy mod 97 and a mixed even/odd operation never generalized.
- **Findings:**
  1. Grokking reproduced, but rarely as cleanly as in the paper. Jumps varied in size and location, sometimes came in two steps, and were sometimes preceded by a slow rise.
  2. Training loss falls to near zero, then spikes by a factor of more than 10⁶. Each spike is a chance to move toward a generalizing minimum, but several spikes can pass without generalization.
  3. Spikes occur on nearly every dataset at nearly every train fraction, including ungrokkable operations and already-grokked models.
  4. "There is no un-grokking": the model never jumped to a worse-generalizing minimum within 2·10⁶ steps.
  5. Timing has no clear upper bound. Some runs grokked only after doubling the step budget, and the grokking-inducing instability rose rather than fell over time.
  6. Train-loss curves oscillated at a 60% split (unexplained).
  7. Symmetric operations gave high early validation accuracy. The author attributes this to commuted duplicates, with position embeddings ignorable.
  8. Practical notes (Appendix A): unpinned dependency versions in `openai/grok` (no reply on GitHub); the `datadir` argument is unused; only `visualize_metrics.py` worked; use the highest `version_N` metrics.csv.
- **Quotes** (the source is English, so these are English):
  - "There is no un-grokking." (§4.4, p. 11)
  - "The most crucial parameter is 'weight_decay' for the AdamW optimizer, which I set to 0.01" (Appendix A, pp. A-1–A-2)
- **Grade:** A (thesis with full method; single seed per configuration, no error bars).
- **Novelty:** independent replication with a critique. It is new relative to the English canon in tying every observed grok to Adam/slingshot loss spikes in the original codebase, and in the "no un-grokking up to 2·10⁶ steps" observation.

#### de-17 / de-18: LMU München, ExPLAIND (ICML 2026)

- **Identity:** "ExPLAIND: Unifying Model, Data, and Training Attribution to Study Model Behavior" (in English; German translation: "ExPLAIND: Vereinheitlichung von Modell-, Daten- und Trainingsattribution").
  - Authors: Florian Eichin, Yupei Du, Philipp Mondorf, Maria Matveev, Barbara Plank, Michael A. Hedderich (LMU München / MCML).
  - ICML 2026 poster, listed in LMU's Open Access repository (record dated 2025). arXiv 2505.20076, v4 dated 2026-06-11.
  - URLs: https://epub.ub.uni-muenchen.de/128628/ (LMU record) and https://arxiv.org/html/2505.20076v4
- **Type:** conference paper. It is an English paper by German-university authors, so **pointer only**.
- **Setup (App. E.2):**
  - Task: a+b mod 113 with a ≥ b; 4,000 train and 2,000 test pairs out of 6,328.
  - Model: 1-layer Transformer, 115-token embedding, no positional encoding, 4 heads × 16 dimensions, MLP width 512, d_model 64.
  - Optimizer: **full batch**, AdamW, fixed lr 1e-3, **weight decay 4.0**, β₁ = 0.98, β₂ = 0.99.
- **Findings:**
  - Training accuracy reaches about 100% around step 800. Test accuracy reaches 100% at step 1,939.
  - Kernel and regularization influences grow together from memorization, peak near step 1,700, then regularization dominates.
  - The decoder (unembedding) is the most influential component for memorization.
  - Regularization acts mainly on the embedding and decoder layers. Attention and MLP get little regularization influence.
  - Cyclic (mod-equivalence) structure appears in the influence kernel around steps 450–500 and shifts toward lower frequencies over training.
- **Quote (English):** "regularization acts primarily on the embedding and decoder layers" (arXiv v4, §5.1).
- **Grade:** B (peer-reviewed conference paper, but a single-run case study).
- **Novelty:** new mechanistic refinement of Nanda's cleanup phase, as embedding and unembedding alignment. It bears directly on ReLSM #186 and #218.

#### de-20: Ultralytics glossary entry "Grokking" (German edition)

- **Identity:** "Grokking: Wenn Netzwerke plötzlich verallgemeinern" (German; English: "Grokking: when networks suddenly generalize").
  - Org: Ultralytics, German edition of its ML glossary. Undated.
  - URL: https://www.ultralytics.com/de/glossary/grokking
- **Type:** blog/glossary with runnable code. It is a translation of the English glossary page.
- **Findings:**
  - Recipe: MLP (one-hot a, b → 256 ReLU → 97), p = 97, 30% of 9,409 pairs for training, full batch, AdamW lr 1e-3, weight_decay 1.0, 20,000 steps.
  - Reported validation accuracy: 0.00 (4k), 0.16 (8k), 0.54 (12k), 0.84 (16k), 0.97 (20k). With wd = 0, validation accuracy stays at 0.00.
  - Also summarizes Omnigrok (weight norm) and Prieto et al. 2025 (Softmax collapse; StableMax and ⊥Grad give grokking without weight decay).
  - **Local re-run by this lane** (code verbatim, torch 2.8.0+cpu): wd = 1.0 gave 0.00 / 0.18 / 0.63 / 0.86 / 0.98; wd = 0.0 gave 0.00 at all checkpoints. Training accuracy was 1.00 from 4k steps.
- **Quote:** „Mit `weight_decay=0.0` bleibt die Validierungsgenauigkeit desselben Laufs während aller 20.000 Schritte bei 0,00.“ (section „Wie Grokking abläuft“)
- **Grade:** C (code with reported results; reproduced here).
- **Novelty:** translation only, but it provides a verified CPU baseline.

#### de-08: the-decoder.de on small vs. large models and grokking in pretraining (2026)

- **Identity:** "Warum große KI-Modelle bestimmte Aufgaben lernen und kleine nicht" (German; English: "Why large AI models learn certain tasks and small ones don't").
  - Author: Jonathan Kemper, THE DECODER, 2026-06-07.
  - URL: https://the-decoder.de/warum-grosse-ki-modelle-bestimmte-aufgaben-lernen-und-kleine-nicht/
- **Type:** news. It reports on Huang et al., arXiv 2505.29548, with Anthropic and Stanford among the authors.
- **Findings (as reported):**
  - OLMo models from 4M to 4B parameters, trained on up to 210B Dolma tokens.
  - Two synthetic tasks were injected: number comparison and modular addition. Frequency ranged from about 1,000 instances per batch to 1 instance every 10 batches.
  - Only larger models learned the rare tasks, by rule rather than memorization. Grokking on modular addition occurred only in the larger models and only at sufficient frequency.
  - A 20M model showed gradient interference ("Update-and-Forget"). A 1B model received a clean gradient signal at each injection.
  - In-model capacity results: only sufficiently large models learned tasks making up 0.25% of the data.
  - The article frames memorization as a precondition for generalization.
- **Quote:** „Dieser Sprung gelang ausschließlich den größeren Modellen, und auch nur dann, wenn die Aufgabe oft genug in den Trainingsdaten vorkam.“ (section „Test an echten Sprachmodellen“)
- **Grade:** D (press summary; the primary paper is English).
- **Novelty:** summary only. It is the only German press item on grokking during LLM pretraining.

#### de-10: the-decoder.de on Google PAIR's grokking explainer (2023)

- **Identity:** "Grokking im Machine Learning: Wenn stochastische Papageien Modelle lernen" (German; English: "Grokking in ML: when stochastic parrots learn models").
  - Author: Maximilian Schreiner, THE DECODER, 2023-08-11.
  - URL: https://the-decoder.de/grokking-im-machine-learning-wenn-stochastische-papageien-modelle-lernen/
- **Type:** news, summarizing Google PAIR's "Do ML models memorize or generalize?".
- **Findings:** PAIR trained more than 1,000 small models. Grokking is "contingent": it vanishes unless model size, weight decay, data size and other hyperparameters are just right.
- **Quote:** „Grokking ein kontingentes Phänomen ist - es verschwindet, wenn die Modellgröße, der Gewichtsabfall, die Datengröße und andere Hyperparameter nicht genau richtig sind.“ (section „Google-Team zeigt …“)
- **Grade:** D. **Novelty:** summary only. It is included because it is the main German press framing: grokking is fragile with respect to hyperparameters.

#### de-24: alphaXiv German translation of Prakash & Martin (HTSR)

- **Identity:** "Grokking und Generalisierungszusammenbruch: Erkenntnisse aus der HTSR-Theorie" (German machine translation of "Grokking and Generalization Collapse: Insights from HTSR theory").
  - Authors: Hari K. Prakash and Charles H. Martin (UCSD / Calculation Consulting). Submitted 2025-06-04.
  - URL: https://www.alphaxiv.org/de/abs/2506.04434
- **Type:** paper. This is a machine-translated abstract and overview, not a German original.
- **Findings:**
  - Setup: 3-layer MLP on 1k MNIST samples, with and without weight decay.
  - Training past 10⁷ iterations reveals "anti-grokking": test accuracy collapses while training accuracy stays perfect.
  - The collapse is flagged by HTSR α < 2 and by "correlation traps". Weight-norm, sparsity and circuit-complexity metrics do not separate it from grokking.
- **Quote:** „die Testgenauigkeit bricht zusammen, während die Trainingsgenauigkeit perfekt bleibt“ (Abstract, German translation)
- **Grade:** D (as a German source; the underlying preprint would be B).
- **Novelty:** none (translation). It is retained because it contradicts de-07's "no un-grokking" at longer horizons; see Contradictions.

### Context and negative-result sources (retained, not evidence-bearing)

| ID | Original title / English | Org, date | Type, grade | Why kept |
|---|---|---|---|---|
| de-02, de-03, de-04, de-05 | „Discovering Transformer Internals (Part 2) – Exploring what Grokking is and how it works?“ / same | 39C3 (Chaos Communication Congress) self-organized workshop, user "derdennisop". Day 3 of 39C3, Dec 2025; the exact date is inferred, not stated on the page. Repo: github.com/DerDennisOP/39c3-transformer-workshop | news (event listing) + code, D | The only German community replication activity found. Quote: „Wir versuchen Nachzuvollziehen was im Paper "Grokking: Generalization Beyond Overfitting…" passiert ist“ (event page). The repo has `datasets/addition`, `models/addition`, a one-line README and no reported results. **No numbers.** |
| de-13 | "Investigating Grokking Phenomena: Neural Network Learning Dynamics through Sparse Crosscoders" / same | TU Darmstadt, Self-Organizing Systems Lab (Prof. Heinz Koeppl, supervisor Philipp Froehlich), 2025-12-25 | thesis **offer** (B.Sc./M.Sc.), D | Shows open German academic interest (crosscoders, curriculum for accelerating grokking). No results yet. Its figure caption cites about 10⁵ steps for modular division (from Power et al.). |
| de-14 | "Bachelor & Master: available thesis topics" (Methoden des maschinellen Lernens) / same | Uni Tübingen, undated | thesis offer, D | An M.Sc. topic "Reverse-Engineering a Discrete Diffusion Model" asks whether grokking occurs on Markov-chain data. No results. |
| de-01 | „Research Bite: Was bedeutet „Grokking" – wenn Können ohne Vorwarnung erscheint“ / "What 'grokking' means: when ability appears without warning" | xm-institute, Dr. Oliver Mack, 2026-07-09 | blog, D | A management and governance framing. Point made: standard metrics give no warning. Quote: „Die Kurven, auf die man beim Training normalerweise schaut, … zeigen über weite Strecken schlicht Stillstand.“ Summary only. |
| de-06 | „Grokking“ (Enzyklopädie) / "Grokking (encyclopedia)" | Schweizer KI Akademie, 2025-09-30 | blog/glossary, D | Correct textbook summary (WD, lazy→rich). Quote: „springt die Testgenauigkeit abrupt auf nahe 100 %“. |
| de-11 | „Emergente Fähigkeiten in LLMs – echt oder Artefakt?“ / "Emergent abilities in LLMs: real or artifact?" | ki-mathias.de, undated | blog, D | Summary. It contains a likely misattribution; see Contradictions. |
| de-15 | „Grokking: Das rätselhafte Phänomen, bei dem KI ganz von selbst Mathematik lernt“ / "Grokking: the puzzling phenomenon where AI learns math by itself" | codedtrip.com/de, Ondřej Barták, 2026-05-26 | blog, D | Summary of Power et al. and Nanda. Quote: „…erst nach einer Million Trainingsschritten, während das Modell die Daten bereits in weniger als tausend Schritten auswendig gelernt hatte.“ |
| de-19 | „Was ist Grokking?“ (KI-Lexikon) / "What is grokking?" | dogado.de, undated | blog, D | **Misinformation example**; see Contradictions. |
| de-23 | „Grokking“ (Wiki) | hyper.ai/de, about 2023 | blog, D | Machine-translated. It muddles the definition. |

Opened but neither evidence-bearing nor retained: de-09 (ki-mathias Hopfield page, headings only; grokking only cross-linked), de-12 (TU Darmstadt project list), de-16 (arXiv 2412.10898, not German-affiliated), de-21 (ETH BA thesis by S. Peter: group datasets used by de-07, with no grokking experiments of its own), and de-22 (ETH GitLab, behind a login wall).

## Contradictions

Contradictions with the standard English account, and quality problems in German sources:

1. **Are loss spikes necessary?**
   - de-07 calls training-loss spikes "a fundamental part of grokking behaviour" (Abstract) and says grokking is "bound to training loss spikes" (Conclusion).
   - The standard mechanistic account (Nanda et al.) has smooth circuit formation followed by weight-decay-driven cleanup, with no spikes required. Ultralytics (de-20) and ExPLAIND (de-18) also grok smoothly at WD 1.0 and 4.0.
   - **Reconciliation:** de-07 used WD 0.01 and the original custom AdamW. That is the slingshot (Thilak et al.) regime, where grokking is slow and spike-driven. The author himself shows a case (Fig. 4.9) where spikes "are not essential to get to better accuracy".
2. **Can a model un-grok?**
   - de-07 says no, within 2·10⁶ steps (Transformer, algorithmic data, WD 0.01).
   - de-24 (Prakash & Martin) reports anti-grokking, a late test-accuracy collapse, beyond 10⁷ iterations (MLP, MNIST subset).
   - These are not directly comparable: different horizon, architecture and data. For a 10⁵-step ReLSM run, de-07's regime is the relevant one.
3. **Misattribution in German popular writing.** ki-mathias.de (de-11) states „Levi et al. (2024) zeigten formal, dass Grokking ein Phasenübergang erster Ordnung ist“.
   - In the English literature, the first-order-phase-transition result is usually Rubin, Seroussi & Ringel (ICLR 2024). Levi, Beck & Bar-Sinai (2024) is "Grokking in linear estimators".
   - I did not open the page's reference [19], so this is flagged as a likely misattribution, not a verified error.
4. **Unsupported applications claims.** dogado.de (de-19) claims grokking was "frequently observed" in Go and chess models, and that it is "used" in robotics and financial analysis. No source is given, and the claims do not match the research usage of the term.
5. **Muddled definition.** hyper.ai/de (de-23) defines grokking as networks achieving good generalization „auch dann noch …, wenn der Trainingsfehler über einen langen Zeitraum hinweg abnimmt“. That garbles "generalization long after the training error has reached zero".
6. **Term collision (noise, not a contradiction).** In German security press, „Grokking“ also names a prompt-injection or phishing abuse of xAI's Grok chatbot (welivesecurity.com/de, Nov 2025). It polluted several searches.

## Gaps

- **German-language theses:** none found with grokking results.
  - Repository searches covered mediaTUM, KITopen, LMU epub, Tübingen, ETH Research Collection, TU Wien reposiTUm, Uni Wien phaidra and DNB, by `site:` queries and free text. The only mediaTUM hit (doc 1780617, a physics/AI-alignment thesis) merely cites Nanda et al.; I did not open it.
  - Only thesis *offers* exist (TU Darmstadt Dec 2025; Tübingen). A TU Darmstadt grokking thesis PDF (`documents_1/ThesisGrokking.pdf`) and two etit thesis-detail pages (IDs 36416, 40707) returned **HTTP 404**. They may have been completed theses; their status is unknown.
- **German technical press:** heise, golem, t3n, c't, iX and Spektrum have no grokking articles (t3n covers only the Grok chatbot). The-decoder.de (2023, 2026) is the only press coverage.
- **Acceleration and numerics in German:** nothing on Grokfast or its variants, StableMax/⊥Grad (beyond the Ultralytics glossary mention), Softmax collapse, weight-decay schedules, bf16/fp16 issues, or slingshot (beyond de-07). Searches for „Grokfast beschleunigen“ and „Softmax-Kollaps“ returned noise or zero results.
- **Video, podcast, forum:** no German YouTube explainer, lecture recording or podcast episode was found. The 39C3 session was a self-organized workshop; I did not check media.ccc.de for a recording, so treat that as unverified. No r/de_EDV or German ML forum threads were found.
- **GitHub with German READMEs:** none with reported grokking numbers. The 39C3 repo's README is empty.
- **Expected but not found:**
  - German replications of Grokfast or of Prieto et al.'s StableMax.
  - MPI, ELLIS, Lamarr or Fraunhofer talks on grokking.
  - A German-language Master's thesis replicating Nanda et al. or Power et al.
  - Austrian (JKU Linz, TU Wien, Uni Wien) grokking work.
- **Dominant genre:** the German-language content that does exist is mostly machine-translated paper summaries (alphaxiv.org/de, chatpaper.ai/de, chatpaper.com/de, hyper.ai/de) and SEO lexicon pages. These carry no new evidence.

## Method

- **Tools:** SearXNG (`language: de`, with `time_range: year` for some queries) as the main engine; WebSearch (standard and extended modes) as the second engine. Pages were opened with SearXNG `web_url_read` or WebFetch. PDFs were downloaded and converted with `pdftotext`.
- **SearXNG queries (21):**
  - „Grokking neuronale Netze verzögerte Generalisierung“
  - „Grokking Masterarbeit“
  - „Grokking KI plötzlich generalisieren“ (year)
  - „"Grokking" Weight Decay modulare Addition Transformer“
  - „grokking site:mediatum.ub.tum.de“
  - „grokking thesis KIT OR TUM OR Universität Tübingen OR ETH Zürich delayed generalization“
  - „heise Grokking KI Modell Generalisierung“
  - „"Grokking" "Auswendiglernen" Netz Generalisierung“
  - „ki-mathias Emergenz … Grokking Ising“
  - „Grokking reposiTUm OR phaidra OR Research Collection thesis“
  - „"Grokking" Bachelorarbeit Universität“
  - „Grokfast Grokking beschleunigen Gradientenfilter“
  - „StableMax Softmax Collapse Grokking numerische Stabilität“
  - „reddit Grokking neuronales Netz Erfahrung Weight Decay deutsch“
  - „Grokking Vortrag Universität Video … Vorlesung“
  - „"Grokking" medium.com deutsch …“
  - „grokking arXiv 2025 Universität/LMU/TUM/ETH/KIT modular arithmetic“ (year; 0 results)
  - „spektrum.de Grokking …“
  - „the-decoder Grokking Sprachmodelle Pretraining …“ (year; 0 results)
  - „youtube "Grokking" KI deutsch erklärt …“
  - „"Softmax-Kollaps" OR "Softmax Collapse" Grokking“ (0 results)
- **WebSearch queries (18):**
  - German WD/Transformer phrasing
  - Bachelor/Master/Dissertation „modulare Arithmetik“
  - site:heise/golem/t3n
  - site: ETH/TU Wien/KIT/TUM repositories
  - German YouTube
  - bcs.tu-darmstadt ThesisGrokking
  - German blog "mod 97" experiment
  - GitHub README „Trainingsgenauigkeit/Gewichtsabnahme“
  - „verzögerte Generalisierung“ blog 2025
  - ExPLAIND LMU
  - Spektrum Grokking
  - site:the-decoder.de
  - pub.tik.ee.ethz.ch grokking
  - Master's thesis Germany 2025–26 (extended)
  - JKU/Wien
  - Fraunhofer/MPI/Lamarr/ELLIS
  - German podcast
  - Seminararbeit/Proseminar (extended)
- **Counts:**
  - About 590 result entries seen. This is approximate: SearXNG returns about 25 per query and three queries returned 0; WebSearch returns about 9 per query.
  - **24 unique sources opened** (de-01 to de-24, in the JSONL), plus 3 URLs that returned 404 and are not counted.
  - **19 retained**: 7 evidence-bearing (de-07, de-08, de-10, de-17, de-18, de-20, de-24) and 12 context or negative sources (de-01 to de-06, de-11, de-13, de-14, de-15, de-19, de-23; de-02 to de-05 form one table row).
  - 5 opened and dropped: de-09, de-12, de-16, de-21, de-22.
- **Verification:** the de-20 code was executed locally (CPU, torch 2.8.0, seed 0) for wd = 1.0 and wd = 0.0. The results are given under de-20.
- **Classification:** English-language works by German-speaking institutions (de-07 ETH, de-17/18 LMU) are marked as pointers. Findings from them are reported only where they are not covered by the standard English canon.
