---
slug: llemma-tokenizer-adopted-on-false-premise
title: Llemma tokenizer adopted on a false premise, against the 16k advice; no ADR
issue: 225
severity: P3
labels: [enhancement, P3-minor, area: docs, reproducibility, audit-2026-10-27]
nf: [NF-46]
found_by: [gemini-c, gemini-a]
status: Demonstrated
related_issues: [#186, #170, #156, #178, #189, #139, #142]
---

## Summary

The Dec-17 tokenizer report ranked "Llama 2 / BPE 32k" as insufficient for arithmetic because its digits agglomerate, and Llemma as "the strongest candidate" because it supposedly retrained the tokenizer to isolate digits. The tokenizer the repository loads is `LlamaTokenizerFast` with the Llama-2 vocabulary (32,000 + 16 Code Llama fill tokens); digit splitting is a Llama-2 property, and Llemma did not retrain it. The stated reason to prefer Llemma over Llama 2 is therefore false, and the same report's own final recommendation was the opposite of what was adopted: a custom 16,384-token vocabulary and "do not use the GPT-2 or Llama pretrained tokenizer". The consequences are already filed (#186, #156, #178, #139, #142); what is missing is the decision record. This is an enhancement to the docs, not a code bug.

## What happens

- [utils.py:29](https://github.com/jpturcotte/ReLSM/blob/774513d570341a00a3ba4b8e3dfb22a2f81d4467/utils.py#L29): `DEFAULT_TOKENIZER_NAME = "EleutherAI/llemma_7b"`, set by commit `e9de18b` / PR [#48](https://github.com/jpturcotte/ReLSM/pull/48) on 2025-12-17, the day of the report.
- Audit probe `gemini-c/probe_tokenizer.py` (offline, cached tokenizer): class `LlamaTokenizerFast`, `vocab_size` 32,016 = `len`; ids 0 to 7 are `<unk>, <s>, </s>, <0x00>...`; id 29871 is `▁`, 29900 is `0`, 29906 is `2`: the Llama-2 layout plus 16 fill tokens. `"12345"` -> `▁ 1 2 3 4 5`; `"Add 7390 and 4223. Answer:"` -> `▁Add ▁ 7 3 9 0 ▁and ▁ 4 2 2 3 . ▁Answer :`.
- The report (Gemini chat `ba6cfceb`, "Tokenizer pour Transformers Maths Langage", 2025-12-17): Option A "Llama 2 / BPE 32k", digits "Incohérent (agglomération)", "Verdict : Insuffisant pour l'objectif Maths" [L497, L507-509]; Option C "Llemma": "Contrairement à Llama 2 qui utilise un tokenizer générique, Llemma ré-entraîne le tokenizer pour qu'il apprenne ... que les chiffres doivent être isolés", "Verdict : Le candidat le plus fort" [L525-529]. Final recommendation: "Pour un modèle ~50M - 200M : Vocabulaire cible de 16 384" [L549]; "Ne pas utiliser le tokenizer pré-entraîné de GPT-2 ou Llama!" [L591].
- Costs the report did not mention and the repository then paid: every number costs one extra bare `▁` token (`"Answer: 11613"` is 8 tokens for 5 digits), the prefix behind #178 and part of the length pressure in #139 and #142; `"Answer:"` + `"="` merges to `:=` (#156). The report's own arithmetic (`V=32000, d=512` -> 16.4M embedding parameters, "environ 32 % du budget") predicted the 39.4 % embedding share that #186 measured; the probe gives 16,392,192 embedding parameters at 32,016 x 512 against 8,388,608 at 16,384.
- Audit probe `gemini-a/probe_recipe_params_and_eos_mask.py` (gemini-a G-A-2): the grokking-recipe model the Dec-17 research plan asked for (`d_model 128, 2 layers, 4 heads`) has 4,620,928 parameters at vocab 32,000, of which 4,096,000 (88.6 %) are the tied embedding; the body is 524,928. At 16,384 it is still 80.0 %. The lever for a recipe-sized model is a ~100-token digit-and-symbol vocabulary, not 16k vs 32k.
- No ADR, PR body or README line states why 32k Llama vocabulary and not the 16k custom one. `docs/review-2026-10-26/project-history-gemini.md` (section 4, item 4) records the 16k recommendation but not that the premise for picking Llemma was wrong.

## Why it matters for the experiments

- Every ReLSM run since Dec 17 (nano 41.6M parameters, 16.4M of them embeddings; #170, #186) sits on a vocabulary choice whose written justification does not hold, and the first-order alternative that the same report asked for (16k) was never tried. The grokking-scale model from the same day's research plan cannot be built sensibly on this vocabulary (88.6 % embeddings).
- A future tokenizer change invalidates every checkpoint, so it is a new-run decision; without a record, the next agent will re-derive the same wrong comparison from the same report.

## How it got here

Gemini Deep Research report, 2025-12-17 16:05; commit `e9de18b` "Default to llemma tokenizer" (PR [#48](https://github.com/jpturcotte/ReLSM/pull/48)) the same day, replacing GPT-2 (50,257) after run F. No Codex task text, PR body or README sentence gives the reason; the only written rationale is the report's false sentence about Llemma retraining the tokenizer. The previous summary of the report (project-history-gemini.md) carried the 16k recommendation forward without checking the premise.

## How we verified it, step by step

1. gemini-c (GC-2) read the report lines [L497-529, L549, L591, L446, L384-410] and ran audit probe `gemini-c/probe_tokenizer.py` on the cached Llemma tokenizer (class, vocab size, ids 29871/29900/29906, digit and prompt tokenisations, embedding parameter counts).
2. gemini-a (G-A-2) built `TransformerConfig(d_model=128, n_layers=2, n_heads=4)` at vocab 32,000 and 16,384 with audit probe `gemini-a/probe_recipe_params_and_eos_mask.py` and counted embedding vs body parameters.
3. Not verified: that Llemma's tokenizer is Llama 2's is inferred from the loaded vocabulary (identical layout, 32,000 + 16); the Llemma paper was not fetched (gemini-c section 5).
4. Coordinator: not re-checked.

## Suggested fix

Write the ADR under #189's hygiene work, in `docs/`, with: (a) what was believed on Dec 17 and the quote; (b) what is true (Llama-2 vocabulary, digit splitting inherited, extra `▁` per number, `:=` merge), with the probe numbers; (c) the consequences already filed (#186, #156, #178, #139, #142, #170) and the 88.6 % figure for a recipe-sized model; (d) the decision going forward: keep 32k, build the 16k custom vocabulary the report asked for, or a ~100-token digit-and-symbol tokenizer for the algorithmic phase. Add one README sentence stating the vocabulary size (32,016) and pointing at the ADR. Correct project-history-gemini.md section 4 item 4.

## Acceptance test

A file `docs/adr/*tokenizer*.md` exists, quotes the Dec-17 premise, states the probe facts (`LlamaTokenizerFast`, 32,016, `"12345"` -> 6 tokens, 16,392,192 embedding parameters at d 512) and names the decision. `grep -n "32,016\|32016" README.md` returns a line next to the tokenizer name.

## Related

- #186: embedding share 39.4 % at nano; the report predicted it.
- #170: README parameter counts; nano is 41.6M because of this vocabulary.
- #156: the `:=` merge that un-supervises compare's `=`.
- #178: the `▁` prefix that biases the soft metrics.
- #139, #142: sequence-length pressure that the extra `▁` per number adds to.
- #189: the ADR and run-record work this belongs under.

## Sources

- Audit report gemini-c, finding GC-2 and section 5; probe `probe_tokenizer.py`; transcript "Tokenizer pour Transformers Maths Langage" (chat `ba6cfceb`)
- Audit report gemini-a, finding G-A-2 and File 3 notes; probe `probe_recipe_params_and_eos_mask.py`
- `docs/review-2026-10-26/project-history-gemini.md`, section 4 item 4
- Merged findings, history item 12 (tokenizer switch)
