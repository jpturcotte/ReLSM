# Posted issues

| # | Sev | Area | Title |
|---|---|---|---|
| [#132](https://github.com/jpturcotte/ReLSM/issues/132) | P1 | model | ACT halt head has no gradient; halting is batch-wide; ponder rewards not halting |
| [#133](https://github.com/jpturcotte/ReLSM/issues/133) | P1 | model | SSM fallback recurrence ignores input v; channels within a group collapse |
| [#134](https://github.com/jpturcotte/ReLSM/issues/134) | P1 | model | SSM fallback scan mixes batch and time: cross-example and future-to-past leak |
| [#135](https://github.com/jpturcotte/ReLSM/issues/135) | P1 | model | Thought loop and memory compressor leak future tokens in latent/act/ssm/ssm_mem |
| [#136](https://github.com/jpturcotte/ReLSM/issues/136) | P1 | training | --fixed_data runs crash at the first train-eval and write no checkpoint |
| [#137](https://github.com/jpturcotte/ReLSM/issues/137) | P1 | training | Grokfast diverges exponentially with gradient accumulation >= 2 |
| [#138](https://github.com/jpturcotte/ReLSM/issues/138) | P1 | training | Resume crashes: torch.load rejects the pickled TransformerConfig |
| [#139](https://github.com/jpturcotte/ReLSM/issues/139) | P1 | data | alg_seq_len=48 silently truncates answers and stalled the DAG-v12 curriculum |
| [#140](https://github.com/jpturcotte/ReLSM/issues/140) | P1 | data | Procedural data replays: AlgorithmicDataset reseeds its RNG on every iterator |
| [#141](https://github.com/jpturcotte/ReLSM/issues/141) | P1 | data | Difficulty schedule stuck at easiest phase with workers while logs report 1.0 |
| [#142](https://github.com/jpturcotte/ReLSM/issues/142) | P1 | eval | Eval ignores max_new_tokens override; 7 of 21 OOD conditions impossible at 32 |
| [#143](https://github.com/jpturcotte/ReLSM/issues/143) | P1 | eval | Training-time eval reseeds the global RNGs at every eval interval |
| [#144](https://github.com/jpturcotte/ReLSM/issues/144) | P1 | eval | Add an oracle-model eval regression test (a perfect answerer scores 0.779 today) |
| [#184](https://github.com/jpturcotte/ReLSM/issues/184) | P1 | meta | Add a CPU invariant test suite that runs on every change |
| [#145](https://github.com/jpturcotte/ReLSM/issues/145) | P2 | model | create_model overrides raise TypeError; shared_loop depth is not configurable |
| [#146](https://github.com/jpturcotte/ReLSM/issues/146) | P2 | model | SSM generation cache is never created; every decode step recomputes the prefix |
| [#147](https://github.com/jpturcotte/ReLSM/issues/147) | P2 | model | Model-wide _init_weights overwrites the SSM dt_proj initialisation |
| [#185](https://github.com/jpturcotte/ReLSM/issues/185) | P2 | training | Crashes leave no crash checkpoint; Ctrl-C is saved as a finished run |
| [#186](https://github.com/jpturcotte/ReLSM/issues/186) | P2 | training | Token embeddings and lm_head are excluded from weight decay, with no switch |
| [#148](https://github.com/jpturcotte/ReLSM/issues/148) | P2 | training | Final partial accumulation is dropped; mid-accumulation checkpoints lose grads |
| [#149](https://github.com/jpturcotte/ReLSM/issues/149) | P2 | training | Gradient and update telemetry windows are misaligned and mislabelled (F26, F27) |
| [#150](https://github.com/jpturcotte/ReLSM/issues/150) | P2 | training | MetricsLogger wipes metrics.json history on resume |
| [#151](https://github.com/jpturcotte/ReLSM/issues/151) | P2 | training | Cross-entropy returns NaN on zero-label batches and poisons loss aggregates |
| [#152](https://github.com/jpturcotte/ReLSM/issues/152) | P2 | training | Weight-norm logging concatenates every parameter into one device tensor per log |
| [#153](https://github.com/jpturcotte/ReLSM/issues/153) | P2 | training | eval/loss/ppl exponentiates a loss that includes the ACT ponder term |
| [#154](https://github.com/jpturcotte/ReLSM/issues/154) | P2 | training | Resume replays the data stream from the start and drops RNG and Grokfast state |
| [#187](https://github.com/jpturcotte/ReLSM/issues/187) | P2 | training | Run arguments and code version are not saved with the checkpoint |
| [#155](https://github.com/jpturcotte/ReLSM/issues/155) | P2 | data | Addition 'OOD' 6-digit condition overlaps the final training phase (3-6 digits) |
| [#156](https://github.com/jpturcotte/ReLSM/issues/156) | P2 | data | Compare '=' answers are never supervised: 'Answer:' + '=' tokenizes to ':=' |
| [#157](https://github.com/jpturcotte/ReLSM/issues/157) | P2 | data | DAG gating fragile: EMA snap, gates above controller band, untimed pause |
| [#158](https://github.com/jpturcotte/ReLSM/issues/158) | P2 | data | lang_loader aliasing replays the alg stream; language PPL uses training data |
| [#159](https://github.com/jpturcotte/ReLSM/issues/159) | P2 | data | Language pipeline prototype: head truncation, seed 42, silent source drops |
| [#160](https://github.com/jpturcotte/ReLSM/issues/160) | P2 | eval | Eval sets differ per process: build_dataset seeds examples with salted hash() |
| [#161](https://github.com/jpturcotte/ReLSM/issues/161) | P2 | eval | Eval JSON keeps only normalized predictions; raw outputs and token IDs are lost |
| [#162](https://github.com/jpturcotte/ReLSM/issues/162) | P2 | eval | "IID" grid is not the training distribution; dyck OOD depth 5/6 is a no-op |
| [#163](https://github.com/jpturcotte/ReLSM/issues/163) | P2 | eval | Long-context eval: raw string equality, AUC 0 when nothing runs, budget override |
| [#164](https://github.com/jpturcotte/ReLSM/issues/164) | P2 | eval | Numeric distance is 1.0 for every answer with 16+ digits, even when exact |
| [#165](https://github.com/jpturcotte/ReLSM/issues/165) | P2 | eval | Numeric answer parser false positives and negatives (42.0 -> 0, 1,234 -> 234) |
| [#166](https://github.com/jpturcotte/ReLSM/issues/166) | P2 | eval | pytest stops at collection on a stale import; 29 pass / 3 fail without it |
| [#167](https://github.com/jpturcotte/ReLSM/issues/167) | P2 | docs | Duplicate train_eval/acc/token key drops teacher-forced accuracy |
| [#168](https://github.com/jpturcotte/ReLSM/issues/168) | P2 | docs | requirements.txt: mamba-ssm hard-required, pytest missing, no lock file |
| [#169](https://github.com/jpturcotte/ReLSM/issues/169) | P2 | docs | README claims about data, results and the IID grid are untested or wrong |
| [#170](https://github.com/jpturcotte/ReLSM/issues/170) | P2 | docs | README preset parameter counts and shared_loop depth are wrong |
| [#188](https://github.com/jpturcotte/ReLSM/issues/188) | P2 | meta | Add CI and a locked environment for pytest, invariants and a CPU smoke run |
| [#189](https://github.com/jpturcotte/ReLSM/issues/189) | P2 | meta | Experiment hygiene: self-contained run records, a run log, and agent rules |
| [#171](https://github.com/jpturcotte/ReLSM/issues/171) | P3 | model | generate() calls self.eval() and never restores the previous training mode |
| [#190](https://github.com/jpturcotte/ReLSM/issues/190) | P3 | model | GQA head counts are not validated; bad values fail late or are silently ignored |
| [#172](https://github.com/jpturcotte/ReLSM/issues/172) | P3 | model | Cached decoding drops the padding mask for past tokens |
| [#191](https://github.com/jpturcotte/ReLSM/issues/191) | P3 | training | --compile checkpoints save _orig_mod. keys and cannot be loaded by eval |
| [#173](https://github.com/jpturcotte/ReLSM/issues/173) | P3 | training | Cosine LR divides by zero at warmup == total; wsd/plateau jump past 40% warmup |
| [#174](https://github.com/jpturcotte/ReLSM/issues/174) | P3 | training | Loss under gradient accumulation weights micro-batch means equally, not tokens |
| [#175](https://github.com/jpturcotte/ReLSM/issues/175) | P3 | training | metrics.json is rewritten in full at every log, so I/O grows quadratically |
| [#192](https://github.com/jpturcotte/ReLSM/issues/192) | P3 | data | Manager-proxy curriculum snapshots slow per-example sampling by 5 to 7 times |
| [#176](https://github.com/jpturcotte/ReLSM/issues/176) | P3 | data | Dyck invalid strings are detectable by counting; compare rarely emits '=' |
| [#193](https://github.com/jpturcotte/ReLSM/issues/193) | P3 | eval | Eval generation autocasts in fp16 on CUDA while training autocasts in bf16 |
| [#194](https://github.com/jpturcotte/ReLSM/issues/194) | P3 | eval | Eval metadata records the current directory's git commit and no dirty flag |
| [#177](https://github.com/jpturcotte/ReLSM/issues/177) | P3 | eval | eval/self_test.py determinism assert compares payloads that include timestamps |
| [#178](https://github.com/jpturcotte/ReLSM/issues/178) | P3 | eval | Soft metrics encode standalone strings, so the SentencePiece prefix biases them |
| [#179](https://github.com/jpturcotte/ReLSM/issues/179) | P3 | eval | test_model_import_optional.py leaks sys.modules and sys.path after it runs |
| [#180](https://github.com/jpturcotte/ReLSM/issues/180) | P3 | docs | Dead code: thought_use_rope, nano_baseline.py, write_summary_md and friends |
| [#181](https://github.com/jpturcotte/ReLSM/issues/181) | P3 | docs | profile_curriculum_memory.py fails as a script and cannot see tensor memory |
