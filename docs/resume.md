# Resuming a training run (`train.py --resume_from`)

A run that stops (killed, crashed, interrupted) continues from one of its checkpoints:

```bash
python train.py <the same arguments> --resume_from OUTPUT_DIR/checkpoint_step_N.pt
```

## Exact and inexact resumes

- **Exact.** A rotating checkpoint (`checkpoint_step_N.pt`, written every `--eval_interval` micro-steps and kept
  `--eval_checkpoints` deep) holds everything the training loop carries from one iteration to the next: weights,
  optimizer and scaler, pending gradients (a checkpoint can fall part-way through a gradient accumulation), the
  Grokfast EMA, RoPE caches, every RNG (Python, NumPy, torch CPU, the training CUDA device), the position of every
  data loader, the task-curriculum and DAG state and their published snapshots, the logging window, the best model
  as it was then, and the metrics history. A resume from it continues as if the run had never stopped: the same
  batches, metrics records (all but timing) and final weights as an uninterrupted run.
- **Inexact.** Anything else needs `--resume_inexact`: a checkpoint written before this change, `best_model.pt`,
  `final_model.pt`, an `_interrupt` checkpoint, changed arguments, code or library versions, `--num_workers > 0`
  (the data position lives in worker processes), `--compile`, Hub language sources (they can change under the same
  name), a local language directory that holds a dataset card (`README*`), a `.py` script, a `.yaml`/`.yml` file
  or links out of it (each can point the datasets library at files the hashes do not cover; an exact source is a
  plain directory of data files), or CUDA without `--cuda_deterministic`. It restores what the file has, the
  data loaders start from their beginning, and the resume is recorded as inexact with its reasons. Without the flag
  the run stops, lists the reasons, and changes nothing. Even an inexact resume of a rotating checkpoint keeps
  `--use_task_curriculum` as the run had it (the loop state it restores belongs to that setting); to change it,
  resume from `best_model.pt` or `final_model.pt`, or start a new run.

Arguments that may differ on an exact resume: `--resume_from`, `--resume_inexact`, `--output_dir`.

## What a resume does to the output directory

Before anything is written, the whole restore is built in memory and checked (a refusal leaves the directory as
it was): a checkpoint with exact-resume state must be complete, as the run that wrote it was set up (weights,
optimizer, scaler, curriculum and DAG state when that run had them, every RNG state loadable, a CUDA run's CUDA
RNG state for its own device) and consistent (its step and token counters agree, its exact flag matches its
reasons); otherwise the resume stops with the reason. Then, in this order:

1. `resume_log.jsonl` gets one line for the attempt (a torn last line from a crash is kept aside as
   `resume_log.jsonl.torn.N`).
2. `summary.json` is moved to `old_<attempt>/` first (the run no longer counts as finished), then the earlier
   attempt's later outputs (`final_model.pt`, checkpoints after the resume step, with their sidecars), and any
   earlier rotating checkpoint the resume checkpoint had already retired (left behind when a process was killed
   between publishing a checkpoint and pruning the oldest one).
3. `metrics.json` is copied whole to `metrics.json.legacy.N`, then rewritten as it was at the checkpoint (#150):
   rows the killed attempt logged after the checkpoint are dropped and logged again, no gap, no duplicate. A
   `metrics.json` in a different destination directory is never adopted: it is moved to `metrics.json.unadopted.N`.
4. `best_model.pt` is set to the best model as it was at the checkpoint. When there was none yet, or it is unknown
   (below), a `best_model.pt` already there is moved to `old_<attempt>/` with its sidecar.

The file named by `--resume_from` and its sidecar are never moved or rewritten by these startup steps. Every
step can be killed and the resume run again. After startup the run behaves as usual: a rotating checkpoint you
resumed from is pruned once it leaves the retained window, and a resume from `final_model.pt` in its own
directory rewrites that file when the run finishes (copy a file elsewhere first to keep it).

Every resume is listed under `resumes` in metrics.json's hyperparameter row and in summary.json (`kind`,
`why_inexact`, the source file and its sha256, the step, the history lineage). `total_time_hours` counts training
time, not downtime: work done after the checkpoint by a killed attempt is not counted.

## Files

- Checkpoints load with `torch.load(weights_only=True)` (#138): the model config is stored as plain values
  (`{"schema": 1, "fields": {...}}`). `utils.load_checkpoint` also reads files written before this change, with an
  allowlist scoped to that one load.
- Every checkpoint, `best_model.pt` and `final_model.pt` are written atomically with a sidecar `NAME.sha256`
  (`{"sha256", "size", "step"}`). A resume refuses a file whose sidecar does not match; a missing sidecar (a file
  written before this change, or killed between the payload and its sidecar) is accepted with a warning.
- Each checkpoint embeds the best model (about twice the size of a checkpoint without it), so a checkpoint is
  complete on its own.
- Every run has a `run_id`, in every checkpoint and in the hyperparameter row; a resume keeps it.

## Resuming from an older checkpoint (`--resume_inexact`)

A checkpoint without exact-resume state restores weights, optimizer, scaler, curriculum and DAG state (their
snapshots are published right away), and takes the metrics history from `metrics.json` **in the checkpoint's own
directory**, rows up to the checkpoint step, if that file is compatible with the checkpoint (same curriculum
settings, consistent token counts). Compatibility is not proof that the rows are this run's: such history is marked
`lineage: unverified`. The best model is recovered from `best_model.pt` beside it when its step and accuracy match
the checkpoint; otherwise it is unknown, and later checkpoints say so.

## Language data

An exact resume of a run whose language stream was already reading replays that stream up to its position: it reads
and tokenizes each example consumed so far, without training on it, then checks that the stream routed the same way
(the original run's routing events: each source that ended or failed, each rebuild of the stream) and that the last
example's hash matches the checkpoint's; a difference stops the resume. Restore time grows with the language
examples consumed; nothing else is replayed.

## CUDA

`--cuda_deterministic` (with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before Python starts) turns on deterministic
kernels, turns off TF32, reduced-precision reductions and FP16 accumulation, and keeps attention on the math
kernel. It is slower. The settings are read back from torch and fingerprinted with the GPU, driver, CUDA, cuDNN and
torch versions; any difference makes a resume inexact. A setting the installed torch does not have (FP16
accumulation before torch 2.7) is recorded as `null` and counts as off.

## Test hooks (environment variables, read only when set)

- `RELSM_TEST_KILL_STEP=MICROSTEP:POINT` with `RELSM_TEST_KILL_MARKER=PATH` (one-shot): end the process
  (`os._exit(86)`) at that point: `step`, `log`, `eval`, `end`, `interrupt` (raises KeyboardInterrupt), or a file
  operation such as `ckpt:save`, `ckpt:payload`, `ckpt:prune#1`, `best:payload`, `final:payload`,
  `summary:write`, `rs:summary`, `rs:move#1`, `rs:logger`, `rs:best`.
- `RELSM_TEST_BATCH_LOG=PATH`: one line per micro-batch, `micro_step sha256`.
- `RELSM_TEST_DIGEST=PATH`: digests of the end-of-run state (weights, optimizer, EMA, buffers, gradients, scaler,
  RNGs).
- `RELSM_TEST_EXPECT_DEVICE=cuda:0`: stop unless training runs on that device.

## Not covered

`--compile`, multi-GPU, and a Hub language stream that fails differently after a restart (detected by the
routing and content checks, then refused; not recovered). The `--mod_table_p` bridge path keeps its own resume
(`--mod_table_resume`).
