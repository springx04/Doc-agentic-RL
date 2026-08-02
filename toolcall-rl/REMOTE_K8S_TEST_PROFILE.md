# Remote Qwen3-VL RL test profile

Last verified: 2026-07-23 (Asia/Shanghai). Reuse this profile for bounded
real-data tests unless the server environment, model mount, or data mount has
changed.

## Fixed remote locations

| Item | Verified value |
| --- | --- |
| Workspace | `/workspace/data` |
| Project | `/workspace/data/OpenClaw-RL` |
| Python environment | `/workspace/data/envs/openclaw-rl-qwen3vl/bin/python` |
| Policy model | `/models/Qwen3-VL-4B-Instruct` |
| Train data | `/workspace/data/OpenClaw-RL/data/real_docvqa_rl_test_20260717-01/train.jsonl` (4 rows) |
| Eval data | `/workspace/data/OpenClaw-RL/data/real_docvqa_rl_test_20260717-01/eval.jsonl` (2 rows) |
| Document root | `/workspace/data/OpenClaw-RL/data` |
| Tool-model cache | `/workspace/data/OpenClaw-RL/openclaw-tool-models-20260717-02` |
| Docling cache | `/workspace/data/OpenClaw-RL/openclaw-tool-models-20260717-02/docling-artifacts` |
| Latest Ray temp directory | `/workspace/data/.ray/q3v4b-real-20260723-03` |
| Latest output directory | `/workspace/data/OpenClaw-RL/outputs/qwen3-vl-4b-docvqa-real-rl-test-20260723-03` |

## Latest verified real-RL smoke result

- `openclaw_real_docvqa_train_05` completed successfully on 2026-07-23:
  one RL update using four real train rows and two real eval rows.
- JSON: `/workspace/data/OpenClaw-RL/outputs/qwen3-vl-4b-docvqa-real-rl-test-20260723-03/rollout_interactions.json`.
  It contains no exact `"tokens"` key.
- Four of eight training samples scored fully correct. The remaining failures
  are model/document-reasoning errors, not terminal-answer formatting or the
  previously observed explanatory-prefix scoring error.
- Eval quality was 0.5 (one of two examples correct). This is a post-one-update
  smoke signal, not a formal measure of training quality.

Do not repeat broad environment/GPU/disk/Git-worktree discovery for ordinary
tests. Reuse this profile; only validate a fresh output/Ray directory and the
target script's data paths.

## Untrained Qwen3-VL baseline evaluation (2026-07-23)

- Purpose: strict evaluation-only baseline using the initial
  `/models/Qwen3-VL-4B-Instruct` weights; no RL rollout or optimizer update.
- Command profile: `--num-rollout 0` (the authoritative no-update switch) and
  `--lr-decay-iters 1` (required because Slime still constructs its scheduler
  in evaluation-only mode).
- Dataset: the two-row real document QA evaluation split above.
- Result: **0 / 2 correct (0.0% exact accuracy)**; both per-sample rewards and
  ANLS quality scores were `0.0`.
- Output: `/workspace/data/OpenClaw-RL/outputs/qwen3-vl-4b-docvqa-baseline-20260723-05`.
- JSON rollout log (with no `tokens` field):
  `rollout_interactions.json`; summary: `training_run_summary.json`.
- The run completed successfully on four GPUs from 03:32:52 to 03:36:04 UTC.
  Its result records `num_rollout: 0` and `checkpoint: null`, which confirms
  this is a pre-training baseline rather than a trained checkpoint evaluation.

## Validated capacity and versions

- Four NVIDIA A100-SXM4-80GB GPUs; each had 78.84 GiB free at verification.
- Free workspace disk: 1200.72 GiB.
- CUDA 12.8; cuDNN 9.16; Torch 2.9.1+cu128; Ray 2.54.0; Transformers 4.57.1;
  SGLang 0.5.7.dev0 at commit `24c91001cf99ba642be791e099d358f4dfe955f5`.
- The test uses gradient checkpointing, dynamic tool images, rollout batch 2,
  four samples per prompt, global batch 8, and four GPUs.

## Remote task policy

- Use `openclaw_real_docvqa_check_05` for the preflight only when this profile
  may have changed, or before choosing a new output directory.
- Use `openclaw_real_docvqa_train_05` for the bounded real-data RL test. The
  user has standing approval for this real training-test task.
- Use `openclaw_rollout_json_export_05` after a successful run. The exporter
  must omit the `tokens` field.
- Never reuse an existing output or Ray temporary directory. Choose a fresh
  dated suffix instead; do not delete old artifacts.

## Code synchronization constraint

The remote workspace is **not a Git working tree**. Local Git diffs cannot be
applied or queried remotely with Git. Treat the local branch
`codex/openclaw-rl-eval-fixes` (commit `b5cafe0e13497c552b8497f588d02aac4f98f9f4`)
as the source of the production reward/rollout changes. Apply only targeted
unified patches under the remote path prefix `OpenClaw-RL/...`; do not copy
models, environments, datasets, logs, checkpoints, or output directories.

Before a later remote code edit, inspect only the affected remote file and
update this profile if a fixed path, package version, task name, or capacity
assumption changes. Do not repeat broad inventory or disk/GPU checks while this
profile remains current.
