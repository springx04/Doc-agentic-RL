#!/usr/bin/env bash
# Independent final evaluation: one clean evaluation per 50 held-out tasks.
set -euo pipefail

: "${CODE_EVAL_RUNTIME_MANIFEST:?set public 50-row eval runtime manifest}"
: "${CODE_EVALUATOR_MANIFEST:?set matching private evaluator manifest}"
: "${CODE_OUTPUT_DIR:?set dedicated Code output directory}"
: "${CODE_ENV_SERVER_URL:?set dedicated Code pool URL}"
: "${CODE_SLIME_ROOT:?set Slime checkout path}"
: "${CODE_HF_CHECKPOINT:?set final checkpoint path}"

CODE_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_GPU_COUNT="${CODE_GPU_COUNT:-2}"
export PYTHONPATH="${CODE_SLIME_ROOT}:${CODE_AGENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CODE_EVALUATOR_MANIFEST
export CODE_STAGE="EVAL"

python "${CODE_SLIME_ROOT}/eval_only.py" \
  --train-backend fsdp \
  --actor-num-gpus-per-node "${CODE_GPU_COUNT}" \
  --num-gpus-per-node "${CODE_GPU_COUNT}" \
  --rollout-num-gpus "${CODE_GPU_COUNT}" \
  --rollout-num-gpus-per-engine 1 \
  --colocate \
  --hf-checkpoint "${CODE_HF_CHECKPOINT}" \
  --eval-prompt-data code_swe_eval "${CODE_EVAL_RUNTIME_MANIFEST}" \
  --eval-input-key text \
  --metadata-key metadata \
  --n-samples-per-eval-prompt 1 \
  --eval-temperature 0 \
  --eval-max-response-len "${CODE_MAX_RESPONSE_TOKENS:-4096}" \
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
  --eval-function-path slime.rollout.sglang_rollout.generate_rollout \
  --custom-generate-function-path slime_adapter.generate \
  --custom-eval-rollout-log-function-path eval_adapter.log_eval_rollout_data \
  "$@"
