#!/usr/bin/env bash
# Launch only after the isolated Code pool, dedicated output root, and two GPUs
# have been explicitly reserved.  This script never falls back to Doc paths.
set -euo pipefail

: "${CODE_TRAIN_MANIFEST:?set Code-only 100-row SWE-Gym JSONL}"
: "${CODE_EVALUATOR_MANIFEST:?set matching private evaluator JSONL}"
: "${CODE_OUTPUT_DIR:?set dedicated Code output directory}"
: "${CODE_ENV_SERVER_URL:?set dedicated Code pool URL}"
: "${CODE_SLIME_ROOT:?set path to the shared Slime checkout}"
: "${CODE_HF_CHECKPOINT:?set model checkpoint path}"

if [ "${CODE_NUM_ROLLOUTS:-25}" -ne 25 ] || [ "${CODE_ROLLOUT_BATCH_SIZE:-4}" -ne 4 ]; then
  echo "This launcher is the fixed 100-task recipe: require 25 rollouts x 4 tasks" >&2
  exit 2
fi

export PYTHONPATH="${CODE_SLIME_ROOT}:${CODE_SLIME_ROOT}/../toolcall-rl/code-agent${PYTHONPATH:+:${PYTHONPATH}}"
export CODE_EVALUATOR_MANIFEST
export CODE_STAGE="${CODE_STAGE:-C}"
export CODE_STAGE_C_GROUP_SIZE="${CODE_STAGE_C_GROUP_SIZE:-4}"
mkdir -p "${CODE_OUTPUT_DIR}" "${CODE_OUTPUT_DIR}/checkpoints" "${CODE_OUTPUT_DIR}/logs"

python "${CODE_SLIME_ROOT}/train.py" \
  --hf-checkpoint "${CODE_HF_CHECKPOINT}" \
  --prompt-data "${CODE_TRAIN_MANIFEST}" \
  --prompt-key text \
  --metadata-key metadata \
  --rollout-function-path slime.rollout.sglang_rollout.generate_rollout \
  --custom-generate-function-path slime_adapter.generate \
  --custom-rm-path slime_adapter.reward_func \
  --custom-convert-samples-to-train-data-path slime_train_data.convert_samples_to_train_data \
  --train-backend fsdp \
  --use-rollout-logprobs \
  --num-rollout "${CODE_NUM_ROLLOUTS:-25}" \
  --rollout-batch-size "${CODE_ROLLOUT_BATCH_SIZE:-4}" \
  --n-samples-per-prompt 1 \
  --advantage-estimator bayes_grpo \
  --global-batch-size "${CODE_GLOBAL_BATCH_SIZE:-64}" \
  --micro-batch-size "${CODE_MICRO_BATCH_SIZE:-1}" \
  --rollout-max-response-len "${CODE_MAX_RESPONSE_TOKENS:-4096}" \
  --save "${CODE_OUTPUT_DIR}/checkpoints" \
  --save-interval "${CODE_SAVE_INTERVAL:-5}" \
  "$@"
