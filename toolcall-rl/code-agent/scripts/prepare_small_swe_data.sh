#!/usr/bin/env bash
set -euo pipefail

# Small default data preparation for a 2xA100 / short-run Code Agent recipe.
# The commands stream only bounded candidate windows; they do not build SWE
# Docker images and do not start training.
CODE_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${CODE_AGENT_DIR}/data/manifests}"
SEED="${CODE_SWE_DATA_SEED:-20260811}"
TRAIN_SAMPLES="${CODE_SWE_TRAIN_SAMPLES:-100}"
EVAL_SAMPLES="${CODE_SWE_EVAL_SAMPLES:-50}"
export PYTHONPATH="${CODE_AGENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_DIR}"
python -m data.preprocess_swe \
  --source swe-gym \
  --num-samples "${TRAIN_SAMPLES}" \
  --seed "${SEED}" \
  --output "${OUTPUT_DIR}/swe_gym_train.jsonl"
python -m data.preprocess_swe \
  --source swe-bench-verified \
  --num-samples "${EVAL_SAMPLES}" \
  --seed "${SEED}" \
  --exclude-jsonl "${OUTPUT_DIR}/swe_gym_train.jsonl" \
  --output "${OUTPUT_DIR}/swe_bench_verified_eval.jsonl"
