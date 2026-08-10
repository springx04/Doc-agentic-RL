#!/usr/bin/env bash
set -euo pipefail

# This wrapper only adapts the already validated Qwen3-VL tool-agent launcher.
# It never downloads data, starts an RL update, or changes the project tool
# protocol.  Set OPENCLAW_EVAL_LAUNCHER on the server when the validated
# launcher has a different filename.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
EVAL_DATA=
OUTPUT_DIR=
MODEL=
OVERWRITE=0
LAUNCHER=${OPENCLAW_EVAL_LAUNCHER:-${REPO_ROOT}/toolcall-rl/run_qwen3_vl_4b_real_docvqa_test_05.py}

usage() {
  cat >&2 <<'EOF'
Usage: bash toolcall-rl/eval_benchmarks/run_eval_only.sh \
  --eval-data /abs/path/eval.jsonl \
  --output-dir /abs/path/output \
  --model /abs/path/model_or_checkpoint \
  [--launcher /abs/path/validated_launcher.py] [--overwrite]
EOF
}

while (($#)); do
  case "$1" in
    --eval-data)
      (($# >= 2)) || { usage; exit 2; }
      EVAL_DATA=$2
      shift 2
      ;;
    --output-dir)
      (($# >= 2)) || { usage; exit 2; }
      OUTPUT_DIR=$2
      shift 2
      ;;
    --model)
      (($# >= 2)) || { usage; exit 2; }
      MODEL=$2
      shift 2
      ;;
    --launcher)
      (($# >= 2)) || { usage; exit 2; }
      LAUNCHER=$2
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "$EVAL_DATA" && -n "$OUTPUT_DIR" && -n "$MODEL" ]] || { usage; exit 2; }
[[ -f "$EVAL_DATA" ]] || { echo "eval data does not exist: $EVAL_DATA" >&2; exit 1; }
[[ -e "$MODEL" ]] || { echo "model/checkpoint does not exist: $MODEL" >&2; exit 1; }
[[ -f "$LAUNCHER" ]] || {
  echo "validated eval launcher does not exist: $LAUNCHER" >&2
  echo "Set OPENCLAW_EVAL_LAUNCHER or pass --launcher after the main process has uploaded it." >&2
  exit 1
}

ARTIFACT=${OUTPUT_DIR}/dump_details/rollout_data/eval_0.pt
if [[ -e "$ARTIFACT" && "$OVERWRITE" != 1 ]]; then
  echo "refusing to overwrite existing rollout artifact: $ARTIFACT" >&2
  echo "Pass --overwrite explicitly for a new run." >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

# The current real-data launcher consumes these variables.  Keeping both the
# generic and BayesTool names makes the wrapper usable with the validated
# baseline bridge as well as the current Qwen3-VL launcher.
export OPENCLAW_BAYESTOOL_EVAL_DATA="$EVAL_DATA"
export OPENCLAW_BAYESTOOL_OUTPUT_DIR="$OUTPUT_DIR"
export OPENCLAW_BAYESTOOL_NUM_ROLLOUT=0
export OPENCLAW_BAYESTOOL_EVAL_SAMPLES_PER_PROMPT=1
export NUM_ROLLOUT=0
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
export OPENCLAW_BAYESTOOL_SAMPLES_PER_PROMPT="${OPENCLAW_BAYESTOOL_SAMPLES_PER_PROMPT:-$N_SAMPLES_PER_PROMPT}"
export N_SAMPLES_PER_EVAL_PROMPT=1
export OPENCLAW_BAYESTOOL_EXPECT_EVAL_ROWS="${OPENCLAW_BAYESTOOL_EXPECT_EVAL_ROWS:-$(awk 'NF { n += 1 } END { print n + 0 }' "$EVAL_DATA")}"
export BAYESTOOL_OUTPUT_DIR="${BAYESTOOL_OUTPUT_DIR:-$OUTPUT_DIR}"
export SAVE_CKPT="${SAVE_CKPT:-$OUTPUT_DIR/checkpoints}"
export HF_CKPT="$MODEL"
export REF_LOAD="${REF_LOAD:-$MODEL}"
export EVAL_DATA="$EVAL_DATA"
export OUTPUT_DIR="$OUTPUT_DIR"

cd "$REPO_ROOT"
case "$LAUNCHER" in
  *.sh)
    bash "$LAUNCHER"
    ;;
  *.py)
    "$PYTHON_BIN" -B "$LAUNCHER"
    ;;
  *)
    echo "launcher must be a .py or .sh file: $LAUNCHER" >&2
    exit 2
    ;;
esac

if [[ ! -f "$ARTIFACT" ]]; then
  echo "eval-only launcher finished without required artifact: $ARTIFACT" >&2
  exit 1
fi
echo "eval-only rollout artifact: $ARTIFACT"
