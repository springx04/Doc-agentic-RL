#!/usr/bin/env bash

# Two-GPU BayesTool-RL launcher for Qwen3-VL-4B-Instruct.
# BayesTool defaults to the FSDP actor path; Megatron remains an explicit
# fail-fast incompatibility until it consumes the question manifest/weights.
# The bridge reads the official Qwen3-VL vision tower, MRoPE, and image token
# configuration from the local HF checkpoint; this intentionally does not
# source the text-only qwen3-4B model definition.

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

NUM_GPUS=${NUM_GPUS:-2}
ACTOR_GPUS=${ACTOR_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-1}
TRAIN_BACKEND=${TRAIN_BACKEND:-fsdp}
if (( NUM_GPUS <= 0 || ACTOR_GPUS <= 0 || ROLLOUT_GPUS <= 0 )); then
    echo "NUM_GPUS, ACTOR_GPUS, and ROLLOUT_GPUS must all be positive" >&2
    exit 1
fi
if (( ACTOR_GPUS + ROLLOUT_GPUS > NUM_GPUS )); then
    echo "ACTOR_GPUS + ROLLOUT_GPUS must be <= NUM_GPUS" >&2
    echo "actor=${ACTOR_GPUS}, rollout=${ROLLOUT_GPUS}, total=${NUM_GPUS}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../slime" >/dev/null 2>&1 && pwd)"
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"${PROJECT_DIR}/Megatron-LM"}
DEFAULT_MODEL_DIR=${DEFAULT_MODEL_DIR:-"${PROJECT_DIR}/models/Qwen3-VL-4B-Instruct"}
SGLANG_SOURCE_DIR=${SGLANG_SOURCE_DIR:-"${SCRIPT_DIR}/../third_party/sglang/python"}
if [[ -d "${SGLANG_SOURCE_DIR}/sglang" ]]; then
    SGLANG_SOURCE_DIR="$(cd -- "${SGLANG_SOURCE_DIR}" >/dev/null 2>&1 && pwd)"
else
    # Local checkouts may rely on an already-installed SGLang package; the
    # server deployment keeps the source tree under third_party/sglang.
    SGLANG_SOURCE_DIR=""
fi

# Resolve the Python and Ray executables from the same runtime environment.
# Non-interactive server jobs do not necessarily activate the conda env, and
# falling back to /opt/conda/bin/python3 would mix incompatible packages.
PYTHON_BIN=${PYTHON_BIN:-${PYTHON:-python3}}
if [[ "${PYTHON_BIN}" != */* ]]; then
    PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable is not available: ${PYTHON_BIN}" >&2
    exit 1
fi
PYTHON_ENV_BIN="$(dirname -- "$(readlink -f "${PYTHON_BIN}")")"
export PATH="${PYTHON_ENV_BIN}:${PATH}"

# cv2 used by the offline OCR backends is a GUI-enabled wheel.  The server
# image already contains its GL dependencies under /opt/conda; expose those
# existing files to the launcher and Ray workers without installing anything.
OCR_LIBRARY_DIR=${OPENCLAW_OCR_LIBRARY_DIR:-}
OCR_LIBRARY_CANDIDATES=()
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    OCR_LIBRARY_CANDIDATES+=("${CONDA_PREFIX}/lib")
fi
OCR_LIBRARY_CANDIDATES+=(
    "/opt/conda/lib"
    "/opt/conda/pkgs/libgl-1.7.0-ha4b6fd6_2/lib"
    "/opt/conda/pkgs/libglib-2.86.0-h1fed272_0/lib"
)
if [[ -z "${OCR_LIBRARY_DIR}" ]]; then
    for candidate in "${OCR_LIBRARY_CANDIDATES[@]}"; do
        if [[ -f "${candidate}/libGL.so.1" && -f "${candidate}/libgthread-2.0.so.0" ]]; then
            OCR_LIBRARY_DIR="${candidate}"
            break
        fi
    done
fi
if [[ -n "${OCR_LIBRARY_DIR}" && -d "${OCR_LIBRARY_DIR}" ]]; then
    export OPENCLAW_OCR_LIBRARY_DIR="${OCR_LIBRARY_DIR}"
    export LD_LIBRARY_PATH="${OCR_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export OPENCLAW_OCR_OFFLINE=${OPENCLAW_OCR_OFFLINE:-1}
export OPENCLAW_OCR_ALLOW_EASYOCR_DOWNLOAD=0
export OPENCLAW_OCR_AUTO_BACKENDS=${OPENCLAW_OCR_AUTO_BACKENDS:-rapidocr,rapidocr_torch,paddleocr,easyocr}
RAY_BIN=${RAY_BIN:-"${PYTHON_ENV_BIN}/ray"}
if [[ ! -x "${RAY_BIN}" ]]; then
    RAY_BIN="$(command -v ray || true)"
fi
if [[ -z "${RAY_BIN}" || ! -x "${RAY_BIN}" ]]; then
    echo "Ray CLI is not available in the Python runtime: ${PYTHON_BIN}" >&2
    exit 1
fi

# Keep a dedicated Qwen3-VL configuration.  The official Megatron-Bridge
# provider loads the vision tower, MRoPE metadata, image/video token IDs, and
# multimodal processor behavior from HF_CKPT; this launcher must not fall back
# to the text-only qwen3-4B model definition.
source "${SLIME_DIR}/scripts/models/qwen3-vl-4B.sh"

HF_CKPT=${HF_CKPT:-${DEFAULT_MODEL_DIR}}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-${PROJECT_DIR}/outputs/qwen3-vl-4b-bayestool-rl}
if [[ -z "${RESUME_LOAD:-}" ]]; then
    if [[ -f "${SAVE_CKPT}/latest_checkpointed_iteration.txt" ]]; then
        RESUME_LOAD="${SAVE_CKPT}"
    else
        # A fresh run must initialize from the HF bridge instead of passing a
        # nonexistent output directory to Megatron's checkpoint loader.
        RESUME_LOAD="${HF_CKPT}"
    fi
fi
BAYESTOOL_OUTPUT_DIR=${BAYESTOOL_OUTPUT_DIR:-${SAVE_CKPT}/bayestool_outputs}
export OPENCLAW_TOOL_OUTPUT_DIR=${OPENCLAW_TOOL_OUTPUT_DIR:-${BAYESTOOL_OUTPUT_DIR}}
PROMPT_DATA=${PROMPT_DATA:-${PROJECT_DIR}/data/document-qa/train.jsonl}
EVAL_DATA=${EVAL_DATA:-${PROJECT_DIR}/data/document-qa/eval.jsonl}
for required_path in "${HF_CKPT}" "${PROMPT_DATA}" "${EVAL_DATA}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required project input is missing: ${required_path}" >&2
        echo "Set HF_CKPT, PROMPT_DATA, and EVAL_DATA explicitly when using an external dataset/model." >&2
        exit 1
    fi
done
BAYESTOOL_STAGE=${BAYESTOOL_STAGE:-c}
case "${BAYESTOOL_STAGE}" in
    a) BAYESTOOL_BRANCH_PROBABILITY=${BAYESTOOL_BRANCH_PROBABILITY:-0.0} ;;
    b) BAYESTOOL_BRANCH_PROBABILITY=${BAYESTOOL_BRANCH_PROBABILITY:-0.10} ;;
    c) BAYESTOOL_BRANCH_PROBABILITY=${BAYESTOOL_BRANCH_PROBABILITY:-0.25} ;;
    d) BAYESTOOL_BRANCH_PROBABILITY=${BAYESTOOL_BRANCH_PROBABILITY:-0.25} ;;
    *) echo "BAYESTOOL_STAGE must be one of a, b, c, d" >&2; exit 1 ;;
esac

# Stage A is an executable replay/training stage, not a policy-RL run with
# branching disabled.  It consumes a completed rollout artifact and produces
# the canonical replay, belief/smoother checkpoint, and one capability
# manifest that later stages can reference.  Refuse to continue without the
# artifact so an accidental Stage-A launch cannot masquerade as training.
if [[ "${BAYESTOOL_STAGE}" == "a" ]]; then
    if [[ -z "${BAYESTOOL_STAGE_A_INPUT:-}" || ! -f "${BAYESTOOL_STAGE_A_INPUT}" ]]; then
        echo "Stage A requires BAYESTOOL_STAGE_A_INPUT=.../rollout_interactions.json" >&2
        exit 1
    fi
    STAGE_A_OUTPUT_DIR=${BAYESTOOL_STAGE_A_OUTPUT_DIR:-${SAVE_CKPT}/bayestool_stage_a}
    STAGE_A_ARGS=(
        --artifact "${BAYESTOOL_STAGE_A_INPUT}"
        --output-dir "${STAGE_A_OUTPUT_DIR}"
        --epochs "${BAYESTOOL_STAGE_A_EPOCHS:-1}"
        --device "${BAYESTOOL_STAGE_A_DEVICE:-cpu}"
    )
    if [[ "${BAYESTOOL_STAGE_A_FIT_Q:-0}" == "1" ]]; then
        STAGE_A_ARGS+=(--fit-q)
    fi
    if [[ "${BAYESTOOL_STAGE_A_FIT_RISK:-0}" == "1" ]]; then
        STAGE_A_ARGS+=(--fit-risk)
    fi
    if [[ -n "${BAYESTOOL_STAGE_A_RISK_LABEL_KEY:-}" ]]; then
        STAGE_A_ARGS+=(--risk-label-key "${BAYESTOOL_STAGE_A_RISK_LABEL_KEY}")
    fi
    exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_bayestool_stage_a.py" "${STAGE_A_ARGS[@]}"
fi

# The defaults below are the full training configuration.  The overrides are
# intentionally environment-based so server smoke runs can bound work without
# creating a second, simplified training implementation.
SAVE_INTERVAL=${SAVE_INTERVAL:-20}
NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
BAYESTOOL_GROUP_SIZE=${BAYESTOOL_GROUP_SIZE:-4}
BAYESTOOL_REALIZATIONS=${BAYESTOOL_REALIZATIONS:-4}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-${BAYESTOOL_REALIZATIONS}}
if (( BAYESTOOL_GROUP_SIZE != 4 && BAYESTOOL_GROUP_SIZE != 8 )); then
    echo "BayesTool requires K=4 or K=8, got ${BAYESTOOL_GROUP_SIZE}" >&2
    exit 1
fi
if (( BAYESTOOL_REALIZATIONS < 4 || BAYESTOOL_REALIZATIONS > 6 || N_SAMPLES_PER_PROMPT != BAYESTOOL_REALIZATIONS )); then
    echo "BayesTool requires an explicit 4-6 primary realization plan; got K=${BAYESTOOL_GROUP_SIZE}, R=${BAYESTOOL_REALIZATIONS}, primary_samples=${N_SAMPLES_PER_PROMPT}" >&2
    exit 1
fi
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-16384}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1}
EVAL_INTERVAL=${EVAL_INTERVAL:-20}
N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-8}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-8192}
EVAL_MAX_CONTEXT_LEN=${EVAL_MAX_CONTEXT_LEN:-16384}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.6}
# The actual Qwen3-VL packed THD layout (`thd_thd_thd` with padding-causal
# masking) is not accepted by the cuDNN fused sub-backend in TE 2.12.0 on the
# target A100 runtime.  TE's exact UnfusedDotProductAttention path is complete
# and keeps packed training semantics intact; use FlashAttention explicitly
# after installing and validating its CUDA extension.
ATTENTION_BACKEND=${ATTENTION_BACKEND:-unfused}
NVTE_DEBUG=${NVTE_DEBUG:-0}
NVTE_DEBUG_LEVEL=${NVTE_DEBUG_LEVEL:-0}

CKPT_ARGS=(
    --megatron-to-hf-mode bridge
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${REF_LOAD}"
    --load "${RESUME_LOAD}"
    --save "${SAVE_CKPT}"
    --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
    --train-backend "${TRAIN_BACKEND}"
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --apply-chat-template
    --rollout-shuffle
    --reward-key score
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --balance-data
)

EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data docqa "${EVAL_DATA}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}"
    --eval-top-p 1
    --eval-reward-key quality
)

BAYESTOOL_ARGS=(
    --advantage-estimator bayes_grpo
    --bayestool-enable
    --bayestool-group-size "${BAYESTOOL_GROUP_SIZE}"
    --bayestool-worlds-per-prompt "${BAYESTOOL_REALIZATIONS}"
    --bayestool-replicas-per-world 1
    --bayestool-posterior-particles 8
    --bayestool-max-action-candidates "${BAYESTOOL_GROUP_SIZE}"
    --bayestool-max-siblings "${BAYESTOOL_GROUP_SIZE}"
    --bayestool-branch-horizon 3
    --bayestool-branch-probability "${BAYESTOOL_BRANCH_PROBABILITY}"
    --bayestool-consensus-threshold 0.75
    --bayestool-decision-regret-threshold 0.08
    --bayestool-local-surprise-threshold 4.0
    --bayestool-family-surprise-threshold 6.0
    --bayestool-global-surprise-threshold 8.0
    --bayestool-change-probability-threshold 0.80
    --bayestool-max-belief-prompt-tokens 1200
    --bayestool-stage "${BAYESTOOL_STAGE}"
    --bayestool-aux-interval 2
    --bayestool-max-switch-bundles-per-rank 8
    --bayestool-max-preinv-bundles-per-rank 8
    --bayestool-switch-loss-weight 0.20
    --bayestool-preinv-loss-weight 0.05
    --bayestool-aux-micro-batch-size 4
    --disable-grpo-std-normalization
    --use-kl-loss
    --kl-loss-coef 0.01
    --kl-loss-type k3
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

if [[ -n "${BAYESTOOL_BELIEF_CHECKPOINT:-}" ]]; then
    BAYESTOOL_ARGS+=(--bayestool-belief-checkpoint "${BAYESTOOL_BELIEF_CHECKPOINT}")
fi
if [[ -n "${BAYESTOOL_Q_CHECKPOINT:-}" ]]; then
    BAYESTOOL_ARGS+=(--bayestool-q-checkpoint "${BAYESTOOL_Q_CHECKPOINT}")
fi
if [[ -n "${BAYESTOOL_RISK_CHECKPOINT:-}" ]]; then
    BAYESTOOL_ARGS+=(--bayestool-risk-checkpoint "${BAYESTOOL_RISK_CHECKPOINT}")
fi
if [[ -n "${BAYESTOOL_META_MANIFEST:-}" ]]; then
    BAYESTOOL_ARGS+=(--bayestool-meta-manifest "${BAYESTOOL_META_MANIFEST}")
fi
if [[ "${BAYESTOOL_ALLOW_HEURISTIC_BELIEF:-0}" == "1" ]]; then
    BAYESTOOL_ARGS+=(--bayestool-allow-heuristic-belief)
fi
if [[ "${BAYESTOOL_ALLOW_HEURISTIC_Q:-0}" == "1" ]]; then
    BAYESTOOL_ARGS+=(--bayestool-allow-heuristic-q)
fi
if [[ "${BAYESTOOL_ALLOW_HEURISTIC_RISK:-0}" == "1" ]]; then
    BAYESTOOL_ARGS+=(--bayestool-allow-heuristic-risk)
fi
if [[ -n "${BAYESTOOL_WORLD_TYPE_PROBABILITIES:-}" ]]; then
    BAYESTOOL_ARGS+=(--bayestool-world-type-probabilities "${BAYESTOOL_WORLD_TYPE_PROBABILITIES}")
fi
if [[ -n "${BAYESTOOL_SESSION_STATE_PROBABILITIES:-}" ]]; then
    BAYESTOOL_ARGS+=(--bayestool-session-state-probabilities "${BAYESTOOL_SESSION_STATE_PROBABILITIES}")
fi

PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend "${ATTENTION_BACKEND}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

CUSTOM_ARGS=(
    --custom-generate-function-path generate_with_retool.generate
    --custom-rm-path generate_with_retool.reward_func
)

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_JOB_ADDRESS=${RAY_JOB_ADDRESS:-http://127.0.0.1:${RAY_DASHBOARD_PORT}}
RAY_DASHBOARD_AGENT_PORT=${RAY_DASHBOARD_AGENT_PORT:-$((RAY_DASHBOARD_PORT + 100))}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-20000}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-29999}
if (( RAY_MIN_WORKER_PORT > RAY_MAX_WORKER_PORT )); then
    echo "RAY_MIN_WORKER_PORT must not exceed RAY_MAX_WORKER_PORT" >&2
    exit 1
fi
if (( RAY_PORT >= RAY_MIN_WORKER_PORT && RAY_PORT <= RAY_MAX_WORKER_PORT )); then
    echo "RAY_PORT must be outside the configured Ray worker port range" >&2
    exit 1
fi
if (( RAY_DASHBOARD_PORT >= RAY_MIN_WORKER_PORT && RAY_DASHBOARD_PORT <= RAY_MAX_WORKER_PORT )); then
    echo "RAY_DASHBOARD_PORT must be outside the configured Ray worker port range" >&2
    exit 1
fi
if (( RAY_DASHBOARD_AGENT_PORT >= RAY_MIN_WORKER_PORT && RAY_DASHBOARD_AGENT_PORT <= RAY_MAX_WORKER_PORT )); then
    echo "RAY_DASHBOARD_AGENT_PORT must be outside the configured Ray worker port range" >&2
    exit 1
fi
if (( RAY_DASHBOARD_AGENT_PORT == RAY_PORT || RAY_DASHBOARD_AGENT_PORT == RAY_DASHBOARD_PORT )); then
    echo "RAY_DASHBOARD_AGENT_PORT must be distinct from Ray head/dashboard ports" >&2
    exit 1
fi

HAS_NVLINK=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi topo -m 2>/dev/null | grep -q 'NV'; then
    HAS_NVLINK=1
fi

RUNTIME_PYTHONPATH="${MEGATRON_LM_PATH}:${SCRIPT_DIR}:${SLIME_DIR}"
if [[ -n "${SGLANG_SOURCE_DIR}" ]]; then
    RUNTIME_PYTHONPATH="${SGLANG_SOURCE_DIR}:${RUNTIME_PYTHONPATH}"
fi

"${RAY_BIN}" start --head --node-ip-address "${MASTER_ADDR}" --port "${RAY_PORT}" --num-gpus "${NUM_GPUS}" \
    --min-worker-port="${RAY_MIN_WORKER_PORT}" --max-worker-port="${RAY_MAX_WORKER_PORT}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_PORT}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH:-}\",
    \"OPENCLAW_OCR_LIBRARY_DIR\": \"${OPENCLAW_OCR_LIBRARY_DIR:-}\",
    \"OPENCLAW_OCR_OFFLINE\": \"${OPENCLAW_OCR_OFFLINE}\",
    \"OPENCLAW_OCR_ALLOW_EASYOCR_DOWNLOAD\": \"0\",
    \"OPENCLAW_OCR_AUTO_BACKENDS\": \"${OPENCLAW_OCR_AUTO_BACKENDS}\",
    \"NVTE_DEBUG\": \"${NVTE_DEBUG}\",
    \"NVTE_DEBUG_LEVEL\": \"${NVTE_DEBUG_LEVEL}\"
  }
}"

"${RAY_BIN}" job submit --address="${RAY_JOB_ADDRESS}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${PYTHON_BIN}" "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${BAYESTOOL_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"
