"""Run a real-data Qwen3-VL RL update with dynamic tool images bridged to Qwen3-VL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# Do not enable ``expandable_segments`` here: SGLang's colocated
# TorchMemorySaver rejects that allocator mode during engine startup.  The
# primary OOM protection is the bounded training trajectory in
# ``generate_with_retool.py`` plus the dynamic-batch diagnostics in the actor.

import run_qwen3_vl_4b_smoke as base
from bayestool.config import validate_stage_capabilities


DATA_DIR = (base.PROJECT / "data").resolve()
SUBSET_DIR = DATA_DIR / "real_docvqa_rl_test_20260717-01"
FULL_DATA_DIR = DATA_DIR / "document-qa"
REAL_TRAIN_DATA = Path(
    os.environ.get("OPENCLAW_BAYESTOOL_TRAIN_DATA", str(FULL_DATA_DIR / "train.jsonl"))
).resolve()
REAL_EVAL_DATA = Path(
    os.environ.get("OPENCLAW_BAYESTOOL_EVAL_DATA", str(FULL_DATA_DIR / "eval.jsonl"))
).resolve()
STAGE_A_DIR = (
    base.PROJECT / "artifacts" / "bayestool-stage-a-20260809-fulltrain" / "checkpoints"
).resolve()
DEFAULT_CHECKPOINTS = {
    "BAYESTOOL_BELIEF_CHECKPOINT": STAGE_A_DIR / "belief_filter.pt",
    "BAYESTOOL_Q_CHECKPOINT": STAGE_A_DIR / "bayes_q_head.pt",
    "BAYESTOOL_RISK_CHECKPOINT": STAGE_A_DIR / "answer_risk.json",
}
REAL_OUTPUT_DIR = Path(os.environ.get("OPENCLAW_BAYESTOOL_OUTPUT_DIR", str(base.PROJECT / "outputs" / "qwen3-vl-4b-docvqa-bayestool-rl-20260807-13"))).resolve()
# Ray creates Unix-domain sockets below this directory.  Keep the default
# deliberately short: the project path may be nested deeply on a mounted
# workspace and can otherwise exceed the Linux 107-byte AF_UNIX limit before
# training even starts.  Operators can still override it for a dedicated
# per-run directory.
REAL_RAY_TEMP_DIR = Path(
    os.environ.get("OPENCLAW_BAYESTOOL_RAY_TEMP_DIR", str(base.WORKSPACE / "ocrl-ray"))
).resolve()
_BASE_TRAINING_ARGV = base.training_argv


def _replace_value(argv: list[str], option: str, value: str) -> None:
    index = argv.index(option)
    argv[index + 1] = value


def _configure() -> None:
    base.TRAIN_DATA = REAL_TRAIN_DATA
    base.EVAL_DATA = REAL_EVAL_DATA
    base.OUTPUT_DIR = REAL_OUTPUT_DIR
    base.CHECKPOINT_DIR = REAL_OUTPUT_DIR / "checkpoints"
    base.RAY_TEMP_DIR = REAL_RAY_TEMP_DIR
    base.DOCUMENT_ROOT = DATA_DIR
    base.DOCUMENT_PROBE = None


def _training_argv() -> list[str]:
    argv = _BASE_TRAINING_ARGV()
    argv.remove("--disable-rewards-normalization")
    # The legacy launcher formula ties global_batch_size to the unexpanded
    # primary count.  BayesTool's explicit manifest owns optimizer batching;
    # remove that option before the strict argument validator sees it.
    if "--num-steps-per-rollout" in argv:
        index = argv.index("--num-steps-per-rollout")
        del argv[index : index + 2]
    _replace_value(argv, "--rollout-batch-size", os.environ.get("OPENCLAW_BAYESTOOL_ROLLOUT_BATCH_SIZE", "2"))
    # Four mandatory realization roles x K independent continuations.  The
    # legacy replica fields below are compatibility metadata only.
    group_size = int(os.environ.get("OPENCLAW_BAYESTOOL_GROUP_SIZE", "4"))
    realization_count = int(os.environ.get("OPENCLAW_BAYESTOOL_REALIZATIONS", "4"))
    if group_size not in {4, 8} or not 4 <= realization_count <= 6:
        raise ValueError(
            f"invalid BayesTool plan defaults: realization_count={realization_count}, K={group_size}"
        )
    samples_per_prompt = os.environ.get(
        "OPENCLAW_BAYESTOOL_SAMPLES_PER_PROMPT",
        str(realization_count),
    )
    if int(samples_per_prompt) != realization_count:
        raise ValueError(
            "OPENCLAW_BAYESTOOL_SAMPLES_PER_PROMPT must equal the explicit "
            f"realization count ({realization_count}); K siblings are expanded "
            "from each primary checkpoint, not requested as primary samples."
        )
    eval_samples_per_prompt = os.environ.get("OPENCLAW_BAYESTOOL_EVAL_SAMPLES_PER_PROMPT", "1")
    _replace_value(argv, "--n-samples-per-prompt", samples_per_prompt)
    _replace_value(argv, "--n-samples-per-eval-prompt", eval_samples_per_prompt)
    _replace_value(argv, "--global-batch-size", os.environ.get("OPENCLAW_BAYESTOOL_GLOBAL_BATCH_SIZE", "32"))
    _replace_value(argv, "--num-rollout", os.environ.get("OPENCLAW_BAYESTOOL_NUM_ROLLOUT", "1"))
    _replace_value(argv, "--eval-interval", os.environ.get("OPENCLAW_BAYESTOOL_EVAL_INTERVAL", "1"))
    # The actor gates actual writes by valid-question count.  Calling the
    # save hook every rollout lets it hit exactly 100 valid questions even
    # when a rollout contains skipped/invalid questions; the hook is cheap
    # when the threshold has not been reached.
    if "--save-interval" in argv:
        _replace_value(argv, "--save-interval", os.environ.get("OPENCLAW_BAYESTOOL_SAVE_INTERVAL", "1"))
    _replace_value(argv, "--advantage-estimator", "bayes_grpo")
    response_len = os.environ.get("OPENCLAW_BAYESTOOL_RESPONSE_LEN", "512")
    _replace_value(argv, "--rollout-max-response-len", response_len)
    _replace_value(argv, "--eval-max-response-len", response_len)
    # Real DocVQA tool observations can span several turns/pages.  4096
    # leaves too little room for the configured 512-token response and causes
    # otherwise valid degraded-world rollouts to be discarded as
    # ``context_overflow``.  Keep the setting overrideable, but make the
    # real-data default large enough for the full tool trace.
    context_len = os.environ.get("OPENCLAW_BAYESTOOL_CONTEXT_LEN", "8192")
    _replace_value(argv, "--rollout-max-context-len", context_len)
    _replace_value(argv, "--eval-max-context-len", context_len)
    _replace_value(argv, "--max-tokens-per-gpu", os.environ.get("OPENCLAW_BAYESTOOL_MAX_TOKENS_PER_GPU", "2048"))
    load_checkpoint = os.environ.get("OPENCLAW_BAYESTOOL_LOAD")
    if load_checkpoint:
        argv.extend(["--load", str(Path(load_checkpoint).expanduser().resolve())])
    stage = os.environ.get("OPENCLAW_BAYESTOOL_STAGE", "c")
    argv.extend(
        [
            "--bayestool-enable",
            "--bayestool-group-size", str(group_size),
            "--bayestool-worlds-per-prompt", os.environ.get("OPENCLAW_BAYESTOOL_WORLDS_PER_PROMPT", str(realization_count)),
            "--bayestool-replicas-per-world", os.environ.get("OPENCLAW_BAYESTOOL_REPLICAS_PER_WORLD", "1"),
            "--bayestool-questions-per-step", os.environ.get("OPENCLAW_BAYESTOOL_QUESTIONS_PER_STEP", "2"),
            "--bayestool-max-questions-per-step", os.environ.get("OPENCLAW_BAYESTOOL_MAX_QUESTIONS_PER_STEP", "8"),
            "--bayestool-stage", stage,
            "--bayestool-branch-probability", os.environ.get("OPENCLAW_BAYESTOOL_BRANCH_PROBABILITY", "1.0"),
            "--bayestool-decision-regret-threshold", os.environ.get("OPENCLAW_BAYESTOOL_REGRET_THRESHOLD", "-1.0"),
            "--bayestool-max-action-candidates", str(group_size),
            "--bayestool-max-siblings", str(group_size),
            "--bayestool-branch-horizon", "3",
            "--bayestool-checkpoint-interval-questions",
            os.environ.get("OPENCLAW_BAYESTOOL_CHECKPOINT_INTERVAL_QUESTIONS", "100"),
            "--bayestool-checkpoint-retention",
            os.environ.get("OPENCLAW_BAYESTOOL_CHECKPOINT_RETENTION", "2"),
        ]
    )
    checkpoint_options = (
        ("--bayestool-belief-checkpoint", "BAYESTOOL_BELIEF_CHECKPOINT"),
        ("--bayestool-q-checkpoint", "BAYESTOOL_Q_CHECKPOINT"),
        ("--bayestool-risk-checkpoint", "BAYESTOOL_RISK_CHECKPOINT"),
        ("--bayestool-meta-manifest", "BAYESTOOL_META_MANIFEST"),
    )
    for option, variable in checkpoint_options:
        value = os.environ.get(variable)
        if not value and stage in {"b", "c", "d"} and variable in DEFAULT_CHECKPOINTS:
            value = str(DEFAULT_CHECKPOINTS[variable])
        if value:
            argv.extend([option, value])
    for option, variable in (
        ("--bayestool-allow-heuristic-belief", "BAYESTOOL_ALLOW_HEURISTIC_BELIEF"),
        ("--bayestool-allow-heuristic-q", "BAYESTOOL_ALLOW_HEURISTIC_Q"),
        ("--bayestool-allow-heuristic-risk", "BAYESTOOL_ALLOW_HEURISTIC_RISK"),
    ):
        if os.environ.get(variable, "0") == "1":
            argv.append(option)
    if os.environ.get("OPENCLAW_BAYESTOOL_GRADIENT_CHECKPOINTING", "1") == "1":
        argv.append("--gradient-checkpointing")
    eval_index = argv.index("--eval-prompt-data")
    argv[eval_index + 1] = "real_docvqa_test_with_dynamic_tool_images"
    return argv


def _validate() -> dict[str, Any]:
    # Ray appends a timestamp/pid session directory and the plasma socket to
    # this path.  Check the worst-case shape before starting Ray so a long run
    # name fails in preflight instead of leaving a half-created output tree.
    ray_socket_suffix = "/session_YYYY-MM-DD_HH-MM-SS_ffffff_99999999/sockets/plasma_store"
    ray_temp_text = str(REAL_RAY_TEMP_DIR)
    if len(ray_temp_text) + len(ray_socket_suffix) >= 107:
        raise RuntimeError(
            "OPENCLAW_BAYESTOOL_RAY_TEMP_DIR is too long for Ray's AF_UNIX "
            f"socket path ({len(ray_temp_text) + len(ray_socket_suffix)} bytes); "
            "use a short directory such as /workspace/data/.ray/bt-r8k4-100"
        )
    required_model_paths = (
        base.DOCLING_ARTIFACTS_DIR / "docling-project--docling-layout-heron",
        base.DOCLING_ARTIFACTS_DIR / "docling-project--docling-models",
        base.TOOL_MODELS_DIR / "deplot" / "config.json",
        base.TOOL_MODELS_DIR / "deplot" / "model.safetensors",
    )
    missing = [str(path) for path in required_model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("uploaded tool model bundle is incomplete: " + json.dumps(missing))
    report = _base_validate()
    stage = os.environ.get("OPENCLAW_BAYESTOOL_STAGE", "c")
    capability_paths = {
        variable: os.environ.get(variable, str(DEFAULT_CHECKPOINTS[variable]))
        for variable in DEFAULT_CHECKPOINTS
        if stage in {"b", "c", "d"}
    }
    report["bayestool_capabilities"] = validate_stage_capabilities(
        stage,
        belief_checkpoint=capability_paths.get("BAYESTOOL_BELIEF_CHECKPOINT"),
        q_checkpoint=capability_paths.get("BAYESTOOL_Q_CHECKPOINT"),
        risk_checkpoint=capability_paths.get("BAYESTOOL_RISK_CHECKPOINT"),
        meta_manifest=os.environ.get("BAYESTOOL_META_MANIFEST"),
        allow_heuristic_belief=os.environ.get("BAYESTOOL_ALLOW_HEURISTIC_BELIEF", "0") == "1",
        allow_heuristic_q=os.environ.get("BAYESTOOL_ALLOW_HEURISTIC_Q", "0") == "1",
        allow_heuristic_risk=os.environ.get("BAYESTOOL_ALLOW_HEURISTIC_RISK", "0") == "1",
    )
    expected_train_rows = int(os.environ.get("OPENCLAW_BAYESTOOL_EXPECT_TRAIN_ROWS", "1000"))
    expected_eval_rows = int(os.environ.get("OPENCLAW_BAYESTOOL_EXPECT_EVAL_ROWS", "200"))
    if report["train_rows"] != expected_train_rows or report["eval_rows"] != expected_eval_rows:
        raise RuntimeError(
            f"real-data run requires exactly {expected_train_rows} train rows and "
            f"{expected_eval_rows} evaluation rows"
        )
    load_checkpoint = os.environ.get("OPENCLAW_BAYESTOOL_LOAD")
    load_path = Path(load_checkpoint).expanduser().resolve() if load_checkpoint else None
    if load_path is not None:
        if not load_path.is_dir():
            raise FileNotFoundError(f"requested checkpoint directory does not exist: {load_path}")
        has_checkpoint_marker = (load_path / "latest_checkpointed_iteration.txt").is_file()
        has_checkpoint_metadata = any(load_path.glob("iter_*/meta.json"))
        if not (has_checkpoint_marker or has_checkpoint_metadata):
            raise FileNotFoundError(
                f"requested checkpoint directory has no loadable checkpoint metadata: {load_path}"
            )
    selection_manifest = os.environ.get("OPENCLAW_BAYESTOOL_SELECTION_MANIFEST")
    report.update(
        {
            "profile": os.environ.get("OPENCLAW_BAYESTOOL_PROFILE", "real_docvqa_full_20260807-13"),
            "output_dir": str(REAL_OUTPUT_DIR),
            "load_checkpoint": str(load_path) if load_path is not None else None,
            "source_dataset": "nielsr/docvqa_1200_examples",
            "selection_manifest": selection_manifest,
            "train_data": str(REAL_TRAIN_DATA),
            "eval_data": str(REAL_EVAL_DATA),
            "tool_models_dir": str(base.TOOL_MODELS_DIR),
            "docling_artifacts_dir": str(base.DOCLING_ARTIFACTS_DIR),
            "dynamic_tool_images": True,
            "rewards_normalization": True,
            "gradient_checkpointing": os.environ.get("OPENCLAW_BAYESTOOL_GRADIENT_CHECKPOINTING", "1") == "1",
            "rollout_batch_size": int(os.environ.get("OPENCLAW_BAYESTOOL_ROLLOUT_BATCH_SIZE", "2")),
            "samples_per_prompt": int(
                os.environ.get(
                    "OPENCLAW_BAYESTOOL_SAMPLES_PER_PROMPT",
                    os.environ.get("OPENCLAW_BAYESTOOL_REALIZATIONS", "4"),
                )
            ),
            "eval_samples_per_prompt": int(os.environ.get("OPENCLAW_BAYESTOOL_EVAL_SAMPLES_PER_PROMPT", "1")),
            "global_batch_size": int(os.environ.get("OPENCLAW_BAYESTOOL_GLOBAL_BATCH_SIZE", "32")),
            "num_rollout": int(os.environ.get("OPENCLAW_BAYESTOOL_NUM_ROLLOUT", "1")),
            "eval_interval": int(os.environ.get("OPENCLAW_BAYESTOOL_EVAL_INTERVAL", "1")),
            "save_interval": int(os.environ.get("OPENCLAW_BAYESTOOL_SAVE_INTERVAL", "1")),
            "checkpoint_interval_questions": int(
                os.environ.get("OPENCLAW_BAYESTOOL_CHECKPOINT_INTERVAL_QUESTIONS", "100")
            ),
            "checkpoint_retention": int(os.environ.get("OPENCLAW_BAYESTOOL_CHECKPOINT_RETENTION", "2")),
            "max_train_sequence_length": int(
                os.environ.get("OPENCLAW_BAYESTOOL_MAX_TRAIN_SEQUENCE_LENGTH", "8192")
            ),
            "expected_train_rows": expected_train_rows,
            "expected_eval_rows": expected_eval_rows,
            "agent_detail_log_dir": str(REAL_OUTPUT_DIR / "dump_details"),
            "tool_output_log_dir": str(REAL_OUTPUT_DIR / "tool_outputs"),
        }
    )
    return report


_configure()
_base_validate = base.validate
base.training_argv = _training_argv
base.validate = _validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(base.validate(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    base.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
