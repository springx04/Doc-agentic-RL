# Document Tool-Call RL

This directory trains language models to answer questions about local PDF,
DOCX, PPTX, image, Markdown, and text documents by selecting and chaining the
repository's document-understanding tools. The rollout/training backend is
slime + SGLang + Megatron.

## Default tools

| Tool | Purpose |
|---|---|
| `render_page` | Render a PDF page or image to PNG |
| `crop_region` | Crop a page/image region |
| `zoom_region` | Crop and enlarge a region |
| `parse_document` | Parse PDF/DOCX/PPTX/image/text to Markdown/JSON |
| `detect_layout` | Locate paragraphs, titles, tables, figures, and formulas |
| `ocr_region` | OCR a page or selected region |
| `extract_table` | Extract table structure and cells |
| `chart_to_table` | Convert a chart region to a table |

## Rollout protocol

The model receives a prompt containing a document path and a question. Tool
calls use the Qwen model's JSON or XML convention, for example:

```xml
<tool_call>
{"name":"parse_document","arguments":{"document_path":"/data/report.pdf","page_range":"1-3"}}
</tool_call>
```

Tool observations are appended inside `<interpreter>` tags with `loss_mask=0`.
Only model-generated reasoning, tool calls, and final-answer tokens are trained.
The trajectory terminates with:

```xml
<final>$1.2 million</final>
```

The default limits are 16 assistant turns, 16 tool calls, 8192 observation
characters per call, and 32 concurrent tool executions.

## Reward

`document_reward.py` extracts the last strict `<final>...</final>` value and
compares it with one or more acceptable answers. Supported metrics are:

- `exact_match`
- `anls` (DocVQA-style normalized Levenshtein similarity)
- `token_f1`
- `contains`
- `json`
- `auto` (maximum of exact match, ANLS, and token F1)

The answer quality `q` is mapped to the GRPO reward `2q - 1`, so the reward is
in `[-1, 1]`. Missing the final tag receives `-1`. Conventional lead-ins such
as `The answer is:` and `答案是：` are ignored for `acc`, while `exact_acc`
retains strict raw-string accuracy. Use `quality` as the evaluation reward key:
it follows each task's configured metric and does not collapse a useful
partial/semantic match to zero. The reward result also logs `acc`, `exact_acc`,
`quality`, `format`, tool-call count, valid-call count, and tool errors.

PRM scripts retain step-wise training, but the judge now evaluates whether each
action chose an appropriate document tool/page/region and retrieved grounded,
relevant evidence. In PRM mode:

```text
final_score = outcome_score + PRM_STEP_COEF * mean(step_scores)
```

## Dataset format

Create a source JSONL manifest:

```json
{"id":"q1","document_path":"/data/report.pdf","question":"What was 2025 revenue?","answers":["$1.2 million","$1.2m"],"metric":"anls"}
```

Convert it to slime's `prompt`/`label`/`metadata` JSONL:

```bash
python toolcall-rl/rl_data_preprocess.py \
  --input /data/document_qa/source.jsonl \
  --output /data/document_qa/train.jsonl \
  --check-files
```

Common aliases such as `file_path`, `query`, `answer`, and `ground_truth` are
accepted. Relative document paths can be prefixed using `--document-root`.
Every referenced file must be mounted at the same path in rollout workers.

For optional document-tool SFT, input rows must contain a standard `messages`
trajectory and can be converted with:

```bash
python toolcall-rl/sft_data_processing.py \
  --input /data/document_tool_sft.jsonl \
  --output /data/document-tool-sft/train.parquet
```

## Training

Single-node Qwen3-4B GRPO:

```bash
export PROMPT_DATA=/data/document_qa/train.jsonl
export EVAL_DATA=/data/document_qa/eval.jsonl
export HF_CKPT=/models/qwen3-4b-document-tool-sft
export REF_LOAD=/models/qwen3-4b-document-tool-sft_torch_dist
cd slime
bash ../toolcall-rl/retool_qwen3_4b_rl.sh
```

PRM + step-wise RL:

```bash
export PROMPT_DATA=/data/document_qa/train.jsonl
export EVAL_DATA=/data/document_qa/eval.jsonl
export PRM_MODEL_PATH=/models/document-prm
cd slime
bash ../toolcall-rl/retool_qwen3_4b_prm_rl.sh
```

Equivalent Qwen3.5-4B, Qwen3.5-27B, and Qwen2.5-32B scripts are provided in
this directory. All RL scripts now default to `document-qa/train.jsonl` and
`document-qa/eval.jsonl`, use the `docqa` evaluation name, and log to the
`document_tool_rl` W&B project.

## Tool runtime

Heavy backends are loaded lazily. Useful environment variables include:

- `OPENCLAW_TOOL_OUTPUT_DIR`
- `OPENCLAW_TOOL_CACHE_DIR`
- `OPENCLAW_DOCLING_ARTIFACTS_PATH`
- `OPENCLAW_DEPLOT_MODEL`
- `OPENCLAW_PADDLEOCR_DET_MODEL`
- `OPENCLAW_PADDLEOCR_REC_MODEL`

Run the lightweight tests with:

```bash
pytest -q toolcall-rl/tests/test_document_reward.py toolcall-rl/tests/test_document_tools.py
```

## Qwen3-VL DocVQA operations

This section documents the reproducible remote workflow using
`/models/Qwen3-VL-4B-Instruct`. Use a new output directory per run; do not
reuse checkpoints.

### Environment and data

- project: `/workspace/data/OpenClaw-RL`
- environment: `/workspace/data/envs/openclaw-rl-qwen3vl`
- full raw test: `data/test.jsonl` (200 QA over 17 PDFs)
- derived rollout manifest: `data/full_docvqa_eval_20260723_07.jsonl`
- document root: `data/`
- Docling cache: `openclaw-tool-models-20260717-02/docling-artifacts`

The derived manifest contains `prompt`, `label`, and `metadata`; evaluation
does not modify the raw test JSONL.

### Launch modes

Preflight before any new evaluation or training run:

```bash
/workspace/data/envs/openclaw-rl-qwen3vl/bin/python -B \
  toolcall-rl/preflight_qwen3_vl_4b.py --mode train \
  --train-data /absolute/train.jsonl --eval-data /absolute/eval.jsonl \
  --output-dir /workspace/data/OpenClaw-RL/outputs/unique-run
```

For ordinary GRPO training use the recipes in the Training section. For an
initial-weight baseline, set `--num-rollout 0` and `--lr-decay-iters 1`;
`smoke_result.json` must show `checkpoint: null` and `num_rollout: 0`.

The fixed bridge task `openclaw_full_docvqa_baseline_07` evaluates the full
200-QA manifest with four GPUs, without updates, at:

```text
/workspace/data/OpenClaw-RL/outputs/qwen3-vl-4b-docvqa-baseline-20260723-07-full200
```

Its initial-weight mean ANLS was `0.09771303258145363`; it is a baseline, not
post-training evidence.

### Logs and workflow exports

Each run retains `smoke_result.json`, `launch_manifest.json`, complete raw
rollouts in `dump_details/rollout_data/eval_0.pt`, and tool text/images under
`tool_outputs/`. The `.pt` archive contains tensors and tokens, so use the
bounded exporter for human review:

```bash
python toolcall-rl/export_rollout_workflows.py --limit 20
```

It writes Markdown and JSON workflow subsets to the run directory. They retain
the prompt, model tool calls, tool-return text, final answer, media references,
labels, and reward fields, while excluding tokens, masks, tensors, and vectors.

## Method details

### Agent and multimodal trajectory

The agent receives a document path and question, calls tools such as
`parse_document`, `detect_layout`, `render_page`, and `ocr_region`, then gets
the result in an `<interpreter>` block as context for the next generation.
Rendered page PNGs are retained as visual evidence references. An episode ends
only at a strict `<final>...</final>` answer or a configured limit.

The Qwen3-VL DocVQA runner uses a bounded multi-turn search loop. After each
tool result it regenerates a decision, maintains page/region navigation state,
and may continue to another page or a specialized tool before producing the
final answer. The default Qwen3-VL tool budget is eight calls and can be
overridden by the runner configuration. A negative answer such as `None` or
`not found` is guarded until all relevant pages have been checked or the search
budget is exhausted.

Evidence stopping is structural rather than keyword-only. A valid candidate
must satisfy the question's text, table row/column, or visual-region
constraints and meet the relation threshold. `final_supported_by_evidence`
requires a document string match, a relation match, and a supporting page or
region. `parse_document` returns only the requested pages; `returned_pages`
must equal the page numbers in `pages`, and `content_truncated` is separate
from the document having unreturned pages.

For table-like questions, flattened or scanned pages may use the fallback
chain `extract_table` -> `render_page`/`detect_layout` -> `crop_region`/`ocr_region`.
For visual questions, a rendered image is attached to the next model turn only
when the decision requires page pixels. Each generation step records whether a
visual input was required and whether a placeholder, image tensor, model
attachment, and consumed image path were actually present.

### Outcome reward and reporting

The scorer applies the configured metric (normally DocVQA ANLS) to the final
answer and logs `quality`, `acc`, `exact_acc`, `format`, and tool counters.
The rollout reward is `2 * quality - 1`. `acc` permits safe normalization of
answer lead-ins and a short field label: `Proposal #: 14-3006-14` can match
`14-3006-14`. `exact_acc` remains raw strict matching. Arbitrary substrings or
unlisted abbreviations are not accepted: `ITC` is not automatically correct
for `ITC Limited`.

### Training versus baseline evidence

Eval-only baselines validate serving, tools, manifests, and rewards but do not
measure RL improvement. Training evidence requires a new checkpointed run and
a separate held-out evaluation. Retain both the raw archive and workflow export
for agent and reward audits.

Rollout infrastructure status is separated from model behavior. Generation
errors, context overflow, infrastructure/media failures, and excluded tool
infrastructure failures use `valid_for_rl: false` and do not contribute to
GRPO group mean/std or policy gradients. Protocol-invalid model actions remain
auditable and can receive a protocol penalty. Action-level fields
(`action_rewards`, `rejected_action_indices`, and `assistant_token_masks`) are
consumed independently of the sequence reward so a rejected premature final
cannot inherit the final answer's positive advantage.

The workflow exporter reports evidence localization and evidence alignment as
separate process signals and validates reward invariants before including a
sample in RL statistics. Group ordering metrics are meaningful only when the
source rollout batch carries group IDs; if `group_index` is absent, the report
uses `null` accuracy and an explicit zero comparable-pair count rather than
misrepresenting missing metadata as zero ordering accuracy.
