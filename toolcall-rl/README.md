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
in `[-1, 1]`. Missing the final tag receives `-1`. The reward result also logs
`acc`, `quality`, `format`, tool-call count, valid-call count, and tool errors.

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
