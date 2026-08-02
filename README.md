# Document Tool-Call RL

Document Tool-Call RL trains language models to answer questions about local
PDF, DOCX, PPTX, image, Markdown, and text documents through grounded,
multi-step tool use. Agents learn to inspect a document, select the appropriate
tool, retrieve evidence, and return a final answer.

## Included components

- [`toolcall-rl/`](./toolcall-rl): document tools, data preparation, rewards,
  rollout logic, SFT utilities, training scripts, and tests.
- [`slime/`](./slime): the RL training and rollout framework.
- [`Megatron-LM/`](./Megatron-LM): the distributed training backend used by the
  provided scripts.

The document-tool suite includes page rendering, cropping and zooming, document
parsing, layout detection, OCR, table extraction, and chart-to-table
conversion. Outcome rewards support exact match, ANLS, token F1, containment,
and JSON evaluation; optional PRM training additionally scores the quality of
intermediate tool choices and retrieved evidence.

## Quick start

Install the document-tool dependencies:

```bash
pip install -r toolcall-rl/requirements.txt
```

Prepare a JSONL document-QA manifest and convert it to the training format:

```bash
python toolcall-rl/rl_data_preprocess.py \
  --input /data/document_qa/source.jsonl \
  --output /data/document_qa/train.jsonl \
  --check-files
```

Then launch a supplied RL recipe:

```bash
cd slime
bash ../toolcall-rl/retool_qwen3_4b_rl.sh
```

See [`toolcall-rl/README.md`](./toolcall-rl/README.md) for dataset schemas,
environment variables, SFT and PRM workflows, and testing instructions.

## Qwen3-VL document-QA workflows

The repository also contains a verified Qwen3-VL-4B workflow for local DocVQA
data on the remote workspace. It separates preflight validation, strict
initial-weight baseline evaluation (`num_rollout=0`), and RL training. The
detailed commands, retained rollout artifacts, and reward/agent method notes
are in [the tool-call README](./toolcall-rl/README.md#qwen3-vl-docvqa-operations).

## Agent design contract

Document QA uses a bounded multi-turn loop: every successful tool result
triggers another model decision until a strict `<final>...</final>` action is
accepted or the search budget is exhausted. The runner tracks parsed,
rendered, cropped, and OCR-checked pages separately. `evidence_sufficient` is
set only by a question-constrained structured evidence candidate; seeing a
page, number, table hint, or answer-page annotation alone is not sufficient.

`parse_document` is page-scoped: `pages` and `returned_pages` describe exactly
the content returned, while `document_has_unreturned_pages` and
`content_truncated` distinguish unvisited pages from truncated content. Table-
like questions can fall back from text parsing to table extraction, layout,
rendering, cropping, or OCR. Visual input is attached only when the next model
decision requires pixels, and each such turn records the actual placeholder,
tensor, forward attachment, and consumed image path.

Infrastructure failures (empty/error generation, context overflow, media or
tool infrastructure failures) are marked `valid_for_rl: false` and excluded
from group statistics and policy gradients. Model protocol errors remain
observable as model actions; rejected actions receive a zero policy mask and a
separate action-level negative signal. Reward reports keep answer correctness,
evidence localization/alignment, tool cost, format validity, and consistency
audits separate from the final scalar reward.
