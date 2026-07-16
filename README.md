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
