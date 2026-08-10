# Four-benchmark evaluation workflow

This directory is the evaluation-only integration for DocVQA 2026 validation,
LongDocURL public, MP-DocVQA validation, and DUDE validation. It reuses the
project document tools and `generate_with_retool.generate`; it does not add a
benchmark-specific agent.

The benchmark files stay on the evaluation server. Set one shared root there:

```bash
export REPO_ROOT=/workspace/data/OpenClaw-RL
export BENCH_ROOT=/data/doc_agentic_benchmarks
export HF_HOME=$BENCH_ROOT/hf_cache
export HF_DATASETS_CACHE=$BENCH_ROOT/hf_datasets_cache
mkdir -p "$BENCH_ROOT"/{raw,documents,eval,gold,results} "$HF_HOME" "$HF_DATASETS_CACHE"
```

Install only adapter dependencies on the server, then clone scorer code into
`toolcall-rl/eval_benchmarks/vendors/`. Do not clone datasets into the Git
checkout and do not download the forbidden test/PNG/OCR artifacts.

Preparation commands are:

```bash
python -m pip install -U huggingface_hub datasets pyarrow pillow pymupdf img2pdf python-Levenshtein

mkdir -p toolcall-rl/eval_benchmarks/vendors
git -C toolcall-rl/eval_benchmarks/vendors clone --depth 1 https://github.com/VLR-CVC/DocVQA2026.git
git -C toolcall-rl/eval_benchmarks/vendors clone --depth 1 https://github.com/dengc2023/LongDocURL.git
git -C toolcall-rl/eval_benchmarks/vendors clone --depth 1 https://github.com/Jordy-VL/DUDEeval.git

hf download VLR-CVC/DocVQA-2026 val.parquet --repo-type dataset \
  --local-dir "$BENCH_ROOT/raw/docvqa2026"
hf download dengchao/LongDocURL LongDocURL_public.jsonl pdf_files.tar.gz \
  --repo-type dataset --local-dir "$BENCH_ROOT/raw/longdocurl"
mkdir -p "$BENCH_ROOT/documents/longdocurl"
tar -xzf "$BENCH_ROOT/raw/longdocurl/pdf_files.tar.gz" -C "$BENCH_ROOT/documents/longdocurl"
hf download lmms-lab/MP-DocVQA --repo-type dataset \
  --include 'data/val-*.parquet' --local-dir "$BENCH_ROOT/raw/mpdocvqa"
curl -L 'https://zenodo.org/records/7763635/files/2023-03-23_DUDE_gt_test_PUBLIC.json?download=1' \
  -o "$BENCH_ROOT/raw/dude/2023-03-23_DUDE_gt_test_PUBLIC.json"
hf download jordyvl/DUDE_loader data/DUDE_train-val-test_binaries.tar.gz \
  --repo-type dataset --local-dir "$BENCH_ROOT/raw/dude"
```

Before the DUDE archive, verify at least 45 GB free. `prepare_dude.py` streams
only validation PDFs from the archive and supports `--delete-archive` after a
successful extraction. It never uses `tar.extractall`.

```bash
python toolcall-rl/eval_benchmarks/prepare_docvqa2026.py --bench-root "$BENCH_ROOT"
python toolcall-rl/eval_benchmarks/prepare_longdocurl.py --bench-root "$BENCH_ROOT"
python toolcall-rl/eval_benchmarks/prepare_mpdocvqa.py --bench-root "$BENCH_ROOT"
python toolcall-rl/eval_benchmarks/prepare_dude.py --bench-root "$BENCH_ROOT" --delete-archive

for benchmark in docvqa2026 longdocurl mpdocvqa dude; do
  python toolcall-rl/eval_benchmarks/validate_prepared_data.py \
    --benchmark "$benchmark" --bench-root "$BENCH_ROOT"
done
```

For a GPU smoke run, first confirm that the main training session has declared
the GPUs idle and send it one wait message immediately before starting. Take
the first eight rows per benchmark into separate files and use the validated
Qwen3-VL launcher through:

```bash
bash toolcall-rl/eval_benchmarks/run_eval_only.sh \
  --eval-data "$BENCH_ROOT/eval/mpdocvqa_smoke8.jsonl" \
  --output-dir "$BENCH_ROOT/results/mpdocvqa/smoke8" \
  --model /models/Qwen3-VL-4B-Instruct
```

The wrapper forces eval-only (`num_rollout=0`) and one sample per question. It
refuses to overwrite an existing `eval_0.pt` unless `--overwrite` is explicit,
checks the eval JSONL row count for the validated launcher, and fails if the
required rollout artifact is absent. The launcher path can be
supplied with `--launcher` or `OPENCLAW_EVAL_LAUNCHER`; the default is the
project's validated `run_qwen3_vl_4b_real_docvqa_test_05.py` bridge.

After a rollout, export and score each benchmark independently:

```bash
python toolcall-rl/eval_benchmarks/export_eval_predictions.py \
  --eval-pt "$BENCH_ROOT/results/mpdocvqa/smoke8/dump_details/rollout_data/eval_0.pt" \
  --output "$BENCH_ROOT/results/mpdocvqa/predictions.jsonl"
python toolcall-rl/eval_benchmarks/score_mpdocvqa.py --bench-root "$BENCH_ROOT"
python toolcall-rl/eval_benchmarks/summarize_agent_metrics.py \
  --benchmark mpdocvqa --bench-root "$BENCH_ROOT"
```

Use `score_docvqa2026.py`, `score_longdocurl.py`, and `score_dude.py` for the
other benchmarks. Official scorer failures are surfaced; no GPT/API grader
or project training reward is substituted. DUDE writes its official GT and
submission artifacts, while LongDocURL writes `official_input.jsonl`.

Before a formal result, write `results/evaluation_integrity.json` with
`write_evaluation_integrity.py`; it records the checkpoint, commit, four
dataset counts, one-sample policy, and shared tool budget.

The four result directories contain `predictions.jsonl`,
`official_metrics.json`, `agent_metrics.json`, and
`per_sample_scores.jsonl`; `summarize_agent_metrics.py --benchmark all` also
writes `results/summary.json` and `results/summary.md`. Navigation metrics are
post-hoc only: gold pages/evidence never enter the prompt or runtime state.
