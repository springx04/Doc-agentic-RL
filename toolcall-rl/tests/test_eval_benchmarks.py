from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import fitz
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_benchmarks.common import images_to_pdf, jsonl_write, make_eval_row  # noqa: E402
from eval_benchmarks.export_eval_predictions import export_predictions  # noqa: E402
from eval_benchmarks.prepare_dude import prepare as prepare_dude  # noqa: E402
from eval_benchmarks.prepare_longdocurl import prepare as prepare_longdocurl  # noqa: E402
from eval_benchmarks.prepare_mpdocvqa import _answers_for_eval  # noqa: E402
from eval_benchmarks.score_docvqa2026 import score as score_docvqa  # noqa: E402
from eval_benchmarks.score_dude import score as score_dude  # noqa: E402
from eval_benchmarks.score_longdocurl import score as score_longdocurl  # noqa: E402
from eval_benchmarks.score_mpdocvqa import score as score_mpdocvqa  # noqa: E402
from eval_benchmarks.summarize_agent_metrics import summarize  # noqa: E402
from eval_benchmarks.validate_prepared_data import validate  # noqa: E402


def _pdf(path: Path, pages: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"page {index + 1}")
    document.save(path)
    document.close()


def _prediction(task_id: str, answer: str) -> dict:
    return {
        "task_id": task_id,
        "final_answer": answer,
        "protocol_valid": True,
        "tool_call_count": 1,
        "valid_tool_call_count": 1,
        "tool_error_count": 0,
        "visited_pages": [1],
        "rendered_pages": [1],
        "ocr_pages": [],
        "tool_calls": [{"tool": "render_page", "arguments": {"page_number": 1}}],
    }


def test_images_to_pdf_preserves_page_order(tmp_path: Path) -> None:
    output = tmp_path / "document.pdf"
    images_to_pdf(
        [Image.new("RGB", (32, 32), "red"), Image.new("RGB", (32, 32), "blue")],
        output,
        expected_page_count=2,
    )
    assert fitz.open(output).page_count == 2
    images_to_pdf([Image.new("RGB", (32, 32), "green")], output, expected_page_count=1)
    assert fitz.open(output).page_count == 1


def test_prepare_and_validate_longdocurl_without_gold_in_prompt(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    document = bench_root / "documents" / "longdocurl" / "doc-1.pdf"
    _pdf(document, pages=2)
    raw = bench_root / "raw" / "longdocurl" / "LongDocURL_public.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "doc_no": "doc-1",
                "question": "What is the page count?",
                "answer": "2",
                "answer_format": "number",
                "task_tag": "understanding",
                "evidence_pages": [2],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = prepare_longdocurl(bench_root, limit=1)
    assert report["num_questions"] == 1
    validation = validate("longdocurl", bench_root, allow_partial=True)
    assert validation["num_questions"] == 1
    prompt = json.loads((bench_root / "eval" / "longdocurl.jsonl").read_text(encoding="utf-8"))["prompt"]
    assert "evidence_pages" not in prompt
    assert "Question: What is the page count?" in prompt


def test_export_and_all_cpu_scorers(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench"
    document = bench_root / "documents" / "synthetic.pdf"
    _pdf(document)
    eval_row = make_eval_row(
        benchmark="mpdocvqa",
        question_id="q1",
        doc_id="doc-1",
        document_path=document,
        page_count=1,
        question="What word is shown?",
        answers=["answer"],
    )
    gold_mp = {
        "task_id": "mpdocvqa:q1",
        "question_id": "q1",
        "doc_id": "doc-1",
        "answers": ["answer"],
        "gold_answer_page_1based": 1,
    }
    eval_path = bench_root / "eval" / "mpdocvqa.jsonl"
    gold_path = bench_root / "gold" / "mpdocvqa.jsonl"
    pred_path = bench_root / "results" / "mpdocvqa" / "predictions.jsonl"
    jsonl_write(eval_path, [eval_row])
    jsonl_write(gold_path, [gold_mp])
    jsonl_write(pred_path, [_prediction("mpdocvqa:q1", "answer")])
    mp_metrics, _ = score_mpdocvqa(pred_path, gold_path)
    assert mp_metrics["anls"] == 1.0
    agent_metrics = summarize("mpdocvqa", bench_root)
    assert agent_metrics["gold_page_visited_rate"] == 1.0

    doc_vendor = tmp_path / "DocVQA2026"
    doc_vendor.mkdir()
    (doc_vendor / "eval_utils.py").write_text(
        "def evaluate_docvqa_prediction(raw, answers):\n"
        "    value = raw.split(':', 1)[1].strip()\n"
        "    return (value in answers, value)\n",
        encoding="utf-8",
    )
    doc_gold = bench_root / "gold" / "docvqa2026.jsonl"
    doc_pred = bench_root / "results" / "docvqa2026" / "predictions.jsonl"
    jsonl_write(
        doc_gold,
        [{"task_id": "docvqa2026:q1", "question_id": "q1", "doc_id": "doc-1", "doc_category": "maps", "answers": ["answer"]}],
    )
    jsonl_write(doc_pred, [_prediction("docvqa2026:q1", "answer")])
    doc_metrics, _ = score_docvqa(doc_pred, doc_gold, doc_vendor)
    assert doc_metrics["accuracy"] == 1.0

    long_vendor = tmp_path / "LongDocURL" / "utils"
    long_vendor.mkdir(parents=True)
    (long_vendor / "__init__.py").write_text("", encoding="utf-8")
    (long_vendor / "utils_score_v3.py").write_text(
        "def eval_score(gold, pred, answer_format):\n    return 1.0 if gold == pred else 0.0\n",
        encoding="utf-8",
    )
    long_gold = bench_root / "gold" / "longdocurl.jsonl"
    long_pred = bench_root / "results" / "longdocurl" / "predictions.jsonl"
    jsonl_write(
        long_gold,
        [{"task_id": "longdocurl:q1", "question_id": "q1", "answer": "answer", "answer_format": "number", "task_tag": "understanding"}],
    )
    jsonl_write(long_pred, [_prediction("longdocurl:q1", "answer")])
    long_metrics, _ = score_longdocurl(long_pred, long_gold, long_vendor.parent, bench_root / "results" / "longdocurl" / "official_input.jsonl")
    assert long_metrics["generalized_accuracy"] == 1.0

    dude_raw = bench_root / "raw" / "dude" / "2023-03-23_DUDE_gt_test_PUBLIC.json"
    dude_raw.parent.mkdir(parents=True)
    dude_source_row = {"questionId": "q1", "docId": "doc-1", "question": "What?", "answers": ["answer"], "data_split": "val"}
    dude_raw.write_text(json.dumps({"data": [dude_source_row]}) + "\n", encoding="utf-8")
    archive = bench_root / "raw" / "dude" / "data" / "DUDE_train-val-test_binaries.tar.gz"
    archive.parent.mkdir(parents=True)
    archive_pdf = tmp_path / "doc-1.pdf"
    _pdf(archive_pdf)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(archive_pdf, arcname="nested/doc-1.pdf")
    dude_report = prepare_dude(bench_root, limit=1)
    assert dude_report["num_questions"] == 1
    dude_gold = bench_root / "gold" / "dude.jsonl"
    dude_pred = bench_root / "results" / "dude" / "predictions.jsonl"
    jsonl_write(dude_pred, [_prediction("dude:q1", "answer")])
    dude_metrics, _ = score_dude(
        dude_pred,
        dude_gold,
        dude_raw,
        tmp_path / "DUDEeval",
        bench_root / "results" / "dude",
        skip_official=True,
    )
    assert dude_metrics["official_pending"] is True
    assert (bench_root / "results" / "dude" / "dude_val_submission.json").is_file()


def test_empty_mp_answers_keep_empty_gold_but_valid_project_label() -> None:
    assert _answers_for_eval([]) == ([], [""])


def test_export_keeps_failed_rollout_in_denominator(tmp_path: Path) -> None:
    import torch

    artifact = tmp_path / "eval_0.pt"
    torch.save(
        {
            "samples": [
                {
                    "metadata": {"task_id": "mpdocvqa:q1", "benchmark": "mpdocvqa"},
                    "response": "",
                    "rollout_status": "generation_error",
                    "tool_execution": {"calls": []},
                }
            ]
        },
        artifact,
    )
    output = tmp_path / "predictions.jsonl"
    report = export_predictions(artifact, output)
    assert report["num_predictions"] == 1
    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert exported[0]["final_answer"] == ""
    assert exported[0]["protocol_valid"] is False
