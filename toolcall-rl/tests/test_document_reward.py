import json
from pathlib import Path
import sys


TOOLCALL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLCALL_DIR))

from document_reward import compute_document_reward, extract_final_answer, normalize_answer
from rl_data_preprocess import transform_record


def test_rejects_final_embedded_in_explanatory_or_multi_action_text():
    assert extract_final_answer("draft <final>old</final> revised <final>New answer</final>") == ("", False)
    assert extract_final_answer("Use <final> and </final> to return the answer.") == ("", False)


def test_missing_final_tag_gets_minimum_reward():
    result = compute_document_reward("The answer is 2024.", "2024")
    assert result["score"] == -1.0
    assert result["format"] == 0.0


def test_abstention_reward_distinguishes_unsupported_and_unnecessary_refusal():
    justified = compute_document_reward(
        "<abstain>the available observations are contradictory</abstain>",
        "2024",
        {"evidence_sufficient": False, "bayestool": {"stop_decision": {"mode": "abstain"}}},
    )
    unnecessary = compute_document_reward(
        "<abstain>I will not answer</abstain>",
        "2024",
        {"evidence_sufficient": True},
    )
    assert justified["abstention"] is True
    assert justified["abstention_justified"] is True
    assert justified["score"] > unnecessary["score"]


def test_semantically_correct_final_span_inside_prose_keeps_positive_correctness():
    result = compute_document_reward("Explanation: the answer is <final>Bengaluru</final>.", "Bengaluru")
    assert result["answer_correctness"] == 1.0
    assert result["format_validity"] == 0.0
    assert result["answer_conciseness"] < 1.0
    assert result["score"] > 0.0


def test_exact_match_accepts_aliases_and_normalizes_punctuation():
    label = json.dumps({"answers": ["US$1.2 million", "$1.2m"], "metric": "exact_match"})
    result = compute_document_reward("<final>US$1.2 million.</final>", label)
    assert result["score"] == 1.0
    assert result["acc"] == 1.0


def test_exact_match_accepts_explicit_answer_prefix_but_keeps_strict_metric():
    result = compute_document_reward("<final>The answer is: Caffeine</final>", "Caffeine")
    assert result["score"] == 1.0
    assert result["acc"] == 1.0
    assert result["exact_acc"] == 0.0


def test_anls_accepts_short_document_field_label_but_keeps_strict_metric():
    label = json.dumps({"answers": ["14-3006-14"], "metric": "anls"})
    result = compute_document_reward("<final>Proposal #: 14-3006-14</final>", label)
    assert result["score"] == 1.0
    assert result["acc"] == 1.0
    assert result["quality"] == 1.0
    assert result["exact_acc"] == 0.0


def test_field_label_rule_does_not_accept_truncated_entity():
    label = json.dumps({"answers": ["ITC Limited"], "metric": "anls"})
    result = compute_document_reward("<final>Company Name: ITC</final>", label)
    assert result["acc"] == 0.0
    assert result["quality"] == 0.0


def test_unstructured_explanation_is_not_promoted_to_exact_match():
    result = compute_document_reward("<final>Caffeine because it is listed first.</final>", "Caffeine")
    assert result["acc"] == 0.0
    assert 0.0 < result["quality"] < 1.0


def test_anls_gives_partial_credit_for_ocr_noise():
    result = compute_document_reward("<final>OpenClaw RL</final>", json.dumps({"answers": ["OpenClaw-RL"], "metric": "anls"}))
    assert 0.0 < result["quality"] <= 1.0


def test_chinese_normalization_preserves_characters():
    source = "".join(chr(code) for code in (0x6587, 0x6863, 0x7406, 0x89e3)) + chr(0xff1a)
    expected = "".join(chr(code) for code in (0x6587, 0x6863, 0x7406, 0x89e3))
    assert normalize_answer(source) == expected
    result = compute_document_reward(f"<final>{expected}</final>", expected)
    assert result["score"] == 1.0


def test_manifest_record_is_converted_to_document_task(tmp_path):
    item = transform_record(
        {"id": "q1", "file_path": "report.pdf", "question": "What is the title?", "answers": ["Annual report"]},
        document_root=tmp_path,
        default_metric="anls",
    )
    assert str(tmp_path / "report.pdf") in item["prompt"]
    assert "<final>" in item["prompt"]
    assert json.loads(item["label"])["answers"] == ["Annual report"]
    assert item["metadata"]["task_id"] == "q1"


def test_pdf_path_manifest_is_converted_to_document_task(tmp_path):
    item = transform_record(
        {"id": "q-pdf", "pdf_path": "train/pdfs/train_000000.pdf", "question": "What is shown?", "answers": ["A"]},
        document_root=tmp_path,
    )
    assert str(tmp_path / "train/pdfs/train_000000.pdf") in item["prompt"]
    assert item["metadata"]["task_id"] == "q-pdf"


def test_answer_page_metadata_is_preserved(tmp_path):
    item = transform_record(
        {
            "id": "q2",
            "file_path": "report.pdf",
            "question": "Who is the supplier?",
            "answers": ["BURKE"],
            "answer_page": 3,
            "answer_bbox": [1, 2, 3, 4],
            "num_pages": 4,
        },
        document_root=tmp_path,
    )
    assert item["metadata"]["answer_page"] == 3
    assert item["metadata"]["answer_bbox"] == [1, 2, 3, 4]
    assert item["metadata"]["page_count"] == 4
