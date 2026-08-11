from __future__ import annotations

import json

from data.preprocess_swe import main, select_rows, with_source_image


def _row(index: int, **overrides):
    value = {
        "instance_id": f"repo-{index}",
        "problem_statement": f"Repair the Python implementation for deterministic smoke case {index}.",
        "image_name": f"swebench/testbed:case-{index}",
        "repo": "example/python-project",
        "base_commit": f"deadbeef{index}",
        "test_patch": f"private test patch {index}",
        "FAIL_TO_PASS": [f"test_case_{index}"],
        "PASS_TO_PASS": ["test_existing"],
    }
    value.update(overrides)
    return value


def test_small_selection_is_reproducible_disjoint_and_leakage_safe():
    rows = [_row(index) for index in range(12)]
    first, first_report = select_rows(rows, num_samples=5, seed=17, data_source="SWE-Gym", candidate_limit=10, excluded_instance_ids={"repo-0"})
    second, second_report = select_rows(rows, num_samples=5, seed=17, data_source="SWE-Gym", candidate_limit=10, excluded_instance_ids={"repo-0"})

    assert first == second
    assert first_report == second_report
    assert "repo-0" not in first_report["instance_ids"]
    assert first_report["filter_counts"]["filtered_overlap"] == 1
    for row in first:
        assert row["environment"] == "code"
        metadata = row["metadata"]
        assert metadata["instance_id"] == metadata["public_instance"]["instance_id"]
        assert metadata["data_source"] == "SWE-Gym"
        assert metadata["image_name"] == metadata["public_instance"]["image_name"]
        visible = json.dumps({"text": row["text"], "public_instance": metadata["public_instance"]})
        assert "private test patch" not in visible
        assert "FAIL_TO_PASS" not in visible
        assert metadata["evaluator_private"]["test_patch"].startswith("private test patch")


def test_selection_filters_bad_metadata_before_sampling():
    rows = [
        _row(0, language="Java"),
        _row(1, image_name=""),
        _row(2, environment_status="broken"),
        _row(3, problem_statement="short"),
        _row(4),
    ]
    selected, report = select_rows(rows, num_samples=1, seed=1, data_source="SWE-Gym", candidate_limit=1)
    assert selected[0]["metadata"]["instance_id"] == "repo-4"
    assert report["filter_counts"]["filtered_non_python"] == 1
    assert report["filter_counts"]["filtered_missing_image"] == 1
    assert report["filter_counts"]["filtered_known_bad_environment"] == 1
    assert report["filter_counts"]["filtered_incomplete_problem_statement"] == 1


def test_swe_gym_uses_only_its_documented_per_instance_image_convention():
    row = _row(9, image_name="")
    enriched = with_source_image(row, source="swe-gym")
    assert enriched["image_name"] == "xingyaoww/sweb.eval.x86_64.repo-9:latest"
    assert with_source_image(row, source="swe-bench-verified").get("image_name", "") == ""


def test_cli_writes_manifest_statistics_and_ids_from_small_snapshot(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(_row(index)) for index in range(4)) + "\n", encoding="utf-8")
    output = tmp_path / "train.jsonl"

    assert main(["--source", "swe-gym", "--input", str(source), "--num-samples", "3", "--seed", "7", "--output", str(output), "--candidate-limit", "4"]) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    report = json.loads(output.with_suffix(".stats.json").read_text(encoding="utf-8"))
    ids = output.with_suffix(".instance_ids.txt").read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(ids) == 3
    assert ids == report["instance_ids"]
    assert report["source"] == "swe-gym"
