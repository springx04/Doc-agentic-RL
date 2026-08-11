"""Build small, leakage-safe Code Agent manifests from SWE datasets.

The command deliberately consumes a bounded streaming prefix instead of
materialising an entire Hub dataset.  It is a *data preparation* tool only:
Docker build and official evaluator execution happen later in isolated Code
environment preflight, never inside rollout or this module.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

try:
    from .leakage_guard import assert_public_safe, is_hidden_key
    from .schema import SWEInstance
except ImportError:  # pragma: no cover - direct PYTHONPATH execution
    from data.leakage_guard import assert_public_safe, is_hidden_key
    from data.schema import SWEInstance


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    config: str
    split: str
    data_source: str


SOURCE_SPECS = {
    "swe-gym": SourceSpec("SWE-Gym/SWE-Gym", "default", "train", "SWE-Gym"),
    "swe-bench-verified": SourceSpec("SWE-bench/SWE-bench_Verified", "default", "test", "SWE-bench Verified"),
}

_EVALUATOR_EVIDENCE_KEYS = frozenset({"test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "fail_to_pass", "pass_to_pass", "eval_script", "test_command"})
_BAD_STATUS_WORDS = frozenset({"broken", "failed", "invalid", "missing", "unavailable", "unsupported", "timeout"})


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if isinstance(value, list):
        return [dict(item) for item in value]
    if isinstance(value, Mapping):
        return [dict(value)]
    raise ValueError("input must contain a JSON object/list or JSONL objects")


def _first_text(raw: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def source_image_name(raw: Mapping[str, Any], *, source: str) -> str:
    """Return an explicit image, or SWE-Gym's documented image convention."""

    image = _first_text(raw, "image_name", "docker_image", "image")
    if image:
        return image
    if source == "swe-gym":
        instance_id = _first_text(raw, "instance_id", "id", "instance")
        if instance_id:
            # SWE-Gym publishes one Docker image per instance under this
            # exact documented repository prefix.  Availability is still
            # checked by the separate Docker preflight before a run.
            return f"xingyaoww/sweb.eval.x86_64.{instance_id.lower()}:latest"
    return ""


def with_source_image(raw: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """Copy a source row and fill only the documented SWE-Gym image alias."""

    value = dict(raw)
    if not _first_text(value, "image_name", "docker_image", "image"):
        image = source_image_name(value, source=source)
        if image:
            value["image_name"] = image
    return value


def _hidden_values(value: Any, *, found: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {} if found is None else found
    if isinstance(value, Mapping):
        for key, item in value.items():
            if is_hidden_key(str(key)):
                result[str(key)] = item
            else:
                _hidden_values(item, found=result)
    elif isinstance(value, list):
        for item in value:
            _hidden_values(item, found=result)
    return result


def _has_bad_environment_status(raw: Mapping[str, Any]) -> bool:
    for key in ("environment_status", "image_status", "build_status", "docker_status", "status"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip().lower() in _BAD_STATUS_WORDS:
            return True
    return bool(raw.get("environment_broken") or raw.get("image_missing") or raw.get("skip") or raw.get("disabled"))


def _python_compatible(raw: Mapping[str, Any]) -> bool:
    language = _first_text(raw, "language", "repo_language", "programming_language")
    return not language or language.lower() in {"python", "python3", "py"}


def normalize_row(raw: Mapping[str, Any], *, data_source: str) -> dict[str, Any]:
    """Normalise source aliases without moving evaluator labels into public data."""

    value = dict(raw)
    instance_id = _first_text(value, "instance_id", "id", "instance")
    problem_statement = _first_text(value, "problem_statement", "text", "problem")
    image_name = _first_text(value, "image_name", "docker_image", "image")
    repository = _first_text(value, "repo", "repository")
    base_revision = _first_text(value, "base_commit", "base_revision", "commit")
    hidden = _hidden_values(value)
    public = {
        "instance_id": instance_id,
        "problem_statement": problem_statement,
        "image_name": image_name,
        "repository": repository,
        "base_revision": base_revision or None,
        "data_source": data_source,
        "task_kind": _first_text(value, "task_kind") or "bugfix",
    }
    # ``SWEInstance`` owns the final public/private split.  Passing hidden
    # values alongside the normalised public fields preserves official test
    # metadata exclusively in evaluator_private.
    return {**public, **hidden}


def eligibility_reason(raw: Mapping[str, Any], *, require_evaluator: bool = True) -> str | None:
    instance_id = _first_text(raw, "instance_id", "id", "instance")
    problem = _first_text(raw, "problem_statement", "text", "problem")
    image = _first_text(raw, "image_name", "docker_image", "image")
    if not instance_id:
        return "missing_instance_id"
    if len(problem) < 24:
        return "incomplete_problem_statement"
    if not image:
        return "missing_image"
    if not _python_compatible(raw):
        return "non_python"
    if _has_bad_environment_status(raw):
        return "known_bad_environment"
    if require_evaluator and not (set(_hidden_values(raw)) & _EVALUATOR_EVIDENCE_KEYS):
        return "missing_evaluator_metadata"
    return None


def iter_hub_rows(spec: SourceSpec, *, dataset: str | None = None, config: str | None = None, split: str | None = None, seed: int | None = None, candidate_limit: int | None = None) -> Iterator[Mapping[str, Any]]:
    """Yield Hub records lazily, with a Dataset Viewer fallback.

    Some Windows builds of ``datasets`` cannot resolve Hub parquet aliases even
    though the public Dataset Viewer is reachable.  The fallback reads the
    same source in pages of at most 100 rows and remains bounded by the caller
    stopping iteration after its candidate window is full.
    """

    dataset_name, config_name, split_name = dataset or spec.dataset, config or spec.config, split or spec.split
    try:
        # The Viewer gives a total row count, so its pages can be sampled in a
        # seed-controlled order. This avoids a hidden first-repository bias
        # while retaining a strictly bounded transfer.
        yield from iter_dataset_viewer_rows(dataset_name, config=config_name, split=split_name, seed=seed, max_rows=candidate_limit)
        return
    except Exception as viewer_error:  # pragma: no cover - depends on Hub/client state
        try:
            from datasets import load_dataset
            stream = load_dataset(dataset_name, config_name, split=split_name, streaming=True)
            yield from stream
            return
        except Exception as dataset_error:
            raise RuntimeError(
                "Unable to stream the SWE source. Install/repair the Hugging Face 'datasets' client or use --input with a small local snapshot. "
                f"Dataset Viewer error: {viewer_error}; datasets error: {dataset_error}"
            ) from dataset_error


def iter_dataset_viewer_rows(dataset: str, *, config: str, split: str, page_size: int = 100, seed: int | None = None, max_rows: int | None = None) -> Iterator[Mapping[str, Any]]:
    """Read the public Dataset Viewer API without downloading source shards."""

    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be in [1, 100]")
    def fetch(offset: int) -> tuple[list[Any], int]:
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": page_size})
        request = urllib.request.Request(f"https://datasets-server.huggingface.co/rows?{query}", headers={"User-Agent": "OpenClaw-RL-CodeAgent/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        page = payload.get("rows")
        if not isinstance(page, list):
            raise RuntimeError("Dataset Viewer returned no rows array")
        total = int(payload.get("num_rows_total", 0))
        if total < 1:
            raise RuntimeError("Dataset Viewer returned no row count")
        return page, total

    first_page, total = fetch(0)
    offsets = list(range(0, total, page_size))
    if seed is not None:
        random.Random(seed).shuffle(offsets)
    emitted = 0
    for offset in offsets:
        page = first_page if offset == 0 else fetch(offset)[0]
        for item in page:
            if isinstance(item, Mapping) and isinstance(item.get("row"), Mapping):
                yield dict(item["row"])
                emitted += 1
                if max_rows is not None and emitted >= max_rows:
                    return
        if len(page) < page_size and seed is None:
            return


def load_excluded_instance_ids(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    values: set[str] = set()
    for row in read_rows(path):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        public = metadata.get("public_instance") if isinstance(metadata.get("public_instance"), Mapping) else {}
        value = _first_text(metadata, "instance_id") or _first_text(public, "instance_id") or _first_text(row, "instance_id")
        if value:
            values.add(value)
    return values


def select_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    num_samples: int,
    seed: int,
    data_source: str,
    candidate_limit: int,
    excluded_instance_ids: set[str] | None = None,
    require_evaluator: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if candidate_limit < num_samples:
        raise ValueError("candidate_limit must be at least num_samples")
    excluded = excluded_instance_ids or set()
    stats: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        stats["scanned"] += 1
        if not isinstance(raw, Mapping):
            stats["malformed"] += 1
            continue
        reason = eligibility_reason(raw, require_evaluator=require_evaluator)
        if reason:
            stats[f"filtered_{reason}"] += 1
            continue
        instance_id = _first_text(raw, "instance_id", "id", "instance")
        if instance_id in excluded:
            stats["filtered_overlap"] += 1
            continue
        if instance_id in seen:
            stats["filtered_duplicate"] += 1
            continue
        try:
            instance = SWEInstance.from_raw(normalize_row(raw, data_source=data_source), data_source=data_source)
        except ValueError:
            stats["filtered_schema"] += 1
            continue
        row = instance.to_manifest_row()
        assert_public_safe(row["text"])
        assert_public_safe(row["metadata"]["public_instance"])
        candidates.append(row)
        seen.add(instance_id)
        stats["eligible"] += 1
        if len(candidates) >= candidate_limit:
            stats["candidate_limit_reached"] += 1
            break
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:num_samples]
    if len(selected) < num_samples:
        raise ValueError(f"only {len(selected)} eligible rows found; requested {num_samples}. Increase --candidate-limit or provide a healthier source snapshot.")
    ids = [str(row["metadata"]["instance_id"]) for row in selected]
    stats["selected"] = len(selected)
    return selected, {"data_source": data_source, "seed": seed, "num_samples": num_samples, "candidate_limit": candidate_limit, "filter_counts": dict(sorted(stats.items())), "instance_ids": ids}


def preprocess(rows: Iterable[Mapping[str, Any]], *, data_source: str = "") -> list[dict[str, Any]]:
    output = []
    for raw in rows:
        instance = SWEInstance.from_raw(raw, data_source=data_source)
        output.append(instance.to_manifest_row())
    return output


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_report(report: Mapping[str, Any], *, output: str | Path, stats_output: str | Path | None = None, ids_output: str | Path | None = None) -> None:
    output_path = Path(output)
    stats_path = Path(stats_output) if stats_output else output_path.with_suffix(".stats.json")
    ids_path = Path(ids_output) if ids_output else output_path.with_suffix(".instance_ids.txt")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ids_path.write_text("\n".join(str(value) for value in report["instance_ids"]) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a small, deterministic, leakage-safe SWE Code JSONL manifest")
    parser.add_argument("--source", required=True, choices=sorted(SOURCE_SPECS))
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", help="small local JSON/JSONL snapshot; skips Hub access")
    parser.add_argument("--dataset", help="override the source's Hugging Face dataset name")
    parser.add_argument("--config", help="override the source's dataset config")
    parser.add_argument("--split", help="override the source's dataset split")
    parser.add_argument("--candidate-limit", type=int, help="maximum eligible streaming candidates to inspect; default is num_samples * 8")
    parser.add_argument("--exclude-jsonl", help="manifest whose instance_id values must not be sampled")
    parser.add_argument("--stats-output")
    parser.add_argument("--instance-ids-output")
    parser.add_argument("--allow-missing-evaluator", action="store_true", help="only for metadata investigation; do not use for train/eval manifests")
    args = parser.parse_args(argv)
    spec = SOURCE_SPECS[args.source]
    candidate_limit = args.candidate_limit or args.num_samples * 8
    raw_rows: Iterable[Mapping[str, Any]] = read_rows(args.input) if args.input else iter_hub_rows(spec, dataset=args.dataset, config=args.config, split=args.split, seed=args.seed, candidate_limit=candidate_limit)
    source_rows = (with_source_image(row, source=args.source) for row in raw_rows)
    selected, report = select_rows(
        source_rows,
        num_samples=args.num_samples,
        seed=args.seed,
        data_source=spec.data_source,
        candidate_limit=candidate_limit,
        excluded_instance_ids=load_excluded_instance_ids(args.exclude_jsonl),
        require_evaluator=not args.allow_missing_evaluator,
    )
    report = {**report, "source": args.source, "dataset": args.dataset or spec.dataset, "config": args.config or spec.config, "split": args.split or spec.split}
    write_jsonl(selected, args.output)
    write_report(report, output=args.output, stats_output=args.stats_output, ids_output=args.instance_ids_output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["SOURCE_SPECS", "SourceSpec", "eligibility_reason", "iter_dataset_viewer_rows", "iter_hub_rows", "load_excluded_instance_ids", "main", "normalize_row", "preprocess", "read_rows", "select_rows", "source_image_name", "with_source_image", "write_jsonl", "write_report"]
