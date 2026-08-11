"""Public/evaluator-separated SWE dataset records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .leakage_guard import assert_public_safe, split_public_private


@dataclass(frozen=True)
class PublicInstance:
    instance_id: str
    problem_statement: str
    image_name: str
    base_revision: str | None = None
    data_source: str = ""
    repository: str = ""
    task_kind: str = "bugfix"
    public_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "instance_id": self.instance_id,
            "problem_statement": self.problem_statement,
            "image_name": self.image_name,
            "base_revision": self.base_revision,
            "data_source": self.data_source,
            "repository": self.repository,
            "task_kind": self.task_kind,
            **self.public_metadata,
        }
        assert_public_safe(value)
        return value


@dataclass(frozen=True)
class EvaluatorPrivate:
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SWEInstance:
    public: PublicInstance
    evaluator_private: EvaluatorPrivate = field(default_factory=EvaluatorPrivate)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any], *, data_source: str = "") -> "SWEInstance":
        # ``preprocess`` writes a transport-safe manifest with public and
        # evaluator-private fields nested under metadata.  Training entry
        # points must be able to reload that exact format without ever moving
        # evaluator labels into the public instance.
        source = dict(raw)
        nested_private: dict[str, Any] = {}
        metadata = source.get("metadata")
        if isinstance(metadata, Mapping) and isinstance(metadata.get("public_instance"), Mapping):
            source = dict(metadata["public_instance"])
            if raw.get("text") and not source.get("problem_statement"):
                source["problem_statement"] = raw["text"]
            private_value = metadata.get("evaluator_private")
            if isinstance(private_value, Mapping):
                nested_private = dict(private_value)
        public_raw, private_raw = split_public_private(source)
        private_raw = {**private_raw, **nested_private}
        instance_id = str(public_raw.get("instance_id") or public_raw.get("id") or public_raw.get("instance") or "")
        problem = str(public_raw.get("problem_statement") or public_raw.get("text") or public_raw.get("problem") or "")
        image = str(public_raw.get("image_name") or public_raw.get("docker_image") or public_raw.get("image") or "")
        base_revision = public_raw.get("base_revision") or public_raw.get("base_commit") or public_raw.get("commit")
        public = PublicInstance(
            instance_id=instance_id,
            problem_statement=problem,
            image_name=image,
            base_revision=str(base_revision) if base_revision else None,
            data_source=str(public_raw.get("data_source") or data_source),
            repository=str(public_raw.get("repo") or public_raw.get("repository") or ""),
            task_kind=str(public_raw.get("task_kind") or "bugfix"),
            public_metadata={key: value for key, value in public_raw.items() if key not in {"instance_id", "id", "instance", "problem_statement", "text", "problem", "image_name", "docker_image", "image", "base_revision", "base_commit", "commit", "data_source", "repo", "repository", "task_kind"}},
        )
        if not public.instance_id or not public.problem_statement:
            raise ValueError("SWE instance requires instance_id and problem_statement")
        assert_public_safe(public.to_dict())
        return cls(public, EvaluatorPrivate(private_raw))

    def to_runtime_row(self) -> dict[str, Any]:
        """Return only policy-safe fields for rollout creation."""

        return {"text": self.public.problem_statement, "environment": "code", "metadata": {"environment": "code", "public_instance": self.public.to_dict()}}

    def to_manifest_row(self) -> dict[str, Any]:
        public = self.public.to_dict()
        # Keep the three routing fields convenient for data tooling while
        # retaining one canonical policy-visible object.  They are duplicated
        # rather than reconstructed from evaluator-private records so the
        # split remains auditable after a JSONL has been copied elsewhere.
        return {
            "text": self.public.problem_statement,
            "environment": "code",
            "metadata": {
                "environment": "code",
                "instance_id": self.public.instance_id,
                "data_source": self.public.data_source,
                "image_name": self.public.image_name,
                "public_instance": public,
                "evaluator_private": dict(self.evaluator_private.values),
            },
        }


__all__ = ["EvaluatorPrivate", "PublicInstance", "SWEInstance"]
