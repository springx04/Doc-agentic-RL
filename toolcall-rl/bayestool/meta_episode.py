"""Persistent-session meta episodes over multiple questions in one document."""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .belief import BeliefRuntime
from .config import BayesToolConfig, default_config
from .training import suffix_meta_returns


@dataclass
class MetaQuestion:
    prompt: str
    label: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaEpisodeState:
    meta_episode_id: str
    episode_content_id: str
    document_path: str
    questions: list[MetaQuestion]
    world_id: str | None = None
    question_index: int = 0
    session_belief: BeliefRuntime | None = None
    task_belief: BeliefRuntime | None = None
    completed_utilities: list[float] = field(default_factory=list)
    task_records: list[dict[str, Any]] = field(default_factory=list)

    def start_question(self, index: int | None = None) -> MetaQuestion:
        if index is not None:
            self.question_index = int(index)
        if not 0 <= self.question_index < len(self.questions):
            raise IndexError("meta episode question index out of range")
        question = self.questions[self.question_index]
        if self.session_belief is not None:
            # Restore only the persistent session/shared posterior and
            # recurrent hidden state.  BeliefRuntime.from_replay_record
            # intentionally leaves page/context history empty for this new
            # question.
            session_record = self.session_belief.export_replay_record()
            self.task_belief = BeliefRuntime.from_replay_record(
                session_record,
                self.session_belief.config,
                document_digest=self.session_belief.document_digest,
                seed=self.question_index,
                model=self.session_belief.model,
                model_version=self.session_belief.model_version,
            )
        else:
            self.task_belief = BeliefRuntime(
                default_config(enabled=True),
                seed=self.question_index,
            )
        return question

    def finish_question(self, utility: float, *, task_record: Mapping[str, Any] | None = None) -> None:
        if self.task_belief is not None:
            if self.session_belief is None:
                self.session_belief = copy.deepcopy(self.task_belief)
            else:
                # Persist only session/shared belief.  Local context and task
                # history are intentionally discarded before the next query.
                self.session_belief._session_probs = self.task_belief._session_probs
                self.session_belief._regime_probs = self.task_belief._regime_probs
                self.session_belief._family_probs = copy.deepcopy(self.task_belief._family_probs)
                self.session_belief._quality_params = copy.deepcopy(self.task_belief._quality_params)
                self.session_belief._cost_params = copy.deepcopy(self.task_belief._cost_params)
                self.session_belief._change_probability = self.task_belief._change_probability
                self.session_belief._ood_score = self.task_belief._ood_score
                self.session_belief._session_hidden = copy.deepcopy(self.task_belief._session_hidden)
                self.session_belief._shared_hidden = copy.deepcopy(self.task_belief._shared_hidden)
        self.completed_utilities.append(float(utility))
        if task_record is not None:
            self.task_records.append(dict(task_record))
        self.task_belief = None
        self.question_index += 1

    @property
    def done(self) -> bool:
        return self.question_index >= len(self.questions)

    def suffix_returns(self, discount: float | None = None) -> list[float]:
        discount = discount if discount is not None else (self.session_belief.config.meta.discount if self.session_belief else 0.95)
        return suffix_meta_returns(self.completed_utilities, discount=discount)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "meta_episode_id": self.meta_episode_id,
            "episode_content_id": self.episode_content_id,
            "document_path": self.document_path,
            "world_id": self.world_id,
            "question_index": self.question_index,
            "question_count": len(self.questions),
            "completed_utilities": list(self.completed_utilities),
            "suffix_returns": self.suffix_returns(),
            "session_belief": self.session_belief.export_replay_record() if self.session_belief else None,
            "meta_trajectory_ids": [f"{self.meta_episode_id}:q{index}" for index in range(len(self.questions))],
        }


def build_meta_episode(
    records: Sequence[Mapping[str, Any]],
    *,
    config: BayesToolConfig | None = None,
    seed: int = 0,
) -> MetaEpisodeState:
    config = config or default_config(enabled=True)
    if not records:
        raise ValueError("cannot build a meta episode from no records")
    document_path = str(records[0].get("document_path") or records[0].get("metadata", {}).get("document_path") or "")
    selected = [record for record in records if str(record.get("document_path") or record.get("metadata", {}).get("document_path") or "") == document_path]
    if len(selected) < config.meta.questions_per_episode_min:
        raise ValueError(
            f"meta episode requires at least {config.meta.questions_per_episode_min} questions for {document_path!r}; "
            f"found {len(selected)}"
        )
    rng = random.Random(seed)
    rng.shuffle(selected)
    count = min(len(selected), config.meta.questions_per_episode_max)
    selected = selected[:count]
    digest = hashlib.sha256((document_path + "|" + "|".join(str(item.get("id", index)) for index, item in enumerate(selected))).encode("utf-8")).hexdigest()[:24]
    episode_content_id = hashlib.sha256(document_path.encode("utf-8", "surrogatepass")).hexdigest()[:24]
    questions = [
        MetaQuestion(
            prompt=str(item.get("question") or item.get("query") or item.get("prompt") or ""),
            label=item.get("label", item.get("answers")),
            metadata=dict(item.get("metadata") or {}) | {
                "question_order": index,
                "episode_content_id": episode_content_id,
                "meta_trajectory_id": f"meta-{digest}:q{index}",
            },
        )
        for index, item in enumerate(selected)
    ]
    return MetaEpisodeState(
        meta_episode_id=f"meta-{digest}",
        episode_content_id=episode_content_id,
        document_path=document_path,
        questions=questions,
    )


__all__ = ["MetaQuestion", "MetaEpisodeState", "build_meta_episode"]
