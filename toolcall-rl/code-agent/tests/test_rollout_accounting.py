from __future__ import annotations

import asyncio
from pathlib import Path

from env.client import LocalCodeEnvClient
from rollout import CodeRollout


class _TokenModel:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    async def generate(self, prompt: str, **_: object) -> dict[str, object]:
        return dict(self.payload)


def _sample(repo: Path) -> dict[str, object]:
    return {
        "instance_id": "accounting",
        "problem_statement": "Inspect the repository and report completion.",
        "image_name": "local",
        "repository_root": str(repo),
    }


def test_rollout_preserves_model_token_accounting_exactly(git_repo: Path) -> None:
    model = _TokenModel(
        {
            "text": "<final>done</final>",
            "token_ids": [7, 8],
            "token_mask": [1, 1],
            "token_logprobs": [-0.25, -0.5],
        }
    )

    result = asyncio.run(
        CodeRollout().run(
            _sample(git_repo),
            model_client=model,
            code_env_client=LocalCodeEnvClient(git_repo),
        )
    )

    assert list(result.tokens) == [7, 8]
    assert list(result.loss_mask) == [1, 1]
    assert list(result.rollout_log_probs) == [-0.25, -0.5]
    assert result.metadata["policy_gradient_eligible"] is True
    assert result.metadata["failure_penalty"] > 0


def test_rollout_rejects_partial_model_token_accounting(git_repo: Path) -> None:
    model = _TokenModel({"text": "<final>done</final>", "token_ids": [7]})

    result = asyncio.run(
        CodeRollout().run(
            _sample(git_repo),
            model_client=model,
            code_env_client=LocalCodeEnvClient(git_repo),
        )
    )

    assert result.metadata["termination_reason"] == "invalid_token_accounting"
    assert result.metadata["valid_for_rl"] is False
    assert list(result.tokens) == []
    assert list(result.loss_mask) == []
    assert list(result.rollout_log_probs) == []
