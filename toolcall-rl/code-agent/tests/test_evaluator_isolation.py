import asyncio

from env.client import LocalCodeEnvClient
from env.evaluator import CleanEvaluator, EvaluatorRequest


def test_clean_evaluator_uses_fresh_lease(git_repo, patch_value):
    async def run():
        client = LocalCodeEnvClient(git_repo)
        interaction = await client.allocate("local", "i")
        assert (await client.apply_patch(interaction.lease_id, patch_value(), cwd=interaction.cwd)).ok
        candidate = await client.diff(interaction.lease_id, cwd=interaction.cwd)
        result = await CleanEvaluator(client).evaluate(EvaluatorRequest("local", "i", candidate, "python -m pytest -q", timeout=60))
        assert result.ok and result.resolved
        await client.close(interaction.lease_id)

    asyncio.run(run())


def test_new_file_patch_survives_diff_and_fresh_evaluation(git_repo):
    async def run():
        client = LocalCodeEnvClient(git_repo)
        interaction = await client.allocate("local", "new-file")
        patch = """diff --git a/helper.py b/helper.py
new file mode 100644
--- /dev/null
+++ b/helper.py
@@ -0,0 +1 @@
+ANSWER = 42
"""
        applied = await client.apply_patch(interaction.lease_id, patch, cwd=interaction.cwd)
        assert applied.ok
        candidate = await client.diff(interaction.lease_id, cwd=interaction.cwd)
        assert "diff --git a/helper.py b/helper.py" in candidate
        evaluated = await CleanEvaluator(client).evaluate(
            EvaluatorRequest("local", "new-file", candidate, "python -c \"from helper import ANSWER; assert ANSWER == 42\"", timeout=60)
        )
        assert evaluated.ok and evaluated.resolved
        await client.close(interaction.lease_id)

    asyncio.run(run())
