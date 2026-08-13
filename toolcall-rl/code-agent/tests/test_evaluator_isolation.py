import asyncio
import subprocess

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


def test_local_lease_baseline_includes_files_ignored_by_host_config(git_repo, tmp_path, monkeypatch):
    """A host-wide excludes file must not alter a disposable task baseline."""

    excludes = tmp_path / "global-excludes"
    excludes.write_text("test_app.py\n", encoding="utf-8")
    global_config = tmp_path / "gitconfig"
    global_config.write_text(f"[core]\n\texcludesFile = {excludes.as_posix()}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))

    async def run():
        client = LocalCodeEnvClient(git_repo)
        lease = await client.allocate("local", "global-excludes")
        root = client.root_for_lease(lease.lease_id)
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "test_app.py"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert tracked.returncode == 0
        await client.close(lease.lease_id)

    asyncio.run(run())
