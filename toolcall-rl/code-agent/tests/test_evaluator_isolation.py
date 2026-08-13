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


def test_clean_evaluator_applies_private_test_patch_without_exposing_it(git_repo):
    evaluator_patch = """diff --git a/hidden_oracle.py b/hidden_oracle.py
new file mode 100644
--- /dev/null
+++ b/hidden_oracle.py
@@ -0,0 +1,2 @@
+from app import VALUE
+
+assert VALUE == 2
"""
    candidate_patch = """diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""

    async def run():
        result = await CleanEvaluator(LocalCodeEnvClient(git_repo)).evaluate(
            EvaluatorRequest("local", "private-test", candidate_patch, "python -c \"import hidden_oracle\"", timeout=60, evaluator_patch=evaluator_patch)
        )
        assert result.ok and result.resolved
        assert "hidden_oracle" not in result.output

    asyncio.run(run())


def test_clean_evaluator_rejects_candidate_patch_to_private_test_path(git_repo):
    private_test_patch = """diff --git a/test_app.py b/test_app.py
--- a/test_app.py
+++ b/test_app.py
@@ -1,4 +1,4 @@
 from app import VALUE

 def test_value():
-    assert VALUE == 2
+    assert VALUE == 3
"""

    async def run():
        result = await CleanEvaluator(LocalCodeEnvClient(git_repo)).evaluate(
            EvaluatorRequest("local", "private-path", private_test_patch, "python -m pytest -q", timeout=60, evaluator_patch=private_test_patch)
        )
        assert result.ok and not result.resolved
        assert "evaluator-private" in result.output

    asyncio.run(run())
