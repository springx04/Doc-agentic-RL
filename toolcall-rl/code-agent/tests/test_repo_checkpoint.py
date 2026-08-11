from env.checkpoint import CodeRepoCheckpoint, patch_sha256, repo_state_digest


def test_repo_checkpoint_digest_is_normalized_but_patch_is_preserved():
    patch = "diff --git a/a.py b/a.py\r\n--- a/a.py\r\n+++ b/a.py\r\n"
    record = CodeRepoCheckpoint.create(instance_id="i", image_name="img", base_revision="r", cwd="/testbed", patch=patch)
    assert record.patch == patch
    assert record.patch_sha256 == patch_sha256(patch)
    assert record.repo_state_digest == repo_state_digest("i", "img", "r", patch)
    assert CodeRepoCheckpoint.from_dict(record.to_dict()) == record
