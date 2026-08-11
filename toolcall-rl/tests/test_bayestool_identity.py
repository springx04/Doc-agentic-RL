import hashlib

from build_bayestool_data import coupling_id
from bayestool.identity import canonical_task_question, make_coupling_id


def test_coupling_identity_uses_question_not_prompt_wrapper(tmp_path):
    document = tmp_path / "document.pdf"
    document.write_bytes(b"document-bytes")
    prompt = (
        f"Document path: {document}\n"
        "Question: What is the invoice number?\n\n"
        "Inspect the document with the available document tools."
    )
    expected_digest = hashlib.sha256(b"document-bytes").hexdigest()
    expected = make_coupling_id(expected_digest, "What is the invoice number?", "")
    record = {"prompt": prompt, "metadata": {"document_path": str(document)}}

    assert canonical_task_question(prompt) == "What is the invoice number?"
    assert coupling_id(record) == expected


def test_builder_can_hash_deployment_path_from_local_document_root(tmp_path):
    document = tmp_path / "train_000001.pdf"
    document.write_bytes(b"deployment-document")
    deployment_path = "/workspace/data/OpenClaw-RL/data/train/pdfs/train_000001.pdf"
    prompt = f"Document path: {deployment_path}\nQuestion: Which page?\n\nAnswer."
    record = {"prompt": prompt, "metadata": {"document_path": deployment_path}}

    expected_digest = hashlib.sha256(b"deployment-document").hexdigest()
    assert coupling_id(record, document_root=tmp_path) == make_coupling_id(
        expected_digest, "Which page?", ""
    )
