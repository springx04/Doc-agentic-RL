import asyncio
from dataclasses import replace
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


TOOLCALL_DIR = Path(__file__).resolve().parents[1]


def _load_generator(monkeypatch):
    """Load the rollout module with the unavailable slime runtime stubbed."""
    fake_slime = types.ModuleType("slime")
    fake_rollout = types.ModuleType("slime.rollout")
    fake_sglang = types.ModuleType("slime.rollout.sglang_rollout")
    fake_utils = types.ModuleType("slime.utils")
    fake_http = types.ModuleType("slime.utils.http_utils")
    fake_types = types.ModuleType("slime.utils.types")
    fake_processing = types.ModuleType("slime.utils.processing_utils")

    class FakeSample:
        class Status:
            COMPLETED = "completed"
            TRUNCATED = "truncated"
            FAILED = "failed"

    class FakeGenerateState:
        def __init__(self, args):
            self.tokenizer = args.fake_tokenizer
            self.processor = args.fake_processor

    fake_sglang.GenerateState = FakeGenerateState
    fake_http.post = None
    fake_types.Sample = FakeSample
    fake_processing.encode_image_for_rollout_engine = lambda image: "encoded-image"

    for name, module in {
        "slime": fake_slime,
        "slime.rollout": fake_rollout,
        "slime.rollout.sglang_rollout": fake_sglang,
        "slime.utils": fake_utils,
        "slime.utils.http_utils": fake_http,
        "slime.utils.types": fake_types,
        "slime.utils.processing_utils": fake_processing,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.syspath_prepend(str(TOOLCALL_DIR))
    module_name = "generate_with_retool_test_module"
    spec = importlib.util.spec_from_file_location(module_name, TOOLCALL_DIR / "generate_with_retool.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, FakeSample


def test_world_injected_failure_is_a_valid_sendable_observation(monkeypatch):
    module, _ = _load_generator(monkeypatch)

    assert module._is_world_injected_observation(
        {"failure_origin": "world_injected", "success": False}
    )
    assert module._is_world_injected_observation(
        {
            "success": False,
            "world_event": {"failure_origin": "world_injected"},
        }
    )
    assert not module._is_world_injected_observation(
        {"failure_origin": "real_infrastructure", "success": False}
    )
    assert not module._is_world_injected_observation({"success": False})


def test_bayestool_dvoi_mapping_is_reduced_only_for_branch_ranking(monkeypatch):
    module, _ = _load_generator(monkeypatch)

    assert module._bayestool_dvoi_score({"render_page": 0.25, "ocr_region": 0.7}) == 0.7
    assert module._bayestool_dvoi_score({"invalid": "not-a-number"}) == 0.0
    assert module._bayestool_dvoi_score(0.4) == 0.4


def test_image_context_compaction_keeps_follow_up_generation_possible(monkeypatch):
    module, _ = _load_generator(monkeypatch)
    segments = [
        {
            "action_token_ids": [1],
            "observation_token_ids": list(range(2, 200)),
            "text_observation_token_ids": [2, 3, 4],
            "image_data": ["rendered-page.png"],
        }
    ]

    context, image_data = module._compact_model_context(
        [0], segments, max_context_length=100, reserve_tokens=32
    )

    assert len(context) <= 68
    assert context == [0, 1, 2, 3, 4]
    assert image_data == []


def test_observation_compaction_preserves_assistant_turn_boundary(monkeypatch):
    module, _ = _load_generator(monkeypatch)
    observation = list(range(2, 202))
    segments = [
        {
            "action_token_ids": [1],
            "observation_token_ids": observation,
            "observation_suffix_token_count": 4,
            "image_data": [],
        }
    ]

    context, image_data = module._compact_model_context(
        [0], segments, max_context_length=100, reserve_tokens=32
    )

    # The fitted observation is 67 tokens: a prefix plus the four-token
    # ``<|im_end|>...assistant`` suffix.  The next request must end at the
    # assistant turn boundary, never in the middle of the observation.
    assert len(context) == 68
    assert context[-4:] == observation[-4:]
    assert image_data == []


def test_training_trajectory_cap_preserves_suffix_alignment(monkeypatch):
    module, _ = _load_generator(monkeypatch)
    response = list(range(30))
    masks = [index % 2 for index in response]
    log_probs = [float(index) for index in response]
    segments = [
        {
            "action_token_start": 0,
            "action_token_end": 2,
            "response_token_start": 0,
            "response_token_end": 8,
            "multimodal_train_input_index": None,
        },
        {
            "action_token_start": 8,
            "action_token_end": 10,
            "response_token_start": 8,
            "response_token_end": 16,
            "multimodal_train_input_index": None,
        },
        {
            "action_token_start": 16,
            "action_token_end": 18,
            "response_token_start": 16,
            "response_token_end": 24,
            "multimodal_train_input_index": None,
        },
    ]
    action_spans = [
        {"token_start": 0, "token_end": 2},
        {"token_start": 8, "token_end": 10},
        {"token_start": 16, "token_end": 18},
        {"token_start": 24, "token_end": 30},
    ]

    capped = module._cap_training_trajectory(
        [100, 101, 102, 103],
        response,
        masks,
        log_probs,
        segments,
        [],
        action_spans,
        max_sequence_length=22,
    )

    trimmed_response, trimmed_masks, trimmed_log_probs, _, metadata = capped
    assert trimmed_response == response[16:]
    assert trimmed_masks == masks[16:]
    assert trimmed_log_probs == log_probs[16:]
    assert metadata["response_start"] == 16
    assert metadata["training_total_length"] == 18
    assert metadata["fallback_tail_cut"] is False


class _FakeTokenizer:
    chat_template = ""

    def __init__(self, responses):
        self.responses = responses

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(1000, 1000 + max(2, len(str(text)) // 20 + 2)))}

    def convert_tokens_to_ids(self, token):
        return 999 if token == "<|im_end|>" else 42

    def decode(self, token_ids):
        return self.responses[tuple(token_ids)]


class _FakeProcessor:
    image_token = "<|image_pad|>"
    vision_start_token = "<|vision_start|>"
    vision_end_token = "<|vision_end|>"
    image_token_id = 42

    def __call__(self, text, images, return_tensors="pt"):
        return {"input_ids": [[1, 42, 2]], "pixel_values": object()}


def _sample(FakeSample, prompt="What is shown on page 1?"):
    sample = FakeSample()
    sample.prompt = prompt
    sample.label = "Page 1 shows a diagram"
    sample.metadata = {}
    sample.rollout_log_probs = []
    sample.status = None
    sample.remove_sample = False
    return sample


def test_tool_prompt_preserves_qwen_role_boundaries_and_function_schema(monkeypatch):
    module, _ = _load_generator(monkeypatch)
    tool = {
        "type": "function",
        "function": {
            "name": "render_page",
            "description": "Render one page.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    prompt = module.format_conversation_with_tools(
        "Document path: fixture.pdf\nQuestion: What is shown?",
        tools=[tool],
        system_prompt="Inspect the document with the available tools.",
        tool_call_format="json",
    )

    assert prompt.startswith("<|im_start|>system\nInspect the document")
    assert "<|im_start|>systemInspect" not in prompt
    assert "<|im_start|>user\nDocument path: fixture.pdf" in prompt
    assert "<|im_start|>userDocument path: fixture.pdf" not in prompt
    assert "<tools>\n{\"type\": \"function\", \"function\":" in prompt
    assert "</tool_call><|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_multi_action_and_placeholder_are_logged_without_execution(monkeypatch):
    module, _ = _load_generator(monkeypatch)

    multi_action = (
        '<tool_call>{"name":"render_page","arguments":{"page_number":1}}</tool_call>'
        '<tool_call>{"name":"render_page","arguments":{"page_number":2}}</tool_call>'
    )
    stats = {}
    trace = []
    _, done = asyncio.run(module.execute_predictions(multi_action, trace, stats, turn=1))
    assert done is True
    assert stats["candidate_action_count"] == 2
    assert stats["executed_action_count"] == 0
    assert stats["protocol_error_count"] == 1
    assert stats["invalid_action_count"] == 2
    assert stats["_terminal_status"] == "model_protocol_error"

    placeholder = (
        '<tool_call>{"name":"render_page","arguments":'
        '{"document_path":"/path/to/file.pdf","page_number":1}}</tool_call>'
    )
    stats = {}
    trace = []
    _, done = asyncio.run(module.execute_predictions(placeholder, trace, stats, turn=1))
    assert done is True
    assert stats["candidate_action_count"] == 1
    assert stats["executed_action_count"] == 0
    assert stats["invalid_action_count"] == 1
    assert stats["protocol_error_count"] == 1


def test_render_page_observation_is_nonempty_and_next_generation_receives_image(monkeypatch, tmp_path):
    render_path = tmp_path / "render.png"
    Image.new("RGB", (8, 8), "white").save(render_path)
    render_action = (
        '<tool_call>{"name":"render_page","arguments":'
        '{"document_path":"/workspace/data/a.pdf","page_number":1}}</tool_call>'
    )
    final_action = "<final>Page 1 shows a diagram.</final>"
    tokenizer = _FakeTokenizer({(101,): render_action, (102,): final_action})
    processor = _FakeProcessor()
    module, FakeSample = _load_generator(monkeypatch)
    module._TOOL_CALL_FORMAT = "json"
    monkeypatch.setattr(module, "GenerateState", lambda args: SimpleNamespace(tokenizer=tokenizer, processor=processor))

    calls = []
    responses = iter(
        [
            {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 101]]}},
            {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 102]]}},
            {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 102]]}},
        ]
    )

    async def fake_post(url, payload):
        calls.append(payload)
        return next(responses)

    async def fake_render(name, arguments):
        assert name == "render_page"
        return json.dumps({"status": "ok", "tool": name, "image_path": str(render_path)})

    monkeypatch.setattr(module, "post", fake_post)
    monkeypatch.setattr(module.tool_registry, "execute_tool", fake_render)
    args = SimpleNamespace(
        fake_tokenizer=tokenizer,
        fake_processor=processor,
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_context_len=4096,
        rollout_max_response_len=32,
        eval_max_context_len=None,
        prm_enable=False,
    )
    result = asyncio.run(
        module.generate(
            args,
            _sample(FakeSample, "What is shown in the image?"),
            {"max_new_tokens": 32},
        )
    )

    assert len(calls) == 2, "render_page must be followed by a real second generation request"
    assert calls[0].get("image_data") is None
    assert calls[1].get("image_data") == ["encoded-image"]
    assert result.response.endswith(final_action)
    assert result.rollout_status == "completed"
    assert result.valid_for_rl is True
    assert result.tool_call_count == 1
    assert result.metadata["generation_steps"][1]["assistant_output_token_count"] > 0
    assert result.metadata["generation_steps"][1]["image_token_count"] == 1


def test_empty_generation_keeps_backend_stop_token_diagnostics(monkeypatch):
    tokenizer = _FakeTokenizer({(999,): "<|im_end|>"})
    processor = _FakeProcessor()
    module, FakeSample = _load_generator(monkeypatch)
    module._TOOL_CALL_FORMAT = "json"
    monkeypatch.setattr(module, "GenerateState", lambda args: SimpleNamespace(tokenizer=tokenizer, processor=processor))

    async def stop_only_generation(url, payload):
        return {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 999]]}}

    monkeypatch.setattr(module, "post", stop_only_generation)
    args = SimpleNamespace(
        fake_tokenizer=tokenizer,
        fake_processor=processor,
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_context_len=4096,
        rollout_max_response_len=32,
        eval_max_context_len=None,
        prm_enable=False,
    )
    result = asyncio.run(module.generate(args, _sample(FakeSample, "Document path: test.pdf\nQuestion: Who is the supplier?"), {"max_new_tokens": 32}))
    step = result.metadata["generation_steps"][0]
    assert step["generation_called"] is True
    assert step["raw_generation_text"] == ""
    assert step["backend_raw_generation_text"] == "<|im_end|>"
    assert step["output_token_count"] == 0
    assert step["finish_reason"] == "stop"
    assert step["generation_error"] is None
    assert result.rollout_status == "generation_empty"
    assert result.valid_for_rl is False
    assert result.metadata["exclude_from_group_statistics"] is True


def test_bayestool_branch_uses_shared_prefix_and_real_candidate_tokens(monkeypatch):
    current = '<tool_call>{"name":"render_page","arguments":{"document_path":"/workspace/a.pdf","page_number":1}}</tool_call>'
    alternate = '<tool_call>{"name":"extract_table","arguments":{"document_path":"/workspace/a.pdf","page_number":1}}</tool_call>'
    third = '<tool_call>{"name":"ocr_region","arguments":{"document_path":"/workspace/a.pdf","page_number":1,"bbox":[0,0,1,1]}}</tool_call>'
    tokenizer = _FakeTokenizer({(101,): current, (102,): alternate, (103,): third})
    module, FakeSample = _load_generator(monkeypatch)
    FakeSample.Status.PENDING = "pending"
    state = SimpleNamespace(tokenizer=tokenizer)
    responses = iter(
        [
            {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.2, 102]]}},
            {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.3, 103]]}},
        ]
    )

    async def fake_post(url, payload):
        return next(responses)

    async def fake_generate(args, child, sampling_params, evaluation=False):
        child.metadata["fake_branch_generation"] = True
        return child

    monkeypatch.setattr(module, "post", fake_post)
    monkeypatch.setattr(module, "generate", fake_generate)
    from bayestool.config import default_config

    config = replace(
        default_config(enabled=True),
        decision_regret_threshold=-1.0,
        branch_probability_when_eligible=1.0,
        max_action_candidates=3,
        max_siblings=3,
    )
    belief = module.BeliefRuntime(config, document_digest="doc", seed=4)
    sample = _sample(FakeSample, "Question")
    sample.index = 7
    sample.group_index = 2
    sample.status = FakeSample.Status.PENDING
    sample.metadata["sibling_group_id"] = "base-group"
    checkpoint = module._capture_bayestool_branch_checkpoint(
        prompt_token_ids=[1, 2],
        context_token_ids=[1, 2, 3],
        context_image_data=[],
        context_segments=[],
        response="prefix",
        response_token_ids=[3],
        loss_masks=[1],
        rollout_log_probs=[-0.1],
        current_images=[],
        multimodal_train_inputs_buffer=[],
        execution_trace=[],
        action_log={"actions": []},
        generation_steps=[],
        step_action_spans=[],
        navigation_state={"question_type": "table", "remaining_tool_budget": 5},
        world_runtime=None,
        belief_runtime=belief,
        bayes_document_digest="doc",
        bayes_coupling_id="coupling",
        world_sample_index=0,
        turn=0,
        tool_call_count=0,
        max_tool_steps=8,
        config=config,
    )
    children, event = asyncio.run(
        module._launch_bayestool_branches(
            args=SimpleNamespace(),
            state=state,
            url="http://router/generate",
            sample=sample,
            sampling_params={"max_new_tokens": 32},
            evaluation=False,
            checkpoint=checkpoint,
            current_text=current,
            current_token_ids=[101],
            current_log_probs=[-0.1],
            im_end_id=999,
            decision_controller=module.DecisionController(config, seed=1),
            bayes_config=config,
        )
    )
    assert event["branch_triggered"] is True
    assert len(children) == 2
    assert all(child.metadata["bayestool_branch_shared_prefix"] for child in children)
    assert all(child.metadata["bayestool_branch_prefix_hash"] == checkpoint["prefix_hash"] for child in children)
    assert all(child.metadata["fake_branch_generation"] for child in children)
    assert all(child.metadata["bayestool_branch_action_key"] for child in children)
    for child in children:
        module._BAYES_BRANCH_CHECKPOINTS.pop(child.metadata["bayestool_branch_resume_id"], None)


def test_explicit_plan_force_branch_constructs_exact_k_even_when_gate_would_skip(monkeypatch):
    module, FakeSample = _load_generator(monkeypatch)
    FakeSample.Status.PENDING = "pending"
    from bayestool.config import default_config
    from bayestool.grouping import make_question_rollout_plan

    def action(tool_name: str) -> dict[str, object]:
        return {
            "kind": "tool",
            "tool": tool_name,
            "arguments": {"document_path": "/workspace/a.pdf", "page_number": 1},
        }

    raw = {
        name: module._bayestool_action_text(action(name))
        for name in ("render_page", "extract_table", "ocr_region", "detect_layout")
    }
    candidates = [
        {
            "key": module.canonical_action_key(action(name)),
            "action": action(name),
            "raw": raw[name],
            "token_ids": [100 + index],
            "log_probs": [-0.1],
            "source": "test",
        }
        for index, name in enumerate(raw)
    ]
    config = replace(
        default_config(enabled=True),
        default_group_size=4,
        max_action_candidates=4,
        max_siblings=4,
        branch_probability_when_eligible=0.0,
    )
    belief = module.BeliefRuntime(config, document_digest="doc", seed=9)
    sample = _sample(FakeSample, "Question")
    sample.index = 3
    sample.status = FakeSample.Status.PENDING
    sample.metadata.update(
        {
            "question_rollout_plan": make_question_rollout_plan("q", group_size=4).to_dict(),
            "decision_group_size": 4,
            "question_id": "q",
            "world_slot_role": "healthy",
            "variant_id": "base",
        }
    )
    checkpoint = module._capture_bayestool_branch_checkpoint(
        prompt_token_ids=[1, 2],
        context_token_ids=[1, 2, 3],
        context_image_data=[],
        context_segments=[],
        response="prefix",
        response_token_ids=[3],
        loss_masks=[1],
        rollout_log_probs=[-0.1],
        current_images=[],
        multimodal_train_inputs_buffer=[],
        execution_trace=[],
        action_log={"actions": []},
        generation_steps=[],
        step_action_spans=[],
        navigation_state={"question_type": "table", "remaining_tool_budget": 5},
        world_runtime=None,
        belief_runtime=belief,
        bayes_document_digest="doc",
        bayes_coupling_id="coupling",
        world_sample_index=0,
        turn=0,
        tool_call_count=0,
        max_tool_steps=8,
        config=config,
    )

    async def fake_generate(args, child, sampling_params, evaluation=False):
        return child

    monkeypatch.setattr(module, "generate", fake_generate)
    children, event = asyncio.run(
        module._launch_bayestool_branches(
            args=SimpleNamespace(),
            state=SimpleNamespace(tokenizer=None),
            url="http://router/generate",
            sample=sample,
            sampling_params={"max_new_tokens": 32},
            evaluation=False,
            checkpoint=checkpoint,
            current_text=raw["render_page"],
            current_token_ids=[100],
            current_log_probs=[-0.1],
            im_end_id=None,
            decision_controller=module.DecisionController(config, seed=2),
            bayes_config=config,
            candidates_override=candidates,
            force_branch=True,
        )
    )
    assert event["force_branch"] is True
    assert event["target_group_size"] == 4
    assert len(children) == 3
    for child in children:
        module._BAYES_BRANCH_CHECKPOINTS.pop(child.metadata["bayestool_branch_resume_id"], None)


def test_bayestool_filter_and_q_checkpoints_load_into_rollout_worker(monkeypatch, tmp_path):
    import torch

    module, _ = _load_generator(monkeypatch)
    from bayestool.belief import ToolWorldFilterNetwork
    from bayestool.config import default_config
    from bayestool.decision import BayesQHead, Q_FEATURE_SCHEMA_VERSION

    config = default_config(enabled=True)
    belief_path = tmp_path / "belief.pt"
    q_path = tmp_path / "qhead.pt"
    torch.save(
        {
            "model_state": ToolWorldFilterNetwork(config).state_dict(),
            "belief_model_version": "stage-a-test",
        },
        belief_path,
    )
    torch.save(
        {
            "model_state": BayesQHead().state_dict(),
            "model_version": "q-test",
            "q_feature_schema_version": Q_FEATURE_SCHEMA_VERSION,
        },
        q_path,
    )

    filter_model, filter_version, q_head, q_version = module._load_bayestool_models(
        SimpleNamespace(
            bayestool_belief_checkpoint=str(belief_path),
            bayestool_q_checkpoint=str(q_path),
        ),
        config,
    )
    assert filter_model is not None
    assert filter_version == "stage-a-test"
    assert q_head is not None
    assert q_version == "q-test"


def test_simple_prompt_generation_diagnostics_survive_100_calls(monkeypatch):
    final_action = "<final>BURKE</final>"
    tokenizer = _FakeTokenizer({(101,): final_action})
    processor = _FakeProcessor()
    module, FakeSample = _load_generator(monkeypatch)
    module._TOOL_CALL_FORMAT = "json"
    monkeypatch.setattr(module, "GenerateState", lambda args: SimpleNamespace(tokenizer=tokenizer, processor=processor))
    call_count = 0

    async def stable_generation(url, payload):
        nonlocal call_count
        call_count += 1
        return {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 101]]}}

    monkeypatch.setattr(module, "post", stable_generation)
    args = SimpleNamespace(
        fake_tokenizer=tokenizer,
        fake_processor=processor,
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_context_len=4096,
        rollout_max_response_len=32,
        eval_max_context_len=None,
        prm_enable=False,
    )
    for _ in range(100):
        result = asyncio.run(
            module.generate(args, _sample(FakeSample, "Document path: test.pdf\nQuestion: Who is the supplier?"), {"max_new_tokens": 32})
        )
        step = result.metadata["generation_steps"][0]
        assert step["generation_called"] is True
        assert step["raw_generation_text"] == final_action
        assert step["output_token_count"] > 0
        assert step["finish_reason"] == "stop"
        assert result.rollout_status == "search_budget_exhausted"
        assert result.metadata["premature_final"] is True
    expected_turns = max(
        int(module.TOOL_CONFIGS["max_turns"]),
        int(module.TOOL_CONFIGS["max_tool_calls"]) + 2,
    )
    assert call_count == 100 * expected_turns


def test_generate_render_crop_final_calls_backend_after_each_image_tool(monkeypatch, tmp_path):
    image_path = tmp_path / "render.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    render_action = (
        '<tool_call>{"name":"render_page","arguments":'
        '{"document_path":"/workspace/data/a.pdf","page_number":1}}</tool_call>'
    )
    crop_action = (
        '<tool_call>{"name":"crop_region","arguments":'
        f'{{"image_path":{json.dumps(str(image_path))},"bbox":[0,0,1,1]}}}}</tool_call>'
    )
    final_action = "<final>Visual answer</final>"
    tokenizer = _FakeTokenizer({(101,): render_action, (102,): crop_action, (103,): final_action})
    processor = _FakeProcessor()
    module, FakeSample = _load_generator(monkeypatch)
    module._TOOL_CALL_FORMAT = "json"
    monkeypatch.setattr(module, "GenerateState", lambda args: SimpleNamespace(tokenizer=tokenizer, processor=processor))

    calls = []
    responses = iter(
        [
            {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 101]]}},
            {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 102]]}},
            {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 103]]}},
        ]
    )

    async def fake_post(url, payload):
        calls.append(payload)
        return next(responses)

    async def fake_visual_tool(name, arguments):
        return json.dumps({"status": "ok", "tool": name, "image_path": str(image_path)})

    monkeypatch.setattr(module, "post", fake_post)
    monkeypatch.setattr(module.tool_registry, "execute_tool", fake_visual_tool)
    args = SimpleNamespace(
        fake_tokenizer=tokenizer,
        fake_processor=processor,
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_context_len=4096,
        rollout_max_response_len=32,
        eval_max_context_len=None,
        prm_enable=False,
    )
    result = asyncio.run(
        module.generate(
            args,
            _sample(FakeSample, "What is shown in the image?"),
            {"max_new_tokens": 32},
        )
    )

    assert len(calls) == 3
    assert calls[0].get("image_data") is None
    assert calls[1].get("image_data") == ["encoded-image"]
    assert calls[2].get("image_data") == ["encoded-image", "encoded-image"]
    assert result.response.endswith(final_action)
    assert result.rollout_status == "completed"
    assert all(step["generation_called"] for step in result.metadata["generation_steps"])
    assert all(step["assistant_output_token_count"] > 0 for step in result.metadata["generation_steps"])
    assert result.metadata["generation_call_count"] == 3


def test_tool_and_generation_errors_are_excluded_from_rl(monkeypatch):
    module, FakeSample = _load_generator(monkeypatch)
    valid_action = '<tool_call>{"name":"render_page","arguments":{"page_number":1}}</tool_call>'

    async def failing_tool(name, arguments):
        raise RuntimeError("tool backend unavailable")

    monkeypatch.setattr(module.tool_registry, "execute_tool", failing_tool)
    stats = {}
    trace = []
    _, done = asyncio.run(module.execute_predictions(valid_action, trace, stats, turn=1))
    assert done is True
    assert stats["_terminal_status"] == "tool_error"
    assert stats["executed_action_count"] == 1
    assert stats["tool_error_count"] == 1

    tokenizer = _FakeTokenizer({})
    processor = _FakeProcessor()
    module._TOOL_CALL_FORMAT = "json"
    monkeypatch.setattr(module, "GenerateState", lambda args: SimpleNamespace(tokenizer=tokenizer, processor=processor))

    async def failing_generation(url, payload):
        raise RuntimeError("generation backend unavailable")

    monkeypatch.setattr(module, "post", failing_generation)
    args = SimpleNamespace(
        fake_tokenizer=tokenizer,
        fake_processor=processor,
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_context_len=4096,
        rollout_max_response_len=32,
        eval_max_context_len=None,
        prm_enable=False,
    )
    result = asyncio.run(module.generate(args, _sample(FakeSample), {"max_new_tokens": 32}))
    assert result.rollout_status == "generation_error"
    assert result.valid_for_rl is False
    assert result.remove_sample is True
    assert result.metadata["generation_steps"][0]["generation_error"]

    reward = asyncio.run(module.reward_func(SimpleNamespace(prm_enable=False), result))
    assert reward["valid_for_rl"] is False
    assert reward["rollout_status"] == "generation_error"
    assert reward["score"] == 0.0

    batched = asyncio.run(module.reward_func(SimpleNamespace(prm_enable=False), [result]))
    assert isinstance(batched, list)
    assert batched[0]["valid_for_rl"] is False
    assert batched[0]["rollout_status"] == "generation_error"


def test_evidence_sufficiency_turns_true_only_for_a_local_field_value(monkeypatch):
    module, _ = _load_generator(monkeypatch)
    sufficient = module._new_navigation_state(
        "Document path: test.pdf\nQuestion: Who is the supplier?"
    )
    sufficient["page_count"] = 4
    sufficient["unvisited_pages"] = [1, 2, 3, 4]
    module._update_navigation_state(
        sufficient,
        "parse_document",
        {"page_numbers": [1]},
        {
            "status": "ok",
            "page_count": 4,
            "returned_pages": [1],
            "has_more_pages": True,
            "pages": [{"page_number": 1, "markdown": "SUPPLIER: BURKE"}],
        },
        True,
    )
    assert sufficient["evidence_sufficient"] is True
    assert sufficient["supporting_pages"] == [1]
    assert sufficient["stop_reason"] == "sufficient_evidence"
    assert sufficient["unvisited_pages"] == [2, 3, 4]

    insufficient = module._new_navigation_state(
        "Document path: test.pdf\nQuestion: Who is the supplier?"
    )
    insufficient["page_count"] = 4
    insufficient["unvisited_pages"] = [1, 2, 3, 4]
    module._update_navigation_state(
        insufficient,
        "parse_document",
        {"page_numbers": [1]},
        {
            "status": "ok",
            "page_count": 4,
            "returned_pages": [1],
            "has_more_pages": True,
            "pages": [{"page_number": 1, "markdown": "Supplier information is listed elsewhere."}],
        },
        True,
    )
    assert insufficient["evidence_sufficient"] is False
    assert insufficient["stop_reason"] == "evidence_insufficient"

    ambiguous = module._new_navigation_state(
        "Document path: test.pdf\nQuestion: What is the name of the company?"
    )
    ambiguous["page_count"] = 4
    ambiguous["unvisited_pages"] = [1, 2, 3, 4]
    module._update_navigation_state(
        ambiguous,
        "parse_document",
        {"page_numbers": [1]},
        {
            "status": "ok",
            "page_count": 4,
            "returned_pages": [1],
            "has_more_pages": True,
            "pages": [
                {
                    "page_number": 1,
                    "markdown": "RJRT CONFIDENTIAL\nR.J. Reynolds Tobacco Company\n(Name/Date)",
                }
            ],
        },
        True,
    )
    assert ambiguous["evidence_sufficient"] is False

    generic = module._new_navigation_state(
        "Document path: test.pdf\nQuestion: What is the page number?"
    )
    generic["page_count"] = 3
    generic["unvisited_pages"] = [1, 2, 3]
    module._update_navigation_state(
        generic,
        "parse_document",
        {"page_numbers": [1]},
        {
            "status": "ok",
            "page_count": 3,
            "returned_pages": [1],
            "has_more_pages": True,
            "pages": [
                {
                    "page_number": 1,
                    "markdown": "A number of projects were delayed. Page 1 contains no answer.",
                }
            ],
        },
        True,
    )
    assert generic["evidence_sufficient"] is False


def test_render_page_result_populates_page_frontier(monkeypatch):
    module, _ = _load_generator(monkeypatch)
    navigation = module._new_navigation_state(
        "Document path: test.pdf\nQuestion: Which logo is shown at the bottom?"
    )
    module._update_navigation_state(
        navigation,
        "render_page",
        {"page_number": 1},
        {
            "status": "ok",
            "page_number": 1,
            "page_count": 4,
            "image_path": "/tmp/page-1.png",
        },
        True,
    )
    assert navigation["page_count"] == 4
    assert navigation["rendered_pages"] == [1]
    assert navigation["unvisited_pages"] == [2, 3, 4]
    assert navigation["evidence_sufficient"] is False


def test_visited_pages_is_the_union_of_text_and_visual_operations(monkeypatch):
    module, _ = _load_generator(monkeypatch)
    navigation = module._new_navigation_state(
        "Document path: test.pdf\nQuestion: Which logo is shown at the bottom?"
    )
    module._update_navigation_state(
        navigation,
        "parse_document",
        {"page_numbers": [1]},
        {"status": "ok", "page_count": 4, "returned_pages": [1], "pages": [{"page_number": 1, "markdown": "Page 1"}]},
        True,
    )
    module._update_navigation_state(
        navigation,
        "render_page",
        {"page_number": 2},
        {"status": "ok", "page_count": 4, "page_number": 2, "image_path": "/tmp/page-2.png"},
        True,
    )
    module._update_navigation_state(
        navigation,
        "crop_region",
        {"image_path": "/tmp/page-2.png", "bbox": [0, 0, 10, 10]},
        {"status": "ok", "page_number": 2, "image_path": "/tmp/crop.png", "bbox": [0, 0, 10, 10]},
        True,
    )
    module._update_navigation_state(
        navigation,
        "ocr_region",
        {"page_number": 3, "bbox": [0, 0, 10, 10]},
        {"status": "ok", "page_number": 3, "text": "logo"},
        True,
    )
    assert navigation["parsed_pages"] == [1]
    assert navigation["rendered_pages"] == [2]
    assert navigation["cropped_pages"] == [2]
    assert navigation["ocr_pages"] == [3]
    assert navigation["visited_pages"] == [1, 2, 3]


def test_structured_table_evidence_rejects_the_wrong_column(monkeypatch):
    module, _ = _load_generator(monkeypatch)
    navigation = module._new_navigation_state(
        "Document path: test.pdf\nQuestion: What was the cholesterol by the 4th week for #1 rats?"
    )
    module._update_navigation_state(
        navigation,
        "extract_table",
        {"page_number": 2, "table_index": 0},
        {
            "status": "ok",
            "page_count": 4,
            "page_number": 2,
            "rows": [
                ["", "#1 rats", "#2 rats"],
                ["Week", "Cholesterol", "Cholesterol"],
                ["4", "103", "133"],
            ],
        },
        True,
    )
    candidates = navigation["evidence_candidates"]
    matching = [item for item in candidates if item["value"] == "103"]
    wrong_column = [item for item in candidates if item["value"] == "133"]
    assert matching and matching[0]["satisfies_question_constraints"] is True
    assert wrong_column and wrong_column[0]["satisfies_question_constraints"] is False
    assert navigation["evidence_sufficient"] is True
    accepted, _, _ = module._can_finish("<final>103</final>", navigation)
    assert accepted is True
    accepted, _, _ = module._can_finish("<final>133</final>", navigation)
    assert accepted is False
    assert navigation["prediction_found_in_document"] is True
    assert navigation["prediction_relation_matched"] is False


def test_flattened_table_fallback_recovers_week_group_value(monkeypatch):
    module, _ = _load_generator(monkeypatch)
    navigation = module._new_navigation_state(
        "Document path: test.pdf\nQuestion: What was the cholesterol by the 4th week for #1 rats?"
    )
    payload = json.loads(_page_result(
        2,
        page_count=3,
        markdown=(
            "Group #1 rats #2 rats #3 rats\n"
            "Cholesterol\n"
            "Week 0 96 157 157\n"
            "Week 4 103 133 111"
        ),
    ))
    module._update_navigation_state(navigation, "parse_document", {"page_numbers": [2]}, payload, True)
    matching = [
        item
        for item in navigation["evidence_candidates"]
        if item.get("value") == "103"
        and item.get("row_key") == "Week 4"
        and "#1" in str(item.get("column_key"))
    ]
    assert matching
    assert matching[0]["metric"].casefold() == "cholesterol"
    assert matching[0]["satisfies_question_constraints"] is True
    assert navigation["evidence_sufficient"] is True


def test_render_then_ocr_does_not_attach_the_full_page_image(monkeypatch, tmp_path):
    module, FakeSample = _load_generator(monkeypatch)
    image_path = tmp_path / "render.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    actions = [
        '<tool_call>{"name":"render_page","arguments":{"document_path":"test.pdf","page_number":1}}</tool_call>',
        '<tool_call>{"name":"ocr_region","arguments":{"document_path":"test.pdf","page_number":1,"bbox":[0,0,16,16]}}</tool_call>',
        "<final>logo</final>",
    ]

    async def fake_tool(name, arguments):
        if name == "render_page":
            return json.dumps({"status": "ok", "tool": name, "page_number": 1, "page_count": 1, "image_path": str(image_path)})
        return json.dumps({"status": "ok", "tool": name, "page_number": 1, "text": "logo"})

    result, calls = _run_scripted_generate(
        monkeypatch,
        module,
        FakeSample,
        "Document path: test.pdf\nQuestion: Read the text in the known bbox region at the bottom.",
        actions,
        fake_tool,
    )
    assert result.rollout_status == "completed"
    assert len(calls) == 3
    assert calls[1].get("image_data") is None
    assert calls[2].get("image_data") is None
    assert result.metadata["image_tensor_count"] == 0
    assert result.metadata["visual_evidence_sufficient"] is True


def test_guard_recovery_masks_only_the_rejected_final_action(monkeypatch):
    module, FakeSample = _load_generator(monkeypatch)
    actions = [
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[1]}}</tool_call>',
        "<final>BURKE</final>",
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[2]}}</tool_call>',
        "<final>BURKE</final>",
    ]

    async def fake_tool(name, arguments):
        page = arguments["page_numbers"][0]
        return _page_result(page, page_count=2, markdown="No supplier evidence" if page == 1 else "SUPPLIER: BURKE")

    result, _ = _run_scripted_generate(
        monkeypatch,
        module,
        FakeSample,
        "Document path: test.pdf\nQuestion: Who is the supplier?",
        actions,
        fake_tool,
        answer_page=2,
    )
    assert result.rollout_status == "completed"
    assert result.metadata["had_evidence_guard_recovery"] is True
    assert result.metadata["rejected_final_count"] == 1
    assert result.metadata["rejected_action_indices"] == [1]
    assert result.metadata["assistant_token_masks"][1]["mask"] == 0
    assert result.metadata["assistant_token_masks"][3]["mask"] == 1
    assert result.metadata["action_rewards"][1] < 0
    rejected_span = result.metadata["assistant_token_masks"][1]
    assert all(value == 0 for value in result.loss_mask[rejected_span["token_start"]:rejected_span["token_end"]])
    assert result.metadata["answer_page_visited"] is True
    assert "answer_page" not in result.metadata["navigation_state"]
    assert result.metadata["ground_truth_metadata_used_in_prompt"] is False
    assert result.metadata["ground_truth_metadata_used_in_action_selection"] is False
    reward = asyncio.run(module.reward_func(SimpleNamespace(prm_enable=False), result))
    assert reward["reward_consistency_valid"] is True
    assert reward["final_supported_by_evidence"] is True


def test_visual_canary_attaches_rendered_image_to_next_model_input(monkeypatch, tmp_path):
    module, FakeSample = _load_generator(monkeypatch)
    from tools import document_tools

    fitz = __import__("fitz")
    pdf_path = tmp_path / "visual-canary.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((72, 120), "VISUAL-CANARY-7391", fontsize=30)
        document.save(str(pdf_path))

    render_action = (
        '<tool_call>{"name":"render_page","arguments":'
        f'{{"document_path":{json.dumps(str(pdf_path))},"page_number":1}}}}</tool_call>'
    )
    final_action = "<final>VISUAL-CANARY-7391</final>"
    tokenizer = _FakeTokenizer({(101,): render_action, (102,): final_action})
    processor = _FakeProcessor()
    monkeypatch.setattr(module, "GenerateState", lambda args: SimpleNamespace(tokenizer=tokenizer, processor=processor))
    calls = []
    responses = iter(
        [
            {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 101]]}},
            {"meta_info": {"finish_reason": {"type": "stop"}, "output_token_logprobs": [[-0.1, 102]]}},
        ]
    )

    async def fake_post(url, payload):
        calls.append(payload)
        return next(responses)

    async def fake_render(name, arguments):
        assert name == "render_page"
        return document_tools.render_page(arguments)

    monkeypatch.setattr(module, "post", fake_post)
    monkeypatch.setattr(module.tool_registry, "execute_tool", fake_render)
    args = SimpleNamespace(
        fake_tokenizer=tokenizer,
        fake_processor=processor,
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_context_len=4096,
        rollout_max_response_len=32,
        eval_max_context_len=None,
        prm_enable=False,
    )
    result = asyncio.run(
        module.generate(
            args,
            _sample(FakeSample, f"Document path: {pdf_path}\nQuestion: What text is shown in the image?"),
            {"max_new_tokens": 32},
        )
    )

    assert result.response.endswith(final_action)
    assert len(calls) == 2
    assert calls[1]["image_data"] == ["encoded-image"]
    visual_step = result.metadata["generation_steps"][1]
    assert visual_step["image_input_count"] == 1
    assert visual_step["image_token_count"] == 1
    assert visual_step["vision_input_attached"] is True
    assert result.metadata["vision_input_attached"] is True


def test_extract_table_failure_returns_visual_fallback_observation(monkeypatch):
    module, _ = _load_generator(monkeypatch)
    navigation = module._new_navigation_state(
        "Document path: test.pdf\nQuestion: What is the total amount in the table?"
    )
    navigation["page_count"] = 2
    navigation["unvisited_pages"] = [2]

    async def failing_table(name, arguments):
        return json.dumps({"status": "error", "tool": name, "error": "libGL.so.1 unavailable"})

    monkeypatch.setattr(module.tool_registry, "execute_tool", failing_table)
    trace = []
    stats = {}
    observation, done = asyncio.run(
        module.execute_predictions(
            '<tool_call>{"name":"extract_table","arguments":{"document_path":"test.pdf","page_number":2,"table_index":0}}</tool_call>',
            trace,
            stats,
            turn=1,
            navigation_state=navigation,
        )
    )

    assert done is False
    assert "Fallback required: render the same page" in observation
    assert trace[-1]["recovered_with_fallback"] is True
    assert stats["tool_error_count"] == 1
    assert navigation["fallback_used"] is True


def test_visual_backend_failure_does_not_end_rollout(monkeypatch):
    module, _ = _load_generator(monkeypatch)
    navigation = module._new_navigation_state(
        "Document path: test.pdf\nQuestion: Which logo is shown at the top?"
    )
    navigation["page_count"] = 1
    navigation["unvisited_pages"] = [1]

    async def failing_layout(name, arguments):
        return json.dumps({"status": "error", "tool": name, "error": "libGL.so.1 unavailable"})

    monkeypatch.setattr(module.tool_registry, "execute_tool", failing_layout)
    trace = []
    stats = {}
    observation, done = asyncio.run(
        module.execute_predictions(
            '<tool_call>{"name":"detect_layout","arguments":{"document_path":"test.pdf","page_number":1}}</tool_call>',
            trace,
            stats,
            turn=1,
            navigation_state=navigation,
        )
    )

    assert done is False
    assert "Layout backend is unavailable" in observation
    assert trace[-1]["recovered_with_fallback"] is True
    assert stats["tool_error_count"] == 1


def test_pagination_and_visual_tool_chains_reach_final(monkeypatch, tmp_path):
    module, _ = _load_generator(monkeypatch)
    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    async def fake_tool(name, arguments):
        if name == "parse_document" and arguments.get("page_numbers") == [5]:
            return json.dumps(
                {
                    "status": "ok",
                    "page_count": 5,
                    "returned_pages": [5],
                    "truncated": False,
                    "has_more_pages": True,
                    "pages": [{"page_number": 5, "markdown": "## Page 5\n\nAnswer evidence"}],
                }
            )
        if name == "parse_document":
            return json.dumps(
                {
                    "status": "ok",
                    "page_count": 5,
                    "returned_pages": [1, 2],
                    "truncated": True,
                    "has_more_pages": True,
                    "pages": [{"page_number": 1, "markdown": "## Page 1"}],
                }
            )
        return json.dumps({"status": "ok", "tool": name, "image_path": str(image_path)})

    monkeypatch.setattr(module.tool_registry, "execute_tool", fake_tool)

    stats = {}
    trace = []
    parse_action = (
        '<tool_call>{"name":"parse_document","arguments":'
        '{"document_path":"/workspace/data/a.pdf"}}</tool_call>'
    )
    follow_up_action = (
        '<tool_call>{"name":"parse_document","arguments":'
        '{"document_path":"/workspace/data/a.pdf","page_numbers":[5]}}</tool_call>'
    )
    _, done = asyncio.run(module.execute_predictions(parse_action, trace, stats, turn=1))
    assert done is False
    _, done = asyncio.run(module.execute_predictions(follow_up_action, trace, stats, turn=2))
    assert done is False
    _, done = asyncio.run(module.execute_predictions("<final>Answer evidence</final>", trace, stats, turn=3))
    assert done is True
    assert stats["_terminal_status"] == "completed"
    assert stats["executed_action_count"] == 2


    stats = {}
    trace = []
    render_action = (
        '<tool_call>{"name":"render_page","arguments":'
        '{"document_path":"/workspace/data/a.pdf","page_number":1}}</tool_call>'
    )
    crop_action = (
        '<tool_call>{"name":"crop_region","arguments":'
        f'{{"image_path":{json.dumps(str(image_path))},"bbox":[0,0,1,1]}}}}</tool_call>'
    )
    _, done = asyncio.run(module.execute_predictions(render_action, trace, stats, turn=1))
    assert done is False
    _, done = asyncio.run(module.execute_predictions(crop_action, trace, stats, turn=2))
    assert done is False
    _, done = asyncio.run(module.execute_predictions("<final>Visual answer</final>", trace, stats, turn=3))
    assert done is True
    assert stats["_terminal_status"] == "completed"
    assert stats["executed_action_count"] == 2


def _run_scripted_generate(monkeypatch, module, FakeSample, prompt, responses, fake_tool, answer_page=None):
    token_responses = {(101 + index,): response for index, response in enumerate(responses)}
    tokenizer = _FakeTokenizer(token_responses)
    processor = _FakeProcessor()
    module._TOOL_CALL_FORMAT = "json"
    monkeypatch.setattr(module, "GenerateState", lambda args: SimpleNamespace(tokenizer=tokenizer, processor=processor))
    backend_responses = iter(
        [
            {
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [[-0.1, 101 + index]],
                }
            }
            for index in range(len(responses))
        ]
    )
    calls = []

    async def fake_post(url, payload):
        calls.append(payload)
        return next(backend_responses)

    monkeypatch.setattr(module, "post", fake_post)
    monkeypatch.setattr(module.tool_registry, "execute_tool", fake_tool)
    args = SimpleNamespace(
        fake_tokenizer=tokenizer,
        fake_processor=processor,
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_context_len=8192,
        rollout_max_response_len=64,
        eval_max_context_len=None,
        prm_enable=False,
    )
    sample = _sample(FakeSample, prompt)
    if answer_page is not None:
        sample.metadata["answer_page"] = answer_page
    return asyncio.run(module.generate(args, sample, {"max_new_tokens": 64})), calls


def _page_result(page_number, page_count=4, markdown="", **extra):
    payload = {
        "status": "ok",
        "tool": "parse_document",
        "page_count": page_count,
        "returned_pages": [page_number],
        "truncated": False,
        "content_truncated": False,
        "has_more_pages": page_number < page_count,
        "document_has_unreturned_pages": page_number < page_count,
        "pages": [{"page_number": page_number, "markdown": markdown}],
    }
    payload.update(extra)
    return json.dumps(payload)


def test_four_page_search_reaches_answer_page_with_multiple_parse_turns(monkeypatch, tmp_path):
    module, FakeSample = _load_generator(monkeypatch)
    from tools import document_tools

    fitz = __import__("fitz")
    pdf_path = tmp_path / "four-page-search.pdf"
    with fitz.open() as document:
        for page_number in range(1, 5):
            page = document.new_page()
            text = "No supplier evidence here" if page_number < 3 else "Supplier: BURKE"
            page.insert_text((72, 72), f"Page {page_number}\n{text}")
        document.save(str(pdf_path))
    monkeypatch.setattr(document_tools, "_docling_convert", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no docling")))

    actions = [
        f'<tool_call>{{"name":"parse_document","arguments":{{"document_path":{json.dumps(str(pdf_path))},"page_numbers":[1]}}}}</tool_call>',
        f'<tool_call>{{"name":"parse_document","arguments":{{"document_path":{json.dumps(str(pdf_path))},"page_numbers":[2]}}}}</tool_call>',
        f'<tool_call>{{"name":"parse_document","arguments":{{"document_path":{json.dumps(str(pdf_path))},"page_numbers":[3]}}}}</tool_call>',
        "<final>Supplier: BURKE</final>",
    ]
    seen_pages = []

    async def fake_tool(name, arguments):
        assert name == "parse_document"
        seen_pages.extend(arguments.get("page_numbers", []))
        return document_tools.parse_document({**arguments, "output_format": "markdown"})

    result, calls = _run_scripted_generate(
        monkeypatch,
        module,
        FakeSample,
        f"Document path: {pdf_path}\nQuestion: Who is the supplier?",
        actions,
        fake_tool,
        answer_page=3,
    )

    assert len(calls) == 4
    assert seen_pages == [1, 2, 3]
    assert result.rollout_status == "completed"
    assert result.response.endswith(actions[-1])
    assert result.metadata["visited_pages"] == [1, 2, 3]
    assert result.metadata["answer_page_visited"] is True
    assert result.metadata["final_supported_by_evidence"] is True
    assert result.metadata["generation_turn_count"] == 4
    assert result.metadata["tool_execution"]["call_count"] == 3
    assert result.metadata["unique_tool_count"] == 1
    assert result.metadata["multi_turn_rollout"] is True
    assert result.metadata["multi_tool_rollout"] is True
    assert result.metadata["completed_multi_tool_rollout"] is True


def test_final_absence_is_blocked_until_another_page_is_checked(monkeypatch):
    module, FakeSample = _load_generator(monkeypatch)
    actions = [
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[1]}}</tool_call>',
        "<final>None</final>",
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[2]}}</tool_call>',
        "<final>BURKE</final>",
    ]

    async def fake_tool(name, arguments):
        page = arguments["page_numbers"][0]
        markdown = "Page 1 has no answer" if page == 1 else "Supplier: BURKE"
        return _page_result(page, page_count=3, markdown=markdown)

    result, calls = _run_scripted_generate(
        monkeypatch,
        module,
        FakeSample,
        "Document path: test.pdf\nQuestion: Who is the supplier?",
        actions,
        fake_tool,
    )

    assert len(calls) == 4
    assert result.rollout_status == "completed"
    assert result.metadata["premature_final"] is True
    assert result.metadata["action_statistics"]["premature_final_count"] == 1
    assert result.metadata["visited_pages"] == [1, 2]
    assert result.metadata["final_supported_by_evidence"] is True
    assert any(item.get("recovery_observation") for item in result.tool_execution_trace)


def test_final_before_first_document_read_is_blocked(monkeypatch):
    module, FakeSample = _load_generator(monkeypatch)
    actions = [
        "<final>None</final>",
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[1]}}</tool_call>',
        "<final>BURKE</final>",
    ]

    async def fake_tool(name, arguments):
        return _page_result(1, page_count=1, markdown="Supplier: BURKE")

    result, calls = _run_scripted_generate(
        monkeypatch,
        module,
        FakeSample,
        "Document path: test.pdf\nQuestion: Who is the supplier?",
        actions,
        fake_tool,
    )

    assert len(calls) == 3
    assert result.rollout_status == "completed"
    assert result.metadata["premature_final"] is True
    assert result.metadata["visited_pages"] == [1]
    assert result.metadata["final_supported_by_evidence"] is True


def test_positive_final_before_first_document_read_is_blocked(monkeypatch):
    module, FakeSample = _load_generator(monkeypatch)
    actions = [
        "<final>BURKE</final>",
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[1]}}</tool_call>',
        "<final>BURKE</final>",
    ]

    async def fake_tool(name, arguments):
        return _page_result(1, page_count=1, markdown="Supplier: BURKE")

    result, calls = _run_scripted_generate(
        monkeypatch,
        module,
        FakeSample,
        "Document path: test.pdf\nQuestion: Who is the supplier?",
        actions,
        fake_tool,
    )

    assert len(calls) == 3
    assert result.rollout_status == "completed"
    assert result.metadata["premature_final"] is True
    assert result.metadata["visited_pages"] == [1]
    assert result.metadata["final_supported_by_evidence"] is True


def test_repeated_guarded_final_is_not_counted_as_protocol_error(monkeypatch):
    module, FakeSample = _load_generator(monkeypatch)
    actions = [
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[1]}}</tool_call>',
        *(["<final>BURKE</final>"] * 9),
    ]

    async def fake_tool(name, arguments):
        return _page_result(1, page_count=4, markdown="Page 1 has no supplier evidence")

    result, calls = _run_scripted_generate(
        monkeypatch,
        module,
        FakeSample,
        "Document path: test.pdf\nQuestion: Who is the supplier?",
        actions,
        fake_tool,
    )

    assert len(calls) == len(actions)
    assert result.rollout_status == "search_budget_exhausted"
    assert result.metadata["premature_final"] is True
    assert result.metadata["action_statistics"]["premature_final_count"] == 9
    assert result.metadata["action_statistics"]["protocol_error_count"] == 0
    assert result.valid_for_rl is True


def test_visual_route_uses_render_crop_and_ocr_after_page_search(monkeypatch, tmp_path):
    module, FakeSample = _load_generator(monkeypatch)
    image_path = tmp_path / "page-3.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    actions = [
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[1]}}</tool_call>',
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[2]}}</tool_call>',
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[3]}}</tool_call>',
        '<tool_call>{"name":"render_page","arguments":{"document_path":"test.pdf","page_number":3}}</tool_call>',
        f'<tool_call>{{"name":"crop_region","arguments":{{"image_path":{json.dumps(str(image_path))},"bbox":[0,0,16,16]}}}}</tool_call>',
        f'<tool_call>{{"name":"ocr_region","arguments":{{"image_path":{json.dumps(str(image_path))}}}}}</tool_call>',
        "<final>logo</final>",
    ]
    tool_names = []

    async def fake_tool(name, arguments):
        tool_names.append(name)
        if name == "parse_document":
            page = arguments["page_numbers"][0]
            return _page_result(
                page,
                markdown="Page 3 contains an image" if page == 3 else f"Page {page}",
                has_images=page == 3,
                visual_content_omitted=page == 3,
                image_regions=[{"page_number": 3, "bbox": [0, 0, 16, 16]}] if page == 3 else [],
            )
        if name == "ocr_region":
            return json.dumps({"status": "ok", "tool": name, "text": "logo"})
        return json.dumps({"status": "ok", "tool": name, "image_path": str(image_path)})

    result, calls = _run_scripted_generate(
        monkeypatch,
        module,
        FakeSample,
        "Document path: test.pdf\nQuestion: Which logo is shown at the bottom?",
        actions,
        fake_tool,
    )

    assert len(calls) == len(actions)
    assert tool_names == ["parse_document", "parse_document", "parse_document", "render_page", "crop_region", "ocr_region"]
    assert result.rollout_status == "completed"
    assert result.metadata["completed_multi_tool_rollout"] is True
    assert result.metadata["unique_tool_count"] == 4
    assert result.metadata["final_supported_by_evidence"] is True


def test_table_route_extracts_table_after_locating_page(monkeypatch):
    module, FakeSample = _load_generator(monkeypatch)
    actions = [
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[1]}}</tool_call>',
        '<tool_call>{"name":"parse_document","arguments":{"document_path":"test.pdf","page_numbers":[2]}}</tool_call>',
        '<tool_call>{"name":"extract_table","arguments":{"document_path":"test.pdf","page_number":2,"table_index":0}}</tool_call>',
        "<final>33.0</final>",
    ]
    tool_names = []

    async def fake_tool(name, arguments):
        tool_names.append(name)
        if name == "parse_document":
            page = arguments["page_numbers"][0]
            return _page_result(
                page,
                markdown="Page 2 contains a table" if page == 2 else f"Page {page}",
                has_tables=page == 2,
                table_count=1 if page == 2 else 0,
                table_extraction_recommended=page == 2,
                table_regions=[{"page_number": 2, "bbox": [10, 10, 100, 100]}] if page == 2 else [],
            )
        return json.dumps(
            {
                "status": "ok",
                "tool": name,
                "page_number": 2,
                "markdown": "| Item | Amount |\n| --- | --- |\n| Total | 33.0 |",
                "rows": [["Item", "Amount"], ["Total", "33.0"]],
            }
        )

    result, calls = _run_scripted_generate(
        monkeypatch,
        module,
        FakeSample,
        "Document path: test.pdf\nQuestion: What is the total amount in the table?",
        actions,
        fake_tool,
        answer_page=2,
    )

    assert len(calls) == 4
    assert tool_names == ["parse_document", "parse_document", "extract_table"]
    assert result.rollout_status == "completed"
    assert result.metadata["answer_page_visited"] is True
    assert result.metadata["final_supported_by_evidence"] is True
    assert result.metadata["multi_tool_rollout"] is True
