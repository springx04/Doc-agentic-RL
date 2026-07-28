import asyncio
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
    result = asyncio.run(module.generate(args, _sample(FakeSample), {"max_new_tokens": 32}))

    assert len(calls) == 2, "render_page must be followed by a real second generation request"
    assert calls[0].get("image_data") is None
    assert calls[1]["image_data"] == ["encoded-image"]
    assert result.response.endswith(final_action)
    assert result.rollout_status == "completed"
    assert result.valid_for_rl is True
    assert result.tool_call_count == 1
    assert result.metadata["generation_steps"][1]["assistant_output_token_count"] > 0
    assert result.metadata["generation_steps"][1]["image_token_count"] == 1


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
