# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise real request orchestration with a CPU-only stand-in for the vLLM engine."""

import ast
import asyncio
import contextlib
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_server_methods():
    source = Path(__file__).resolve().parents[3] / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    module = ast.parse(source.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "vLLMHttpServer")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name in {"generate", "_park_until_admitted", "set_global_steps"}
    ]
    # Loading the methods avoids importing vLLM/CUDA; their bodies are unchanged.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    compiled = ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[]))
    namespace = {
        "torch": torch,
        "logger": logging.getLogger(__name__),
        "normalize_token_ids": list,
        "SamplingParams": lambda **kwargs: SimpleNamespace(prompt_logprobs=None, **kwargs),
        "TokensPrompt": dict,
        "EmbedsPrompt": dict,
        "TokenOutput": SimpleNamespace,
        "qwen2_5_vl_dedup_image_tokens": lambda tokens, processor: tokens,
        "validate_topk_logprob_request": lambda *args: None,
        "warn_topk_logprob_budget": lambda *args: None,
        "extract_prompt_logprobs": lambda **kwargs: None,
        "RLInsightLogger": SimpleNamespace(trace_state=lambda *args, **kwargs: contextlib.nullcontext()),
        "ray": SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(get_actor_name=lambda: "cpu-request-test")),
    }
    exec(compile(compiled, str(source), "exec"), namespace)
    return type("RequestServer", (), {method.name: namespace[method.name] for method in methods})


def _make_server():
    server = _load_server_methods()()
    server._disaggregation_role = "null"
    server.config = SimpleNamespace(
        engine_kwargs={"vllm": {"enable_prompt_embeds": True}},
        max_model_len=32,
        response_length=4,
        prompt_length=8,
        full_determinism=False,
        enable_rollout_routing_replay=False,
        mtp=None,
        get=lambda key, default: default,
    )
    server.model_config = SimpleNamespace(processor=None)
    server.global_steps = 3
    server._weight_update_epoch = 0
    server._submission_paused = False
    server._rejecting = False
    server._admitting = 0
    server._resume_event = asyncio.Event()
    server._preprocess_sampling_params = lambda params: None
    server._postprocess_output = lambda output, *args: output
    server._dp_rank_for_session = lambda session: None
    server.lora_as_adapter = False
    server.received_prompts = []
    server.advance_after_output = False

    async def generate(**kwargs):
        server.received_prompts.append(kwargs["prompt"])
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[11, 12], finish_reason="length")],
            num_cached_tokens=0,
        )
        if server.advance_after_output:
            # A completed request must retain its behavior-policy version even
            # when the next update publishes before the caller consumes it.
            await server.set_global_steps(8)

    server.engine = SimpleNamespace(generate=generate)
    return server


def test_parked_request_captures_admitted_policy_and_keeps_it_in_output():
    async def run():
        server = _make_server()
        server._submission_paused = True
        server.advance_after_output = True
        request = asyncio.create_task(server.generate([1, 2], {}, "parked"))
        await asyncio.sleep(0)
        assert not request.done()
        await server.set_global_steps(7)
        server._submission_paused = False
        server._resume_event.set()
        output = await request
        assert server.received_prompts[0]["cache_salt"] == "verl:step:7:update:1"
        assert output.extra_fields["global_steps"] == 7
        assert output.extra_fields["policy_version"] == "step:7:update:1"
        assert server.global_steps == 8
        assert server._admitting == 0

    asyncio.run(run())


def test_prompt_embeddings_reach_engine_without_multimodal_fields():
    async def run():
        server = _make_server()
        embeddings = torch.randn(2, 8)
        output = await server.generate([1, 2], {}, "embeddings", prompt_embeds=embeddings)
        prompt = server.received_prompts[0]
        assert prompt["prompt_embeds"] is embeddings
        assert "prompt_token_ids" not in prompt
        assert "multi_modal_data" not in prompt
        assert output.token_ids == [11, 12]
        await server.set_global_steps(3)
        await server.generate([1, 2], {}, "same-step-update", prompt_embeds=embeddings)
        assert server.received_prompts[1]["cache_salt"] != prompt["cache_salt"]

    asyncio.run(run())


def test_embedding_contract_rejects_misaligned_or_disabled_inputs_before_admission():
    async def run():
        server = _make_server()
        with pytest.raises(ValueError, match="shape"):
            await server.generate([1, 2], {}, "misaligned", prompt_embeds=torch.randn(3, 8))
        server.config.engine_kwargs["vllm"]["enable_prompt_embeds"] = False
        with pytest.raises(ValueError, match="enable_prompt_embeds"):
            await server.generate([1, 2], {}, "disabled", prompt_embeds=torch.randn(2, 8))
        assert not server.received_prompts
        assert server._admitting == 0

    asyncio.run(run())
