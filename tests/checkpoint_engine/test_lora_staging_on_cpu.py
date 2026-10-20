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

import asyncio
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.checkpoint_engine.base import CheckpointEngineManager, CheckpointEngineWorker
from verl.workers.engine_workers import ActorRolloutRefWorker


def test_adapter_only_export_is_sent_with_prepared_version():
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = OmegaConf.create(
        {
            "model": {"lora": {"merge": False}},
            "rollout": {"load_format": "auto", "checkpoint_engine": {"backend": "mooncake", "lora_staging": True}},
        }
    )
    export_calls = []

    def export(**kwargs):
        export_calls.append(kwargs)
        return iter([("lora_A", torch.ones(2))]), {"r": 8}

    sent = []

    async def send(weights, global_steps):
        sent.append((global_steps, list(weights)))

    worker.actor = SimpleNamespace(engine=SimpleNamespace(get_per_tensor_param=export, is_param_offload_enabled=False))
    worker.checkpoint_engine = SimpleNamespace(send_weights=send)
    config = ActorRolloutRefWorker.prepare_lora_sync.__wrapped__(worker, global_steps=7)
    asyncio.run(ActorRolloutRefWorker.update_weights.__wrapped__(worker, global_steps=7, adapter_only=True))
    assert config == {"r": 8}
    assert export_calls == [{"layered_summon": False, "base_sync_done": True}]
    assert sent[0][0] == 7
    assert sent[0][1][0][0] == "lora_A"
    assert not hasattr(worker, "_pending_lora_export")


def test_staging_copies_reused_buffer_and_only_commit_updates_policy():
    worker = CheckpointEngineWorker.__new__(CheckpointEngineWorker)
    worker._staged_lora = None
    worker.rollout_config = SimpleNamespace(
        checkpoint_engine=SimpleNamespace(lora_staging_max_megabytes=1, update_weights_bucket_megabytes=1)
    )
    received_versions = []

    async def receive(global_steps):
        received_versions.append(global_steps)
        bucket = torch.tensor([1.0, 2.0])
        yield "lora_A", bucket
        bucket.fill_(3)
        yield "lora_B", bucket
        bucket.zero_()

    applied = []

    async def update(weights, **kwargs):
        applied.append((list(weights), kwargs))

    worker.checkpoint_engine = SimpleNamespace(receive_weights=receive)
    worker.server_adapter = SimpleNamespace(update_weights=update)

    async def run():
        assert await CheckpointEngineWorker.stage_lora_weights.__wrapped__(worker, {"r": 8}, 7) == 16
        assert not applied
        await CheckpointEngineWorker.commit_staged_lora.__wrapped__(worker, 7)

    asyncio.run(run())
    assert received_versions == [7]
    tensors, kwargs = applied[0]
    torch.testing.assert_close(tensors[0][1], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(tensors[1][1], torch.tensor([3.0, 3.0]))
    assert kwargs == {"global_steps": 7, "peft_config": {"r": 8}, "base_sync_done": True}
    assert worker._staged_lora is None


def test_staging_limit_drains_transfer_without_publishing():
    worker = CheckpointEngineWorker.__new__(CheckpointEngineWorker)
    worker._staged_lora = None
    worker.rollout_config = SimpleNamespace(
        checkpoint_engine=SimpleNamespace(lora_staging_max_megabytes=1, update_weights_bucket_megabytes=1)
    )
    drained = []

    async def receive(global_steps):
        for _ in range(3):
            yield "adapter", torch.ones(512 * 1024, dtype=torch.uint8)
        drained.append(True)

    worker.checkpoint_engine = SimpleNamespace(receive_weights=receive)
    with pytest.raises(ValueError, match="exceeding"):
        asyncio.run(CheckpointEngineWorker.stage_lora_weights.__wrapped__(worker, {"r": 8}, 7))
    assert drained == [True]
    assert worker._staged_lora is None


def test_manager_finishes_staging_before_pausing_and_publishes_before_resume(monkeypatch):
    events = []
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.backend = "mooncake"

    async def record(event, value=None):
        events.append(event)
        return value

    manager.actor_wg = SimpleNamespace(
        world_size=1,
        prepare_lora_sync=lambda **kwargs: [{"r": 8}],
        update_weights=lambda **kwargs: [record("send", {"checkpoint/sent": 16})],
        execute_checkpoint_engine=lambda methods: [events.append("sender_finalize")],
    )
    rollout = SimpleNamespace(
        world_size=1,
        stage_lora_weights=lambda **kwargs: [record("staged", 16)],
        commit_staged_lora=lambda **kwargs: [record("commit")],
        discard_staged_lora=lambda: [events.append("discard")],
        execute_checkpoint_engine=lambda methods: [events.append("receiver_finalize")],
    )
    manager.build_process_group = lambda group: events.append("prepare")
    manager.abort_replicas = lambda: record("pause")
    manager.release_kv_cache_replicas = lambda: record("release_kv")
    manager.resume_kv_cache_replicas = lambda: record("resume_kv")
    manager.resume_generation_replicas = lambda: record("resume")
    monkeypatch.setattr("verl.checkpoint_engine.base.ray.get", lambda refs: refs)
    metrics = asyncio.run(manager._update_staged_lora_weights(rollout, 7))
    assert events == [
        "prepare",
        "send",
        "staged",
        "sender_finalize",
        "receiver_finalize",
        "pause",
        "release_kv",
        "commit",
        "discard",
        "resume_kv",
        "resume",
    ]
    assert metrics["checkpoint/sent"] == 16
    assert metrics["checkpoint/lora_staged_bytes_per_worker"] == 16
