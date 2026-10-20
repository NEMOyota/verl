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

import pickle

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.experimental.agent_loop.compact_output import CompactAgentLoopOutput, snapshot_raw_prompts
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask


@pytest.mark.parametrize("multimodal_positions", [False, True])
def test_compact_rollout_roundtrip_preserves_training_inputs(multimodal_positions):
    prompt_width, response_width = 1024, 2048
    prompts = torch.full((2, prompt_width), 99, dtype=torch.long)
    responses = torch.full((2, response_width), 99, dtype=torch.long)
    prompts[0, -3:], prompts[1, -2:] = torch.tensor([1, 2, 3]), torch.tensor([4, 5])
    responses[0, :4], responses[1, :2] = torch.tensor([6, 7, 8, 9]), torch.tensor([10, 11])
    attention_mask = torch.zeros((2, prompt_width + response_width), dtype=torch.long)
    attention_mask[0, prompt_width - 3 : prompt_width + 4] = 1
    attention_mask[1, prompt_width - 2 : prompt_width + 2] = 1
    response_mask = torch.zeros_like(responses)
    # The zero in the middle denotes a tool observation, not padding.
    response_mask[0, :4], response_mask[1, :2] = torch.tensor([1, 0, 1, 1]), 1
    log_probs = torch.zeros_like(responses, dtype=torch.float32)
    log_probs[0, :4], log_probs[1, :2] = torch.tensor([-0.1, 0.0, -0.3, -0.4]), -0.5
    rewards = torch.zeros_like(log_probs)
    rewards[0, 3], rewards[1, 1] = 1.0, 0.5
    position_ids = compute_position_id_with_mask(attention_mask)
    if multimodal_positions:
        position_ids = position_ids.unsqueeze(1).repeat(1, 4, 1)
        position_ids[:, 2] += 7
    routed_experts = torch.zeros((2, prompt_width + response_width, 2, 2), dtype=torch.int16)
    routed_experts[0, prompt_width - 3 : prompt_width + 4] = 3
    raw_prompt = np.empty(2, dtype=object)
    raw_prompt[:] = [
        [{"role": "user", "content": "private question " * 1000}],
        [{"role": "user", "content": "another question " * 1000}],
    ]
    prompt_embeds = np.empty(2, dtype=object)
    prompt_embeds[0] = torch.ones((3, 128), dtype=torch.bfloat16)
    prompt_embeds[1] = torch.zeros((2, 128), dtype=torch.bfloat16)
    source = DataProto(non_tensor_batch={"raw_prompt": raw_prompt, "prompt_embeds": prompt_embeds})
    output = DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat((prompts, responses), dim=1),
                "response_mask": response_mask,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "rollout_log_probs": log_probs,
                "rm_scores": rewards,
                "routed_experts": routed_experts,
            },
            batch_size=2,
        ),
        non_tensor_batch={
            "raw_prompt": raw_prompt,
            "prompt_embeds": prompt_embeds,
            "uid": np.array(["group-a", "group-a"], dtype=object),
            "min_global_steps": np.array([7, 7]),
            "max_global_steps": np.array([7, 8]),
        },
        meta_info={"metrics": [{"generate_sequences": 1.0}, {"generate_sequences": 2.0}]},
    )
    wire = CompactAgentLoopOutput.pack(output, snapshot_raw_prompts(source), source.non_tensor_batch)
    serialized = pickle.dumps(wire, protocol=5)
    restored = pickle.loads(serialized).unpack(source)
    assert wire.reuse_raw_prompt
    assert wire.reuse_prompt_embeds
    assert "prompt_embeds" not in wire.non_tensor_batch
    assert restored.non_tensor_batch["prompt_embeds"] is prompt_embeds
    assert len(serialized) < len(pickle.dumps(output, protocol=5)) // 4
    assert restored.batch.keys() == output.batch.keys()
    for key in output.batch.keys():
        torch.testing.assert_close(restored.batch[key], output.batch[key], rtol=0, atol=0)
    for key in output.non_tensor_batch:
        assert pickle.dumps(restored.non_tensor_batch[key]) == pickle.dumps(output.non_tensor_batch[key])
    assert restored.meta_info == output.meta_info


def test_changed_prompt_is_returned_instead_of_using_local_original():
    source = DataProto(non_tensor_batch={"raw_prompt": np.array(["original"], dtype=object)})
    snapshot = snapshot_raw_prompts(source)
    output = DataProto(
        batch=TensorDict(
            {
                "prompts": torch.tensor([[1, 2]]),
                "responses": torch.tensor([[3, 4]]),
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            },
            batch_size=1,
        ),
        non_tensor_batch={"raw_prompt": np.array(["changed by agent"], dtype=object)},
    )
    wire = CompactAgentLoopOutput.pack(output, snapshot)
    assert not wire.reuse_raw_prompt
    assert wire.unpack(source).non_tensor_batch["raw_prompt"].tolist() == ["changed by agent"]
