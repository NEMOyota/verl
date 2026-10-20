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
"""Compact representation used only on the AgentLoopWorker -> Manager RPC."""

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto


def snapshot_raw_prompts(batch: DataProto) -> list[bytes] | None:
    """Snapshot before running tools, so a modified prompt is never replaced by the original."""
    prompts = batch.non_tensor_batch.get("raw_prompt")
    return None if prompts is None else [pickle.dumps(prompt, protocol=5) for prompt in prompts]


@dataclass
class _PackedTensor:
    values: torch.Tensor
    left_padding: torch.Tensor
    right_padding: torch.Tensor
    sequence_dim: int
    width: int
    spans: list[tuple[int, int]]

    @classmethod
    def pack(cls, tensor: torch.Tensor, sequence_dim: int, spans: list[tuple[int, int]]):
        rows = tensor.movedim(sequence_dim, 1)
        left, right, values = [], [], []
        for row, (start, end) in zip(rows, spans, strict=True):
            left_value, right_value = row[0], row[-1]
            # Position IDs may have nonzero and per-channel padding. Preserve those
            # values rather than assuming all padding is zero.
            if not torch.all(row[:start] == left_value) or not torch.all(row[end:] == right_value):
                return tensor
            left.append(left_value)
            right.append(right_value)
            values.append(row[start:end])
        # cat/stack allocate independent storage: serializing a view could otherwise
        # retain the entire original padded tensor's storage.
        return cls(
            torch.cat(values),
            torch.stack(left),
            torch.stack(right),
            sequence_dim,
            rows.shape[1],
            spans,
        )

    def unpack(self) -> torch.Tensor:
        rows = self.values.new_empty((len(self.spans), self.width, *self.values.shape[1:]))
        offset = 0
        for i, (start, end) in enumerate(self.spans):
            rows[i, :start] = self.left_padding[i]
            rows[i, start:end] = self.values[offset : offset + end - start]
            rows[i, end:] = self.right_padding[i]
            offset += end - start
        return rows.movedim(1, self.sequence_dim).contiguous()


@dataclass
class CompactAgentLoopOutput:
    """Wire payload; the Manager restores the exact padded training interface."""

    tensors: dict[str, torch.Tensor | _PackedTensor]
    non_tensor_batch: dict[str, np.ndarray]
    meta_info: dict[str, Any]
    batch_size: int
    reuse_raw_prompt: bool
    reuse_prompt_embeds: bool

    @classmethod
    def pack(
        cls,
        output: DataProto,
        raw_prompt_snapshot: list[bytes] | None,
        input_non_tensor_batch: dict[str, np.ndarray] | None = None,
    ):
        """Pack the standard worker result without changing training data or metadata."""
        batch = output.batch
        prompt_width = batch["prompts"].shape[1]
        response_width = batch["responses"].shape[1]
        attention_mask = batch["attention_mask"]
        prompt_lengths = attention_mask[:, :prompt_width].sum(-1).tolist()
        response_lengths = attention_mask[:, prompt_width:].sum(-1).tolist()
        prompt_spans = [(prompt_width - length, prompt_width) for length in prompt_lengths]
        response_spans = [(0, length) for length in response_lengths]
        sequence_spans = [
            (prompt_width - prompt_length, prompt_width + response_length)
            for prompt_length, response_length in zip(prompt_lengths, response_lengths, strict=True)
        ]
        layouts = {
            "prompts": (1, prompt_spans),
            "responses": (1, response_spans),
            "response_mask": (1, response_spans),
            "rollout_log_probs": (1, response_spans),
            "rm_scores": (1, response_spans),
            "attention_mask": (1, sequence_spans),
            "position_ids": (-1, sequence_spans),
            "routed_experts": (1, sequence_spans),
            "teacher_ids": (1, sequence_spans),
            "teacher_logprobs": (1, sequence_spans),
        }
        tensors = {}
        for key, tensor in batch.items():
            if key == "input_ids":
                # This field is constructed by concatenating prompts and responses.
                continue
            if key in layouts and prompt_width > 0 and response_width > 0:
                sequence_dim, spans = layouts[key]
                tensors[key] = _PackedTensor.pack(tensor, sequence_dim, spans)
            else:
                tensors[key] = tensor

        non_tensor_batch = output.non_tensor_batch.copy()
        raw_prompt = non_tensor_batch.get("raw_prompt")
        reuse_raw_prompt = (
            raw_prompt_snapshot is not None
            and raw_prompt is not None
            and raw_prompt.shape == (len(output),)
            and raw_prompt.dtype == object
            and len(raw_prompt_snapshot) == len(raw_prompt)
            and all(
                pickle.dumps(prompt, protocol=5) == original
                for prompt, original in zip(raw_prompt, raw_prompt_snapshot, strict=True)
            )
        )
        if reuse_raw_prompt:
            del non_tensor_batch["raw_prompt"]
        # Embeddings are read-only inputs. Only reuse the exact inherited array;
        # an agent that replaces it with a new output retains the normal return path.
        # Identity checking avoids copying or hashing potentially GB-sized tensors.
        reuse_prompt_embeds = (
            input_non_tensor_batch is not None
            and "prompt_embeds" in input_non_tensor_batch
            and "prompt_embeds" in non_tensor_batch
            and non_tensor_batch["prompt_embeds"] is input_non_tensor_batch["prompt_embeds"]
        )
        if reuse_prompt_embeds:
            del non_tensor_batch["prompt_embeds"]
        return cls(tensors, non_tensor_batch, output.meta_info, len(output), reuse_raw_prompt, reuse_prompt_embeds)

    def unpack(self, source: DataProto) -> DataProto:
        """Restore a worker result using its corresponding manager-side input chunk."""
        tensors = {
            key: value.unpack() if isinstance(value, _PackedTensor) else value for key, value in self.tensors.items()
        }
        tensors["input_ids"] = torch.cat((tensors["prompts"], tensors["responses"]), dim=1)
        non_tensor_batch = self.non_tensor_batch.copy()
        if self.reuse_raw_prompt:
            raw_prompt = np.empty(self.batch_size, dtype=object)
            raw_prompt[:] = list(source.non_tensor_batch["raw_prompt"])
            non_tensor_batch["raw_prompt"] = raw_prompt
        if self.reuse_prompt_embeds:
            non_tensor_batch["prompt_embeds"] = source.non_tensor_batch["prompt_embeds"]
        return DataProto(
            batch=TensorDict(tensors, batch_size=self.batch_size),
            non_tensor_batch=non_tensor_batch,
            meta_info=self.meta_info,
        )
