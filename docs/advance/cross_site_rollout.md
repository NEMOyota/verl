# Cross-site rollout transport

Last updated: 10/20/2026

These options reduce transport and synchronization overhead when a trainer and
rollout servers run at different sites. They do not establish data confidentiality
or implement a remote LM Head and distributed autograd.

## Receive LoRA before pausing generation

See [LoRA staging](lora_staging.md). With an identical frozen base preloaded at
both sites, `actor_rollout_ref.rollout.checkpoint_engine.lora_staging=true`
exports only the unmerged adapter and receives it before closing the rollout
admission gate. The default remains `false`.

## Compact trajectory return

Set `actor_rollout_ref.rollout.agent.compact_output=true` when the
`AgentLoopWorker` to `AgentLoopManager` boundary crosses the network. The worker
sends actual sequence lengths, without duplicated `input_ids`, and the manager
reconstructs the existing padded training batch. Tool masks, behavior logprobs,
reward data, position IDs and group/version metadata retain their meaning.

The original prompt is restored from the manager's input only when it has not
changed. Precomputed embeddings inherited from that input are also reused
locally under the read-only input contract. This option does not change where
agents, reward code or tools execute, and does not move them automatically.

## Precomputed text embeddings

For the standard, non-resuming vLLM client, enable:

```yaml
actor_rollout_ref:
  rollout:
    name: vllm
    engine_kwargs:
      vllm:
        enable_prompt_embeds: true
```

The single-turn agent accepts `prompt_embeds` in its per-sample non-tensor
inputs, or a custom agent can pass it to `LLMServerClient.generate`.
Use a floating-point CPU tensor of shape `[prompt_length, hidden_size]`, computed
from exactly the token sequence produced by the configured chat template.
The caller still supplies `prompt_ids` for training alignment and length
accounting. The engine receives an `EmbedsPrompt`; these IDs and other input
metadata still cross the RPC boundary, so this is not an embedding-only privacy
protocol.

Keep `repetition_penalty=1` and `prompt_logprobs=None` with this input mode:
the engine does not have the original token history for prefix-dependent
penalties or prompt-token scoring. Generated-token logprobs are still available.

This path supports text-only requests without PD disaggregation or fully
asynchronous partial-rollout resume. A resumed prefix needs embeddings for its
newly generated tokens as well; passing the original embedding matrix would
misalign the request. The LM Head and generated-token embedding lookups still
run on the rollout server. A real enterprise-side LM Head needs a separate
per-token transport and training backward path.

## KV cache versions

The vLLM server captures the committed training step and successful weight-update
counter after a request passes the admission gate. It sets `cache_salt` from both
values and returns that request's version in `TokenOutput.extra_fields`.
Re-publishing at the same training step therefore produces a new cache key.
Existing prefix-cache resets and synchronization barriers remain in place.
Partial-rollout clients retain their existing min/max training-step metadata;
one resumed trajectory can still span several policies.

With the matching vllm-ascend Mooncake layerwise connector, also configure a
nonempty `kv_connector_extra_config.cache_namespace` containing the tenant,
immutable base-model revision and a unique training-run identifier. Use the same
namespace and update sequence on producers and consumers. Choose a new run
namespace when restarting with reset update counters. The namespace prevents
accidental cross-run key reuse; it is not access control or encryption.

## Deployment validation

The accompanying checks exercise CPU tensor reconstruction, adapter staging,
request-version selection and cache-key agreement. They do not measure NPU or
WAN performance. On the deployment, compare adapter stage time, generation pause
time, useful trajectories per second and transferred bytes, with the same model
and task quality target.
