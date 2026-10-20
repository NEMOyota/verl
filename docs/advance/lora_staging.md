# Staging LoRA updates before pausing rollout

Last updated: 10/20/2026

For a disaggregated enterprise trainer and operator-hosted vLLM rollout, enable:

```yaml
actor_rollout_ref:
  rollout:
    load_format: auto
    checkpoint_engine:
      backend: mooncake
      lora_staging: true
      lora_staging_max_megabytes: 4096
      update_weights_bucket_megabytes: 256
```

Use an already configured checkpoint transport; `mooncake` is one example.
Both sites must load the **same frozen base weights** from an immutable model
revision before starting. The mode requires unmerged LoRA and an engine that
exports adapter-only tensors, such as the FSDP or Megatron engines. It does not
initialize a dummy rollout model or verify base-model identity automatically.
The adapter must fit within the CPU staging limit, and its largest individual
tensor must fit in the transfer bucket.

The manager exports only the adapter, receives and copies it into CPU memory
while the old policy continues generating, and finalizes the WAN transfer.
Only then does it close the existing request admission gate, release KV,
load the complete adapter, clear caches, publish the policy version, and
resume generation. It uses the existing single-adapter replacement path;
requests do not run concurrently against two installed adapters.

Budget CPU memory for one complete adapter **per rollout checkpoint worker**.
Transport device buckets remain allocated alongside the running model and its
KV cache during staging; the CPU byte limit does not cover those allocations.
The local adapter loader also needs its normal tensor-copy workspace.
No tensor is written to disk by this staging path.

Metrics separate `checkpoint/lora_stage_seconds` from
`checkpoint/lora_pause_seconds`; `checkpoint/lora_staged_bytes_per_worker`
reports the largest received adapter payload. Transfer failures leave the
active adapter untouched. A failure after commit begins leaves generation
paused because some ranks may already have installed new weights; restart or
explicit recovery is required before serving again.

This path is opt-in and defaults to disabled. CPU checks cover adapter-only
export, reused-buffer ownership, staging-before-pause ordering and the byte
limit. Accelerator and WAN performance still need validation on the deployment.
