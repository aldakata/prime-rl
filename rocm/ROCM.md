# prime-rl on ROCm (MPCDF Viper, AMD MI300A)

Branch `exp/brutus-rocm` = `exp/brutus` + the fixes below. Status (2026-11-04): a 10-step
parity run of `obf_mix2` (Qwen3.5-4B + LoRA r32, batch 32 x group 8, oversampling 4) runs
end to end on one Viper `apu` node. LoRA updates load (every `/load_lora_adapter` call
returns 200) and mismatch KL stays at 3-4e-4. Truncation and correct-trace lengths match the
CUDA run (job 12698279).

All paths here are specific to `/ptmp/atatjer`.

## Cluster facts that matter

- **Nodes:** `vipa*` nodes, 2x MI300A per node (gfx942, ~110 GiB each), 96 cores, 220 GB
  Slurm memory, no NVLink.
- **Partitions:**
  - `apu`: exclusive, 2 GPUs, 24 h max, 128 nodes per job max.
  - `apu1`: 1 GPU, shared node.
  - `apudev`: 15 min.
  - Only `viper11-13` can submit to `apu*`.
- **Unified memory:** GPU allocations count as process RSS and against the job's 220 GB
  memory cgroup. A trainer that is fine on CUDA can get the job killed (`OUT_OF_MEMORY`).
  Use full activation checkpointing (`ac` at its default), and log `memory.current` of the
  cgroup (`rl.sbatch` does this).
- **Slurm GPU env vars:** Slurm sets `CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` and
  `GPU_DEVICE_ORDINAL`. HIP aborts when `GPU_DEVICE_ORDINAL` disagrees with the visible
  set, so `unset GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES` and pin with
  `CUDA_VISIBLE_DEVICES` (plus `HIP_VISIBLE_DEVICES`).
- **Several vLLM servers on one APU:** start them one after another. vLLM's memory
  profiling counts the other processes' allocations, so starting them together gives
  negative KV cache sizes.

## Stack

- **Container:** `rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-pytorch_2.13.0_vllm-0.29.0`, as
  `/ptmp/atatjer/containers/vllm-rocm10.1-0.29.0.sif`.
  - Contents: vLLM 0.29.1.dev0, torch 2.13.0+rocm10.1.0, Python 3.14, transformers 5.18,
    triton 3.8, ROCm flash_attn 2.8.3.
- **Venv:** `/ptmp/atatjer/venvs/prl`, built by `setup_venv.sh`. It layers on the
  container's site-packages and pins against the container freeze.
  - **Never `uv sync`**: `uv.lock` pins CUDA wheels.
  - `setup_venv.sh` was reconstructed from the interactive setup and hasn't been re-run
    from scratch.
- **apptainer:** run with `--rocm -B /ptmp,/viper`. It resets `PATH`, so prepend the venv
  with `APPTAINERENV_PREPEND_PATH`. `ce.sh` runs one command in the container with the venv.

## Running

```
cd /ptmp/atatjer/brutus
sbatch -p apu -t 03:00:00 rl.sbatch       # CONFIG=obf_mix2_rocm.toml READER_GPU=0 RL_ARGS=... are optional
```

`rl.sbatch` does the following, in order:
1. Starts the 0.8B and 4B reader vLLM servers, one after another, on APU `$READER_GPU`
   (ports 8300/8301).
2. Starts the cgroup memory logger.
3. Runs `rl @ configs/$CONFIG` in the container on both APUs: inference on APU 0, the
   trainer on APU 1.

Logs go to `/ptmp/atatjer/brutus/logs/<jobid>/` and `<output_dir>/logs/`.

The run config is `brutus-unchained/configs/viper/obf_mix2_rocm.toml`. It is `obf_mix2.toml`
with these changes:
- 1 train + 1 infer GPU
- no wandb
- local paths for the reference chars, LID model and taskset cache
- `gpu_memory_utilization` 0.45
- full activation checkpointing

## Fix log

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | trainer picks an attention impl unsupported on ROCm | `resolve_auto_attn` reads the gfx arch (9,4) as a CUDA SM | `trainer/model.py`: on HIP, resolve to `flash_attention_2` (only ROCm FA2 build exists) |
| 2 | `rl` fails listing GPUs | `get_physical_gpu_ids` uses pynvml | `entrypoints/rl.py`: on HIP use `torch.cuda.device_count()` |
| 3 | `ImportError` in inference server | vLLM 0.29 moved `cli_args`, `ErrorResponse`, `RequestResponseMetadata`, usage protocol classes | `inference/vllm/server.py`, `serving_tokens.py`: try 0.26 paths, fall back to `vllm.entrypoints.launchers.cli_args`, `vllm.entrypoints.serve.engine.protocol`, `vllm.entrypoints.generate.base.protocol` |
| 4 | **policy server silently never received LoRA updates**; 404 on `/load_lora_adapter` | vLLM 0.29 `run_server` lives in `vllm.entrypoints.launchers.api_server.entry` and resolves `build_app`/`init_app_state` from its own globals; patching `vllm.entrypoints.openai.api_server` (now a re-export shim) had no effect, so none of prime-rl's routes (`/load_lora_adapter`, `/update_weights`, `/pause`, `/resume`, `/liveness`, `/init_broadcaster`) nor `PrimeRlServingTokens` were installed | `server.py`: also patch `launchers.api_server.entry` (commit 350af96c). Check: inference log lists `Route: /load_lora_adapter` and shows `POST /load_lora_adapter 200` |
| 5 | job `OUT_OF_MEMORY` at step ~7 | unified memory: trainer peak 70 GiB + policy server + 2 readers > 220 GB cgroup | full AC (peak 23.7 GiB), policy util 0.45, readers 0.08/0.15 |
| 6 | 4B reader refuses to start: `max_num_seqs (1024) exceeds available Mamba cache blocks` | default max-num-seqs with small util | readers `--max-num-seqs 256` |
| 7 | `rl: executable not found`, then `FileNotFoundError: vllm-router` | apptainer resets PATH; router not in container | `APPTAINERENV_PREPEND_PATH`; install `vllm-router` wheel; `torchrun` shim in the venv |

Environment-only issues (all handled in `setup_venv.sh`):
- `/opt/python` is itself a venv; `setup_venv.sh` adds a `.pth` so ours sees it.
- Constraint conflicts:
  - mcp, typer, harbor/litellm vs verifiers: dropped those constraints and skipped
    `verifiers[harbor]`.
  - torchao 0.18 imports flydsl: pinned to 0.17.0.
  - wandb_gql missing: pinned wandb 0.27.0.
- fasttext needs `-include cstdint` to build with the container's GCC.
- ring_flash_attn is already shimmed by `prime_rl._compat`.

## Throughput (rollout generation is the bottleneck)

`vllm bench throughput` of Qwen3.5-4B on one APU, util 0.55, 256 prompts, input 1024 /
output 2048 tokens. All numbers are output tokens/s.

| variant | out tok/s |
|---|---|
| baseline (ROCM_ATTN, Triton paged-attention fallback, Triton GDN decode) | 6393 |
| `--max-num-seqs 192` | 5365 |
| no prefix caching | 6461 |
| `--attention-backend TRITON_ATTN` | 7432 |
| AITER unified attention | fails: read-only JIT dir in the container |
| prime-rl plugin, `fp32_lm_head` | 6470 |
| `--enable-lora` (r32, max_loras 8), adapter loaded | 4748 |
| LoRA + TRITON_ATTN | 5218 |

Takeaways:
- Preemption and fp32_lm_head aren't the cause.
- LoRA costs about 26%.
- TRITON_ATTN gets back about 10%.
- In the RL run, generation peaks at ~3.7k tok/s, with most rollouts running to the
  8192-token cap.

## Multi-node

prime-rl's bundled `multi_node_rl.sbatch.j2` doesn't run on Viper as-is:
- It runs `source .venv/bin/activate` and `uv sync --all-extras --all-packages`
  (CUDA wheels).
- It runs every component with `uv run`, on the host rather than in the container.
- Its node cleanup pattern-kills `vllm`/`python` processes and calls `nvidia-smi`.
- It has no place for the reward-reader vLLM servers.

`rocm/viper_rl.sbatch.j2` is a Viper template for `slurm.template_path`:
- every component runs in `apptainer exec --rocm` with the venv;
- no `uv`, no pattern kills: cleanup kills only the PIDs the script recorded;
- reward-reader **pools**: R vLLM replicas (one per APU) on extra nodes after the main
  nodes, behind one `vllm-router` on the orchestrator node's `localhost:<port>`, so the
  taskset keeps a single reader URL. The orchestrator starts only after every replica and
  the router answer;
- resumes with `--ckpt.resume-step -1` when `<output_dir>/checkpoints/step_*` exists, so
  jobs can be chained with `--dependency=afterany`.

The pools and site paths go in `[slurm.template_extra]` (a free-form dict this branch adds
to `SlurmConfig`; the bundled templates ignore it). See the header of the template for all
keys.

```toml
[slurm]
template_path = "/u/atatjer/src/prime-rl-brutus/rocm/viper_rl.sbatch.j2"
project_dir = "/u/atatjer/src/prime-rl-brutus"
partition = "apu"
account = "mmgt_apu"
time = "01:30:00"

[slurm.template_extra]
env = { HF_HOME = "/ptmp/atatjer/hf", HF_HUB_OFFLINE = "1", WANDB_MODE = "disabled" }

[[slurm.template_extra.pools]]
name = "reader"
model = "Qwen/Qwen3.5-4B"
port = 8300          # router port; reader_base_url = "http://localhost:8300/v1"
replicas = 2
util = 0.85
policy = "round_robin"   # the default; cache_aware pinned 168 of 178 requests to one replica
args = "--max-num-seqs 256 --max-model-len 16384 --max-num-batched-tokens 2048 --enable-prefix-caching --language-model-only --attention-backend TRITON_ATTN"
```

Render in the container, submit on the host (there is no `sbatch` in the container):

```
cd /ptmp/atatjer/brutus
~/src/prime-rl-brutus/rocm/ce.sh rl @ configs/<cfg>.toml --dry-run     # writes <output_dir>/rl.sbatch
sbatch <output_dir>/rl.sbatch
sbatch --dependency=afterany:<jobid> <output_dir>/rl.sbatch   # chain: resumes from the latest checkpoint
```

Viper gotchas the template handles:
- `apu` caps `--cpus-per-task` at 48 (that gives the task all 96 hardware threads).
  Without `-c`, srun steps get 1 CPU and fail with "More processors requested than
  permitted".
- `#` followed by `{` in bash (`${#arr[@]}`) opens a Jinja comment; avoid it in the template.

Tested (job 12702230, single_node + one 2-replica 4B reader pool, 2 nodes, 5 steps): the
job completed, the router served all reader traffic, and mismatch KL was 2-4e-4.
Resubmitting the same `rl.sbatch` with `max_steps = 7` (job 12702673) resumed from
`step_5` and ran steps 6-7. The
`multi_node` branch (trainer and policy inference on separate nodes) runs: lanhid job 12712011
(3 nodes: 2 policy APUs behind vllm-router, 2 trainer APUs, 2 x 27B readers) ran clean.

Lessons from the lanhid/hid4b launches (2026-11-05):
- **The orchestrator and env servers need a visible GPU.** On ROCm, importing torchao queries the
  device (`torchao.utils.is_MI300`) and fails with "No CUDA GPUs are available" if
  `CUDA_VISIBLE_DEVICES` is empty (job 12708157).
- **Routers must never eject a worker.** A policy API server stalled for ~100 s at a step
  boundary, probably while it serialized a burst of finished 8k-token responses with logprobs.
  The router's default health check (5 s timeout, 3 failures) dropped both workers, and 1629
  rollouts failed with "No available workers" (job 12709100). The template now passes
  `--health-check-timeout-secs 60 --health-failure-threshold 1000000 --disable-circuit-breaker`.
  prime-rl's own single-node router uses threshold 20 and timeout 30.
- **Size in-flight rollouts to the KV cache, not the batch.** The policy router times requests out
  at 1800 s (`--request-timeout-secs`). With `max_inflight_episodes` x ~10k tokens above the KV
  cache, vLLM keeps preempting the longest requests until they time out, so the long rollouts are
  the ones dropped (a biased loss). Results on 1 APU:
  - 512 in flight: 79% errors (job 12708192).
  - 256 in flight on a 1.62M-token cache: 9% errors (job 12711052).
  - 160 per APU: no preemption and no errors.
  - The KV cache is ~1.62M tokens for Qwen3.5-4B at util 0.75 and ~1.9M at 0.85.
- **27B readers with `reader_skip_wrong` are nearly idle.** Scoring takes median 0.1 s against
  250-350 s of generation, so 2 replicas are enough.
