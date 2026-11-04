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

`slurm.template_path` takes a custom template, so the plan is a Viper template:
- the same node split and router logic;
- every component launched in `apptainer exec --rocm` with the venv;
- no `uv`, no pattern kills;
- extra reader nodes whose URLs are passed into the env config.
