#!/bin/bash
#SBATCH -J brutus-rocm
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -o /ptmp/atatjer/brutus/logs/rl-%j.out
# usage: sbatch -p apudev -t 00:15:00 rl.sbatch   |   sbatch -p apu -t 06:00:00 rl.sbatch
source /etc/profile.d/modules.sh; module load apptainer/1.5.2
SIF=/ptmp/atatjer/containers/vllm-rocm10.1-0.29.0.sif
export HF_HOME=/ptmp/atatjer/hf HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
export APPTAINERENV_PREPEND_PATH=/ptmp/atatjer/venvs/prl/bin APPTAINERENV_VIRTUAL_ENV=/ptmp/atatjer/venvs/prl
L=/ptmp/atatjer/brutus/logs/$SLURM_JOB_ID; mkdir -p $L
echo "slurm env: GDO=$GPU_DEVICE_ORDINAL CUDA=$CUDA_VISIBLE_DEVICES HIP=$HIP_VISIBLE_DEVICES ROCR=$ROCR_VISIBLE_DEVICES"
unset GPU_DEVICE_ORDINAL HIP_VISIBLE_DEVICES
AX="apptainer exec --rocm -B /ptmp,/viper $SIF"
RFLAGS="--max-num-seqs 256 --max-model-len 16384 --max-num-batched-tokens 2048 --enable-prefix-caching --language-model-only"
# readers on APU $READER_GPU (default 0 = the inference APU), one at a time: vLLM's memory profiling counts the other servers' allocations
PIDS=""
READER_GPU=${READER_GPU:-0}; CONFIG=${CONFIG:-obf_mix2_rocm.toml}
start_reader() {  # model port util name
  CUDA_VISIBLE_DEVICES=$READER_GPU HIP_VISIBLE_DEVICES=$READER_GPU $AX vllm serve $1 --port $2 --gpu-memory-utilization $3 $RFLAGS > $L/$4.log 2>&1 &
  local pid=$!; PIDS="$PIDS $pid"; local t0=$(date +%s)
  until curl -sf localhost:$2/v1/models >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "reader $4 died"; tail -n 20 $L/$4.log; kill $PIDS 2>/dev/null; exit 1; }
    sleep 5; done
  echo "reader $4 (:$2, pid $pid) up after $(( $(date +%s)-t0 ))s"
}
start_reader Qwen/Qwen3.5-0.8B 8300 0.08 reader08
start_reader Qwen/Qwen3.5-4B 8301 0.15 reader4b
# memory log: MI300A GPU memory is host memory and counts against the job cgroup
CG=/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
( while true; do echo "$(date +%T) cgroup=$(( $(cat $CG/memory.current 2>/dev/null || echo 0)/1073741824 ))G free=$(free -g | awk '/Mem/{print $7}')G top: $(ps -u $USER -o rss=,comm= --sort=-rss | head -6 | awk '{printf "%s:%dG ", $2, $1/1048576}')"; sleep 30; done ) > $L/mem.log 2>&1 & MEMLOG=$!
cd ~/src/prime-rl-brutus
CUDA_VISIBLE_DEVICES=0,1 $AX rl @ /ptmp/atatjer/brutus/configs/$CONFIG $RL_ARGS > $L/rl.log 2>&1 & P3=$!
echo "rl pid $P3"; wait $P3; echo "rl exit $?"
kill $PIDS $MEMLOG 2>/dev/null; wait
