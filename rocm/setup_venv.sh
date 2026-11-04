#!/bin/bash
# Build the prime-rl venv on top of the ROCm vLLM container (MPCDF Viper, MI300A).
# Run INSIDE the container:  ./ce.sh-style:  apptainer exec --rocm -B /ptmp,/viper $SIF bash setup_venv.sh
# Never `uv sync` this repo on ROCm: uv.lock pins CUDA torch/vLLM wheels.
# Reconstructed from the interactive setup of 2026-11-04 (see ROCM.md); paths are /ptmp-specific.
set -euo pipefail
V=${V:-/ptmp/atatjer/venvs/prl}
PRL=${PRL:-$HOME/src/prime-rl-brutus}
BRUTUS=${BRUTUS:-$HOME/src/brutus-unchained}
W=${W:-/ptmp/atatjer/wheels}
CONS=${CONS:-/ptmp/atatjer/cons3.txt}

# 1) venv that sees the container's site-packages. /opt/python is itself a venv, so
#    --system-site-packages alone does not expose it: add a .pth.
python3 -m venv --system-site-packages $V
echo /opt/python/lib/python3.14/site-packages > $V/lib/python3.14/site-packages/_container_site.pth
source $V/bin/activate

# 2) constraints = container freeze (torch/vllm/triton/flash_attn stay the ROCm builds),
#    minus packages prime-rl needs newer/older versions of.
pip freeze --path /opt/python/lib/python3.14/site-packages \
  | grep -viE '^(verifiers|renderers|prime|prime-.*|harbor|mcp|mcp-types|typer)==' > $CONS
export PIP_CONSTRAINT=$CONS PIP_IGNORE_REQUIRES_PYTHON=1 PIP_FIND_LINKS=$W

# 3) fasttext (LID model) needs <cstdint> with the container's GCC
mkdir -p $W
CFLAGS="-include stdint.h" CXXFLAGS="-include cstdint" pip wheel --no-deps -w $W fasttext-wheel==0.9.2

# 4) prime-rl and its workspace packages (prime-rl itself --no-deps: its pins are CUDA)
pip install --no-deps -e $PRL
pip install -e $PRL/packages/prime-rl-configs -e $PRL/deps/pydantic-config -e $PRL/deps/renderers -e $PRL/deps/verifiers
pip install 'prime>=0.6.19' tomli-w tomli tyro tensorboard beartype jaxtyping loguru torchdata aiolimiter tenacity \
  orjson rich setproctitle pybase64 pandas msgspec nvidia-ml-py wandb==0.27.0 wandb-workspaces==0.4.3 fasttext-wheel==0.9.2
pip install --no-deps 'torchtitan @ git+https://github.com/pytorch/torchtitan@23e4dfc' \
  'dion @ git+https://github.com/microsoft/dion@d891eeb' \
  'flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention' \
  ring-flash-attn==0.1.8 modelexpress==0.3.0 torchao==0.17.0 vllm-router==0.1.26

# 5) brutus envs
pip install -e $BRUTUS/obfuscation/gym_env -e $BRUTUS/obfuscation/env

# 6) torchrun shim: the container's torchrun script points at /opt/python's interpreter
cat > $V/bin/torchrun <<'SH'
#!/bin/bash
exec /ptmp/atatjer/venvs/prl/bin/python -m torch.distributed.run "$@"
SH
chmod +x $V/bin/torchrun
python -c "import prime_rl, vllm, torch; print('ok', torch.__version__, vllm.__version__, torch.version.hip)"
