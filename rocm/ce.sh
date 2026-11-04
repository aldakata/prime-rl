#!/bin/bash
# run a command inside the ROCm vLLM container with /ptmp and /viper bound and the prime-rl venv active
source /etc/profile.d/modules.sh; module load apptainer/1.5.2 >/dev/null 2>&1
exec apptainer exec --rocm -B /ptmp,/viper /ptmp/atatjer/containers/vllm-rocm10.1-0.29.0.sif bash -c "export PATH=/ptmp/atatjer/venvs/prl/bin:\$HOME/.local/bin:\$PATH VIRTUAL_ENV=/ptmp/atatjer/venvs/prl; $*"
