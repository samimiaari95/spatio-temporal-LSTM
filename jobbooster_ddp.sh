#!/bin/bash -x
# author: Sami Miaari
# e-mail: s.miaari@fz-juelich.de

#SBATCH --job-name=ensemble400px
#SBATCH --account=esmtst
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --gres=gpu:4
#SBATCH --partition=develbooster
#SBATCH --time=02:00:00
#SBATCH --mail-user=s.miaari@fz-juelich.de
#SBATCH --mail-type=ALL

cd /p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/env11_juwels.ini

# Check if an integer argument is provided
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <integer>"
  exit 1
fi

# Get the integer argument
ens_member="$1"

# Validate if the input is indeed an integer
if [[ ! "$ens_member" =~ ^-?[0-9]+$ ]]; then
  echo "Error: Input must be an integer."
  exit 1
fi

# Use the integer value in your script
echo "Received integer: $ens_member"
export ENS_MEMBER=$ens_member  # Set the environment variable

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
export MASTER_PORT=54123

export NCCL_SOCKET_IFNAME=ib0

srun env -u CUDA_VISIBLE_DEVICES python -u -m torchrun_jsc \
       --nproc_per_node=gpu \
       --nnodes="$SLURM_JOB_NUM_NODES" \
       --rdzv_id="$SLURM_JOB_ID" \
       --rdzv_endpoint="$MASTER_ADDR":"$MASTER_PORT" \
       --rdzv_backend static \
       /p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/main.py -t
