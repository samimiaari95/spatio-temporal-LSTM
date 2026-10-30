#!/bin/bash -x
# author: Sami Miaari
# e-mail: s.miaari@fz-juelich.de

#SBATCH --job-name=predictstatsmap_prediction
#SBATCH --account=xxxx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/outputs/EU_74_-48_69_20/statisticsEUmap/logs/prediction_%j.out
#SBATCH --error=/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/outputs/EU_74_-48_69_20/statisticsEUmap/logs/prediction_%j.err
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu-devel
#SBATCH --time=1:00:00
#SBATCH --mail-user=s.miaari@fz-juelich.de
#SBATCH --mail-type=ALL

set -euo pipefail

cd /p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DCadapted_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini

export NCCL_SOCKET_IFNAME=ib0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

STATISTICS_ROOT=/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/outputs/EU_74_-48_69_20/statisticsEUmap

if [[ "$#" -gt 1 ]]; then
  echo "Usage: sbatch $0 [run_name]" >&2
  exit 2
fi
if [[ "$#" -eq 1 ]]; then
  if [[ -n "${RUN_NAME:-}" ]]; then
    if [[ "$RUN_NAME" != "$1" ]]; then
      echo "RUN_NAME and the supplied run_name argument do not match" >&2
      exit 2
    fi
  fi
  RUN_NAME="$1"
fi

if [[ -z "${RUN_NAME:-}" ]]; then
  mapfile -t READY_RUN_DIRS < <(
    while IFS= read -r feature_path; do
      candidate_dir="${feature_path%/feature_inputs/era5land_features.npy}"
      if [[ -f "${candidate_dir}/metadata/preprocessing.json" ]]; then
        printf '%s\n' "$candidate_dir"
      fi
    done < <(
      find "$STATISTICS_ROOT" -mindepth 3 -maxdepth 3 -type f \
        -path '*/feature_inputs/era5land_features.npy' -print
    )
  )
  if [[ "${#READY_RUN_DIRS[@]}" -ne 1 ]]; then
    if [[ "${#READY_RUN_DIRS[@]}" -eq 0 ]]; then
      echo "No fully preprocessed run was found under ${STATISTICS_ROOT}" >&2
    else
      echo "More than one fully preprocessed run was found; specify one:" >&2
      printf '  %s\n' "${READY_RUN_DIRS[@]##*/}" >&2
    fi
    echo "Usage: sbatch $0 <run_name> (or submit through submit_statisticsmap_pipeline.sh)" >&2
    exit 2
  fi
  RUN_NAME="${READY_RUN_DIRS[0]##*/}"
  echo "RUN_NAME was not supplied; using the only fully preprocessed run: ${RUN_NAME}"
fi

if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid RUN_NAME: use only letters, numbers, '.', '_' or '-'" >&2
  exit 2
fi

RUN_DIR="${STATISTICS_ROOT}/${RUN_NAME}"
if [[ ! -s "${RUN_DIR}/feature_inputs/era5land_features.npy" ||
      ! -f "${RUN_DIR}/metadata/preprocessing.json" ]]; then
  echo "Preprocessed features and metadata are missing for run ${RUN_NAME}: ${RUN_DIR}" >&2
  exit 1
fi

echo "Starting four-GPU inference for ${RUN_NAME}"

srun env -u CUDA_VISIBLE_DEVICES torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  --module LSTM_model.data.predict_era5land_statisticsmap \
  --output-root "$STATISTICS_ROOT" \
  --run-name "$RUN_NAME" \
  --members 0-99 \
  --expected-world-size 4 \
  --require-gpu
