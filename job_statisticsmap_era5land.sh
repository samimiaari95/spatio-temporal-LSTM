#!/bin/bash -x
# author: Sami Miaari
# e-mail: s.miaari@fz-juelich.de

#SBATCH --job-name=era5land_lstm_statsmap
#SBATCH --account=xxxxx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/outputs/EU_74_-48_69_20/statisticsEUmap/run_20261029_era5land_mask/logs/job_%j.out
#SBATCH --error=/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/outputs/EU_74_-48_69_20/statisticsEUmap/run_20261029_era5land_mask/logs/job_%j.err
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu-devel
#SBATCH --time=00:20:00
#SBATCH --mail-user=s.miaari@fz-juelich.de
#SBATCH --mail-type=ALL

cd /p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM
source /p/project1/cslts/miaari1/python_scripts/DailyScriptBox/envs/loadenv.JURECA-DCadapted_JUWELS_2025_GCC-OpenMPI_Std-Py-AI.ini
module load netcdf4-python

srun python -u -m LSTM_model.data.predict_era5land_statisticsmap \
  --mask-path /p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/inputs/EU_74_-48_69_20/mapping_0stdroll6months.npy \
  --era5-dir /p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/inputs/EU_74_-48_69_20/raw/ERA5-Land \
  --input-dir /p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/inputs/EU_74_-48_69_20 \
  --output-root /p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM/outputs/EU_74_-48_69_20/statisticsEUmap \
  --run-name run_20261029_era5land_mask \
  --members 0-99
