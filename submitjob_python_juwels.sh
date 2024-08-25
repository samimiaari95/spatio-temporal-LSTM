#!/bin/bash
#
# author: Sami Miaari
# e-mail: s.miaari@fz-juelich.de

#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --account=esmtst
#SBATCH --time=02:00:00
#SBATCH --job-name="multitraining"
#SBATCH --ntasks=48
#SBATCH --ntasks-per-node=48
#SBATCH --mail-user=s.miaari@fz-juelich.de
#SBATCH --mail-type=ALL
#

cd /p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM
source /p/project1/cslts/miaari1/python_scripts/spatio-temporal-LSTM/env11_juwels.ini
python multitrain_LSTM.py