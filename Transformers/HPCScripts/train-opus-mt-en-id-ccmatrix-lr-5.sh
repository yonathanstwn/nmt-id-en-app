#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/JobOut/train-opus-mt-en-id-ccmatrix-lr-5/%j.out
#SBATCH --job-name=fypTrain
#SBATCH --time=0-48:00
git lfs install
export TRANSFORMERS_CACHE=/scratch/users/k2036348/.cache
export HF_HOME=/scratch/users/k2036348/.cache
module load anaconda3/2021.05-gcc-9.4.0
source /users/${USER}/.bashrc
source activate /scratch/users/k2036348/nmt-id-en-app/venv
cd /scratch/users/k2036348/nmt-id-en-app/
cd Transformers/
python main.py TRAIN_OPUS_EN_ID_CCMATRIX_LR_5