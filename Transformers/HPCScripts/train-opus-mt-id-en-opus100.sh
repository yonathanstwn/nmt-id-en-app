#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/JobOut/%j-train-opus-mt-id-en-opus100.out
#SBATCH --job-name=fypTrain
#SBATCH --time=0-48:00
#SBATCH --mem=10000
#SBATCH --gres=gpu
export TRANSFORMERS_CACHE=/scratch/users/k2036348/.cache
export HF_HOME=/scratch/users/k2036348/.cache
module load anaconda3/2021.05-gcc-9.4.0
module load cuda
source /users/${USER}/.bashrc
source activate /scratch/users/k2036348/nmt-id-en-app/Transformers/HPCScripts/venv
git lfs install
cd /scratch/users/k2036348/nmt-id-en-app/
cd Transformers/
python main.py TRAIN_OPUS_ID_EN_OPUS100