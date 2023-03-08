#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/train-opus-mt-en-id-ccmatrix-lr-3.out
#SBATCH --job-name=fypTrain
#SBATCH --time=0-48:00
export TRANSFORMERS_CACHE=/scratch/users/k2036348/.cache
export HF_HOME=/scratch/users/k2036348/.cache
module load anaconda3/2021.05-gcc-9.4.0
source /users/${USER}/.bashrc
source activate /scratch/users/k2036348/nmt-id-en-app/venv
cd /scratch/users/k2036348/nmt-id-en-app/
cd Transformers/
python main2.py