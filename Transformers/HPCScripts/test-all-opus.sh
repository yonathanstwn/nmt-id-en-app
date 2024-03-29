#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/JobOut/%j-test-all-opus.out
#SBATCH --job-name=fypTest
#SBATCH --time=0-24:00
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
python main.py TEST_ALL_OPUS