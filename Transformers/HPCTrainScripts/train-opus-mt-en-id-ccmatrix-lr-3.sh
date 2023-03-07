#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/train-opus-mt-en-id-ccmatrix-lr-3.out
#SBATCH --job-name=fypTrain
#SBATCH --time=0-48:00
cd /scratch/users/k2036348/nmt-id-en-app/
conda init
conda activate venv
cd Transformers/
python main2.py