#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/JobOut/%j-scrape5.out
#SBATCH --job-name=scrape
#SBATCH --time=0-48:00
#SBATCH --mem=4000
export HF_HOME=/scratch/users/k2036348/.cache
module load anaconda3/2021.05-gcc-9.4.0
source /users/${USER}/.bashrc
source activate /scratch/users/k2036348/nmt-id-en-app/Scraper5/venv
cd /scratch/users/k2036348/nmt-id-en-app/Scraper5
python main.py
