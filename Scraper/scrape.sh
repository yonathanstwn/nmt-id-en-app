#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/JobOut/%j-scrape.out
#SBATCH --job-name=scrape
#SBATCH --time=0-24:00
#SBATCH --mem=4000
module load anaconda3/2021.05-gcc-9.4.0
source /users/${USER}/.bashrc
source activate /scratch/users/k2036348/nmt-id-en-app/Scraper/venv
cd /scratch/users/k2036348/nmt-id-en-app/Scraper
python main.py