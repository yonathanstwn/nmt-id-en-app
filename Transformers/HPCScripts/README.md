# HPC Scripts

This folder contains bash scripts to be run on the KCL CREATE High-Performance Cluster (HPC).
For more information: https://docs.er.kcl.ac.uk/CREATE/running_jobs/

To submit a job using a CPU, run:

`sbatch -p cpu <BASH_FILE>.sh`

To submit a job using a GPU, run:

`sbatch -p gpu <BASH_FILE>.sh`

# Environment

It is recommended to use Conda as the package manager in the HPC.

Before running the scripts, you must create a Conda virtual environment and install the Conda packages.

To install the packages, run the following in the terminal:

`conda create --prefix ./venv --file conda_requirements.txt`

Ensure your current working directory is this folder.

You can activate the virtual environment using:

`conda activate venv/`