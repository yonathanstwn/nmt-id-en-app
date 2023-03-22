# Overview

This is a Python scraper program that aims to generate translations from English to colloquial Indonesian using the OpenAI GPT-3.5 LLM model through their API endpoints.

The generated dataset will then be used to finetune a much smaller model that translates English to colloquial Indonesian. The motivation is that this much smaller model will be more memory-efficient, with a much less parameter count, such that it enables faster inference, needs less storage, and much more specialized. 

To run the scraper, please run `main.py` using:

`python main.py`

Alternatively, if you are running it on the HPC, run the bash script `scrape.sh` using:

`sbatch -p cpu scrape.sh`

This will queue and run a batch job when the required resources are available in the HPC.


# Setup

Please ensure you have installed all the requirements from the root folder which includes the `openai` python library.

Create a json file in the `Scraper` folder called `api_key.json` with the following schema:

```json
{
    "key": "key123"
}
```

Alternatively, if you are using the HPC, it is recommended to use Conda.

Before running the scripts, you must create a Conda virtual environment and install the Conda packages.

To install the packages, run the following in the terminal:

`conda create --prefix ./venv --file conda_requirements.txt`

Ensure your current working directory is this folder.

You can activate the virtual environment using:

`conda activate venv/`