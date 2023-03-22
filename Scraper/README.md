# Overview

...


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