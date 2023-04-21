# EColIndo
The First Specialised NMT Model for English to Colloquial Indonesian Utilising GPT-Generated Dataset for Fully Unsupervised Learning

# Overview
The repository can be seen to have three main parts:
- `Transformers` contains all the training-related code.
- `Scraper` contains the generation part of the parallel English to colloquial Indonesian corpus.
- `WebApp`, `TrainAPI`, `InferenceAPI` are the three microservices that demonstrates the resulting fine-tuned NMT models, specifically EColIndo.

# EColIndo Models

The highlight of the project are the EColIndo models. There are several versions:
- https://huggingface.co/yonathanstwn/opus-ecolindo-best-loss-bleu
- https://huggingface.co/yonathanstwn/nllb-ecolindo-best-bleu
- https://huggingface.co/yonathanstwn/nllb-ecolindo-best-loss

They are available to try directly from the HuggingFace inference interface.

# Environment
Python version: Python 3.7.4

List of packages are found in requirements.txt and can be installed with:
`pip install -r requirements.txt`

Please ensure Python version is the same.

# Web Application
To run the web app, you need to run the three Django microservices locally. Ensure you have installed all requirements and activated the virtual environment.
Please follow the steps:
```
cd WebApp
python manage.py runserver
cd ..
cd InferenceAPI
python manage.py runserver 8001
cd ..
cd TrainAPI
python manage.py runserver 8002
```
