from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate
import numpy as np
import json
import os

# Code highly inspired from the tutorial:
# https://huggingface.co/docs/transformers/tasks/translation

# Download model and tokenizer if not downloaded yet, otherwise load them from local directory
def load_model_and_tokenizer(hf_repo):
    local_dir = "./models/opus-mt-en-id"
    if os.path.exists(local_dir) and os.path.isdir(local_dir):
        # Load local model from filepath
        return AutoModelForSeq2SeqLM.from_pretrained(local_dir), AutoTokenizer.from_pretrained(local_dir)
    else:
        # Download from HuggingFace repository
        tokenizer = AutoTokenizer.from_pretrained(hf_repo)
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_repo)
        # Save locally
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        # Return model and tokenizer
        return model, tokenizer
    
