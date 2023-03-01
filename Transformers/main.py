from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from datasets import load_dataset


# File script config
DOWNLOAD_MODEL = False
DOWNLOAD_DATASET = True
TRAIN_MODEL = True
HF_MODEL_REPOSITORY = "Helsinki-NLP/opus-mt-en-id"
MODEL_FILEPATH = "./models/opus-mt-en-id"
HF_DATASET_REPOSITORY = "yhavinga/ccmatrix"
DATASET_FILEPATH = './datasets/ccmatrix-en-id'

if DOWNLOAD_MODEL:
    # Download from HuggingFace repository
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPOSITORY)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_REPOSITORY)

    # Save locally
    tokenizer.save_pretrained(MODEL_FILEPATH)
    model.save_pretrained(MODEL_FILEPATH)

# Load local model from filepath
tokenizer = AutoTokenizer.from_pretrained(MODEL_FILEPATH)
model = AutoModel.from_pretrained(MODEL_FILEPATH)

# Test model
tokens = tokenizer("I like to go to the beach and study",
                   return_tensors="pt",
                   padding=True,
                   truncation=True,
                   max_length=512)

# Load dataset from cache or by downloading from HuggingFace if not cached
dataset = load_dataset(HF_DATASET_REPOSITORY, "en-id")
print(dataset['train'][:100]['translation'])

# Train model


