from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import evaluate
import numpy as np
import json

# Code highly inspired from the tutorial:
# https://huggingface.co/docs/transformers/tasks/translation

# File script config
DOWNLOAD_MODEL = False
DOWNLOAD_DATASET = True
TRAIN_MODEL = True
HF_MODEL_REPOSITORY = "Helsinki-NLP/opus-mt-en-id"
MODEL_FILEPATH = "./models/opus-mt-en-id"
HF_DATASET_REPOSITORY = "yhavinga/ccmatrix"

# Get tokens from access_tokens.json file
# Raise exception if file is not found
try:
    with open('access_tokens.json', 'r') as f:
        ACCESS_TOKENS = json.load(f)
except FileNotFoundError as e:
    raise Exception("Please read README.md to configure your HuggingFace access tokens")

if DOWNLOAD_MODEL:
    # Download from HuggingFace repository
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPOSITORY)
    model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_REPOSITORY)

    # Save locally
    tokenizer.save_pretrained(MODEL_FILEPATH)
    model.save_pretrained(MODEL_FILEPATH)

# Load local model from filepath
tokenizer = AutoTokenizer.from_pretrained(MODEL_FILEPATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_FILEPATH)

# Load dataset from cache or by downloading from HuggingFace if not cached
dataset = load_dataset(HF_DATASET_REPOSITORY, "en-id")['train'].select(range(1000000)).train_test_split(test_size=0.1)

# Tokenize helper function
def tokenize_helper(dataset):
    inputs = [example['en'] for example in dataset['translation']]
    targets = [example['id'] for example in dataset['translation']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=256, truncation=True)
    return model_inputs

# Batch tokenize dataset
tokenized_dataset = dataset.map(tokenize_helper, batched=True)

# Create batches of tokenized examples and dynamically apply padding
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Evaluate
sacrebleu = evaluate.load("sacrebleu")

# Helper function to clean predictions and labels and put them into appropriate format
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

# Compute metrics function that calculates BLEU score to be passed to the Seq2SeqTrainer later 
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    # Batch decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Batch decode labels/targets
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Use helper function to clean predictions and labels and put them into appropriate format
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Compute BLEU score
    result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": round(result["score"], 4)}
    
    return result

# Train model
training_args = Seq2SeqTrainingArguments(
    output_dir='models/opus-mt-en-id-ccmatrix-lr-4',
    evaluation_strategy="epoch",
    save_strategy='epoch',
    learning_rate=1e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    num_train_epochs=3,
    predict_with_generate=True,
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_token=ACCESS_TOKENS['upload_token']
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
                               
# Inference 
def translate(model, tokenizer, input_text):
    input_tokens = tokenizer(input_text, return_tensors='pt').input_ids
    output_tokens = model.generate(input_tokens, max_new_tokens=500, do_sample=True, top_k=30, top_p=0.95)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text

