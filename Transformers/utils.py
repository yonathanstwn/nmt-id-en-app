"""
Code highly inspired from the tutorial:
https://huggingface.co/docs/transformers/tasks/translation

This file contains utility functions which are used to facilitate the main API
functions to be used in api.py file 
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np
import json
import os


def load_model_and_tokenizer(hf_repo, force_download=False):
    """Download model and tokenizer if not downloaded yet, otherwise load them from local directory"""
    local_dir = "./models/" + hf_repo.split("/")[1]
    if os.path.exists(local_dir) and os.path.isdir(local_dir) and not force_download:
        # Load local model from filepath
        return AutoModelForSeq2SeqLM.from_pretrained(local_dir), AutoTokenizer.from_pretrained(local_dir)
    else:
        # Download from HuggingFace repository
        tokenizer = AutoTokenizer.from_pretrained(hf_repo, force_download=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            hf_repo, force_download=True)
        # Save locally
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        # Return model and tokenizer
        return model, tokenizer


def get_access_tokens():
    """Get tokens from access_tokens.json file"""
    try:
        with open('access_tokens.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError as e:
        # Raise exception if file is not found
        raise Exception(
            "Please read README.md to configure your HuggingFace access tokens")


def tokenize_dataset(dataset, tokenizer, lang_pair):
    """
    Tokenize all example pairs in the dataset.
    Dataset needs to be in a specific format where it contains a 'translation' column
    which contains a dictionary of the parallel sentences with the same language codes
    passed in the lang_pair parameter as the keys of the dictionary.
    """

    source_lang = lang_pair.split("-")[0]
    target_lang = lang_pair.split("-")[1]

    def tokenize_helper(dataset):
        """Tokenize helper function"""
        inputs = [example[source_lang] for example in dataset['translation']]
        targets = [example[target_lang] for example in dataset['translation']]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=256, truncation=True)
        return model_inputs

    # Batch tokenize dataset
    return dataset.map(tokenize_helper, batched=True)


def create_compute_metrics_function(tokenizer):
    """
    Create and returns a compute metrics function that calculates 
    BLEU score to be passed to the Seq2SeqTrainer later
    """

    def postprocess_text(preds, labels):
        """
        Helper function to clean predictions and labels and 
        put them into appropriate format
        """
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        """
        The actual compute metrics function to be passed to the Seq2SeqTrainer later.
        It may only take one variable containing a pair (prediction, label/target)
        but needs to use the tokenizer hence the need for the higher order function
        to create and return this function.
        """
        # Init variables
        preds, labels = eval_preds
        sacrebleu = evaluate.load("sacrebleu")

        # Batch decode predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Batch decode labels/targets
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Use helper function to clean predictions and labels and put them into appropriate format
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels)

        # Compute BLEU score
        result = sacrebleu.compute(
            predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": round(result["score"], 4)}

        return result

    return compute_metrics


def init_training_args(model_name, upload_token, lr, epochs_n, warmup_steps):
    """Initialise an object containing the training arguments/hyperparameters"""
    return Seq2SeqTrainingArguments(
        output_dir='models/' + model_name,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=lr,
        optim="adamw_hf",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        generation_num_beams=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        num_train_epochs=epochs_n,
        predict_with_generate=True,
        load_best_model_at_end=True,
        save_total_limit=2,
        push_to_hub=True,
        hub_token=upload_token,
        hub_strategy="checkpoint"
    )


def init_trainer(model, args, tokenized_dataset, tokenizer, compute_metrics):
    """Initialize trainer object that provides an API to later train the model"""
    return Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

