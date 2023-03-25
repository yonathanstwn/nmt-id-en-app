"""
Main public API functions to be called by outside scripts
"""

import utils
from transformers import Seq2SeqTrainingArguments
import json


def train(hf_model_repo,
          model_name,
          dataset,
          lang_pair,
          base_model_dir=None,
          lr=1e-4,
          epochs_n=25,
          warmup_steps=4000,
          **kwargs
          ):
    """
    Main train function to train/finetune a HuggingFace pretrained model 
    using the specific parameters specified
    """
    model, tokenizer = utils.load_model_and_tokenizer(
        hf_model_repo, base_model_dir, **kwargs)
    access_tokens = utils.get_access_tokens()
    tokenized_dataset = utils.tokenize_dataset(dataset, tokenizer, lang_pair)
    compute_metrics_function = utils.create_compute_metrics_function(tokenizer)
    training_args = utils.init_training_args(
        model_name, access_tokens['upload_token'], lr, epochs_n, warmup_steps)
    trainer = utils.init_trainer(
        model, training_args, tokenized_dataset, tokenizer, compute_metrics_function)
    trainer.train()
    trainer.push_to_hub()


def test(hf_model_repo, dataset, lang_pair, **kwargs):
    """
    Tests the model using the test dataset provided to calculate loss and bleu metrics.
    Returns dictionary with keys: test_loss, test_bleu, test_runtime, etc.
    """

    # Basic setup
    model, tokenizer = utils.load_model_and_tokenizer(hf_model_repo, save=False, **kwargs)
    tokenized_test_dataset = utils.tokenize_dataset(
        dataset, tokenizer, lang_pair)
    compute_metrics_function = utils.create_compute_metrics_function(tokenizer)

    # Empty training args as this is for testing only
    training_args = Seq2SeqTrainingArguments('temp-test-trainer', 
                                             per_device_eval_batch_size=32,
                                             predict_with_generate=True)

    # Empty train and validation datasets
    train_val_dataset = {'train': [], 'validation': []}
    trainer = utils.init_trainer(
        model, training_args, train_val_dataset, tokenizer, compute_metrics_function)

    # Test
    test_results = trainer.predict(tokenized_test_dataset)

    return test_results


def translate(model, tokenizer, input_text, **kwargs):
    """Main translate function to translate an input text"""
    input_tokens = tokenizer(input_text, return_tensors='pt').input_ids
    output_tokens = model.generate(
        input_tokens, max_new_tokens=500, do_sample=True, top_k=30, top_p=0.95, **kwargs)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text
