"""
Main public API functions to be called by outside scripts
"""
import sys
sys.path.append('..')
import shutil
from Transformers import utils
from transformers import Seq2SeqTrainingArguments


def train(hf_model_repo,
          model_name,
          dataset,
          lang_pair,
          base_model_dir=None,
          lr=1e-4,
          epochs_n=25,
          warmup_steps=4000,
          push_to_hub=True,
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

    upload_token = None
    if push_to_hub:
        upload_token = access_tokens['upload_token']

    training_args = utils.init_training_args(
        model_name, upload_token, lr, epochs_n, warmup_steps)
    trainer = utils.init_trainer(
        model, training_args, tokenized_dataset, tokenizer, compute_metrics_function)
    trainer.train()
    
    if push_to_hub:
        trainer.push_to_hub()


def test(hf_model_repo, dataset, lang_pair, **kwargs):
    """
    Tests the model using the test dataset provided to calculate loss and bleu metrics.
    Returns dictionary with keys: test_loss, test_bleu, test_runtime, etc.
    """

    # Basic setup
    model, tokenizer = utils.load_model_and_tokenizer(hf_model_repo, save_model=False, **kwargs)
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

    # Delete empty temp-test-trainer folder which was side effect of initializing trainer
    shutil.rmtree('temp-test-trainer')

    # Return metrics of test results
    return test_results.metrics


def translate(model, tokenizer, input_text, **kwargs):
    """Main translate function to translate an input text"""
    input_tokens = tokenizer(input_text, return_tensors='pt').input_ids
    output_tokens = model.generate(
        input_tokens, max_new_tokens=500, do_sample=True, top_k=30, top_p=0.95, **kwargs)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text
