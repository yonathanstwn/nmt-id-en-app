"""
Main public API functions to be called by outside scripts
"""

import utils


def train(hf_model_repo,
          model_name,
          dataset,
          lang_pair,
          lr=1e-4,
          epochs_n=25
          ):
    """
    Main train function to train/finetune a HuggingFace pretrained model 
    using the specific parameters specified
    """
    model, tokenizer = utils.load_model_and_tokenizer(hf_model_repo)
    access_tokens = utils.get_access_tokens()
    tokenized_dataset = utils.tokenize_dataset(dataset, tokenizer, lang_pair)
    compute_metrics_function = utils.create_compute_metrics_function(tokenizer)
    training_args = utils.init_training_args(
        model_name, access_tokens['upload_token'], lr, epochs_n)
    trainer = utils.init_trainer(model, training_args, tokenized_dataset, tokenizer, compute_metrics_function)
    trainer.train()
    trainer.push_to_hub()

def test(dataset):
    pass

def translate(model, tokenizer, input_text):
    """Main translate function to translate an input text"""
    input_tokens = tokenizer(input_text, return_tensors='pt').input_ids
    output_tokens = model.generate(
        input_tokens, max_new_tokens=500, do_sample=True, top_k=30, top_p=0.95)
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return output_text
