"""
Main runner file for training and testing all the different models and datasets.
"""

import os
import sys
from enum import Enum
import api
import datasetLoaders
import json


class RunnerConfig(Enum):
    """Utility class to define enum constants for different models, datasets, training configurations"""

    #################################
    ###### HELSINKI OPUS MODEL ######
    #################################

    # Helsinki-OPUS-en-id model finetuned with ccmatrix dataset using varying learning rates
    TRAIN_OPUS_EN_ID_CCMATRIX_LR_3 = 'TRAIN_OPUS_EN_ID_CCMATRIX_LR_3'
    TRAIN_OPUS_EN_ID_CCMATRIX_LR_4 = 'TRAIN_OPUS_EN_ID_CCMATRIX_LR_4'
    TRAIN_OPUS_EN_ID_CCMATRIX_LR_5 = 'TRAIN_OPUS_EN_ID_CCMATRIX_LR_5'

    # Helsinki-OPUS-id-en model finetuned with ccmatrix with warmup steps and no warmup steps
    TRAIN_OPUS_ID_EN_CCMATRIX_WARMUP = 'TRAIN_OPUS_ID_EN_CCMATRIX_WARMUP'
    TRAIN_OPUS_ID_EN_CCMATRIX_NO_WARMUP = 'TRAIN_OPUS_ID_EN_CCMATRIX_NO_WARMUP'

    # Helsinki-OPUS-en-id model finetuned with ccmatrix dataset with lr=1e-5, epochs=5, 5M training examples
    TRAIN_OPUS_EN_ID_CCMATRIX_V2 = 'TRAIN_OPUS_EN_ID_CCMATRIX_V2'

    # Helsinki-OPUS-id-en model finetuned with ccmatrix dataset with lr=1e-5, epochs=5, 5M training examples
    TRAIN_OPUS_ID_EN_CCMATRIX_V2 = 'TRAIN_OPUS_ID_EN_CCMATRIX_V2'

    # Helsinki-OPUS-en-id model finetuned with OPUS100 with default hyperparameters
    TRAIN_OPUS_EN_ID_OPUS100 = 'TRAIN_OPUS_EN_ID_OPUS100'

    # Helsinki-OPUS-id-en model finetuned with OPUS100 with default hyperparameters
    TRAIN_OPUS_ID_EN_OPUS100 = 'TRAIN_OPUS_ID_EN_OPUS100'

    # Helsinki-OPUS-en-id model finetuned with OpenSubtitles with default hyperparameters
    TRAIN_OPUS_EN_ID_OPEN_SUBTITLES = 'TRAIN_OPUS_EN_ID_OPEN_SUBTITLES'

    # Helsinki-OPUS-id-en model finetuned with OpenSubtitles with default hyperparameters
    TRAIN_OPUS_ID_EN_OPEN_SUBTITLES = 'TRAIN_OPUS_ID_EN_OPEN_SUBTITLES'

    # Helsinki-OPUS-en-id model finetuned with "jakartaresearch/inglish" with lr=1e-5 as it is optimal from previous training stats
    TRAIN_OPUS_EN_ID_JAKARTA = 'TRAIN_OPUS_EN_ID_JAKARTA'

    # Helsinki-OPUS-id-en model finetuned with "jakartaresearch/inglish" with lr=1e-5 as it is optimal from previous training stats
    TRAIN_OPUS_ID_EN_JAKARTA = 'TRAIN_OPUS_ID_EN_JAKARTA'

    # Tests all successful and best Helsinki-OPUS model checkpoints based on val_bleu and val_loss
    TEST_ALL_OPUS = 'TEST_ALL_OPUS'

    #################################
    ###### FACEBOOK NLLB MODEL ######
    #################################

    # NLLB (english to indonesian) finetuned with ccmatrix dataset with lr=1e-5, epochs=5, warmup_steps=4000
    TRAIN_NLLB_EN_ID_CCMATRIX = 'TRAIN_NLLB_EN_ID_CCMATRIX'

    # NLLB (indonesian to english) finetuned with ccmatrix dataset with lr=1e-5, epochs=5, warmup_steps=4000
    TRAIN_NLLB_ID_EN_CCMATRIX = 'TRAIN_NLLB_ID_EN_CCMATRIX'

    # Tests all successful and best NLLB model checkpoints based on val_bleu and val_loss
    TEST_ALL_NLLB = 'TEST_ALL_NLLB'

    #################################
    ########## ECOLINDO  ############
    #################################

    # Finetune Helsinki-OPUS-en-id model with GPT-generated ecolindo dataset
    TRAIN_OPUS_ECOLINDO = 'TRAIN_OPUS_ECOLINDO'

    # Finetune NLLB (english to indonesian) model with GPT-generated ecolindo dataset
    TRAIN_NLLB_ECOLINDO = 'TRAIN_NLLB_ECOLINDO'


def append_to_test_results_file(results):
    """
    Append to results part of the test_results.json file.
    If file is not created yet, create new test_results.json file.
    """

    # If file is not created yet, create file
    if not os.path.isfile("test_results.json"):
        with open("test_results.json", 'w') as f:
            init_json_structure = {
                "results": [],
                "number_of_tests": 0
            }
            json.dump(init_json_structure, f, indent=4)

    # Main algorithm to append test results to the json file
    with open("test_results.json", 'r') as f:
        data = json.load(f)
    data['results'].append(results)
    data['number_of_tests'] += 1
    with open("test_results.json", 'w') as f:
        json.dump(data, f, indent=4)


def test_all_datasets(hf_model_repo, lang_pair, **kwargs):
    """
    Test a model with all the test datasets available.

    Test datasets:
    - opus100_testset
    - tatoeba_testset
    - flores101_dev_testset
    - flores101_devtest_testset
    - flores200_dev_testset
    - flores200_devtest_testset
    """

    # Dictionary to link dataset names to actual dataset objects
    dataset_names_to_datasets = {
        'opus100_testset': datasetLoaders.get_opus100_test_ds(),
        'tatoeba_testset': datasetLoaders.get_tatoeba_test_ds(),
        'flores101_dev_testset': datasetLoaders.get_flores_test_ds('dev', '101'),
        'flores101_devtest_testset': datasetLoaders.get_flores_test_ds('devtest', '101'),
        'flores200_dev_testset': datasetLoaders.get_flores_test_ds('dev', '200'),
        'flores200_devtest_testset': datasetLoaders.get_flores_test_ds('devtest', '200')
    }

    # Test the model on every dataset and record individual results to the test_results.json file. 
    for ds_name, ds in dataset_names_to_datasets.items():
        results = api.test(hf_model_repo, ds['test'], lang_pair, **kwargs)
        model_results = {
            "model": hf_model_repo,
            "test_dataset": ds_name,
            "language_pair": lang_pair,
            "test_loss": results["test_loss"],
            "test_bleu": results["test_bleu"],
            "test_runtime": results["test_runtime"]
        }
        append_to_test_results_file(model_results)


def main(runner_config):
    """Main runner function which simply matches training configuration to the corresponding function calls"""

    ############################################
    ###### CCMATRIX ENGLISH -> INDONESIAN ######
    ############################################

    # Helsinki-OPUS-en-id model finetuned with ccmatrix dataset using varying learning rates
    if runner_config == RunnerConfig.TRAIN_OPUS_EN_ID_CCMATRIX_LR_3.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-en-id",
                  'opus-mt-en-id-ccmatrix-lr-3', dataset, 'en-id', lr=1e-3)
    elif runner_config == RunnerConfig.TRAIN_OPUS_EN_ID_CCMATRIX_LR_4.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-en-id",
                  'opus-mt-en-id-ccmatrix-lr-4', dataset, 'en-id')
    elif runner_config == RunnerConfig.TRAIN_OPUS_EN_ID_CCMATRIX_LR_5.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-en-id",
                  'opus-mt-en-id-ccmatrix-lr-5', dataset, 'en-id', lr=1e-5)
    elif runner_config == RunnerConfig.TRAIN_OPUS_EN_ID_CCMATRIX_V2.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds(
            size=5_000_000, split_percentage=0.05)
        api.train("Helsinki-NLP/opus-mt-en-id",
                  'opus-mt-en-id-ccmatrix-v2', dataset, 'en-id', lr=1e-5, epochs_n=5)

    ############################################
    ###### CCMATRIX INDONESIAN -> ENGLISH ######
    ############################################

    # Helsinki-OPUS-id-en model finetuned with ccmatrix with warmup steps and no warmup steps
    elif runner_config == RunnerConfig.TRAIN_OPUS_ID_EN_CCMATRIX_WARMUP.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-id-en",
                  'opus-mt-id-en-ccmatrix-warmup', dataset, 'id-en')
    elif runner_config == RunnerConfig.TRAIN_OPUS_ID_EN_CCMATRIX_NO_WARMUP.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-id-en",
                  'opus-mt-id-en-ccmatrix-no-warmup', dataset, 'id-en', warmup_steps=0)
    elif runner_config == RunnerConfig.TRAIN_OPUS_ID_EN_CCMATRIX_V2.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds(
            size=5_000_000, split_percentage=0.05)
        api.train("Helsinki-NLP/opus-mt-id-en",
                  'opus-mt-id-en-ccmatrix-v2', dataset, 'id-en', lr=1e-5, epochs_n=5)

    ############################################
    ###### OPUS100 ENGLISH -> INDONESIAN #######
    ############################################

    # Helsinki-OPUS-en-id model finetuned with OPUS100 with default hyperparameters
    elif runner_config == RunnerConfig.TRAIN_OPUS_EN_ID_OPUS100.value:
        dataset = datasetLoaders.get_opus100_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-en-id",
                  'opus-mt-en-id-opus100', dataset, 'en-id')

    ############################################
    ###### OPUS100 INDONESIAN -> ENGLISH #######
    ############################################

    # Helsinki-OPUS-id-en model finetuned with OPUS100 with default hyperparameters
    elif runner_config == RunnerConfig.TRAIN_OPUS_ID_EN_OPUS100.value:
        dataset = datasetLoaders.get_opus100_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-id-en",
                  'opus-mt-id-en-opus100', dataset, 'id-en')

    ###################################################
    ###### OPEN_SUBTITLES ENGLISH -> INDONESIAN #######
    ###################################################

    # Helsinki-OPUS-en-id model finetuned with OpenSubtitles with default hyperparameters
    elif runner_config == RunnerConfig.TRAIN_OPUS_EN_ID_OPEN_SUBTITLES.value:
        dataset = datasetLoaders.get_open_subtitles_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-en-id",
                  'opus-mt-en-id-open-subtitles', dataset, 'en-id')

    ###################################################
    ###### OPEN_SUBTITLES INDONESIAN -> ENGLISH #######
    ###################################################

    # Helsinki-OPUS-id-en model finetuned with OpenSubtitles with default hyperparameters
    elif runner_config == RunnerConfig.TRAIN_OPUS_ID_EN_OPEN_SUBTITLES.value:
        dataset = datasetLoaders.get_open_subtitles_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-id-en",
                  'opus-mt-id-en-open-subtitles', dataset, 'id-en')

    ###################################################
    ###### JAKARTA ENGLISH -> INDONESIAN ##############
    ###################################################

    # Helsinki-OPUS-en-id model finetuned with "jakartaresearch/inglish" with lr=1e-5 as it is optimal from previous training stats
    elif runner_config == RunnerConfig.TRAIN_OPUS_EN_ID_JAKARTA.value:
        dataset = datasetLoaders.get_jakarta_research_inglish_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-en-id",
                  'opus-mt-en-id-jakarta', dataset, 'en-id', lr=1e-5)

    ###################################################
    ###### JAKARTA INDONESIAN -> ENGLISH ##############
    ###################################################

    # Helsinki-OPUS-id-en model finetuned with "jakartaresearch/inglish" with lr=1e-5 as it is optimal from previous training stats
    elif runner_config == RunnerConfig.TRAIN_OPUS_ID_EN_JAKARTA.value:
        dataset = datasetLoaders.get_jakarta_research_inglish_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-id-en",
                  'opus-mt-id-en-jakarta', dataset, 'id-en', lr=1e-5)

    ###################################################
    ###### NLLB CCMATRIX ENGLISH -> INDONESIAN ########
    ###################################################

    # NLLB (english to indonesian) finetuned with ccmatrix dataset with lr=1e-5, epochs=5, warmup_steps=4000
    elif runner_config == RunnerConfig.TRAIN_NLLB_EN_ID_CCMATRIX.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds()
        api.train("facebook/nllb-200-distilled-600M",
                  'nllb-en-id-ccmatrix', dataset, 'en-id', base_model_dir="nllb-en-id",
                  lr=1e-5, epochs_n=10, src_lang="eng_Latn", tgt_lang="ind_Latn")

    ###################################################
    ###### NLLB CCMATRIX INDONESIAN -> ENGLISH ########
    ###################################################

    # NLLB (indonesian to english) finetuned with ccmatrix dataset with lr=1e-5, epochs=5, warmup_steps=4000
    elif runner_config == RunnerConfig.TRAIN_NLLB_ID_EN_CCMATRIX.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds()
        api.train("facebook/nllb-200-distilled-600M",
                  'nllb-id-en-ccmatrix', dataset, 'id-en', base_model_dir="nllb-id-en",
                  lr=1e-5, epochs_n=10, src_lang="ind_Latn", tgt_lang="eng_Latn")
        
    ###################################################
    ############ OPUS ECOLINDO TRAIN ##################
    ###################################################

    # Finetune Helsinki-OPUS-en-id model with GPT-generated ecolindo dataset
    elif runner_config == RunnerConfig.TRAIN_OPUS_ECOLINDO.value:
        dataset = datasetLoaders.get_ecolindo_train_val_ds()
        api.train("Helsinki-NLP/opus-mt-en-id",
                  'opus-ecolindo', dataset, 'english-colloquial_indo', lr=1e-5)

    ###################################################
    ############ NLLB ECOLINDO TRAIN ##################
    ###################################################

    # Finetune NLLB (english to indonesian) model with GPT-generated ecolindo dataset
    elif runner_config == RunnerConfig.TRAIN_NLLB_ECOLINDO.value:
        dataset = datasetLoaders.get_ecolindo_train_val_ds()
        api.train("facebook/nllb-200-distilled-600M",
                  'nllb-ecolindo', dataset, 'english-colloquial_ind', base_model_dir="nllb-en-id",
                  lr=1e-5, epochs_n=10, src_lang="eng_Latn", tgt_lang="ind_Latn")

    ###################################################
    ###### TESTING ALL HELSINKI-OPUS MODELS ###########
    ###################################################

    # Tests all successful and best Helsinki-OPUS model checkpoints based on val_bleu and val_loss
    elif runner_config == RunnerConfig.TEST_ALL_OPUS.value:
        
        # All models for English -> Indonesian translations
        hf_model_repo_list = [
            "Helsinki-NLP/opus-mt-en-id"]
        lang_pair = 'en-id'
        for hf_model_repo in hf_model_repo_list:
            test_all_datasets(hf_model_repo, lang_pair)
        
        # All models for Indonesian -> English translations
        hf_model_repo_list = [
            "Helsinki-NLP/opus-mt-id-en"]
        lang_pair = 'id-en'
        for hf_model_repo in hf_model_repo_list:
            test_all_datasets(hf_model_repo, lang_pair)

    ###################################################
    ########### TESTING ALL NLLB MODELS ###############
    ###################################################

    # Tests all successful and best NLLB model checkpoints based on val_bleu and val_loss
    elif runner_config == RunnerConfig.TEST_ALL_NLLB.value:
        
        # All models for English -> Indonesian translations
        hf_model_repo_list = [
            "facebook/nllb-200-distilled-600M"]
        lang_pair = 'en-id'
        for hf_model_repo in hf_model_repo_list:
            test_all_datasets(hf_model_repo, lang_pair, src_lang="eng_Latn", tgt_lang="ind_Latn")
        
        # All models for Indonesian -> English translations
        hf_model_repo_list = [
            "facebook/nllb-200-distilled-600M"]
        lang_pair = 'id-en'
        for hf_model_repo in hf_model_repo_list:
            test_all_datasets(hf_model_repo, lang_pair, src_lang="ind_Latn", tgt_lang="eng_Latn")


if __name__ == '__main__':
    main(sys.argv[1])
