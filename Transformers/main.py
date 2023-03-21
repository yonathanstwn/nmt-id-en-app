"""
Main runner file for training and testing all the different models and datasets.
"""

import sys
from enum import Enum
import api
import datasetLoaders


class RunnerConfig(Enum):
    """Utility class to define enum constants for different models, datasets, training configurations"""

    # Helsinki-OPUS-en-id model finetuned with ccmatrix dataset using varying learning rates
    TRAIN_OPUS_EN_ID_CCMATRIX_LR_3 = 'TRAIN_OPUS_EN_ID_CCMATRIX_LR_3'
    TRAIN_OPUS_EN_ID_CCMATRIX_LR_4 = 'TRAIN_OPUS_EN_ID_CCMATRIX_LR_4'
    TRAIN_OPUS_EN_ID_CCMATRIX_LR_5 = 'TRAIN_OPUS_EN_ID_CCMATRIX_LR_5'

    # Helsinki-OPUS-id-en model finetuned with ccmatrix with warmup steps and no warmup steps
    TRAIN_OPUS_ID_EN_CCMATRIX_WARMUP = 'TRAIN_OPUS_ID_EN_CCMATRIX_WARMUP'
    TRAIN_OPUS_ID_EN_CCMATRIX_NO_WARMUP = 'TRAIN_OPUS_ID_EN_CCMATRIX_NO_WARMUP'

    # Helsinki-OPUS-en-id model finetuned with ccmatrix dataset with lr=1e-5 and 10M training examples
    TRAIN_OPUS_EN_ID_CCMATRIX_V2 = 'TRAIN_OPUS_EN_ID_CCMATRIX_V2'

    # Helsinki-OPUS-id-en model finetuned with ccmatrix dataset with lr=1e-5 and 10M training examples
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
        dataset = datasetLoaders.get_ccmatrix_train_val_ds(size=5_000_000, split_percentage=0.05)
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
        dataset = datasetLoaders.get_ccmatrix_train_val_ds(size=5_000_000, split_percentage=0.05)
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


if __name__ == '__main__':
    main(sys.argv[1])
