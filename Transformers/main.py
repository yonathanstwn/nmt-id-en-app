"""
Main runner file for training and testing all the different models and datasets.
"""

import sys
from enum import Enum
import api
import datasetLoaders


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

    #################################
    ###### FACEBOOK NLLB MODEL ######
    #################################

    # NLLB (english to indonesian) finetuned with ccmatrix dataset with lr=1e-5, epochs=5, warmup_steps=4000
    TRAIN_NLLB_EN_ID_CCMATRIX = 'TRAIN_NLLB_EN_ID_CCMATRIX'

    # NLLB (indonesian to english) finetuned with ccmatrix dataset with lr=1e-5, epochs=5, warmup_steps=4000
    TRAIN_NLLB_ID_EN_CCMATRIX = 'TRAIN_NLLB_ID_EN_CCMATRIX'

    # NLLB (english to indonesian) finetuned with opus100 dataset with lr=1e-5, epochs=5, warmup_steps=4000
    TRAIN_NLLB_EN_ID_OPUS100 = 'TRAIN_NLLB_EN_ID_OPUS100'

    # NLLB (indonesian to english) finetuned with opus100 dataset with lr=1e-5, epochs=5, warmup_steps=4000
    TRAIN_NLLB_ID_EN_OPUS100 = 'TRAIN_NLLB_ID_EN_OPUS100'


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
                  lr=1e-5, epochs_n=5, src_lang="eng_Latn", tgt_lang="ind_Latn")

    ###################################################
    ###### NLLB CCMATRIX INDONESIAN -> ENGLISH ########
    ###################################################

    # NLLB (indonesian to english) finetuned with ccmatrix dataset with lr=1e-5, epochs=5, warmup_steps=4000
    elif runner_config == RunnerConfig.TRAIN_NLLB_ID_EN_CCMATRIX.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds()
        api.train("facebook/nllb-200-distilled-600M",
                  'nllb-id-en-ccmatrix', dataset, 'id-en', base_model_dir="nllb-id-en",
                  lr=1e-5, epochs_n=5, src_lang="ind_Latn", tgt_lang="eng_Latn")

    ###################################################
    ###### NLLB OPUS100 ENGLISH -> INDONESIAN ########
    ###################################################

    # NLLB (english to indonesian) finetuned with opus100 dataset with lr=1e-5, epochs=5, warmup_steps=4000
    elif runner_config == RunnerConfig.TRAIN_NLLB_EN_ID_OPUS100.value:
        dataset = datasetLoaders.get_opus100_train_val_ds()
        api.train("facebook/nllb-200-distilled-600M",
                  'nllb-en-id-opus100', dataset, 'en-id', base_model_dir="nllb-en-id",
                  lr=1e-5, epochs_n=10, src_lang="eng_Latn", tgt_lang="ind_Latn")

    ###################################################
    ###### NLLB OPUS100 INDONESIAN -> ENGLISH ########
    ###################################################

    # NLLB (indonesian to english) finetuned with opus100 dataset with lr=1e-5, epochs=5, warmup_steps=4000
    elif runner_config == RunnerConfig.TRAIN_NLLB_ID_EN_OPUS100.value:
        dataset = datasetLoaders.get_opus100_train_val_ds()
        api.train("facebook/nllb-200-distilled-600M",
                  'nllb-id-en-opus100', dataset, 'id-en', base_model_dir="nllb-id-en",
                  lr=1e-5, epochs_n=10, src_lang="ind_Latn", tgt_lang="eng_Latn")


if __name__ == '__main__':
    main(sys.argv[1])
