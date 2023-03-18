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


def main(runner_config):
    """Main runner function which simply matches training configuration to the corresponding function calls"""

    # Helsinki-OPUS-en-id model finetuned with ccmatrix dataset using varying learning rates
    if runner_config == RunnerConfig.TRAIN_OPUS_EN_ID_CCMATRIX_LR_3.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds(size=1_000_000)
        api.train("Helsinki-NLP/opus-mt-en-id",
                  'opus-mt-en-id-ccmatrix-lr-3', dataset, 'en-id', lr=1e-3)
    elif runner_config == RunnerConfig.TRAIN_OPUS_EN_ID_CCMATRIX_LR_4.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds(size=1_000_000)
        api.train("Helsinki-NLP/opus-mt-en-id",
                  'opus-mt-en-id-ccmatrix-lr-4', dataset, 'en-id')
    elif runner_config == RunnerConfig.TRAIN_OPUS_EN_ID_CCMATRIX_LR_5.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds(size=1_000_000)
        api.train("Helsinki-NLP/opus-mt-en-id",
                  'opus-mt-en-id-ccmatrix-lr-5', dataset, 'en-id', lr=1e-5)

    # Helsinki-OPUS-id-en model finetuned with ccmatrix with warmup steps and no warmup steps
    elif runner_config == RunnerConfig.TRAIN_OPUS_ID_EN_CCMATRIX_WARMUP.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds(size=1_000_000)
        api.train("Helsinki-NLP/opus-mt-id-en",
                  'opus-mt-id-en-ccmatrix-warmup', dataset, 'id-en')
    elif runner_config == RunnerConfig.TRAIN_OPUS_ID_EN_CCMATRIX_NO_WARMUP.value:
        dataset = datasetLoaders.get_ccmatrix_train_val_ds(size=1_000_000)
        api.train("Helsinki-NLP/opus-mt-id-en",
                  'opus-mt-id-en-ccmatrix-no-warmup', dataset, 'id-en')


if __name__ == '__main__':
    main(sys.argv[1])
