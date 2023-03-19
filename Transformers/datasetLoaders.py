"""
Utility functions to load and preprocess specific datasets since
different HuggingFace datasets may have different required arguments
to load and may come in different formats (e.g. different columns), so
need to be preprocessed to fit the required format for utils.tokenize_dataset
function.

Dataset needs to be in a specific format where it contains a 'translation' column
which contains a dictionary of the parallel sentences with the same language codes
passed in the lang_pair parameter as the keys of the dictionary.

Datasets are loaded from cache or by downloading from HuggingFace if not cached.

train_test_split function splits dataset into DatasetDict with keys 'train' and 'test'
but when we want to use this split for 'train' and 'validation', we need to rename keys
for the training and validation dataset loaders.
"""

from datasets import load_dataset


#############################################
####### Train and Validation Datasets #######
#############################################

def get_ccmatrix_train_val_ds(size=1_000_000, split_percentage=0.1):
    """
    Load "yhavinga/ccmatrix" dataset for training and validation datasets
    It only contains train part so must split into train and validation
    """
    dataset = load_dataset("yhavinga/ccmatrix", 'en-id', split='train').select(
        range(size)).train_test_split(test_size=split_percentage, shuffle=False)
    dataset['validation'] = dataset.pop('test')
    return dataset


def get_opus100_train_val_ds():
    """
    Load "opus100" dataset for training and validation datasets.
    Dataset already contains train, validation, test parts.
    To be more memory efficient, test part is removed.
    """
    dataset = load_dataset("opus100", "en-id")
    dataset.pop('test')
    return dataset


def get_open_subtitles_train_val_ds(size=1_000_000, split_percentage=0.1):
    """
    Load "open_subtitles" dataset for training and validation datasets
    It only contains train part so must split into train and validation
    """
    dataset = load_dataset("open_subtitles", lang1="en", lang2="id")['train'].select(
        range(size)).train_test_split(test_size=split_percentage, shuffle=False)
    dataset['validation'] = dataset.pop('test')
    return dataset

#############################################
################ Test Datasets ##############
#############################################

def get_opus100_test_ds():
    """
    Load "opus100" dataset for test dataset.
    Dataset already contains train, validation, test parts.
    To be more memory efficient, train and validation parts are removed.
    """
    dataset = load_dataset("opus100", "en-id")
    dataset.pop('train')
    dataset.pop('validation')
    return dataset

def get_tatoeba_test_ds():
    """
    Load "Tatoeba Translation Challenge" dataset for test dataset.
    It contains test and validation parts so remove validation part
    """
    dataset = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-ind")
    dataset.pop('validation')
    return dataset
