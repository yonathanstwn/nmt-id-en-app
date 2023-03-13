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

def get_ccmatrix_train_val_ds(lang_pair, size=1_000_000, split='train', split_percentage=0.1):
    """Load "yhavinga/ccmatrix" dataset for training and validation datasets"""
    dataset = load_dataset("yhavinga/ccmatrix", lang_pair, split=split).select(
        range(size)).train_test_split(test_size=split_percentage, shuffle=False)
    dataset['validation'] = dataset.pop('test')
    return dataset

