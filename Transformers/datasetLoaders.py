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

from datasets import load_dataset, DatasetDict, Dataset


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


def get_jakarta_research_inglish_train_val_ds():
    """
    Load "jakartaresearch/inglish" dataset for training and validation datasets
    Need to do preprocessing to fit the required format of utils.tokenize_dataset function
    """

    def preprocess(dataset):
        return {'translation': {'en': dataset['english'], 'id': dataset['indonesian']}}

    dataset = load_dataset("jakartaresearch/inglish")
    train_ds = dataset['train'].map(preprocess)
    val_ds = dataset['validation'].map(preprocess)
    train_ds = train_ds.remove_columns(['english', 'indonesian'])
    val_ds = val_ds.remove_columns(['english', 'indonesian'])
    return DatasetDict({
        'train': train_ds,
        'validation': val_ds
    })


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
    It contains test and validation parts so remove validation part.
    Need to do preprocessing to fit the required format of utils.tokenize_dataset function.
    """

    def preprocess(dataset):
        return {'translation': {'en': dataset['sourceString'], 'id': dataset['targetString']}}

    dataset = load_dataset("Helsinki-NLP/tatoeba_mt", "eng-ind")
    dataset.pop('validation')
    return DatasetDict({
        'test': dataset['test'].map(preprocess).remove_columns(['sourceLang', 'targetlang',
                                                                'sourceString', 'targetString'])
    })


def get_flores_test_ds(split, version):
    """
    Load Facebook's FLORES Evaluation Benchmark dataset for test dataset.
    Available splits are "dev" and "devtest"
    Available versions are "200" and "101" passed as input string
    for FLORES-200 and FLORES-101 respectively.
    Need to preprocess data to fit format of utils.tokenize_dataset function
    Comprises of 4 different test datasets depending on the split and version parameters.
    """
    if version == '101':
        hf_repo = "gsarti/flores_101"
        eng_code = 'eng'
        indo_code = 'ind'
    elif version == '200':
        hf_repo = "facebook/flores"
        eng_code = "eng_Latn"
        indo_code = "ind_Latn"
    else:
        raise Exception(
            "Invalid FLORES version. Please input either '200' or '101' ")

    english_dataset = load_dataset(hf_repo, eng_code)
    indo_dataset = load_dataset(hf_repo, indo_code)
    english_dataset = english_dataset[split].remove_columns(
        [col for col in english_dataset[split].column_names if col != "sentence"])
    indo_dataset = indo_dataset[split].remove_columns(
        [col for col in indo_dataset[split].column_names if col != "sentence"])
    combined_dataset_ls = []
    for i in range(len(english_dataset['sentence'])):
        combined_dataset_ls.append(
            {'en': english_dataset['sentence'][i], 'id': indo_dataset['sentence'][i]})
    return DatasetDict({
        'test': Dataset.from_dict({'translation': combined_dataset_ls})
    })
