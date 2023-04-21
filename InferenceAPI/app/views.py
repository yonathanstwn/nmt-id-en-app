import sacrebleu
import sys
sys.path.append('..')
from Transformers.api import translate
from Transformers.utils import load_model_and_tokenizer
import os
from django.http import HttpResponse
from config.settings import MODEL_NAMES

models_storage_dir = '/'.join(os.getcwd().split("/")[:-1]) + '/Transformers'

def _load_models():
    """Load models specified in settings and store in models storage in Transformers/models folder"""
    current_dir = os.getcwd()
    # change to models storage dir
    os.chdir(models_storage_dir)
    for lang_pair, model_name in MODEL_NAMES.items():
        m, t = load_model_and_tokenizer(model_name)
        models[lang_pair] = m
        tokenizers[lang_pair] = t
    # change back to current dir after loading
    os.chdir(current_dir)


# Load models when starting app
print("LOADING MODELS...")
models = {}
tokenizers = {}
_load_models()
print("LOADING MODELS DONE")


def translations(request):
    """Translate API"""
    lang_pair = request.POST.get('lang_pair')
    source_sentence = request.POST.get('source_sentence')
    target_sentence = translate(
        models[lang_pair], tokenizers[lang_pair], source_sentence)
    return HttpResponse(target_sentence)


def update(request):
    """Update API for when models just get updated from re-training to update the local models here"""
    _load_models()


def check_feedback(request):
    """API for checking quality of user feedback with backtranslation"""

    # BLEU THRESHOLD CONSTANT
    BLEU_THRESHOLD = 40

    source_sentence = request.POST.get('source_sentence')
    feedback_sentence = request.POST.get('feedback_sentence')
    lang_pair = request.POST.get('lang_pair')
    reverse_lang_pair = "-".join(lang_pair.split("-")[::-1])
    reverse_model = models[reverse_lang_pair]
    reverse_tokenizer = tokenizers[reverse_lang_pair]
    backtranslation = translate(
        reverse_model, reverse_tokenizer, feedback_sentence)
    print(backtranslation.lower().strip())
    print(source_sentence.lower().strip())
    result = sacrebleu.raw_corpus_bleu([source_sentence.lower().strip()], [[backtranslation.lower().strip()]])
    print(result.score)
    if result.score > BLEU_THRESHOLD:
        return HttpResponse('True')
    else:
        return HttpResponse('False')
