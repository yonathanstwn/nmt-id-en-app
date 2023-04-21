from django.http import HttpResponse
import sys
sys.path.append('..')
from Transformers.api import translate
from Transformers.utils import load_model_and_tokenizer
import os
from django.http import HttpResponse
from config.settings import MODEL_NAMES
from Transformers.api import train


def train(request):
    training_data = request.POST.get('training_data')
    lang_pair = request.POST.get('lang_pair')
    model_name = MODEL_NAMES[lang_pair]
    return HttpResponse()