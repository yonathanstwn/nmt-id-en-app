from django.http import HttpResponse
from django.shortcuts import render

def train(request):
    training_data = request.POST.get('training_data')
    model = 
    return HttpResponse()