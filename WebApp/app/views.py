from django.http import Http404
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.shortcuts import redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
import requests
from app.models import UserData
from app.models import UserFeedback
from config.settings import TRAIN_API, INFERENCE_API


def login_prohibited(view_function):
    def modified_view_function(request):
        if request.user.is_authenticated:
            return redirect(settings.REDIRECT_URL_WHEN_LOGGED_IN)
        else:
            return view_function(request)
    return modified_view_function


def _process_lang_params(fromL, toL):
    # Language list
    # FID: Formal Indonesian
    # CID: Colloquial Indonesian
    # EN: English
    lang_ls = ['FID', 'CID', 'EN']

    # Check URL validity
    if fromL == toL or fromL not in lang_ls or toL not in lang_ls:
        raise Http404("Invalid language pair")
    if fromL != 'EN' and toL != 'EN':
        raise Http404(
            "Cannot translate between formal and colloquial Indonesian. Must include English in the language pair.")

    # Handle from or source language
    fromL_disp = 'Indonesian'
    if fromL == 'EN':
        fromL_disp = 'English'

    # Handle to or destination language
    is_formal = True
    toL_disp = 'English'
    if toL == 'CID':
        toL_disp = 'Indonesian (Colloquial)'
        is_formal = False
    elif toL == 'FID':
        toL_disp = 'Indonesian (Formal)'

    return {'fromL_disp': fromL_disp,
            'toL_disp': toL_disp,
            'is_formal': is_formal,
            'fromL': fromL,
            'toL': toL
            }


@login_required
def main(request):
    if request.method == 'GET':
        fromL = request.GET.get('fromL') or 'EN'
        toL = request.GET.get('toL') or 'FID'
        response_data = _process_lang_params(fromL, toL)
        return render(request, 'main.html', response_data)
    else:
        fromL = request.POST['fromL']
        toL = request.POST['toL']
        srcText = request.POST['sourceText']
        outText = ''
        response = requests.post(INFERENCE_API + "translations/",
                                 data={'lang_pair': fromL + '-' + toL, 'source_sentence': srcText})
        outText = response.content.decode('utf-8')
        response_data = _process_lang_params(fromL, toL)
        try:
            user_data = UserData.objects.get(user=request.user)
        except:
            user_data = UserData.objects.create(user=request.user, translations_count=0, feedback_count=0)
        user_data.translations_count += 1
        user_data.save()
        return render(request, 'main.html', {**response_data, 'sourceText': srcText, 'outText': outText})


@login_required
def feedback(request):
    if request.POST.get('action') == 'redirect':
        fromL = request.POST['fromL']
        toL = request.POST['toL']
        srcText = request.POST['sourceText']
        outText = request.POST['outText']
        response_data = _process_lang_params(fromL, toL)
        return render(request, 'feedback.html', {**response_data, 'sourceText': srcText, 'outText': outText})
    else:
        fromL = request.POST['fromL']
        toL = request.POST['toL']
        srcText = request.POST['sourceText']
        feedbackText = request.POST['feedbackText']
        # check feedback quality
        response = requests.post(INFERENCE_API + "check_feedback/", data={
                                'source_sentence': srcText, 'feedback_sentence': feedbackText, 'lang_pair': fromL + '-' + toL})
        if response.content.decode('utf-8') == 'True':
            UserFeedback.objects.create(
                src_text=srcText, target_text=feedbackText, lang_pair=fromL + '-' + toL)
        try:
            user_data = UserData.objects.get(user=request.user)
        except:
            user_data = UserData.objects.create(user=request.user, translations_count=0, feedback_count=0)
        user_data.feedback_count += 1
        user_data.save()
        return redirect('main')


@login_prohibited
def log_in(request):
    form = {}

    next = request.GET.get('next') or ''
    if request.method == 'POST':
        form = dict(request.POST)
        next = request.POST.get('next') or settings.REDIRECT_URL_WHEN_LOGGED_IN
        # Checks for less than 3 even though there are only 2 fields
        # because there is an extra csrf token request data
        if len([x for x in dict(request.POST).values() if x != ['']]) < 3:
            messages.add_message(request, messages.ERROR,
                                 "Please fill in all fields.")
        else:
            user = authenticate(
                username=request.POST['email'], password=request.POST['password'])
            if user is not None:
                login(request, user)
                return redirect(next)
            messages.add_message(request, messages.ERROR,
                                 "Invalid email or password.")

    return render(request, 'login.html', {'form': form, 'next': next})


@login_required
def log_out(request):
    logout(request)
    return redirect(settings.LOGIN_URL)


@login_prohibited
def sign_up(request):
    form = {}

    if request.method == 'POST':
        form = dict(request.POST)
        # Checks for less than 6 even though there are only 5 fields
        # because there is an extra csrf token request data
        if len([x for x in dict(request.POST).values() if x != ['']]) < 6:
            messages.add_message(request, messages.ERROR,
                                 "Please fill in all fields.")
        else:
            if User.objects.filter(email=request.POST['email']).exists():
                messages.add_message(
                    request, messages.ERROR, "Email address is taken. Please use another one.")
            elif request.POST['password'] != request.POST['confirmPassword']:
                messages.add_message(request, messages.ERROR,
                                     "Passwords do not match. Please try again.")
            else:
                user = User.objects.create_user(
                    username=request.POST['email'],
                    email=request.POST['email'],
                    password=request.POST['password'],
                    first_name=request.POST['firstName'],
                    last_name=request.POST['lastName']
                )
                login(request, user)
                return redirect('main')

    return render(request, 'signup.html', {'form': form})

def profile(request):
    name = request.user.get_full_name()
    email = request.user.email
    try:
        translations_count = UserData.objects.get(user=request.user).translations_count
        feedback_count = UserData.objects.get(user=request.user).feedback_count
    except:
        translations_count = 0
        feedback_count = 0
    return render(request, 'profile.html', {
        'name': name,
        'email': email,
        'translations_count': translations_count,
        'feedback_count': feedback_count,
    }
    )