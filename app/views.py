from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.shortcuts import redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout


def login_prohibited(view_function):
    def modified_view_function(request):
        if request.user.is_authenticated:
            return redirect(settings.REDIRECT_URL_WHEN_LOGGED_IN)
        else:
            return view_function(request)
    return modified_view_function


@login_required
def main(request):
    return render(request, 'main.html')


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
            user = authenticate(username=request.POST['email'], password=request.POST['password'])
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
