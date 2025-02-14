from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.contrib import messages

from Adviser.forms import UserRegistrationForm
from .models import Transaction, Budget
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# Dashboard view - logged-in users only
@login_required
def dashboard(request):
    try:
        # Get user's budget or handle the case where no budget exists
        user_budget = Budget.objects.get(user=request.user)
    except Budget.DoesNotExist:
        user_budget = None  # Handle cases where no budget exists for the user

    # Get user's transactions
    transactions = Transaction.objects.filter(user=request.user)

    return render(request, "dashboard.html", {
        "budget": user_budget,
        "transactions": transactions
    })

# Home view - Redirect to dashboard if logged in, else show home page
def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')  # Redirect to the dashboard if logged in
    return render(request, 'home.html')  # Show home page if not logged in

# Transactions view - List of user's transactions
@login_required
def transactions(request):
    transactions = Transaction.objects.filter(user=request.user).order_by("-date")
    return render(request, "transactions.html", {"transactions": transactions})

# Chatbot view - OpenAI integration
def chatbot(request):
    if request.method == "POST":
        user_message = request.POST.get("message")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_message}],
        )
        bot_response = response["choices"][0]["message"]["content"]
        return render(request, "chatbot.html", {"bot_response": bot_response})
    return render(request, "chatbot.html")

# Registration view - Register a new user
def register(request):
    if request.user.is_authenticated:
        return redirect('dashboard')  # Redirect to dashboard if already logged in
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)
            return redirect('dashboard')  # Redirect to dashboard after successful registration
    else:
        form = UserRegistrationForm()
    return render(request, 'registration/register.html', {'form': form})

# Login view - Log in an existing user
def login_user(request):
    if request.user.is_authenticated:
        return redirect('dashboard')  # Redirect to dashboard if already logged in
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')  # Redirect to dashboard after successful login
        else:
            messages.error(request, "Invalid username or password")
            return redirect('login')  # Stay on the login page if authentication fails
    return render(request, 'authentication/login.html')
