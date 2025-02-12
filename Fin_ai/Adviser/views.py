from django.shortcuts import render
from .models import Transaction, Budget
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def dashboard(request):
    user_budget = Budget.objects.get(user=request.user)
    transactions = Transaction.objects.filter(user=request.user)
    return render(request, "dashboard.html", {"budget": user_budget, "transactions": transactions})

def transactions(request):
    transactions = Transaction.objects.filter(user=request.user).order_by("-date")
    return render(request, "transactions.html", {"transactions": transactions})

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
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from .models import Budget

@login_required  # Ensure the user is logged in
def dashboard(request):
    try:
        user_budget = Budget.objects.get(user=request.user)
    except Budget.DoesNotExist:
        user_budget = None  # Handle cases where no budget exists for the user

    return render(request, 'dashboard.html', {'user_budget': user_budget})
