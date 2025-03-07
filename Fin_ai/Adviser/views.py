from decimal import Decimal
import subprocess
import traceback
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.contrib import messages
import requests
from django.db.models import Sum
from .forms import BudgetForm, TransactionForm
from together import Together

from Adviser.forms import ExpenseForm, UserRegistrationForm
from .models import Category, Expense, Transaction, Budget
import openai
import os
from django.shortcuts import get_object_or_404

openai.api_key = os.getenv("OPENAI_API_KEY")


from django.http import JsonResponse
from .rag import get_rag_response

def get_total_spent(user):
    total_transactions = Transaction.objects.filter(user=user).aggregate(Sum('amount'))['amount__sum'] or 0
    total_expenses = Expense.objects.filter(user=user).aggregate(Sum('amount'))['amount__sum'] or 0
    
    return total_transactions + total_expenses
# Dashboard view - logged-in users only
@login_required
def dashboard(request):
    # Retrieve all budgets for the logged-in user
    budgets = Budget.objects.filter(user=request.user)

    # Calculate total budget limit
    total_limit = sum(budget.limit for budget in budgets)

    # Calculate total amount spent across all transactions
    amount_spent =get_total_spent(request.user)

    # Calculate remaining budget
    remaining_budget = total_limit - amount_spent

    context = {
        'user': request.user.username,
        'total_limit': total_limit,
        'amount_spent': amount_spent,
        'remaining_budget': max(remaining_budget, 0),  # Avoid negative values
    }
    return render(request, 'dashboard.html', context)

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

def chatbot_query(request):
    if request.method == "POST":
        user_query = request.POST.get("query")
    else:
        user_query = request.GET.get("query")

    if not user_query:
        return render(request, "chatbot.html", {"bot_response": "Please enter a question!"})

    response = get_rag_response(user_query, request)

    # Format the response here if needed
    bot_response2 = response.replace("\n", "<br>")  # Add line breaks for better readability
    formatted_response = bot_response2.replace("**", "").replace("**", "")

    return render(request, "chatbot.html", {"bot_response": formatted_response})

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


@login_required
def transactions_view(request):
    # Fetch transactions from the database for the logged-in user
    transactions = Transaction.objects.filter(user=request.user).order_by("-date")
    return render(request, 'transactions.html', {'transactions': transactions})

# Expense Management
@login_required
def add_expense(request):
    if request.method == 'POST':
        form = ExpenseForm(request.POST, user=request.user)  # Pass user here
        if form.is_valid():
            expense = form.save(commit=False)
            expense.user = request.user
            expense.save()
            return redirect('manage_expenses')
    else:
        form = ExpenseForm(user=request.user)  # Pass user here as well

    return render(request, 'add_expense.html', {'form': form})

@login_required
def manage_expenses(request):
    expenses = Expense.objects.filter(user=request.user)
    return render(request, 'manage_expenses.html', {'expenses': expenses})

@login_required
def edit_expense(request, expense_id):
    expense = get_object_or_404(Expense, id=expense_id, user=request.user)

    if request.method == 'POST':
        form = ExpenseForm(request.POST, instance=expense, user=request.user)  # Pass user
        if form.is_valid():
            form.save()
            return redirect('manage_expenses')
    else:
        form = ExpenseForm(instance=expense, user=request.user)  # Pass user

    return render(request, 'edit_expense.html', {'form': form})

@login_required
def delete_expense(request, expense_id):
    expense = get_object_or_404(Expense, id=expense_id, user=request.user)
    if request.method == 'POST':
        expense.delete()
        return redirect('manage_expenses')

@login_required
def manage_budgets(request):
    budgets = Budget.objects.filter(user=request.user)
    return render(request, 'manage_budgets.html', {'budgets': budgets})

@login_required
def add_budget(request):
    if request.method == 'POST':
        form = BudgetForm(request.POST, user=request.user)
        if form.is_valid():
            budget = form.save(commit=False)
            budget.user = request.user
            budget.save()
            return redirect('manage_budgets')
    else:
        form = BudgetForm(user=request.user)
    return render(request, 'add_budget.html', {'form': form})

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import BudgetForm
from .models import Budget

@login_required
def edit_budget(request, budget_id):
    budget = get_object_or_404(Budget, id=budget_id, user=request.user)
    
    if request.method == 'POST':
        form = BudgetForm(request.POST, instance=budget, user=request.user)  # Pass user
        if form.is_valid():
            form.save()
            messages.success(request, 'Budget updated successfully!')
            return redirect('manage_budgets')
    else:
        form = BudgetForm(instance=budget, user=request.user)  # Pass user

    return render(request, 'edit_budget.html', {'form': form})

@login_required
def delete_budget(request, budget_id):
    budget = get_object_or_404(Budget, id=budget_id, user=request.user)
    if request.method == 'POST':
        budget.delete()
        messages.success(request, 'Budget deleted successfully!')
        return redirect('manage_budgets')
    return render(request, 'confirm_delete.html', {'budget': budget})
@login_required
def add_category(request):
    if request.method == "POST":
        name = request.POST.get("name")
        user = request.user
        if name:
            Category.objects.create(name=name, user=user)
            
            return redirect("manage_budgets")
    return render(request, "add_category.html")


@login_required
def transactions_view(request):
    transactions = Transaction.objects.filter(user=request.user).order_by('-date')
    return render(request, 'transactions.html', {'transactions': transactions})


@login_required
def add_transaction(request):
    if request.method == "POST":
        category_id, amount, description, name = request.POST.get('category_id'), request.POST.get('amount'), request.POST.get('description'), request.POST.get('category_name')
        if not name:
            return render(request, 'add_transaction.html', {'categories': Category.objects.filter(user=request.user), 'error': "Transaction name is required."})
        category = Category.objects.filter(id=category_id, user=request.user).first() if category_id else None
        Transaction.objects.create(user=request.user, category=category, amount=Decimal(amount), description=description, name=name)
        return redirect('transactions')
    return render(request, 'add_transaction.html', {'categories': Category.objects.filter(user=request.user)})

@login_required
def edit_transaction(request, transaction_id):
    transaction = get_object_or_404(Transaction, id=transaction_id, user=request.user)

    if request.method == 'POST':
        form = TransactionForm(request.POST, instance=transaction, user=request.user)  # Pass user
        if form.is_valid():
            form.save()
            return redirect('transactions')
    else:
        form = TransactionForm(instance=transaction, user=request.user)  # Pass user

    return render(request, 'edit_transaction.html', {'form': form})

@login_required
def delete_transaction(request, transaction_id):
    transaction = get_object_or_404(Transaction, id=transaction_id, user=request.user)
    if request.method == 'POST':
        transaction.delete()
        return redirect('transactions')
    return render(request, 'delete_transaction.html', {'transaction': transaction})
