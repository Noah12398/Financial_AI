from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.contrib import messages
import requests
from .forms import BudgetForm
from together import Together

from Adviser.forms import ExpenseForm, UserRegistrationForm
from .models import Category, Expense, Transaction, Budget
import openai
import os
from django.shortcuts import get_object_or_404

openai.api_key = os.getenv("OPENAI_API_KEY")


from django.http import JsonResponse
from .rag import get_rag_response

def chatbot_query(request):
    user_query = request.GET.get("query")
    if not user_query:
        return JsonResponse({"error": "Query parameter is required"}, status=400)

    response = get_rag_response(user_query)
    return JsonResponse({"response": response})

# Dashboard view - logged-in users only
@login_required
def dashboard(request):
    # Retrieve all budgets for the logged-in user
    budgets = Budget.objects.filter(user=request.user)

    # Calculate total budget limit
    total_limit = sum(budget.limit for budget in budgets)

    # Calculate total amount spent across all transactions
    amount_spent = sum(transaction.amount for transaction in Transaction.objects.filter(user=request.user))

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
def chatbot(request):
    bot_response = None

    if request.method == "POST":
        user_message = request.POST.get("message")
        api_key = os.getenv("TOGETHER_API_KEY")

        # Initialize Together client with the API key
        client = Together(api_key=api_key)

        # Call Llama 3.1 API via Together SDK
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
            messages=[{"role": "user", "content": user_message}],
            temperature=0.5,
            max_tokens=500
        )

        # Extract the chatbot's response
        bot_response = response.choices[0].message.content if response.choices else "No response from the AI."

    return render(request, "chatbot.html", {"bot_response": bot_response})
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
def add_expense(request):
    if request.method == 'POST':
        form = ExpenseForm(request.POST)
        if form.is_valid():
            expense = form.save(commit=False)
            expense.user = request.user  # Assign the logged-in user to the expense
            expense.save()
            return redirect('transactions')  # Redirect to transactions page or another page
    else:
        form = ExpenseForm()
    
    return render(request, 'add_expense.html', {'form': form})

@login_required
def transactions_view(request):
    # Fetch transactions from the database for the logged-in user
    transactions = Transaction.objects.filter(user=request.user).order_by("-date")
    return render(request, 'transactions.html', {'transactions': transactions})

@login_required
def manage_expenses(request):
    expenses = Expense.objects.filter(user=request.user)
    return render(request, 'manage_expenses.html', {'expenses': expenses})

@login_required
def edit_expense(request, expense_id):
    expense = get_object_or_404(Expense, id=expense_id, user=request.user)
    if request.method == 'POST':
        form = ExpenseForm(request.POST, instance=expense)
        if form.is_valid():
            form.save()
            return redirect('manage_expenses')
    else:
        form = ExpenseForm(instance=expense)
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


# 🟢 Add Budget
@login_required
def add_budget(request):
    if request.method == 'POST':
        form = BudgetForm(request.POST)
        if form.is_valid():
            budget = form.save(commit=False)
            budget.user = request.user
            budget.save()
            messages.success(request, 'Budget added successfully!')
            return redirect('manage_budgets')
    else:
        form = BudgetForm()
    return render(request, 'add_budget.html', {'form': form})


# 🟡 Edit Budget
@login_required
def edit_budget(request, budget_id):
    budget = get_object_or_404(Budget, id=budget_id, user=request.user)
    if request.method == 'POST':
        form = BudgetForm(request.POST, instance=budget)
        if form.is_valid():
            form.save()
            messages.success(request, 'Budget updated successfully!')
            return redirect('manage_budgets')
    else:
        form = BudgetForm(instance=budget)
    return render(request, 'edit_budget.html', {'form': form})


# 🔴 Delete Budget
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
        if name:
            Category.objects.create(name=name)
            return redirect("manage_budgets")
    return render(request, "add_category.html")


@login_required
def transactions_view(request):
    transactions = Transaction.objects.filter(user=request.user).order_by('-date')
    return render(request, 'transactions.html', {'transactions': transactions})

@login_required
def add_transaction(request):
    if request.method == 'POST':
        category_id = request.POST.get('category')
        amount = request.POST.get('amount')
        description = request.POST.get('description')

        if category_id and amount and description:
            category = Category.objects.get(id=category_id)
            Transaction.objects.create(
                user=request.user,
                category=category,
                amount=amount,
                description=description
            )
            return redirect('transactions')
    categories = Category.objects.all()
    return render(request, 'add_transaction.html', {'categories': categories})

@login_required
def delete_transaction(request, transaction_id):
    transaction = get_object_or_404(Transaction, id=transaction_id, user=request.user)
    if request.method == 'POST':
        transaction.delete()
        return redirect('transactions')
    return render(request, 'delete_transaction.html', {'transaction': transaction})