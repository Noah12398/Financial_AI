from django import forms
from django.contrib.auth.models import User

from Adviser.models import Budget, Category, Expense, Transaction

from django import forms
from django.contrib.auth.models import User

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Enter password'})
    )
    password_confirm = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Confirm password'})
    )

    class Meta:
        model = User
        fields = ['username', 'email']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter username'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter email'}),
        }

    def clean_password_confirm(self):
        password = self.cleaned_data.get('password')
        password_confirm = self.cleaned_data.get('password_confirm')
        
        if password != password_confirm:
            raise forms.ValidationError("Passwords do not match")
        return password_confirm

# Adviser/forms.py
class ExpenseForm(forms.ModelForm):
    class Meta:
        model = Expense
        fields = ['category', 'amount', 'date', 'name']
        widgets = {
            'category': forms.Select(attrs={'class': 'form-control'}),
            'amount': forms.NumberInput(attrs={'class': 'form-control'}),
            'date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'name': forms.TextInput(attrs={'class': 'form-control'}),  # Add this line
        }
    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            self.fields['category'].queryset = Category.objects.filter(user=user)

class BudgetForm(forms.ModelForm):
    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            self.fields['category'].queryset = Category.objects.filter(user=user)  # Properly filter categories

    class Meta:
        model = Budget
        fields = ['category', 'limit', 'name']
        widgets = {
            'category': forms.Select(attrs={'class': 'form-control', 'onchange': 'updateBudgetName()'}),
            'limit': forms.NumberInput(attrs={'class': 'form-control'}),
            'name': forms.HiddenInput(),  # Hide the name field
        }


class TransactionForm(forms.ModelForm):
    def __init__(self, *args, user=None, **kwargs):
        super().__init__(*args, **kwargs)
        if user:
            self.fields['category'].queryset = Category.objects.filter(user=user)  # Filter categories by user

    class Meta:
        model = Transaction
        fields = ['name', 'category', 'amount', 'description']
        widgets = {
            'name': forms.TextInput(),
            'category': forms.Select(),
            'amount': forms.NumberInput(),
            'description': forms.Textarea(attrs={ 'rows': 3}),
        }

