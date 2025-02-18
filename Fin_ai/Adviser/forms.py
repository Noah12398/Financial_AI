from django import forms
from django.contrib.auth.models import User

from Adviser.models import Budget, Category, Expense

class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    password_confirm = forms.CharField(widget=forms.PasswordInput)
    
    class Meta:
        model = User
        fields = ['username', 'email']
    
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
        fields = ['category', 'amount', 'date']


class BudgetForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['category'].queryset = Category.objects.all()

    class Meta:
        model = Budget
        fields = ['category', 'limit']
        widgets = {
            'category': forms.Select(attrs={'class': 'form-control'}),
            'limit': forms.NumberInput(attrs={'class': 'form-control'}),
        }