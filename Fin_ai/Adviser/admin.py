from django.contrib import admin
from .models import Expense, Transaction, Category, Budget

# Register your models here
admin.site.register(Expense)
admin.site.register(Transaction)
admin.site.register(Category)
admin.site.register(Budget)
