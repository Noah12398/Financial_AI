from django.db import models
from django.contrib.auth.models import User

class Expense(models.Model):
    CATEGORY_CHOICES = [
        ('food', 'Food'),
        ('rent', 'Rent'),
        ('shopping', 'Shopping'),
        ('transport', 'Transport'),
        # Add more categories as needed
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.CharField(max_length=100, choices=CATEGORY_CHOICES)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField()
    
    def __str__(self):
        return f"{self.user.username} - {self.category} - {self.amount}"

class Transaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Added user field
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.CharField(max_length=255)
    date = models.DateField()

    def __str__(self):
        return f"{self.description} - {self.amount}"


class Category(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Budget(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    limit = models.DecimalField(max_digits=10, decimal_places=2)
    class Meta:
        db_table = 'BudgetTable'
    def __str__(self):
        return f"{self.user.username} - {self.category.name} - Limit: {self.limit}"


