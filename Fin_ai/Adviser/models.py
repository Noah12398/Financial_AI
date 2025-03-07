from django.db import models
from django.contrib.auth.models import User

class Category(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Now each category is user-specific
    name = models.CharField(max_length=100)

    class Meta:
        unique_together = ('user', 'name')  # Ensures each user has unique category names

    def __str__(self):
        return f"{self.user.username} - {self.name}"

    
class Expense(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)  # ForeignKey to Category table
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    date = models.DateField()
    name = models.CharField(max_length=255)
    def __str__(self):
        return f"{self.user.username} - {self.category} - {self.amount}"




class Budget(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    limit = models.DecimalField(max_digits=10, decimal_places=2)
    name = models.CharField(max_length=255)

    class Meta:
        db_table = 'BudgetTable'  # Rename to avoid conflict

class Transaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.CharField(max_length=255)
    date = models.DateField(auto_now_add=True,editable=False)
    name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.description} - ${self.amount} on {self.date}"