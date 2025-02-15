from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path("", views.register, name="register"),
    path("transactions/", views.transactions, name="transactions"),
    path("chatbot/", views.chatbot, name="chatbot"),
    path('register/', views.register, name='register'),
    path("dashboard/", views.dashboard, name="dashboard"),  # Dashboard path
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('expenses/', views.manage_expenses, name='manage_expenses'),
    path('expenses/add/', views.add_expense, name='add_expense'),
    path('expenses/edit/<int:expense_id>/', views.edit_expense, name='edit_expense'),
    path('expenses/delete/<int:expense_id>/', views.delete_expense, name='delete_expense'),
    path('budget/', views.manage_budgets, name='manage_budgets'),
    path('budgets/add/', views.add_budget, name='add_budget'),
    path('budgets/edit/<int:budget_id>/', views.edit_budget, name='edit_budget'),
    path('budgets/delete/<int:budget_id>/', views.delete_budget, name='delete_budget'),
]

