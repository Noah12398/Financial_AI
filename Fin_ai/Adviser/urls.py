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

]

