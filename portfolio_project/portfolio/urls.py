from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("banks/", views.bank_sector, name="bank_sector"),
    path("banks/<str:symbol>/", views.bank_detail, name="bank_detail"),
]