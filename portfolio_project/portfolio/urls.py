from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("banks/", views.bank_sector, name="bank_sector"),
    path("banks/<str:symbol>/", views.bank_detail, name="bank_detail"),
    # Portfolio URLs
    path("portfolio/", views.portfolio_list, name="portfolio_list"),
    path("portfolio/create/", views.create_portfolio, name="create_portfolio"),
    path("portfolio/<int:portfolio_id>/", views.portfolio_detail, name="portfolio_detail"),
    path("portfolio/<int:portfolio_id>/add/", views.add_stock, name="add_stock"),
    path("portfolio/stock/<int:stock_id>/delete/", views.delete_stock, name="delete_stock"),
    path("portfolio/<int:portfolio_id>/cluster/", views.portfolio_cluster, name="portfolio_cluster"),
    path("api/portfolios/", views.get_portfolios, name="get_portfolios"),
]
