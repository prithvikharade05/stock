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
    # Stock Prediction URLs
    path("stock-prediction/", views.stock_prediction, name="stock_prediction"),
    path("api/predict-stock/", views.predict_stock_api, name="predict_stock_api"),
    # ARIMA Price Forecast URLs
    path("arima-prediction/", views.arima_prediction, name="arima_prediction"),
    path("api/btc-arima/", views.btc_arima_api, name="btc_arima_api"),
    path("api/stock-arima/", views.stock_arima_api, name="stock_arima_api"),
    # CNN+LSTM Deep Learning URLs
    path("cnn-lstm-prediction/", views.cnn_lstm_prediction, name="cnn_lstm_prediction"),
    path("api/cnn-lstm-predict/", views.cnn_lstm_predict_api, name="cnn_lstm_predict_api"),
    # Live Ticker API
    path("api/live-ticker/", views.live_ticker, name="live_ticker"),
]
