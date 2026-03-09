from django.db import models
from django.contrib.auth.models import User

# Create your models here.

# --------------------------------
# PORTFOLIO MODELS
# --------------------------------
class Portfolio(models.Model):
    name = models.CharField(max_length=100)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    @property
    def total_value(self):
        total = 0
        for stock in self.stocks.all():
            total += stock.current_value
        return total
    
    @property
    def total_stocks(self):
        return self.stocks.count()

class PortfolioStock(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='stocks')
    stock_symbol = models.CharField(max_length=20)
    stock_name = models.CharField(max_length=100)
    quantity = models.IntegerField(default=0)
    purchase_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    added_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.stock_name} ({self.stock_symbol})"
    
    @property
    def current_price(self):
        try:
            import yfinance as yf
            ticker = yf.Ticker(self.stock_symbol + ".NS")
            price = ticker.info.get('currentPrice')
            return price if price else 0
        except:
            return 0
    
    @property
    def current_value(self):
        return self.quantity * self.current_price
