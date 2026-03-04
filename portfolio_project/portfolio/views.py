from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from .models import Portfolio, PortfolioStock
import yfinance as yf
import pandas as pd
import math

# --------------------------------
# BANK UNIVERSE (NSE)
# --------------------------------
BANK_STOCKS = {
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bank of Baroda": "BANKBARODA.NS",
    "Punjab National Bank": "PNB.NS",
    "IDFC First Bank": "IDFCFIRSTB.NS",
    "Federal Bank": "FEDERALBNK.NS",
}

# --------------------------------
# HOME
# --------------------------------
def home(request):
    return render(request, "portfolio/home.html")

# --------------------------------
# BANK SECTOR TABLE
# --------------------------------
def bank_sector(request):
    banks = []

    for name, symbol in BANK_STOCKS.items():
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # 1-year historical close min/max
            year_low = None
            year_high = None
            try:
                hist = ticker.history(period="1y")
                if not hist.empty and "Close" in hist.columns:
                    year_low = float(hist["Close"].min())
                    year_high = float(hist["Close"].max())
            except Exception:
                year_low = None
                year_high = None

            current = info.get("currentPrice")
            discount = None
            try:
                if year_high and current is not None:
                    discount = round(((year_high - float(current)) / year_high) * 100, 2)
            except Exception:
                discount = None

            banks.append({
                "name": name,
                "symbol": symbol.replace(".NS", ""),
                "ltp": current,
                "high": info.get("dayHigh"),
                "low": info.get("dayLow"),
                "market_cap": round(info["marketCap"] / 1e7, 2) if info.get("marketCap") else None,
                "pe": info.get("trailingPE"),
                "year_low": year_low,
                "year_high": year_high,
                "discount_pct": discount,
            })
        except Exception:
            banks.append({
                "name": name,
                "symbol": symbol.replace(".NS", ""),
                "ltp": None,
                "high": None,
                "low": None,
                "market_cap": None,
                "pe": None,
                "year_low": None,
                "year_high": None,
                "discount_pct": None,
            })

    return render(request, "portfolio/bank_sector.html", {"banks": banks})

# --------------------------------
# RSI CALCULATION
# --------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --------------------------------
# STOCK DETAIL PAGE
# --------------------------------
def bank_detail(request, symbol):
    period = request.GET.get("period", "6mo")

    stock = yf.Ticker(symbol.upper() + ".NS")
    nifty = yf.Ticker("^NSEI")

    # -------- PRICE DATA --------
    df = stock.history(period=period)
    df_nifty = nifty.history(period=period)

    if df.empty or df_nifty.empty:
        return render(request, "portfolio/bank_detail.html", {
            "symbol": symbol.upper(),
            "period": period,
            "data": [],
            "quarterly": [],
            "pe_data": [],
            "quarterly_eps": [],
        })

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df["NIFTY_Close"] = df_nifty["Close"]
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    df["MA30"] = df["Close"].rolling(30).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = calculate_rsi(df["Close"])
    df["RS_Line"] = df["Close"] / df["NIFTY_Close"]

    df.dropna(inplace=True)
    df["Date"] = df["Date"].astype(str)

    price_data = df.to_dict("records")

    # -------- PE RATIO --------
    pe_data = []
    eps = stock.info.get("trailingEps")

    if eps and isinstance(eps, (int, float)) and eps > 0:
        for r in price_data:
            pe = r["Close"] / eps
            if math.isfinite(pe):
                pe_data.append({"date": r["Date"], "pe": round(pe, 2)})

    # -------- QUARTERLY REVENUE --------
    quarterly = []
    q = stock.quarterly_income_stmt

    if q is not None and not q.empty:
        q = q.T.tail(12)
        for idx, row in q.iterrows():
            revenue = row.get("Total Revenue") or row.get("Interest Income")
            profit = row.get("Net Income")

            if pd.isna(revenue) or pd.isna(profit) or revenue <= 0:
                continue

            quarter = f"{idx.year}-Q{((idx.month - 1)//3) + 1}"

            quarterly.append({
                "quarter": quarter,
                "revenue": round(float(revenue) / 1e7, 2),
                "profit": round(float(profit) / 1e7, 2),
                "margin": round((profit / revenue) * 100, 2),
            })

    # -------- QUARTERLY EPS (LAST 3 YEARS) --------
    quarterly_eps = []
    shares = stock.info.get("sharesOutstanding")

    if q is not None and shares and shares > 0:
        for idx, row in q.iterrows():
            ni = row.get("Net Income")
            if pd.isna(ni):
                continue

            eps_val = ni / shares
            quarter = f"{idx.year}-Q{((idx.month - 1)//3) + 1}"

            quarterly_eps.append({
                "quarter": quarter,
                "eps": round(float(eps_val), 2)
            })

    return render(request, "portfolio/bank_detail.html", {
        "symbol": symbol.upper(),
        "period": period,
        "data": price_data,
        "quarterly": quarterly,
        "pe_data": pe_data,
        "quarterly_eps": quarterly_eps,
    })

# --------------------------------
# STOCK PORTFOLIO VIEWS
# --------------------------------
def portfolio_list(request):
    portfolios = Portfolio.objects.all().order_by('-created_at')
    return render(request, "portfolio/portfolio_list.html", {"portfolios": portfolios})

def create_portfolio(request):
    if request.method == "POST":
        name = request.POST.get("name")
        if name:
            portfolio = Portfolio.objects.create(name=name)
            messages.success(request, f"Portfolio '{name}' created successfully!")
            return redirect("portfolio_detail", portfolio_id=portfolio.id)
    return redirect("portfolio_list")

def portfolio_detail(request, portfolio_id):
    portfolio = get_object_or_404(Portfolio, id=portfolio_id)
    stocks = portfolio.stocks.all()
    
    # Get current prices for all stocks
    stock_data = []
    for stock in stocks:
        try:
            ticker = yf.Ticker(stock.stock_symbol + ".NS")
            current_price = ticker.info.get('currentPrice', 0)
        except:
            current_price = 0
        
        stock_data.append({
            'id': stock.id,
            'stock_symbol': stock.stock_symbol,
            'stock_name': stock.stock_name,
            'quantity': stock.quantity,
            'purchase_price': stock.purchase_price,
            'current_price': current_price,
            'current_value': stock.quantity * current_price if current_price else 0,
            'added_at': stock.added_at,
        })
    
    total_value = sum(s['current_value'] for s in stock_data)
    
    return render(request, "portfolio/portfolio_detail.html", {
        "portfolio": portfolio,
        "stocks": stock_data,
        "total_value": total_value,
    })

def add_stock(request, portfolio_id):
    if request.method == "POST":
        portfolio = get_object_or_404(Portfolio, id=portfolio_id)
        symbol = request.POST.get("stock_symbol")
        name = request.POST.get("stock_name")
        quantity = int(request.POST.get("quantity", 0))
        purchase_price = request.POST.get("purchase_price")
        
        if symbol and quantity > 0:
            # Check if stock already exists in portfolio
            existing_stock = portfolio.stocks.filter(stock_symbol=symbol).first()
            if existing_stock:
                existing_stock.quantity += quantity
                if purchase_price:
                    existing_stock.purchase_price = purchase_price
                existing_stock.save()
                messages.success(request, f"Updated {name} quantity in portfolio!")
            else:
                PortfolioStock.objects.create(
                    portfolio=portfolio,
                    stock_symbol=symbol,
                    stock_name=name,
                    quantity=quantity,
                    purchase_price=purchase_price if purchase_price else None
                )
                messages.success(request, f"Added {name} to portfolio!")
        
        return redirect("portfolio_detail", portfolio_id=portfolio.id)
    
    return redirect("portfolio_list")

def delete_stock(request, stock_id):
    stock = get_object_or_404(PortfolioStock, id=stock_id)
    portfolio_id = stock.portfolio.id
    stock.delete()
    messages.success(request, "Stock removed from portfolio!")
    return redirect("portfolio_detail", portfolio_id=portfolio_id)

def get_portfolios(request):
    portfolios = Portfolio.objects.all().values('id', 'name')
    return JsonResponse({"portfolios": list(portfolios)})
