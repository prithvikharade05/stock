from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib import messages
from django.views.decorators.http import require_POST
from .models import Portfolio, PortfolioStock
from .ml_engine import cluster_portfolio
from .arima_engine import run_arima_forecast, convert_indian_symbol
from .cnn_lstm_model import run_cnn_lstm_forecast, convert_symbol
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
# RETURNS CALCULATION
# --------------------------------
def calculate_returns(symbol):
    """
    Calculate returns for various time periods.
    
    Args:
        symbol: Stock symbol (e.g., 'HDFCBANK')
    
    Returns:
        Dictionary with return percentages for 1W, 1M, 3M, 6M, YTD, 1Y, 3Y, 5Y
    """
    import datetime as dt
    
    returns = {
        "1W": None,
        "1M": None,
        "3M": None,
        "6M": None,
        "YTD": None,
        "1Y": None,
        "3Y": None,
        "5Y": None,
    }
    
    try:
        # Fetch maximum historical data for accurate calculations
        ticker = yf.Ticker(symbol + ".NS")
        hist = ticker.history(period="5y")
        
        if hist is None or hist.empty or "Close" not in hist.columns:
            return returns
        
        close_prices = hist["Close"].dropna()
        
        if len(close_prices) < 2:
            return returns
        
        latest_price = close_prices.iloc[-1]
        today = close_prices.index[-1]
        
        # Helper function to calculate percentage return
        def calc_return(start_price, end_price):
            if start_price is None or end_price is None:
                return None
            if start_price <= 0 or not math.isfinite(start_price):
                return None
            if end_price <= 0 or not math.isfinite(end_price):
                return None
            return round(((end_price - start_price) / start_price) * 100, 2)
        
        # Helper to get price X days ago
        def get_price_days_ago(days):
            target_date = today - dt.timedelta(days=days)
            # Find the closest date on or before target_date
            past_prices = close_prices[close_prices.index <= target_date]
            if len(past_prices) > 0:
                return past_prices.iloc[-1]
            return None
        
        # Helper to get price at start of year
        def get_price_ytd():
            year_start = dt.datetime(today.year, 1, 1)
            ytd_prices = close_prices[close_prices.index >= year_start]
            if len(ytd_prices) > 0:
                return ytd_prices.iloc[0]
            return None
        
        # 1 Week return (7 days)
        price_1w = get_price_days_ago(7)
        returns["1W"] = calc_return(price_1w, latest_price)
        
        # 1 Month return (~30 days)
        price_1m = get_price_days_ago(30)
        returns["1M"] = calc_return(price_1m, latest_price)
        
        # 3 Month return (~90 days)
        price_3m = get_price_days_ago(90)
        returns["3M"] = calc_return(price_3m, latest_price)
        
        # 6 Month return (~180 days)
        price_6m = get_price_days_ago(180)
        returns["6M"] = calc_return(price_6m, latest_price)
        
        # Year-to-Date return
        price_ytd = get_price_ytd()
        returns["YTD"] = calc_return(price_ytd, latest_price)
        
        # 1 Year return (~365 days)
        price_1y = get_price_days_ago(365)
        returns["1Y"] = calc_return(price_1y, latest_price)
        
        # 3 Year return (~1095 days)
        price_3y = get_price_days_ago(1095)
        returns["3Y"] = calc_return(price_3y, latest_price)
        
        # 5 Year return (use first available price if 5 years not available)
        if len(close_prices) >= 2:
            first_price = close_prices.iloc[0]
            returns["5Y"] = calc_return(first_price, latest_price)
        
    except Exception as e:
        # Return default None values on any error
        pass
    
    return returns

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

    # -------- RETURNS CALCULATION --------
    stock_returns = calculate_returns(symbol.upper())

    return render(request, "portfolio/bank_detail.html", {
        "symbol": symbol.upper(),
        "period": period,
        "data": price_data,
        "quarterly": quarterly,
        "pe_data": pe_data,
        "quarterly_eps": quarterly_eps,
        "returns": stock_returns,
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

# --------------------------------
# PORTFOLIO CLUSTERING
# --------------------------------
@require_POST
def portfolio_cluster(request, portfolio_id):
    """
    API endpoint for clustering portfolio stocks.
    
    Expected POST data:
    - features: List of feature names to use
    - k: Number of clusters (2-6)
    """
    import json
    
    try:
        # Parse request body
        data = json.loads(request.body)
        
        # Get selected features
        selected_features = data.get('features', [
            'pe_ratio',
            'discount_1y_high',
            'return_1m',
            'return_3m',
            'return_6m',
            'ltp_to_1y_high_ratio'
        ])
        
        # Get K value
        k_value = int(data.get('k', 3))
        k_value = max(2, min(6, k_value))  # Clamp to 2-6
        
        # Get portfolio
        portfolio = get_object_or_404(Portfolio, id=portfolio_id)
        
        # Get stock symbols from portfolio
        stock_symbols = list(portfolio.stocks.values_list('stock_symbol', flat=True))
        
        if len(stock_symbols) < 2:
            return JsonResponse({
                'success': False,
                'message': 'Need at least 2 stocks in portfolio for clustering',
                'data': []
            })
        
        # Run clustering
        result = cluster_portfolio(stock_symbols, selected_features, k_value)
        
        return JsonResponse(result)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'message': 'Invalid JSON request',
            'data': []
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}',
            'data': []
        })

# --------------------------------
# STOCK PRICE PREDICTION
# --------------------------------
from django.views.decorators.csrf import csrf_exempt
from .regression_engine import predict_stock

def stock_prediction(request):
    """Render the stock prediction page."""
    return render(request, "portfolio/stock_prediction.html")

@csrf_exempt
@require_POST
def predict_stock_api(request):
    """
    API endpoint for stock price prediction.
    
    Expected POST data:
    - symbol: Stock ticker symbol
    - model_type: "linear" or "logistic"
    """
    import json
    
    try:
        # Parse request body
        data = json.loads(request.body)
        
        # Get parameters
        symbol = data.get('symbol', '').strip().upper()
        model_type = data.get('model_type', 'linear')
        
        # Validate symbol
        if not symbol:
            return JsonResponse({
                'success': False,
                'error': 'Please provide a stock symbol.'
            })
        
        # Validate model_type
        if model_type not in ['linear', 'logistic']:
            model_type = 'linear'
        
        # Run prediction
        result = predict_stock(symbol, model_type)
        
        return JsonResponse(result)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON request'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error: {str(e)}'
        })


# --------------------------------
# LIVE NIFTY 50 TICKER API
# --------------------------------
def live_ticker(request):
    """
    API endpoint for live NIFTY 50 stock prices.
    Returns JSON with stock symbols, prices, and changes.
    """
    # NIFTY 50 Stock Symbols
    NIFTY_50_STOCKS = [
        {"symbol": "NIFTY50", "ticker": "^NSEI", "name": "NIFTY 50"},
        {"symbol": "BANKNIFTY", "ticker": "^NSEBANK", "name": "NIFTY Bank"},
        {"symbol": "RELIANCE", "ticker": "RELIANCE.NS", "name": "Reliance Industries"},
        {"symbol": "TCS", "ticker": "TCS.NS", "name": "Tata Consultancy"},
        {"symbol": "HDFCBANK", "ticker": "HDFCBANK.NS", "name": "HDFC Bank"},
        {"symbol": "INFY", "ticker": "INFY.NS", "name": "Infosys"},
        {"symbol": "ICICIBANK", "ticker": "ICICIBANK.NS", "name": "ICICI Bank"},
        {"symbol": "KOTAKBANK", "ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank"},
        {"symbol": "SBIN", "ticker": "SBIN.NS", "name": "State Bank of India"},
        {"symbol": "LT", "ticker": "LT.NS", "name": "Larsen & Toubro"},
        {"symbol": "ITC", "ticker": "ITC.NS", "name": "ITC Limited"},
        {"symbol": "HINDUNILVR", "ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever"},
        {"symbol": "ASIANPAINT", "ticker": "ASIANPAINT.NS", "name": "Asian Paints"},
        {"symbol": "BAJFINANCE", "ticker": "BAJFINANCE.NS", "name": "Bajaj Finance"},
        {"symbol": "MARUTI", "ticker": "MARUTI.NS", "name": "Maruti Suzuki"},
        {"symbol": "M&M", "ticker": "M&M.NS", "name": "Mahindra & Mahindra"},
        {"symbol": "SUNPHARMA", "ticker": "SUNPHARMA.NS", "name": "Sun Pharma"},
        {"symbol": "TATASTEEL", "ticker": "TATASTEEL.NS", "name": "Tata Steel"},
    ]
    
    stocks_data = []
    
    for stock in NIFTY_50_STOCKS:
        try:
            ticker = yf.Ticker(stock["ticker"])
            info = ticker.info
            
            # Get current price
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            
            # Get previous close for change calculation
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
            
            # Calculate change percentage
            change_pct = None
            if current_price and prev_close and prev_close > 0:
                change_pct = round(((current_price - prev_close) / prev_close) * 100, 2)
            
            if current_price:
                stocks_data.append({
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "price": round(float(current_price), 2),
                    "change": change_pct
                })
        except Exception as e:
            # Skip stocks that fail to load
            continue
    
    return JsonResponse({
        "success": True,
        "stocks": stocks_data,
        "timestamp": pd.Timestamp.now().isoformat()
    })

# --------------------------------
# CNN + LSTM DEEP LEARNING PREDICTION
# --------------------------------

def cnn_lstm_prediction(request):
    """Render the CNN+LSTM prediction page."""
    return render(request, "portfolio/cnn_lstm_prediction.html")


@csrf_exempt
@require_POST
def cnn_lstm_predict_api(request):
    """
    API endpoint for CNN+LSTM stock price prediction.
    
    Expected POST data:
    - symbol: Stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'BTC-USD')
    """
    import json
    
    try:
        # Parse request body
        data = json.loads(request.body)
        
        # Get symbol
        symbol = data.get('symbol', '').strip().upper()
        
        # Validate symbol
        if not symbol:
            return JsonResponse({
                'success': False,
                'error': 'Please provide a stock symbol.'
            })
        
        # Convert to appropriate ticker format
        ticker = convert_symbol(symbol)
        
        # Run CNN+LSTM forecast
        result = run_cnn_lstm_forecast(ticker, period="4y", forecast_days=5)
        
        if result.get('success'):
            return JsonResponse({
                'success': True,
                'historical_dates': result.get('historical_dates', []),
                'historical_prices': result.get('historical_prices', []),
                'predictions': result.get('predictions', []),
                'prediction_dates': result.get('prediction_dates', []),
                'accuracy': result.get('accuracy'),
                'mae': result.get('mae'),
                'rmse': result.get('rmse'),
                'r2': result.get('r2'),
            })
        else:
            return JsonResponse({
                'success': False,
                'error': result.get('error', 'Failed to generate prediction')
            })
            
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON request'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error: {str(e)}'
        })

# --------------------------------
# ARIMA PRICE FORECAST
# --------------------------------

def arima_prediction(request):
    """Render the ARIMA prediction page."""
    return render(request, "portfolio/arima_prediction.html")


@csrf_exempt
@require_POST
def btc_arima_api(request):
    """
    API endpoint for BTC/USD ARIMA forecast.
    """
    import json
    
    try:
        # Run ARIMA forecast for BTC-USD
        result = run_arima_forecast("BTC-USD", period="4y", forecast_days=7)
        
        if result.get('success'):
            return JsonResponse({
                'success': True,
                'historical_dates': result.get('historical_dates', []),
                'historical_prices': result.get('historical_prices', []),
                'forecast_dates': result.get('forecast_dates', []),
                'forecast_prices': result.get('forecast_prices', []),
                'model_order': result.get('model_order', []),
            })
        else:
            return JsonResponse({
                'success': False,
                'error': result.get('error', 'Failed to generate forecast')
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error: {str(e)}'
        })


@csrf_exempt
@require_POST
def stock_arima_api(request):
    """
    API endpoint for Indian stock ARIMA forecast.
    
    Expected POST data:
    - symbol: Stock ticker symbol (e.g., 'RELIANCE', 'TCS', 'HDFCBANK')
    """
    import json
    
    try:
        # Parse request body
        data = json.loads(request.body)
        
        # Get symbol
        symbol = data.get('symbol', '').strip().upper()
        
        # Validate symbol
        if not symbol:
            return JsonResponse({
                'success': False,
                'error': 'Please provide a stock symbol.'
            })
        
        # Convert to NSE format
        ticker = convert_indian_symbol(symbol)
        
        # Run ARIMA forecast
        result = run_arima_forecast(ticker, period="4y", forecast_days=7)
        
        if result.get('success'):
            return JsonResponse({
                'success': True,
                'historical_dates': result.get('historical_dates', []),
                'historical_prices': result.get('historical_prices', []),
                'forecast_dates': result.get('forecast_dates', []),
                'forecast_prices': result.get('forecast_prices', []),
                'model_order': result.get('model_order', []),
            })
        else:
            return JsonResponse({
                'success': False,
                'error': result.get('error', 'Failed to generate forecast')
            })
            
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON request'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Error: {str(e)}'
        })
